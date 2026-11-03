---
layout: post
title: "TRL에서 Hub Bucket으로 1조 매개변수 전송: 델타 가중치 동기화"
description: "TRL의 Delta Weight Sync가 Hub Bucket을 활용해 대규모 모델 체크포인트를 효율적으로 동기화하는 방법을 설명합니다."
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/delta-weight-sync/thumbnail.png
image: assets/images/blog/posts/2026-05-27-delta-weight-sync/thumbnail.png
authors:
  - user: aminediroHF
  - user: qgallouedec
  - user: kashif
  - user: lewtun
  - user: edbeeching
  - user: albertvillanova
  - user: lvwerra
  - user: sergiopaniego
slug: "delta-weight-sync"
source_url: "https://huggingface.co/blog/delta-weight-sync"
source_published_date: "2026-05-27"
source_published_at: "2026-05-27T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Shipping a Trillion Parameters With a Hub Bucket: Delta Weight Sync in TRL](https://huggingface.co/blog/delta-weight-sync)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/delta-weight-sync -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# TRL에서 Hub Bucket으로 1조 매개변수 전송: 델타 가중치 동기화

> **TL;DR**, 왜냐하면 당신은 모델을 훈련해야 하고 우리는 그것을 존중합니다:
> 
> - Async RL에는 더러운 비밀이 있습니다: 매 단계마다 트레이너는 추론 엔진으로 전체 모델을 전송해야 합니다. bf16으로 7B인 경우 14 GB입니다. 프런티어 1T 모델 체크포인트의 경우 테라바이트 규모입니다. 매 단계마다.
> - 사실은 그럴 필요가 없다는 겁니다. 두 연속 RL 옵티마이저 스텝 사이에서, **대략 99%의 bf16 가중치는 비트 동일성(bit-identical)**(최악의 경우 98% 이하)이 유지됩니다. 실제 델타는 아주 작습니다.
> - [a TRL PR](https://github.com/huggingface/trl/pull/5417)를 통해 변경된 요소만 인코딩하는 **희소 safetensors 파일**을 업로드하고 **Hugging Face Bucket**에 업로드하고 vLLM이 가져가도록 합니다. Qwen3-0.6B에서 매 스텝 페이로드는 1.2 GB에서 **20~35 MB**로 떨어집니다.
> - 하이라이트: 우리는 한 대의 박스에 트레이너, **Hugging Face Space**에 vLLM, 다른 Space에 Wordle 환경이 위치하고, 가중치는 하나의 Hub bucket를 통해 흐르게 하는 완전 분리형 훈련을 수행했습니다. 공유 클러스터도, RDMA도, VPN도 없습니다.
>
> Async RL은 이제 훨씬 저렴해졌습니다. 계속 읽으세요.

<figure class="image text-center">
  <iframe src="https://aminedirohf-delta-weight-sync-timeline.static.hf.space" width="100%" height="460" frameborder="0" scrolling="no"></iframe>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">동일한 가중치를 전송하는 두 가지 방식입니다. 빨간색은 토큰이 생성되지 않는 실제 경과 시간을 나타냅니다.</figcaption>
</figure>

---

## 1. 원 테라바이트 문제 {#section-1}

이전에 [the landscape of async RL training](https://huggingface.co/blog/async-rl-training-landscape)에 관한 글을 읽으셨다면 핵심은 이미 아실 겁니다. 어떤 방식으로 \"actor model\"이라고 표기하든, NCCL 백엔드의 색상이 어떻게 칠해져 있든, 결국 같은 뿌리 문제에 부딪힙니다: **가중치 동기화**.

추론 엔진은 N번째 스텝의 정책을 말합니다. 트레이너는 막 끝난 N+1 스텝의 새 가중치를 다른 쪽으로 옮겨야 하며, 추론 엔진이 완전히 잘못된 정책으로 벗어나기 시작하기 전에 반대 편으로 이동해야 합니다. 이것은 동기식이든 비동기식이든 중요한 경로에 놓여 있습니다: 차단 전송은 토큰을 생성하지 않는 GPU의 비활성 대기 시간으로 낭비됩니다. 희소 델타 경로를 사용하면 그 비활성 시간을 초 단위로 줄이고, 트레이너는 추론 엔진이 준비되었는지 기다릴 필요 없이, 옵티마이저 스텝이 끝나는 순간 "weights ready"를 게시하고 가중치를 공유 버킷에 업로드하는 반면, 추론 엔진은 자신의 시간에 가져갑니다.

Fireworks는 이 주제에 대해 아주 기억에 남는 수치를 제시합니다 [Frontier RL Is Cheaper Than You Think](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think): frontier 1T 매개변수 체크포인트를 fp8로 설정한 경우(그들의 설정), 전체 스냅샷은 **1024 GiB**이며, 이는 일반적인 지혜가 롤아웃 플릿을 업데이트할 때마다 전송해야 한다고 말하는 수치입니다. 이 정도 수치는 사람들이 메가-클러스터, RDMA 패브릭, 그리고 지역 간 전용 링크를 그리도록 만듭니다. 인접 체크포인트 간의 평균 델타는 **20.3 GiB, 즉 전체 모델의 1.98%**이고, "bf16 형식의 가중치 중 98% 이상이 연속 체크포인트 간에 비트 등가를 유지한다"고 측정되었습니다.

Cursor의 [Composer 2 report](https://huggingface.co/papers/2603.24477)가 평행한 이야기를 들려줍니다. 그들은 훈련과 추론을 서로 다른 지역에서 실행하고, 공유 S3 버킷(그들의 정확한 표현)으로 이를 함께 엮습니다. 트레이너는 매 학습 단계마다 압축된 가중치 차이를 업로드합니다. 각 클러스터는 공유 델타 체인에서 독립적으로 다운로드 및 재구성하며, "훈련 클러스터에 직접 연결할 필요가 없다"고 말합니다. 양쪽은 파라미터에 대해 직접 말하지 않습니다. 버킷이 와이어(전송)의 역할을 한다.

두 논문은 세 가지에 동의하며, 이 글의 나머지 부분은 사실상 오픈 소스 번역에 충실하므로 천천히 반복하고자 합니다:

1. 인접한 두 RL 스텝 사이에서는 대부분의 가중치가 실제로 변하지 않습니다.
2. 바뀐 부분만 보내면 대역폭 비용이 약 100분의 1로 줄어듭니다.
3. 이 작은 변경분을 공유 객체 저장소를 통해 전달하면 트레이너와 추론 클러스터를 같은 데이터 센터에 둘 필요가 없습니다.

이 이야기를 `pip install`로 읽을 수 있는 버전이 필요했을 뿐이다. 그래서 하나를 썼다.

## 2. bf16 RL 가중치가 거의 항상 희소한 이유 {#section-2}

모든 것을 연결하기 전에, 이 게임이 왜 실제로 이길 수 있는지 이해하는 것이 가치 있습니다. "가중치의 98%는 변하지 않는다"는 주장은 시연에서만 통하는 수가 아니라 실제 상황에서도 작동합니다. 그것은 아닙니다. 이는 RL이 사용하는 학습률에서 bf16 산술이 작동하는 방식에서 비롯됩니다.

bf16 숫자는 맨티사 비트가 7개 있습니다. 연속하는 두 개의 2의 거듭제곱 사이에는 정확히 \(2^7 = 128\)개의 표현 가능한 값이 있으며, 따라서 인접 bf16 숫자들 사이의 간격은 대략 \(|w| \cdot 2^{-7}\)입니다. 업데이트가 그 간격의 절반 아래에 있을 때, 즉 \(|\Delta w| < |w|/256\)일 때 bf16 캐스트에 의해 흡수됩니다. 이것이 PULSE가 그림 3에서 보여주는 "bf16 가시 임계값"입니다.

이제 Adam이 하는 일을 보자. RL 학습률이 예를 들어 \(3 \times 10^{-6}\)일 때, 단일 가중치에 대한 업데이트는:

\\(\\Delta w = -\\eta \\cdot \\frac{\\hat{m}}{\\sqrt{\\hat{v}} + \\epsilon}\\)

정규화된 스텝 \\(\\hat{m}/(\\sqrt{\\hat{v}}+\\epsilon)\\)은 대략 1 정도이므로 \\( |\\Delta w| \\approx \\eta \\approx 3 \\times 10^{-6}\\)입니다. 대부분의 가중치에서 \\(|w|\\)는 대략 \(10^{-2}\\)에서 \(10^{-1}\\) 사이에 위치합니다(PULSE가 대표적인 LLM 가중치에 대해 중앙값 0.019를 보고). 그 규모에서의 임계값 \\(|w|/256\\)은 대략 \(4 \\times 10^{-5}\\)에서 \(4 \\times 10^{-4}\\) 정도로, 업데이트보다 큽니다.

다시 말해 옵티마이저는 속삭이지만 bf16은 이를 듣지 못합니다. 업데이트는 반올림에 흡수되고 가중치의 바이트 표현은 바뀌지 않으므로, 추론 엔진의 관점에서는 이 가중치가 변하지 않은 것입니다. 이를 수억 개의 매개변수에 적용하면 근사 없이도 99%가 넘는 희소성을 얻을 수 있습니다.

이것은 정확히 PULSE 논문에서 형식적으로 제시된 주장과 일치합니다 ([Mihai & Belilovsky, 2026](https://huggingface.co/papers/2602.03839)). 그들은 두 가지 임계값을 정의합니다. **흡수 한계** \(10\eta\)는 Adam 업데이트의 보수적 최악의 경우이고, **유효 한계** \(\\eta\\)는 실제로 우리가 살아가는 영역입니다. **bf16 가시 임계값**은 \(|w|/256\\)입니다. 업데이트가 가시 임계값 아래에 있을 때 흡수되고 bf16 바이트는 변하지 않습니다. 그들의 그림 3은 대표적인 LLM 가중치 구름에 대해 두 임계값을 그래프로 보여주고, 결론은 분명합니다: \(\\eta = 3 \\times 10^{-6}\\)에서 흡수 한계 자체가 거의 모든 가중치의 가시 임계값 아래에 위치합니다. 그들은 Qwen2.5(0.5B/1.5B/7B), Llama-3.2-3B, Gemma-3-4B에 대해 이를 실험적으로 측정했고, 매 스텝 평균 희소도는 **약 99%이며 표준 편차는 0.2~0.4%가 400 스텝에 걸쳐 관찰됩니다**. 최악의 스텝도 98%를 넘지 않습니다. 따라서 <1%의 변경은 운 좋게 나온 것이 아니라 산술적으로 보장된 결과입니다.

우리는 이를 분석적으로 예측할 필요가 없습니다(실제로 Adam의 \(m\)와 \(v\) 통계로 변경 마스크를 예측해 보았지만 재현율은 약 30%에 불과했습니다). 어떤 바이트가 바뀌었는지만 관찰하면 됩니다. 이는 옵티마이저 스텝 전후에 바로 계산되는 매개변수별 작은 불리언 텐서입니다.

<figure class="image text-center">
  <iframe src="https://aminedirohf-delta-weight-sync-bf16-ulp.static.hf.space" width="100%" height="780" frameborder="0" scrolling="no"></iframe>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">학습률을 RL 영역까지 낮추면서 bf16으로 다시 캐스팅하는 마커가 원래 눈금으로 되돌아가는 모습을 확인해 보세요. 왼쪽 아래의 256개 요소 그리드는 작은 모델 전체에서 나타나는 효과를 집계한 것입니다.</figcaption>
</figure>

## 3. HF Buckets와 아키텍처 {#section-3}

여기서 이야기의 두 번째 요소가 등장하고, 이 글은 Fireworks/Cursor를 번역한 글에서 벗어나 Hugging Face의 글이 되기 시작합니다.

### 3.1 버킷이란 무엇인가?

**Bucket**은 고주파 객체 저장을 위해 Hugging Face Hub에 설계된 리포 타입입니다. 커밋 의식도, PR 워크플로우도, LFS의 특이사항도 없습니다. 파일을 추가하고, 파일을 나열하고, 파일을 다운로드합니다. 파이썬 인터페이스는 두 개의 함수입니다:

```python
from huggingface_hub import batch_bucket_files, download_bucket_files

# Trainer side
batch_bucket_files("my-org/wordle-deltas", add=[(buffer, "deltas/step_000042.safetensors")])

# Inference side
download_bucket_files("my-org/wordle-deltas", files=[("deltas/step_000042.safetensors", local_path)])
```


그것이 다입니다. 두 함수 호출로 가중치가 비행 중에 있습니다.

배경적으로, 버킷은 Hub의 콘텐츠 정의 청크 저장 계층인 **Xet**에 의해 뒷받침됩니다. Xet은 업로드하는 모든 파일을 살펴보고, 실제 콘텐츠에 따라 청크로 잘라(고정된 오프셋이 아니라), 버킷에 이미 있는 모든 것과 중복 제거를 수행합니다. 이 맥락에서 특히 기분 좋은 결과는, 희소 인코딩을 작성하지 않고 매 단계마다 전체 앵커를 업로드하더라도 Xet은 여전히 변경된 청크만 전송한다는 점입니다. 희소 인코딩 + Xet 스택: 이동된 부분에 대해서만 비용을 지불하고, 한 번만 비용을 지불합니다.

이것은 Fireworks와 Cursor가 도달하는 "공유 S3 버킷"의 오픈 소스 버전과 동일하지만, 저장 계층은 이미 콘텐츠 해싱을 인식하고, 기존의 Hugging Face 토큰은 이미 권한이 있으며, 다른 스택(Spaces, 데이터셋, 모델)과도 네이티브하게 결합합니다.

### 3.2 세 상자

전체 아키텍처는 정확히 세 개의 상자와 하나의 공유 기저층으로 구성됩니다:

- **Trainer.** 원하는 곳 어디든. 하나의 GPU, 여덟 개의 GPU, USB에 연결된 H100이 있는 노트북도 상관없습니다. 모델 가중치를 소유하고, 옵티마이저를 실행하며, 희소 델타를 방출합니다.
- **HF Bucket.** 하나의 리포지토리, 두 접두사: `anchors/`은 때때로 전체 스냅샷, `deltas/`은 그 사이의 희소 패치에 사용됩니다. 이것이 양측이 합의하는 유일한 항목입니다.
- **vLLM 롤아웃 서버.** 원하는 곳 어디든, 그리고 중요하게는 반드시 트레이너가 있는 곳일 필요는 없습니다. 버킷에서 데이터를 가져와 델타를 적용하고 롤아웃을 제공합니다.
- **Environment.** 롤아웃 서버에 일반적인 방식으로 연결(HTTP, 함수 호출 등)됩니다.

내면화해야 할 특징은 Cursor의 논문에서도 강하게 주장했고 이 글의 본문에서도 그대로 유지되는 점입니다: **트레이너와 롤아웃 서버는 가중치에 대해 서로 이야기하지 않는다**. 그들은 `{"repo_id": ..., "filename": ...}`를 담은 아주 작은 POST만 주고받으며, 그것이 전체 제어 Plane입니다. 실제 바이트 전송은 양측과 버킷 사이에서 병렬로 이루어지며, 공유 네트워크 패브릭은 필요하지 않습니다.

실제로 이것이 중요한 이유:

- 롤아웃 서버는 다른 지역에 있을 수 있고, 다른 클라우드에 있을 수 있으며, Hugging Face Space 뒤에 NAT로 있을 수 있습니다. 상관없습니다.
- N개의 추론 복제본은 같은 버킷에서 같은 델타를 내려받을 수 있으며, Xet는 바이트를 중복 제거합니다.
- 트레이너는 추론 복제본의 존재 여부나 위치를 알 필요가 없고, 하나가 고장나도 문제가 되지 않습니다.

트레이너가 쓰고, 복제본이 읽습니다. Hugging Face Hub가 연결 작업을 처리합니다.

## 4. 프로토콜 {#section-4}

이제 후드를 벗겨보겠습니다. 프로토콜은 네 부분으로 구성됩니다: 와이어 포맷, 버킷 레이아웃, vLLM 확장의 30줄, 트레이너 측 변경 탐지기. honestly, 코드가 생각보다 적습니다.

### 4.1 Safetensors를 와이어 포맷으로

우리는 디스크에 저장되고 와이어 포맷으로 사용할 [safetensors](https://github.com/huggingface/safetensors)를 선택했습니다. 이것은 이미 Hub의 표준 체크포인트 포맷이며, 거의 모든 프레임워크가 이를 읽을 수 있고, 헤더에는 임의의 문자열 메타데이터를 담을 수 있습니다. 그 메타데이터 필드가 바로 프로토콜을 숨기는 위치입니다.

버킷에는 두 가지 종류의 파일이 있습니다.

**앵커**는 일반 체크포인트처럼 보입니다: 파라미터당 하나의 텐서, 전체 bf16 가중치, (N)회 동기화마다 기록합니다(기본값은 N=10).

```
anchors/step_000010.safetensors
  ├── model.layers.0.self_attn.q_proj.weight   (bf16, full)
  ├── model.layers.0.self_attn.k_proj.weight   (bf16, full)
  └── ...
metadata:
  sparse=False, model_version=10, sparsity=0.0
```


**델타**가 핵심 포인트입니다. 실제로 변경된 각 파라미터에 대해 두 항목을 저장합니다: 요소 인덱스의 평탄한 int32 텐서와 해당 인덱스의 값을 담은 bf16 텐서.

```
deltas/step_000011.safetensors
  ├── model.layers.0.self_attn.q_proj.weight.indices   (int32, [num_changed])
  ├── model.layers.0.self_attn.q_proj.weight.values    (bf16,  [num_changed])
  ├── model.layers.0.mlp.gate_proj.weight.indices
  ├── model.layers.0.mlp.gate_proj.weight.values
  └── ...
metadata:
  sparse=True, model_version=11, sparsity=0.9938, changed_params=[...]
```


이 선택의 몇 가지 멋진 결과:

- 델타는 _파일_입니다. Python에서 `safe_open(...)`로 열어 그 안의 모든 텐서를 검사할 수 있습니다. 독점적 프레이밍이나 길이 접두사, 버전 핸드쉐이크가 없습니다.
- 메타데이터는 스스로를 설명합니다. 수신자는 `sparse=True/False`를 읽고 분기합니다. 별도의 매니페스트가 없습니다.
- 추론 측에서 mmap를 통한 제로 카피이므로, 몇 초마다 이 작업을 수행할 때 특히 중요합니다.

주기는 간단합니다: 매 N번째 스텝마다 앵커를 만들고, 그 사이에 델타를 넣습니다. 두 파일은 각각 `anchors/` 및 `deltas/` 접두사 아래 같은 버킷에 들어갑니다. 새로운 추론 복제본은 가장 최근의 앵커를 가져오고 그 뒤의 델타를 재생합니다.

<figure class="image text-center">
  <iframe src="https://aminedirohf-delta-weight-sync-anchor-delta.static.hf.space" width="100%" height="600" frameborder="0" scrolling="no"></iframe>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">Ten training steps. Anchor (full snapshot) on step 1 and step 6, sparse delta on every other step. Files land in the bucket as you watch.</figcaption>
</figure>

### 4.2 트레이너 측: 옵티마이저 훅으로부터의 불리언 마스크

트레이너는 어떤 bf16 요소가 실제로 반전되었는지 알아야 합니다. 이를 위해 옵티마이저에 사전-스텝(pre-step)과 사후-스텝(post-step) 훅을 등록하는 아주 작은 `BF16ChangeDetector`를 사용합니다:

```python
class BF16ChangeDetector:
    def __init__(self, model, optimizer):
        self._pre_step_bf16: dict[str, torch.Tensor] = {}
        self._validated_masks: dict[str, torch.Tensor] = {}
        optimizer.register_step_pre_hook(self._pre_step_hook)
        optimizer.register_step_post_hook(self._post_step_hook)

    def _pre_step_hook(self, opt, args, kwargs):
        for p in self._params:
            self._pre_step_bf16[name_of(p)] = p.detach().to(torch.bfloat16).cpu().clone()

    def _post_step_hook(self, opt, args, kwargs):
        for p in self._params:
            self._validated_masks[name_of(p)] = (
                p.detach().to(torch.bfloat16).cpu() != self._pre_step_bf16[name_of(p)]
            )
```


PR의 실제 코드에는 약간의 배선이 더 있습니다(Accelerate가 이를 서로 다른 Python 객체로 래핑하기 때문에 파라미터 객체를 모델 매개변수에 매칭하는 `data_ptr()` 등), 그러나 아이디어는 냅킨 위에서도 맞습니다: 스냅샷, 스텝, 차이.

이것이 실제 진실입니다. 우리는 마스크를 Adam의 \(m\)과 \(v\) 통계로 예측하는 더 우아한 경로를 시도했지만 기억은 약 30% 정도였습니다(나중에 더 자세히 다룹니다). 따라서 바이트를 비교하는 간단한 방법을 사용합니다. 트레이너 측에서 bf16 CPU 스냅샷 한 번만 비용으로 지불합니다.

새로운 `_sync_weight` 흐름의 네 가지 단계는:

1. 추론이 계속 실행되는 동안 업로드합니다. 트레이너는 마스킹된 요소를 safetensors 버퍼로 인코딩하고 이를 버킷에 푸시합니다. 이 단계 동안 vLLM은 여전히 기존 정책을 문제 없이 서비스합니다.
2. vLLM을 일시 중지합니다. 짧은 HTTP 호출, 수백 밀리초.
3. `/update_weights`를 신호합니다. 버킷 좌표를 보냅니다. vLLM이 다운로드하고, 델타를 적용하고, 응답합니다.
4. 재개합니다. vLLM이 다시 작동합니다.

로그 라인이 이야기를 들려줍니다:

```
Delta: 1234567/200000000 elements changed (sparsity=99.38%)
[delta_engine] uploaded user/wordle-deltas/deltas/step_000042.safetensors (27.4 MB, ...)
Weight sync: done. Total 9.4s (inference paused 1.1s)
```


중요한 줄은 괄호 안에 있습니다. 추론은 **1.1초** 동안 일시 중지되었습니다. 나머지 9.4초는 업로드에 소요되었고, 이는 롤아웃 서버가 여전히 토큰을 생성하던 동안 발생했습니다. NCCL로는 전체 동기화 시간을 일시 중지 시간으로 지불했습니다. 여기에선 이를 백그라운드 시간으로 지불합니다.

<figure class="image text-center">
  <iframe src="https://aminedirohf-delta-weight-sync-diff.static.hf.space" width="100%" height="700" frameborder="0" scrolling="no"></iframe>
  <figcaption style="font-size: 12px; color: #6b7280; margin-top: 4px;">A single sync, end to end. Switch between delta-over-bucket and NCCL broadcast, and try the replica count toggle to see the fan-out story.</figcaption>
</figure>

### 4.3 vLLM 측: 30줄 확장

vLLM은 이를 위한 깔끔한 추상화를 `WeightTransferEngine`라고 부릅니다. 우리는 본질적으로 다음과 같은 `receive_weights` 메서드를 가진 `DeltaWeightTransferEngine`를 구현합니다:

```python
def receive_weights(self, update_info, load_weights):
    download_bucket_files(update_info.repo_id, files=[(update_info.filename, local_path)])
    with safe_open(local_path, framework="pt", device="cpu") as f:
        meta = PatchMetadata.from_metadata_dict(f.metadata())
        if not meta.sparse:
            # Anchor: feed every tensor and snapshot for future deltas
            for name in f.keys():
                tensor = f.get_tensor(name)
                self._bf16_snapshot[name] = tensor.clone()
                load_weights([(name, tensor)])
        else:
            # Delta: apply (indices, values) to snapshot, hand full tensor to vLLM
            for name in json.loads(meta.changed_params):
                indices = f.get_tensor(f"{name}.indices").long()
                values = f.get_tensor(f"{name}.values")
                snap = self._bf16_snapshot[name].flatten()
                snap[indices] = values
                self._bf16_snapshot[name] = snap.reshape(self._bf16_snapshot[name].shape)
                load_weights([(name, self._bf16_snapshot[name])])
```


이를 vLLM의 `--worker-extension-cls` 플래그를 통해 등록합니다. 이는 **vLLM을 포크할 필요가 없다는 것**을 의미합니다. vLLM과 같은 이미지에 TRL을 설치하고, CLI를 우리의 클래스에 맞춰 실행하면 끝입니다.

언급할 만한 점: vLLM 자체도 희소 가중치 전송을 네이티브로 도입하려는 진행 중인 작업이 있으며, [vllm-project/vllm#40096](https://github.com/vllm-project/vllm/pull/40096). 이는 `receive_sparse_weights()`와 `trainer_send_sparse_weights()`를 `WeightTransferEngine` 기본 클래스에 직접 추가하고, 패치를 `(indices, values)`로 인코딩하고 `index_copy_()`를 통해 현장에서 적용하여 GPU/CPU 검증 왕복을 제거합니다. PR은 Qwen3-1.7B에서 희소 패치에 대해 0.16 MB를 0.40 ms에 전송했다고 보고했고, 전체 밀집 전송은 942 MB를 192 ms에 수행했습니다.

우리 구현의 한 가지 솔직한 주의점: 추론 쪽에서 전체 텐서를 재구성할 수 있도록 CPU bf16 스냅샷을 유지합니다(현재 vLLM의 `load_weights`은 전체 텐서를 기대합니다). [#40096](https://github.com/vllm-project/vllm/pull/40096)(또는 후속)이 채택되어 제자리에서 희소 `load_weights` 경로를 공개하면, 인덱스를 GPU에서 직접 적용하고 스냅샷을 제거할 수 있습니다.

## 5. Spaces에서 실제로 가동하기 {#section-5}

지금까지 설명한 모든 내용은 노트북에서도 작동하지만, Hub Bucket를 통해 가중치를 라우팅하는 이유는 트레이너와 롤아웃 서버가 서로 가까이 있을 필요가 없다는 점입니다. 그래서 네트워크를 공유하지 않는 세 대의 기계로 완전히 분리된 훈련을 수행했습니다:

- 한 대의 GPU로 구동되는 트레이너 박스.
- 확장 클래스를 탑재한 vLLM이 구동되는 Hugging Face Space(Docker SDK, L4 GPU).
- 두 번째 Hugging Face Space(CPU)에서 Wordle 환경 서버를 256개의 동시 세션으로 구동.
- 중간에 Hugging Face Hub 버킷.

설정은 실제로 몇 차례 `hf` CLI 호출에 불과합니다. vLLM Space의 `Dockerfile`은 기본적인 upstream vLLM 이미지에 `pip install trl@...` 및 엔트리포인트를 더한 것과 같습니다:

```dockerfile
FROM vllm/vllm-openai:latest
RUN pip install "trl @ git+https://github.com/huggingface/trl.git@delta-weight-sync"
ENV VLLM_SERVER_DEV_MODE=1
EXPOSE 7860
ENTRYPOINT ["vllm", "serve", "Qwen/Qwen3-1.7B", \
    "--host", "0.0.0.0", "--port", "7860", \
    "--worker-extension-cls", "trl.experimental.async_grpo.delta_engine.DeltaWorkerExtension", \
    "--weight-transfer-config", "{\"backend\":\"nccl\"}", \
    "--max-model-len", "32768", \
    "--gpu-memory-utilization", "0.8"]
```


Space로 배포하기:

```bash
hf repos create $USER/vllm-wordle-inference \
    --type space --space-sdk docker --flavor l4x1 \
    --secrets HF_TOKEN=$HF_TOKEN
hf upload $USER/vllm-wordle-inference examples/scripts/openenv/vllm_space/ --type space
```


그리고 훈련은 지구 반대편 어디서나 HTTPS로 연결할 수 있는 곳에서 시작합니다:

```bash
python examples/scripts/openenv/async_wordle.py \
    --vllm-server-url https://$USER-vllm-wordle-inference.hf.space \
    --env-url https://openenv-wordle.hf.space \
    --delta-sync-repo-id $USER/wordle-deltas \
    --model Qwen/Qwen3-1.7B
```


트레이너는 포트를 열지 않습니다. Spaces도 트레이너의 IP를 보지 못합니다. Wordle 환경도 둘의 존재를 모릅니다. 이들은 모두 Hugging Face Hub에 연결됩니다. 즉시 EOS 안전성 점검에서 훈련이 수렴했고, 이후 실제 Wordle 롤아웃에서도 보상은 상승했고, 델타 페이로드는 20~35 MB 대역에 머물렀으며, 동기화당 추론이 일시 중지되는 창은 약 1초였습니다. 전체 실행 로그는 동반 PR에 링크되어 있습니다.

## 6. 이로써 실제로 어떤 가능성이 열리나요? {#section-6}

몇 가지가 있고, 우리는 그것들이 크다고 생각합니다.

**클러스터 없이 비동기 RL 훈련.** 하나의 GPU와 Hugging Face 계정만 있으면 이제 실제로 분리된 훈련을 할 수 있습니다. 트레이너는 GPU에 있고, 롤아웃 파견은 Spaces에 있으며, 환경은 다른 Space에 있으며, 가중치는 버킷을 통해 이동합니다. 이것은 예전에는 공동 배치된 시스템(전송 용량을 해치는 제약이 많음) 또는 공유 네트워킹을 가진 실제 클러스터가 필요했습니다. 그러나 이제는 그렇지 않습니다.

**다중 복제 추론, 무료로.** 두 개의 vLLM Space를 세워도 되고, 열 개를 세워도 됩니다. 모두 같은 버킷에서 데이터를 가져옵니다. Xet는 저장소를 콘텐츠 주소 지정으로 다루므로 연속 앵커가 휴식 시 청크를 공유하고, Hub의 엣지 캐시는 같은 파일의 반복 다운로드를 저렴하게 제공합니다. 글로벌 분산 롤아웃 파견대를 원하십니까? 이제 그것은 소규모 DevOps 작업일 뿐이며 연구 프로젝트가 아닙니다.

**디버깅 가능한 와이어 포맷.** 델타는 safetensors 파일입니다. 노Notebook에서 `safe_open`로 불러오고, 키를 나열하고, 인덱스를 점검하고, 희소성을 직접 계산할 수 있습니다. 불투명한 NCCL 스트림에서 tcpdump를 수십 시간을 보낸 경험이 이를 높이 평가하게 만듭니다.

**프런티어 규모로 가는 경로.** 20~35 MB 수치는 Qwen3-0.6B에 대한 것입니다. 다이얼을 높이면 곡선이 어떻게 보일지 흥미로운 문제입니다. 냅킨 계산으로 한 번 해봅시다.

Llama-3.1-405B를 예로 들자. bf16에서 디스크상 810 GB입니다. PULSE는 RL 학습률에서 매 스텝 평균 희소도를 약 99%로 측정하므로 실제 델타는 매개변수의 약 1% 정도입니다. 그들의 배포 측정 인코딩은 7B 모델에서 108 MB에 도달하며, 이는 PULSE가 보고한 약 130배 감소에 해당합니다. 405B로 선형 확장하면 델타는 대략 매 스텝 6 GB에 도달합니다.

wall-clock 기준으로는 무엇을 얻을 수 있을까요? NCCL은 클러스터 내부에서는 빠르다고 해도, 넉넉하게 잡아 총 브로드캐스트 대역폭을 100 GB/s로 가정해 보겠습니다. 전체 동기화에는 `810 GB / 100 GB/s ≈ 8 seconds`가 걸리므로, 매 스텝마다 추론이 약 8초 동안 멈춥니다. 델타 경로를 사용하면 트레이너는 6 GB를 버킷으로 백그라운드에서 스트리밍하고, 생성은 계속되며, 롤아웃 서버의 실제 일시 중지 창은 적용 단계뿐입니다. 이 규모에서의 적용 시간은 몇 초에 불과합니다. 따라서 클러스터를 떠나도 델타는 가시 중지 시간을 약 4배 줄이고, 와이어 상의 바이트는 약 130배 줄입니다.

이제 클러스터를 떠납니다. NCCL은 클라우드 간에는 작동하지 않습니다. 만약 당신이 `us-east`의 롤아웃 파견대, 다른 하나의 `eu-west` 파견대, 그리고 어쩌면 Hugging Face Space 하나를 원한다면, 버킷 기반 경로가 유일한 경로입니다. 사용 가능한 인터넷 대역폭이 1 GB/s일 때 단일 전체 방송은 13분이 걸리지만, 델타는 6초 만에 처리합니다.

Fireworks 프레임에서 1 TB급 모델에 대한 측정 수치를 보면, 1024 GiB의 전체 스냅샷에 비해 20.3 GiB의 델타가 나타나 약 50배 감소합니다. PULSE의 더 촘촘한 희소 인코딩은 이를 더 줄일 수 있습니다(델타당 약 15 GB로 추정하면 약 65배 근접). 어쨌든 당신은 일반 오브젝트 스토리지로 가중치를 싣는 것이 해킹이 아니라 합리적인 아키텍처가 되는 영역에 들어선 것입니다.

## 7. 우리가 아직 남긴 과제 {#section-7}

완료했다고 주장하진 않습니다. 솔직한 목록은 이렇습니다.

- **Two CPU bf16 snapshots, one too many.** 트레이너는 변경 탐지기에 쓰기 위해 하나를 보관하고, 롤아웃 서버는 vLLM의 `load_weights`를 재구성하기 위해 하나를 보관합니다. 첫 번째 것은 누군가 정밀한 분석 마스크를 찾을 때까지 남아 있습니다. 두 번째는 vLLM이 sparse `load_weights` API를 얻으면 사라집니다. PR은 곧 나올 예정입니다.
- **고정된 앵커 주기.** 현재는 매 \(N\\) 스텝마다 전체 앵커를 덤프합니다. 누적 드리프트가 X를 초과할 때 앵커를 잡는 적응 정책은 긴 실행에서 앵커 비용을 줄여 줍니다.
- **다중 노드 FSDP2 트레이너.** `BF16ChangeDetector`는 프로세스당 옵티마이저 훅을 기반으로 합니다. FSDP2에 자연스럽게 일반화할 수 있을 것으로 보이지만 다중 노드 규모에서는 아직 측정하지 않았습니다. PR에는 담당자가 명시된 미완료 작업 표시가 남아 있습니다.
- **옵티마이저와의 훅 연결.** \((m, v)\\)만으로 마스크를 예측하려는 우리의 시도는 재현이 낮아, 분석적 bf16 임계값이 교과서 공식이 말하는 것보다 더 미묘한 작용을 한다는 것을 의미합니다. 이를 해결한 사람의 이야기를 듣고 싶습니다.
- **와이어 상 압축과의 스태킹.** 희소 safetensors와 청크당 gzip은 직교합니다. 아직 이 둘을 결합해 보지 않았습니다. 큰 압축 이득은 기대하지 않습니다.

## 8. 시도해보기 {#section-8}

- PR: [huggingface/trl#5417](https://github.com/huggingface/trl/pull/5417). 브랜치는 `delta-weight-sync`.
- 전체 Wordle 예제: `examples/scripts/openenv/async_wordle.py`.
- Spaces Dockerfiles: `examples/scripts/openenv/vllm_space/`와 `examples/scripts/openenv/wordle_space/`.
- 배경 읽을거리: 우리의 [async RL landscape post](https://huggingface.co/blog/async-rl-training-landscape), [Fireworks 1 TB post](https://fireworks.ai/blog/frontier-rl-is-cheaper-than-you-think), [Cursor Composer 2 report](https://huggingface.co/papers/2603.24477).
