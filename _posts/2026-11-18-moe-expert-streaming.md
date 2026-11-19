---
layout: post
title: "180B 파라미터, 32GB RAM: Transformers로 MoE 추론의 한계까지 밀어붙이기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/moe-expert-streaming/thumbnail.png
image: assets/images/blog/posts/2026-11-18-moe-expert-streaming/thumbnail.png
authors:
  - user: tarekziade
slug: "moe-expert-streaming"
source_url: "https://huggingface.co/blog/moe-expert-streaming"
source_published_date: "2026-11-18"
source_published_at: "2026-11-18T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [180B Parameters, 32GB of RAM: Pushing MoE Inference to the Limit with Transformers](https://huggingface.co/blog/moe-expert-streaming)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/moe-expert-streaming -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 180B 파라미터, 32GB RAM: Transformers로 MoE 추론의 한계까지 밀어붙이기

*SSD에서 MoE 전문가를 직접 스트리밍하면 토큰 생성 성능이 Strata의 6% 이내까지 따라간다.*

온라인 모델에 덜 의존하고 더 큰 모델을 로컬에서 실행하려고 노력해 왔다.
집에는 RTX 4090 두 개가 장착된 PC가 있고(이 PC를 **Orthrus**라고 부르자. 그리스 신화에 나오는 머리 둘 달린 사냥개다), 업무용으로는 통합 메모리 32G의 MacBook M5가 있다.

Orthrus는 Windows 11에서 실행한다. 아이들이 늘 게임을 하기 때문이다. Ubuntu와 듀얼 부트를 구성해 두었지만, 결국 아이들의 게임이 2테라바이트 디스크 대부분을 차지했다.

나는 RMD(rooms moving development)의 열렬한 팬이기도 하다. 부엌에서 커피를 홀짝이며 한 시간, 사무실 책상에서 두 시간, 그다음은 소파. 고정된 데스크톱 타워는 나와 맞지 않는다.

첫 실험은 Orthrus에서 [Qwen 3.8 27B](https://huggingface.co/Qwen/Qwen3.8-27B)를 [`llama.app`](https://llama.app/)와 함께 실행하고, [`pi`](https://pi.dev/)를 통해 노트북에서 사용하는 것이었다. 이 모델은 코딩 작업을 맡고, Opus 같은 온라인 모델은 오케스트레이션을 담당하며, 두 번째 RTX는 Neovim에서 자동 완성을 위한 FIM 모델을 실행한다.

소셜 네트워크에서 이 이야기를 했더니 매우 강한 피드백을 받았다. [Strata](https://github.com/niko1221/strata)와 [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)를 사용하라는 것이었다.

## Strata란 무엇인가? {#section-1}

Strata는 Qwen 같은 MoE 모델을 위한 추론 엔진이다. 활성 상태인 전문가만 VRAM에 남고, 나머지는 RAM으로, 또는 디스크로 이동한다. 전문가는 MoE 모델 가중치의 90% 이상을 차지하므로, 이 모델처럼 180B인 모델도 훨씬 적은 메모리의 GPU에서 실행할 수 있다. 내 RTX 4090처럼 말이다!

노트북에서도 작동한다. 정확히 그렇게 하는 [이 블로그 게시물](https://modelfit.io/blog/strata-qwen38-flash-next-mac/)을 우연히 발견했는데, 32 GB Mac M4에서 Qwen3.8-Flash-Next를 [llama.cpp](https://github.com/ggml-org/llama.cpp)보다 빠른 13.5 tok/s로 실행한다.

코딩 에이전트를 사용하는 방식에는 너무 느리지만, 흥미로운 가려움증을 긁어볼 만한 주제다. MoE 모델 중에는 MacBook의 통합 메모리에 들어가지 않으면서도 괜찮은 속도로 실행할 수 있는 모델이 아주 많다.

## 다른 엔진은 어떻게 오프로딩하는가? {#section-2}

Strata를 본격적으로 사용하기 전에, MoE 모델이 들어맞지 않을 때 llama.cpp, [vLLM](https://github.com/vllm-project/vllm), [Transformers](https://github.com/huggingface/transformers)가 어떻게 처리하는지 살펴봤다. 중요한 질문은 세 가지다. 전문가는 어디에 존재하는가, 전문가 matmul은 어디에서 실행되는가, 그리고 각 토큰마다 무엇이 이동하는가?

**llama.cpp**는 `mmap`로 GGUF 파일을 매핑하고 전문가를 CPU에 둘 수 있다(선택을 대신해 줄 `--cpu-moe`, `--n-cpu-moe N`, `-ot "exps=CPU"` 또는 `--fit on`). 이 전문가들은 CPU에서 *계산*되므로 PCIe를 가로지르는 것은 활성값뿐이며, 가중치는 절대 이동하지 않는다. 전문가가 RAM에 들어갈 때는 훌륭하다. 들어가지 않으면 페이지 캐시가 thrash한다. GPU 전문가 캐시(`--moe-cache-mib`)도 막 추가되었지만 그래프를 분할하고(토큰당 2개 분할이 50개가 된다), 모든 것이 들어맞는 경우에도 약 10%의 성능을 잃는다.

**vLLM**의 `--cpu-offload-gb N`는 N GB의 가중치를 pinned RAM에 두고, 커널이 PCIe를 통해 해당 위치에서 직접 읽는다. MoE 레이어에서는 라우팅된 전문가만 해당하지만, 캐시되는 것은 없다. 즉, 자주 사용되는 전문가는 모든 토큰마다 다시 PCIe를 통과하고, 디스크 계층도 없다. 기본적으로 오프로딩되는 가중치에는 모든 토큰마다 읽는 dense 레이어가 포함되지만, `--cpu-offload-params experts`을 사용하면 전문가로 제한할 수 있다. 거의 모든 전문가를 건드리는 vLLM의 대규모 배치에는 잘 맞지만, GPU 하나에서 사용자 한 명이 사용하는 경우에는 덜 적합하다.

**Transformers**는 [Accelerate](https://github.com/huggingface/accelerate)(`device_map="auto"`, `offload_folder`)를 통해 한 번에 하나의 *모듈*씩 오프로딩한다. 훅이 `forward` 전에 해당 모듈의 가중치를 GPU로 복사한다. MoE에서 이 모듈은 레이어 전체의 전문가 뱅크이므로, 각 토큰마다 오프로딩된 모든 레이어의 모든 전문가를 복사한 뒤 그중 8개 또는 10개를 사용한다. 캐시도 없고 라우팅 개념도 없다. 게다가 Mac에서는 CPU와 GPU가 같은 메모리를 공유하므로 “CPU로 오프로딩”해도 아무것도 확보되지 않는다.

이 모든 것을 RAM이 충분한 단일 A10G(24 GB)에서 [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)로 실행했고, Harry Mellor는 더 나은 플래그를 사용해 vLLM을 다시 실행했다.

| 엔진                                                          | 디코드 (tok/s) |
|-----------------------------------------------------------------|----------------|
| Strata, Q8_0                                                    | 82 - 88        |
| llama.cpp, Q8_0, `--fit on`                                     | 63 - 64        |
| llama.cpp, BF16, `--fit on`                                     | 35 - 36        |
| vLLM, FP8, `--cpu-offload-gb 16 --cpu-offload-params experts`   | 20.1           |
| vLLM, FP8, 동일한 플래그, `--enforce-eager`                        | 14.7           |
| vLLM, FP8, `--cpu-offload-gb 16`                                | 9.7            |
| vLLM, BF16, `--cpu-offload-gb 50 --cpu-offload-params experts`  | 7.3            |
| vLLM, BF16, `--cpu-offload-gb 50`                               | 3.3            |
| Transformers + Accelerate                                       | 0.16           |

전문가만 오프로딩하면 vLLM의 속도가 두 배가 되고, 그 위에 FP8을 사용하면 더 빨라진다(전문가당 바이트 수가 절반이고 모델의 더 많은 부분을 GPU에 올릴 수 있다). 레이어 그룹을 미리 복사하는 vLLM의 다른 오프로딩 백엔드는 1 tok/s 미만으로 실행됐다.

전문가가 RAM에 들어간다면 llama.cpp를 이기기 어렵다. 들어가지 않는다면 라우터가 선택한 것만 제한된 캐시를 통해 스트리밍해야 한다. SSD가 병목이 되고 Python이 병목이 되지 않을 만큼 빠르게, 가능하면 모델의 빠른 전문가 커널을 재사용하면서 말이다. 이것이 32 GB Mac에서 Qwen3.8-Flash-Next를 실행하는 문제이며, Transformers는 기본적으로 이를 처리할 수 없었다.

## Transformers로 32 GB Mac에서 Qwen3.8-Flash-Next 작동시키기 {#section-3}

노트북에서 Strata는 이미 내게 너무 느렸고, Transformers가 조만간 Strata를 앞설 가능성도 없었다. 하지만 Accelerate 경로는 너무 동떨어져 있어서, 내 작업 방식에는 아마 여전히 너무 느릴 것이라는 점을 알면서도 노트북에서 대형 모델을 실행하는 것을 목표로 Transformers 자체의 오프로딩을 고쳐보기로 했다.

모든 코드는 AI와 함께 만들었다. Transformers PR, 모델별 패키지, Metal 커널까지 모두 포함한다. 내가 한 일은 무엇을 시도할지 고르고, 프로파일을 읽고, 무엇을 남길지 결정하는 것이었다. 모든 내용은 다음 Space에 있다: [tarekziade/moe-expert-streaming](https://huggingface.co/spaces/tarekziade/moe-expert-streaming).

### 작게 시작하기

메모리에 들어가는 [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924)부터 시작했다. 그래서 모든 토큰을 vanilla Transformers와 비교할 수 있었다.

첫 번째 버전은 전문가를 디스크에 두고, 디바이스에 LRU 캐시를 두며, 한 번에 하나의 전문가를 실행하는 Python 루프를 사용하는 방식이었다. 모든 전문가가 이미 캐시된 상태에서는 10.1 tok/s가 나왔다. vanilla는 약 32였다. I/O는 괜찮았지만 내 루프가 문제였다. vanilla는 `grouped_mm`를 사용해 한 레이어의 모든 전문가를 하나의 커널에서 실행하기 때문이다.

이 프로젝트의 핵심 아이디어이자 해결책은 캐시를 **슬롯 풀**로 바꾸는 것이었다. 모든 레이어가 공유하는 사전 할당 디바이스 텐서 하나와, 전문가 ID에서 슬롯으로 매핑하는 레이어별 테이블을 사용하는 방식이다. 라우터의 전문가 ID를 슬롯으로 다시 매핑하면 풀은 일반적인 전문가 뱅크처럼 보이므로 `grouped_mm`가 그대로 실행된다. 사용자 지정 전문가 matmul도 필요 없다. 그 결과 vanilla의 38.1에 비해 34.6 tok/s까지 올라갔다.

### `experts_offload`

그다음 독립 실행형 로더를 `from_pretrained`의 옵션으로 바꾸고, [huggingface/transformers#50717](https://github.com/huggingface/transformers/pull/50717)에 반영했다.
시험해 보려면 해당 PR에서 Transformers를 설치한다.

```bash
pip install "git+https://github.com/huggingface/transformers.git@refs/pull/50717/head"
```


그런 다음 OLMoE에서 실제 목표인 Qwen3.8-Flash-Next로 이동했다.

```python
path = snapshot_download("tarekziade/Qwen3.8-Flash-Next-expert-streaming")

model = AutoModelForCausalLM.from_pretrained(
    f"{path}/dense",
    dtype=torch.bfloat16,
    device_map={"": "mps", "experts.*": "disk"},
    quantization_config=MetalConfig(bits=8),
    experts_offload={
        "cache_bytes": 12 << 30,
        "quantized_experts": f"{path}/experts-q3",
    },
)
```


[해당 리포지토리](https://huggingface.co/tarekziade/Qwen3.8-Flash-Next-expert-streaming)에는 프로젝트의 [변환 스크립트](https://github.com/tarekziade/moe-expert-streaming#readme) 출력이 들어 있다. dense 가중치, 3비트로 양자화된 전문가, 그리고 n-gram 테이블이다.

`"disk"`에 매핑된 전문가는 Accelerate의 오프로딩 대신 스트리밍 경로를 사용한다. 로더는 해당 텐서를 건너뛰고 MoE 레이어는 오프로딩 구현으로 전환한다. 캐시 미스는 8개의 I/O 스레드가 읽으며, 전문가당 하나의 연속 읽기를 수행한다. 다음 두 레이어의 라우터는 현재 hidden state에서 미리 실행되어 어떤 전문가를 프리패치할지 추측한다. 전문가를 오프라인에서 양자화하면 읽어야 할 데이터가 줄어든다. 표준 전문가 레이아웃을 사용하는 MoE 모델에서 작동한다.

API는 아직 Transformers 유지 관리자들과 논의 중이다. `device_map` 형식은 그 논의에서 나왔으며, 양자화된 전문가는 아마도 자체 옵션 대신 [#50245](https://github.com/huggingface/transformers/pull/50245)의 레이어별 양자화 계획을 따르게 될 것이다.

### 32 GB Mac에서 121 GB의 전문가 실행하기

Qwen3.8-Flash-Next에는 512개 전문가(활성화되는 전문가는 10개)를 가진 레이어가 48개 있으며, FP8 기준 121 GB다. 전문가를 스트리밍하는 것만으로는 충분하지 않다. 자체 디스크 경로가 필요한 51 GB의 n-gram 임베딩 테이블도 있기 때문이다. 따라서 다음과 같이 구성했다.

- 3비트로 양자화한 전문가: 디스크에서 49 GB, 디바이스에 12 GiB 캐시
- 8비트 dense 레이어(~5.4 GB, BF16 대비 perplexity +0.2%)
- n-gram 테이블은 디스크에 유지하고, 필요할 때 행을 읽음

변환에는 약 5시간이 걸린다. 로드한 뒤에는 약 18 GiB를 사용하고, 6K 프롬프트에서는 25.4 GiB까지 올라간다. 거의 한계에 다다른 상태다. 브라우저를 닫지 않으면 macOS가 스와핑을 시작한다.

모델별 구성 요소(n-gram 테이블과 Metal 커널)는 PR 위에 얹힌 작은 패키지인 [moe-expert-streaming](https://github.com/tarekziade/moe-expert-streaming)에 들어 있다. 첫 번째 버전은 약 10 tok/s로 실행됐다.

그런 다음 같은 M5에 같은 프롬프트를 사용해 Strata를 설치했다.

|                               | Transformers | Strata | 차이  |
|-------------------------------|--------------|--------|------|
| 디코드, 200 토큰              | 10.0 tok/s   | 14.8   | 1.5x |
| 프리필, 4K 프롬프트           | 68 tok/s     | 220    | 3.2x |
| 6K 문서 이후 디코드            | 7.3 tok/s    | 13.7   | 1.9x |

Strata 0.1.60은 IQ3_XXS 전문가를 사용했다. 두 실행 모두 greedy로 수행했고, thinking, MTP 및 speculative draft는 끄고 1,024토큰 프리필 청크를 사용했다. 전체 설정과 원시 결과는 Space의 [세부 정보 페이지](https://tarekziade-moe-expert-streaming.static.hf.space/details.html)에 있다.

최근 전문가 오프로딩을 지원하게 된 [vllm-metal](https://github.com/vllm-project/vllm-metal)도 시도했다. 이는 MLX를 통해 Apple GPU에서 실행되는 vLLM이다. 이 프로젝트와 마찬가지로 제한된 캐시를 통해 MLX 체크포인트에서 라우팅된 전문가를 스트리밍한다. 하지만 위에서 설명한 n-gram 문제 때문에 32 GB에서 Qwen3.8-Flash-Next를 실행할 수 없다. 스트리밍되는 것은 전문가뿐이고 n-gram 테이블은 메모리에 남기 때문이다. Hub의 MLX 체크포인트만 해도 32 GB이며, 여기에 약 5 GB의 dense 레이어와 전문가 캐시가 추가되기 전의 수치다.

## 격차 줄이기 {#section-4}

대부분의 시간은 전문가에 쓰이지 않았다. 도움이 된 것은 다음과 같다.

- **GPU 작업을 일찍 커밋하기.** MPS의 [PyTorch](https://github.com/pytorch/pytorch)는 다음 동기화까지 작은 커널을 보류하므로, Python이 작업하는 동안 GPU가 유휴 상태가 됐다. 각 블록 뒤에 커밋을 강제하자 +10%의 성능 향상이 있었다. 이 문제는 정말 PyTorch에서 수정되어야 한다.
- **Metal 커널**: macOS에서 빠른 경로가 없는 부분인 Gated DeltaNet과 hyper-connections에 사용했다.
- **Strata 자체의 가중치.** [Strata의 GGUF 가중치](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)로 전환해 토큰당 dense 가중치 트래픽을 5.4 GB에서 3.9 GB로, 개별 전문가 크기를 2.15 MB에서 1.75 MB로 줄였다. 전문가 크기별로 별도의 캐시 풀도 사용했다.
- **프리필**: 더 큰 청크를 사용하고 현재 레이어를 계산하는 동안 다음 레이어의 전문가를 읽어 68에서 약 228 tok/s까지 높였다.
- **긴 문서**: 인덱서가 선택한 블록에 대해서만 어텐션을 수행하고, 토큰마다 복사하는 대신 제자리에서 커지는 KV 캐시를 사용했다.
- **캐시 미스**: Strata는 이를 숨기지 않고 나처럼 기다린다. 어떤 캐시 정책도 LRU를 크게 앞서지 못했다. 도움이 된 것은 SSD가 그렇지 않으면 유휴 상태일 때 레이어마다 추측한 전문가를 하나만 읽는 것이었다.

현재 실행되는 토큰 하나의 MoE 레이어는 다음과 같다.

![For one layer and one token, the router picks 10 of 512 experts. Eight are already in the expert cache and two are read from the SSD into free cache slots. The experts then run on the GPU from the cache slots, while the SSD reads one guessed expert for the next layer, which enters the cache only if it's used.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-expert-streaming/expert-streaming.svg)

같은 M5에서 같은 프롬프트를 사용해 이러한 최적화를 적용한 최종 결과는 다음과 같다.

|                               | Transformers | Strata | 차이  |
|-------------------------------|--------------|--------|-------|
| 디코드, 200 토큰              | 13.9 tok/s   | 14.8   | 1.06x |
| 프리필, 4K 프롬프트           | 228 tok/s    | 220    | 1.0x  |
| 6K 문서 이후 디코드            | 12.4 tok/s   | 13.7   | 1.1x  |

이는 짧은 디코드에서 Strata보다 약 6% 느리고, 프리필은 약간 더 빠른 결과다. 그리고 이 모든 것이 Transformers를 통해 이루어졌다.

남아 있는 대부분의 시간은 호스트 왕복에 쓰이며, 토큰당 약 10 ms다. 동기화 후 다시 깨어나는 데 걸리는 시간은 그중 약 1 ms에 불과하다. 나머지는 Python이 다음 레이어를 큐에 넣는 동안 GPU가 할 일이 없어 기다리는 시간이다. Strata는 토큰 전체의 작업을 미리 큐에 넣고, 호스트가 전문가를 읽는 동안 GPU는 각 레이어에서 Metal shared event를 기다린다. PyTorch는 이를 노출하지 않는다. CPU가 쓰는 플래그를 커널이 확인하며 대기하도록 만들어 흉내 내보았지만, GPU가 그 쓰기를 확인하는 데 1초가 걸리기도 했고, 영원히 확인하지 않기도 했다. 진정한 해결책은 PyTorch의 MPS 스트림에 네이티브 op를 추가하는 것이겠지만 멈춤이 발생할 위험이 있어 구현하지 않았다.

## 결론 {#section-5}

약 14 tok/s로 실행되어 Strata에 가깝지만, 내가 코딩 에이전트를 사용하는 방식에는 여전히 너무 느리다. 진짜 작업은 여전히 Orthrus가 맡고 있다.

Transformers는 Strata를 앞서지 못했고, 남은 약 6%를 줄이려면 스택을 더 깊이 변경해야 한다. 그 시작점은 레이어별 동기화를 제어하는 것인데, PyTorch는 현재 이를 제공하지 않는다. 흥미로운 점은 어떻게 이 정도까지 따라잡았는가이다. 라우팅된 전문가만 스트리밍하고, `grouped_mm`가 그대로 실행되는 제한된 캐시를 사용하며, 몇 개의 목적에 맞춘 Metal 커널을 사용했다. 프로젝트에는 C++ 확장이 없다. Python과 `torch.mps.compile_shader`로 런타임에 컴파일되는 Metal 커널, 그리고 Hub에서 미리 빌드된 [커널](https://huggingface.co/kernels-community)만 사용한다.
이 실험을 위해 작성한 모든 Metal 커널은 이제 [Hub에 게시되어](https://tarekziade-moe-expert-streaming.static.hf.space/details.html#published-kernels) 있으며, 프로젝트는 그곳에서 커널을 로드한다.

하지만 Strata를 앞서는 것이 유일한 목표였던 것은 아니다. 이 프로젝트를 만드는 일은 시스템 관점에서 MoE와 LLM이 어떻게 작동하는지 배우는 훌륭한 방법이었다. 실제 라우팅이 얼마나 희소한지, 전문가 캐시가 얼마나 도움이 되는지, 토큰마다 메모리 대역폭이 어디로 소비되는지, 호스트와 GPU가 서로를 기다리는 데 얼마나 많은 시간을 쓰는지를 알 수 있었다. Transformers를 사용하면 이 모든 것을 한 번에 하나의 모듈씩 쉽게 들여다볼 수 있다.

범용 전문가 스트리밍 구현은 [huggingface/transformers#50717](https://github.com/huggingface/transformers/pull/50717)에 있으며, 벤치마크는 [Space](https://huggingface.co/spaces/tarekziade/moe-expert-streaming)에 있다.
접근 방식과 API에 대한 피드백을 환영한다.

훌륭한 피드백을 보내준 Pedro, Lysandre, Arthur와 vLLM 테스트를 도와준 Harry에게 감사한다.

마지막으로 [Strata](https://github.com/niko1221/strata)를 만든 Niko Veit에게 감사한다.
Strata는 소비자용 하드웨어에서 전문가 스트리밍으로 무엇이 가능한지 보여주는 기준을 세웠고, 목표로 삼을 수 있는 수치가 있었기에 이 작업을 여기까지 밀어붙일 수 있었다.
