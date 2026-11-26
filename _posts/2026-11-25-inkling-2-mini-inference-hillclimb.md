---
layout: post
title: "RTX PRO 6000 한 대에서 Inkling 2 Mini 추론 힐 클라이밍"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/inkling-2-mini-inference-hillclimb/thumbnail.png
image: assets/images/blog/posts/2026-11-25-inkling-2-mini-inference-hillclimb/thumbnail.png
authors:
  - user: nielsr
slug: "inkling-2-mini-inference-hillclimb"
source_url: "https://huggingface.co/blog/inkling-2-mini-inference-hillclimb"
source_published_date: "2026-11-25"
source_published_at: "2026-11-25T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Hill-climbing Inkling 2 Mini inference on one RTX PRO 6000](https://huggingface.co/blog/inkling-2-mini-inference-hillclimb)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/inkling-2-mini-inference-hillclimb -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# RTX PRO 6000 한 대에서 Inkling 2 Mini 추론 힐 클라이밍

[Inkling 2 Mini](https://huggingface.co/blog/inkling2mini)는 지난주 출시되었으며 로컬 AI와 소형 오픈 소스 모델의 새로운 최고 수준을 제시합니다. 총 파라미터 수는 \~26B(토큰당 활성 파라미터 3B)이고 멀티모달 입력과 1M 토큰 컨텍스트 윈도우를 지원합니다. [Artificial Analysis Intelligence Index v5](https://artificialanalysis.ai/models/open-source)에서 동급 파라미터 규모의 모델 중 선두를 차지했으며, Gemma 4 31B와 Qwen3.8-27B 같은 모델을 상대로 48개 벤치마크 중 32개에서 앞서 코딩, 도구 사용, 추론에 강력한 오픈 옵션이 되었습니다. Hugging Face 생태계에서 이를 사용하는 방법에 대한 자세한 블로그 글은 [여기](https://huggingface.co/blog/inkling2mini)에 작성했습니다.

Inkling 2 Mini의 뛰어난 성능을 설명할 수 있는 흥미로운 요소 중 하나는 독특한 아키텍처입니다. Inkling 2 Mini는 물리적 레이어 24개로 구성된 루프형 [Mixture-of-Experts (MoE)](https://huggingface.co/blog/moe) 모델이며, 가운데 12개 레이어가 세 번 실행되므로 각 토큰은 48개 레이어를 통과합니다. 체크포인트는 mixed NVFP4와 MXFP8 형식으로 제공되며 speculative decoding을 위한 DSpark drafter가 포함되어 있습니다. LLM 추론의 대표적인 프레임워크인 vLLM과 SGLang은 모두 nightly 빌드에서 이를 지원하지만, 공개된 서빙 레시피는 NVIDIA B200 및 GB300 GPU를 대상으로 합니다(`sm100`). 따라서 NVIDIA RTX PRO 6000 같은 `sm120` 카드는 제외되어 있으며, 이 경우 커널 호환성이 달라질 수 있습니다.

NVIDIA RTX PRO 6000 96 GB 한 대에서 이를 얼마나 빠르게 서빙할 수 있는지 확인하고자 했습니다. 이 GPU는 `nvidia-rtx-pro-6000` [Inference Endpoint](https://huggingface.co/docs/inference-endpoints)를 구동하는 하드웨어입니다. 플래그를 직접 튜닝하는 대신 코딩 에이전트에 [`inference-optimisation`](https://github.com/sybil-solutions/inference-optimisation-skill) skill, `hf` CLI, [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-overview) 액세스 권한을 제공하고 오후 동안 실험을 실행하도록 했습니다.

유효한 벤치마크 46회 실행에서 vLLM nightly와 DSpark speculative decoding을 사용했을 때 단일 사용자 디코드 속도는 **136에서 190.6 tok/s(+40%)**로 증가했습니다(동일 GPU에서 [LocalMaxxing의 167.7 tok/s 커뮤니티 SGLang + DSpark 벤치마크](https://www.localmaxxing.com/en/inference-benchmarks/cmwqfbexc089gmi01hd6hiqhn)보다 빠름). 8사용자 집계 처리량은 **\~570에서 716 tok/s(+26%)**로 증가했습니다. 추측 실행 여부와 관계없이 greedy 출력은 바이트 단위로 동일했고, 긴 컨텍스트 needle retrieval은 최대 250K 토큰에서 3/3을 통과했으며, 가장 좋은 구성은 고정된 이미지 다이제스트를 사용하는 Inference Endpoint로 현재 실행 중입니다. 모든 로그, 스크립트, 원시 측정값은 [`NielsRogge/inkling-2-mini-hillclimb`](https://github.com/NielsRogge/inkling-2-mini-hillclimb)에 있습니다.

## 설정 {#section-1}

`inference-optimisation` skill은 에이전트에 고정된 실험 프로토콜과 작은 엔진 비종속 벤치마크 스크립트([`bench.py`](https://github.com/sybil-solutions/inference-optimisation-skill/blob/main/skills/inference-optimisation/scripts/bench.py))를 제공합니다.

1. 튜닝 전에 하드웨어를 측정하고 이론적 상한을 추정합니다.
2. 한 번에 변수 하나만 변경하고, 모든 후보를 두 번의 control run 사이에서 검증합니다(`control → candidate → control`). 다른 지표를 저하시키지 않으면서 두 control run을 모두 앞설 때만 변경을 유지합니다.
3. 회귀와 막다른 시도를 포함해 모든 실행을 `PLAN.md`에 기록합니다.
4. 승자를 선언하기 전에 긴 컨텍스트, 동시성, greedy 동등성 검사를 통과시킵니다.

`bench.py`는 단일 사용자 디코드, 동시성 N에서의 집계 디코드, 여러 길이에서의 cold prefill, TTFT를 측정하며, 토큰 수는 서버의 usage 응답에서 직접 가져옵니다. 모든 실행은 코드, 설명, 이야기, 비교 표, bash 스크립트, 역사 요약으로 구성된 고정된 6개 프롬프트 세트를 사용했습니다. temperature는 1.0(모델 기본값), thinking은 활성화, 출력 토큰은 512개로 설정했으며, speculative decoding이 활성화된 경우에는 매번 12회 반복했습니다.

이 skill은 로컬 GPU를 대상으로 작성되었으므로 Hugging Face 인프라에 맞게 조정했습니다.

- **HF Jobs를 컴퓨팅에 사용.** `rtx-pro-6000` Job flavor는 `nvidia-rtx-pro-6000` Inference Endpoint와 동일한 GPU를 사용하므로($2.75/h로 동일), Job 수치를 그대로 옮겨 사용할 수 있습니다. 약 150줄의 드라이버(`jobs/driver.py`)는 서버 구성 목록을 받아 각각을 시작하고, 측정하지 않는 warmup을 실행한 뒤, `bench.py`를 수정 없이 호출하고, 서버를 종료하며, 실행마다 한 줄의 ledger를 출력합니다. Job은 별도 머신에서 실행되므로 에이전트는 SGLang(`lmsysorg/sglang:inkling-2-mini`)과 vLLM nightly(`vllm/vllm-openai:nightly`)를 병렬로 스윕했습니다.
- **엔진 소스 확인에 CPU Jobs를 사용.** 플래그가 문서화되어 있지 않을 때 에이전트는 저렴한 `cpu-upgrade` Job을 실행해 컨테이너 내부에 설치된 Python 패키지를 grep했습니다. 이를 통해 SGLang의 `SGLANG_OPT_*` 환경 변수와 vLLM의 `speculative-config` 필드를 찾았습니다.
- **배포에 Inference Endpoints를 사용.** 검증을 통과한 정확한 이미지 다이제스트와 플래그를 사용해 `hf endpoints deploy`로 생성했습니다.

baseline은 기존 조직 endpoint 구성인 SGLang과 `--attention-backend fa4`, 그리고 speculative decoding을 사용하지 않는 설정이었습니다.

## 결과 {#section-2}

![Single-user decode speed of Inkling 2 Mini on one RTX PRO 6000 for four configurations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inkling-2-mini-inference-hillclimb/results.png)

_단일 사용자 디코드 속도(12개 요청의 중앙값). vLLM + DSpark는 측정된 acceptance rate에서 메모리 대역폭 상한의 \~95%에 도달합니다._

| 구성(1× RTX PRO 6000) | 1사용자 디코드(tok/s) | 8사용자 집계(tok/s) | Prefill 4K / 32K(tok/s) | TTFT |
|---|---:|---:|---:|---:|
| SGLang, speculation 없음(baseline) | 134–139 | 543–598 | 18.7–19.1K / 16.9–19.1K | 41–114 ms |
| vLLM, speculation 없음 | 157.0 | 638 | 20.7K / 20.7K | 33 ms |
| SGLang + DSpark(초안 8개) | 166.6 | — ¹ | 18.0–19.5K / 18.9–19.1K | 38–51 ms |
| **vLLM + DSpark(초안 7개)** | **190.6** | **716** | 20.7K / 20.6K | 40 ms |
| 배포된 endpoint(인터넷 경유) | 194.9 | 673 | 11.7K / 19.2K ² | 198 ms ² |

¹ SGLang의 RTX PRO 6000 DSpark 레시피는 `--max-running-requests`을 4로 제한합니다. 이를 8로 늘리면 660 tok/s에 도달하지만 여전히 vLLM보다 느립니다.
² 클라이언트에서 endpoint까지의 네트워크 왕복 시간 약 160 ms를 포함합니다.

전체 속도 향상은 vLLM nightly로 전환한 것(추측 실행이 없는 SGLang 대비 +15%, 136 → 157 tok/s), 7개의 초안과 확률적 초안 샘플링으로 DSpark를 활성화한 것(추가로 +21%, 네 번의 실행에서 157 → 190.6 tok/s: 189.3, 190.4, 191.1, 190.7), 그리고 vLLM의 DSpark 구성이 96 GB에서 실제로 시작되도록 메모리 플래그 두 개를 추가한 것에서 비롯되었습니다.

```bash
vllm serve thinkingmachines/Inkling-2-Mini \
  --tool-call-parser inkling \
  --enable-auto-tool-choice \
  --reasoning-parser inkling \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```


`num_speculative_tokens: 7`와 `draft_sample_method: "probabilistic"`은 모두 직접 검증했습니다. 더 짧은 초안 수(예: 5)는 두 엔진이 체크포인트의 고정 블록 크기를 적용하기 때문에 시작 시 거부되며, `greedy` draft sampling으로 전환하면 temperature 1.0에서 vLLM 디코드 속도가 11% 감소했습니다([제대로 작동하지 않은 것](#what-didnt-work) 참조).

### 약 190 tok/s에서 멈추는 이유

튜닝 전에 에이전트는 Job 내부에서 RTX PRO 6000을 측정했습니다(**1,469 GB/s** copy bandwidth, **416 TFLOPS** BF16, **792 TFLOPS** FP8). 또한 safetensors 헤더에서 텐서 크기를 읽었습니다. 각 토큰이 48번의 레이어 방문을 거치므로 일반 디코드는 attention 및 dense weight 약 2.7 GB, routed expert 약 2 GB(레이어마다 128개 중 8개), LM head 약 0.9 GB를 다시 읽습니다. 1,469 GB/s에서는 단일 사용자 대역폭 상한이 **약 256 tok/s**가 됩니다. speculation이 없을 때 vLLM은 157 tok/s로 이 상한의 61%에 도달하며, 이는 주로 48번의 방문에 걸친 커널 실행 지연에 의해 제한됩니다.

Speculative decoding은 MoE에서 메모리 계산을 바꿉니다. 8개 위치를 함께 검증하면 각 레이어에서 routed expert의 약 6%가 아니라 이들의 합집합인 약 40%를 활성화해 읽습니다. temperature 1.0에서 측정된 단계당 acceptance가 약 2.3토큰(위치별 acceptance는 첫 번째 초안의 0.6에서 0.33, 0.19로 감소)일 때 대역폭 상한은 **약 201 tok/s**로 떨어집니다. 190.6 tok/s에서 vLLM + DSpark는 이미 **하드웨어 대역폭의 약 95%**에 도달했으므로, 플래그를 더 튜닝해도 개선 여지는 거의 없습니다. 190 tok/s를 넘으려면 temperature 1.0에서 더 높은 acceptance를 내는 drafter나 레이어마다 더 적은 expert에 접근하는 검증 경로가 필요합니다.

## 제대로 작동하지 않은 것 {#section-3}

대부분의 후보 플래그는 지원되지 않거나, 결과가 불안정하거나, 더 느렸습니다.

### vLLM

| 조절 항목 | 결과 |
|---|---|
| 더 짧은 draft length(3–6) | 시작 시 거부됨. `num_speculative_tokens = 7` 필요(checkpoint block size − 1). |
| `draft_sample_method: "greedy"` | temperature 1.0에서 −11%(168.6 vs \~190.4 tok/s). |
| `rejection_sample_method: "block"` | 190.1 vs control의 190.3–190.5로, 효과 없음. |
| `enable_adaptive_verification` | Inkling 2 drafter에서 시작 시 assertion error 발생. |
| `draft_lm_head_quantization: "fp8"` | DeepSeek DSpark drafters에만 지원됨. |

또한 `rejection_sample_method: "fly"`와 `"synthetic"`은 출력 분포를 변경하기 때문에 의도적으로 사용하지 않았습니다. 우리는 엄밀하게 손실 없는 속도 향상을 원했습니다.

### SGLang

| 조절 항목 | 결과 |
|---|---|
| 더 짧은 draft length(3–6) | 시작 시 거부됨. `--speculative-num-draft-tokens 8` 필요(번들된 drafter의 경우 `gamma + 1`). |
| `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` | 한 번은 +4%였지만 새로운 control을 대상으로 재현되지 않음. |
| `SGLANG_OPT_USE_FUSED_DRAFT_SAMPLE=1` | +2%로, 실행 간 노이즈 범위 내. |
| `SGLANG_OPT_MOE_QUANT_ONCE=1` | +1%로, 실행 간 노이즈 범위 내. |
| `SGLANG_DSPARK_USE_GREEDY_DRAFT=1` | temperature 1.0에서 −9%(156.8 vs \~172 tok/s). |

## 벤치마크에서 주의할 점 {#section-4}

진행 과정에서 몇 가지 측정 문제가 발견되었습니다.

- **측정된 prefill 중 Triton JIT 실행.** vLLM의 첫 4K prefill은 Triton이 측정 대상 요청 중 커널을 컴파일했기 때문에 20K가 아니라 202 tok/s로 측정되었습니다. 프롬프트 길이와 동시성 전반에 걸쳐 측정하지 않는 warmup을 실행해 4K prefill 문제를 해결했지만, 각 Job의 첫 실행에서는 마지막 청크의 shape 차이로 인해 32K prefill이 여전히 낮게 측정되었습니다(ledger에 표시).
- **GPU 메모리를 점유한 고아 서버.** 초기 드라이버는 `setsid … &`에서 저장한 PID를 사용해 서버를 중지했지만, 이 값이 process group ID와 일치한다고 보장할 수는 없습니다. 이전 서버가 포트 8000에서 계속 실행되어 다음 시작을 방해했고, `kill -0`을 여전히 통과하는 좀비 프로세스가 되었습니다. `subprocess.Popen(start_new_session=True)`로 프로세스를 관리하고 GPU 메모리와 포트 8000이 모두 해제될 때까지 기다리도록 수정해 해결했습니다.
- **vLLM 로그에 숨겨진 OOM 오류.** vLLM + DSpark가 두 번이나 로그 끝부분에서 `"Engine core initialization failed"`와 함께 종료되었지만, 실제 원인은 200줄 위에 있었습니다. 먼저 기본 1M 토큰 KV cache에는 drafter와 함께 사용할 수 있는 여유 공간이 48.1 GiB뿐인 상황에서 52.6 GiB가 필요했습니다(`--max-model-len 262144`로 해결). 다음으로 vLLM v0.34.0이 기본 `--gpu-memory-utilization`를 `0.92`에서 `1.0`로 올렸기 때문에 KV cache가 남은 VRAM을 차지했고, drafter의 5 GiB 할당에서 OOM이 발생했습니다(`--gpu-memory-utilization 0.85`로 해결). 이제 드라이버는 서버 로그의 마지막 부분만이 아니라 모든 오류 줄을 출력합니다.
- **엔진별 변동성.** speculation을 사용한 동일한 실행에서 vLLM의 변동 폭은 0.4% 이내였지만 SGLang은 ±4.5% 변동했습니다. 이 정도로 폭이 넓었기 때문에 반복 control로 배제하기 전 첫 실행에서는 `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1`가 +4%의 성능 향상처럼 보였습니다.

## 검증 및 배포 {#section-5}

배포 전에 정확한 vLLM nightly 다이제스트(`vllm/vllm-openai@sha256:eaa63fd6…`, build `0.34.0rc1.dev1235+gf2f9edcac`)를 고정하고 Job 내부에서 acceptance gate를 실행했습니다.

| Gate | 결과 |
|---|---|
| 약 50% 컨텍스트에서 needle retrieval(126,537 토큰) | 3/3, TTFT 12.5 s |
| 약 95% 컨텍스트에서 needle retrieval(249,970 토큰) | 3/3, TTFT 28.7 s |
| 동시성 soak(4 × 약 58K 토큰 요청) | 4/4 성공, 각각 retrieval 3/3 |
| Greedy equivalence, DSpark vs speculation 없음(프롬프트 6개, temp 0) | 6/6 바이트 단위 동일 |
| 고정 이미지에서 단일 사용자 디코드 | 189.9 / 190.2 tok/s |

이 테스트를 통과한 후 단일 `hf endpoints deploy` 명령으로 endpoint를 배포하고, 이미지 다이제스트와 모델 revision을 모두 고정했습니다(인증된 액세스, 60분 동안 유휴 상태이면 scale-to-zero).

```bash
hf endpoints deploy inkling2mini-tuned-dspark --namespace <your-org> \
  --repo thinkingmachines/Inkling-2-Mini --revision 7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6 \
  --framework custom --custom-image "vllm/vllm-openai@sha256:eaa63fd6ee3aac77f3158e80ed79ec425db3a53f7cd666bf1954e96abfaf19a7" \
  --health-route /health --port 8000 \
  --vendor aws --region us-east-2 --accelerator gpu --instance-type nvidia-rtx-pro-6000 --instance-size x1 \
  --min-replica 0 --max-replica 1 --scale-to-zero-timeout 60 --type authenticated \
  --container-command "vllm serve" \
  --container-args "/repository --served-model-name thinkingmachines/Inkling-2-Mini --host 0.0.0.0 --port 8000 --tool-call-parser inkling --enable-auto-tool-choice --reasoning-parser inkling --max-model-len 262144 --gpu-memory-utilization 0.85 --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\"}'"
```


실행 중인 endpoint에 `bench.py`를 사용한 결과 단일 사용자에서 194.9 tok/s가 나왔으며, 네트워크 지연 시간을 고려하면 Job 실행 결과와 일치했습니다.

이번 실행의 모든 결과(`PLAN.md`, `THEORY.md`, `jobs/driver.py`, `submit.sh`, `deploy.sh`, 원시 `ledger-*.jsonl` 파일)는 [`NielsRogge/inkling-2-mini-hillclimb`](https://github.com/NielsRogge/inkling-2-mini-hillclimb)에 있습니다. 다른 모델에서 이 루프를 재사용하려면 후보 및 control 서버 명령을 `configs/plan-*.json` 파일에 넣고 대상 endpoint에 맞는 HF Jobs GPU flavor로 제출하면 됩니다.
