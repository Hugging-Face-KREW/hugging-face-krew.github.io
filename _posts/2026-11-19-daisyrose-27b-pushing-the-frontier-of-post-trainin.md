---
layout: post
title: "DaisyRose-27B: 사후 학습 효율성의 한계를 넓히다"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6aeabf9f2efd990530425e58/faP4Su889q_DIgN3wmU9M.png
image: assets/images/blog/posts/2026-11-18-daisyrose-27b-pushing-the-frontier-of-post-trainin/thumbnail.png
authors:
  - user: RadixArk
slug: "daisyrose-27b-pushing-the-frontier-of-post-trainin"
source_url: "https://huggingface.co/blog/RadixArk/daisyrose-27b-pushing-the-frontier-of-post-trainin"
source_published_date: "2026-11-18"
source_published_at: "2026-11-18T19:54:56+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [DaisyRose-27B: Pushing the Frontier of Post-Training Efficiency](https://huggingface.co/blog/RadixArk/daisyrose-27b-pushing-the-frontier-of-post-trainin)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/RadixArk/daisyrose-27b-pushing-the-frontier-of-post-trainin -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# DaisyRose-27B: 사후 학습 효율성의 한계를 넓히다

[![DaisyRose-27B](https://www.radixark.com/images/daisyrose-27b/hero.png)](https://www.radixark.com/images/daisyrose-27b/hero.png)

오늘 RadixArk의 첫 번째 모델인 DaisyRose-27B를 공개합니다. 이 모델은 Qwen3.8-27B를 기반으로 구축되었으며 [Miles](https://miles.radixark.com/)를 사용해 강화 학습으로 학습되었습니다. 최대 1M 토큰의 세션을 사용하는 장기 에이전트 작업으로 18일 동안 144 H200 GPU에서 엄선된 1,200개의 작업만을 학습했습니다. DaisyRose-27B는 Terminal-Bench 4.1 pass@1을 5.6%에서 12.0%로 향상합니다. Terminal-Bench 4.1뿐 아니라 다양한 추론, 지시 따르기, 코딩 벤치마크에서도 성능 향상을 보입니다.

체크포인트는 Apache 2.0 라이선스로 제공됩니다:

- [BF16](https://huggingface.co/RadixArk/DaisyRose-27B)

- [FP8](https://huggingface.co/RadixArk/DaisyRose-27B-FP8)

- [NVFP4](https://huggingface.co/RadixArk/DaisyRose-27B-NVFP4)

이 모델은 Qwen3.8-27B의 아키텍처, 토크나이저, 채팅 템플릿, 이미지 이해 기능 및 256K 토큰 컨텍스트 윈도우를 그대로 유지합니다. 더 긴 학습 세션에서는 여러 궤적에 걸쳐 작업할 수 있도록 harness summarization을 사용합니다. 모델 속성과 추론 설정은 [모델 카드](https://huggingface.co/RadixArk/DaisyRose-27B)를 참조하세요.

## DaisyRose-27B 실행 {#section-1}

DaisyRose-27B는 SGLang과 DSpark speculative decoding을 사용해 서빙하는 것을 권장합니다. [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/RadixArk/DaisyRose-27B)에서는 다양한 GPU와 체크포인트 형식에 대한 실행 구성을 제공합니다.

포트 `30000`에서 서빙되는 모델 이름 `DaisyRose-27B`으로 서버를 시작한 후에는 OpenAI Python 클라이언트를 사용해 요청을 보낼 수 있습니다:

```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="none",
)

response = client.chat.completions.create(
    model="DaisyRose-27B",
    messages=[
        {
            "role": "user",
            "content": "How many positive divisors does 360 have?",
        }
    ],
    temperature=1.0,
    top_p=0.95,
    extra_body={"top_k": 20, "min_p": 0.0},
)

print(response.choices[0].message.reasoning_content)
print(response.choices[0].message.content)
```


이 모델은 기본적으로 답변하기 전에 사고합니다. 추론은 `reasoning_content`에, 최종 답변은 `content`에 반환됩니다. 이러한 설정은 [모델 카드의 빠른 시작 가이드](https://huggingface.co/RadixArk/DaisyRose-27B#3-getting-started)를 따릅니다.

## 벤치마크 결과 {#section-2}

[![DaisyRose-27B compared with eight models across seven benchmarks](https://www.radixark.com/images/daisyrose-27b/benchmarks.png)](https://www.radixark.com/images/daisyrose-27b/benchmarks.png)

| 벤치마크 | Qwen3.8-27B | DaisyRose-27B |
| --- | --- | --- |
| Terminal Bench 4.1 | 5.6 | 12.0 |
| Terminal Bench 2.1 | 73.0 | 75.5 |
| DeepSWE v1.1 | 42.2 | 46.2 |
| SWE Bench Pro v2 | 91.3 | 94.7 |
| HLE (text-only) | 33.9 | 34.3 |
| IFBench | 79.5 | 82.4 |
| GPQA Diamond | 89.2 | 90.7 |

DaisyRose-27B는 위의 모든 벤치마크에서 기존 Qwen3.8-27B 모델보다 높은 점수를 기록합니다. 가장 큰 향상은 Terminal Bench 4.1에서 나타났으며, pass@1이 5.6%에서 12.0%로 상승했습니다.

점수는 각 벤치마크에서 BF16 실행과 FP8 실행 중 더 높은 값을 사용했습니다. 두 형식을 모두 평가한 경우 두 결과의 차이는 1점 미만입니다. [모델 카드](https://huggingface.co/RadixArk/DaisyRose-27B#2-evaluations)에는 비교 자료와 평가 설정이 포함되어 있습니다.

이번 출시를 가능하게 한 학습 설정의 핵심은 세 가지였습니다. 신중하게 설계한 작업, 효율적인 학습 백엔드, 안정적인 샌드박스입니다.

## 고품질 학습 데이터 {#section-3}

[Eigent AI](https://www.eigent.ai/)와 협력해 작업을 구축했으며, 머신 러닝, 소프트웨어 엔지니어링, 과학 및 기타 분야의 대규모 작업 풀에서 시작했습니다. 함께 수천 개의 후보 작업을 구축했고, 이 중 약 1,200개가 최종 학습 세트에 포함되었습니다.

다음 세 가지 기준을 사용해 작업을 선택했습니다:

- 각 작업은 원래 모델이 저지르는 실수를 대상으로 합니다. 모든 작업에 Qwen3.8-27B를 실행하고, 모델이 더 이상 저지르지 않기를 바라는 실수를 중심으로 각 작업을 재설계했습니다. 그런 다음 해결률이 유용한 학습 범위에 들어올 때까지 난이도를 조정했습니다.

- 각 작업은 서로 의존하는 여러 단계를 요구합니다. 에이전트는 장시간의 작업 전반에 걸쳐 추론하고, 계획하고, 실행해야 합니다. 각 시도에는 샌드박스에서 최대 1.5시간이 주어지며, 100회가 넘는 도구 호출과 256K가 넘는 출력 토큰을 사용할 수 있습니다.

- 작업은 실제로 문제를 해결했을 때 보상을 제공해야 합니다. 모든 작업을 모델의 시도와 함께 검토했습니다. 의도한 문제를 해결하지 않고도 통과할 수 있거나 설계 결함이 있는 작업은 수정하거나 제거했습니다.

## 효율적인 학습 백엔드 {#section-4}

RadixArk는 대규모 모델 사후 학습을 위한 오픈 소스 강화 학습 프레임워크인 Miles를 사용해 DaisyRose-27B를 완전 비동기 모드로 학습했습니다. 에이전트가 작업을 계속 수행하는 동안 모델이 학습하며, 어느 쪽도 다른 쪽을 기다리지 않습니다. 128개의 H200이 에이전트 작업에 대한 추론을 실행하는 동안, 추가 16개가 트레이너를 실행하고 궤적에서 학습합니다. 각 단계가 끝나면 업데이트된 모델이 에이전트를 실행하는 128개 GPU로 전달됩니다. 학습은 에이전트 학습에 맞게 구축된 내부 학습 백엔드에서 실행됩니다. 이 백엔드는 일부 세션이 1M 토큰에 달하는 매우 긴 세션 몇 개에서 각 단계마다 학습하며, harness summarization을 통해 이를 여러 궤적으로 묶습니다.

학습 비용이 57% 낮습니다. 이번 출시에서 각 GPU의 최대 연산량 중 37%가 백엔드에서 유용한 학습 작업에 사용되었습니다(model FLOPs utilization, 즉 MFU). Megatron을 기반으로 구축된 오픈 소스 트레이너는 비슷한 실행에서 16%에 도달했습니다. 따라서 동일한 학습에 필요한 GPU 시간이 57% 줄어듭니다. mixture-of-experts 모델에서는 향상이 더욱 큽니다. Qwen3.6-35B-A3B에서는 6.2배, 약 300B 파라미터의 GLM-5.3-Flash에서는 10배 이상입니다.

[![GPU efficiency: RadixArk’s internal backend compared with the open-source trainer](https://www.radixark.com/images/daisyrose-27b/gpu-efficiency.png)](https://www.radixark.com/images/daisyrose-27b/gpu-efficiency.png)

엔드투엔드로 2배 빠릅니다. 에이전트가 작업에 사용하는 시간까지 포함하면, 이번 출시 학습 실행은 144개 GPU 전체에서 초당 약 16,800개 토큰을 처리했습니다. 오픈 소스 트레이너는 동일한 GPU에서 약 8,300개에 도달했습니다.

[![End-to-end training speed: 16.8k versus 8.3k tokens per second on 144 H200s](https://www.radixark.com/images/daisyrose-27b/training-speed.png)](https://www.radixark.com/images/daisyrose-27b/training-speed.png)

수 주 동안 안정적으로 유지됩니다. 이번 출시 실행은 144개의 H200에서 수 주 동안 학습했으며, 총 63,000 GPU-hours가 사용되었습니다. 첫 단계부터 마지막 단계까지 속도와 효율이 안정적으로 유지되었습니다.

[![Training speed and GPU efficiency across the DaisyRose-27B release run](https://www.radixark.com/images/daisyrose-27b/training-stability.png)](https://www.radixark.com/images/daisyrose-27b/training-stability.png)

144개의 H200에서 진행한 DaisyRose-27B 출시 학습 실행의 모든 학습 단계를 총 학습 시간에 대해 표시한 그래프입니다. 점선은 비슷한 실행에서 오픈 소스 트레이너가 보인 결과입니다.

## 견고한 샌드박스 서비스 {#section-5}

모든 학습 시도는 샌드박스에서 실행됩니다. 샌드박스는 작업 환경을 새로 복사한 공간으로, 에이전트가 명령을 실행하고 파일을 편집한 후 평가를 받습니다.

이 서비스는 오픈 소스 [AgentENV](https://github.com/kvcache-ai/AgentENV) 프로젝트를 기반으로 구축되었으며, 저희도 이 프로젝트에 수정 사항을 기여하고 있습니다. 9월 중순부터 11월 중순까지 이 프로젝트를 위해 약 944,000개의 샌드박스를 실행했으며, 최대 1,566개가 동시에 실행되었습니다.

서비스는 한 대의 머신에서 최대 12대까지 확장되었습니다. 학습이 실행되는 동안에도 대부분의 업그레이드를 진행하면서 총 10회의 업그레이드를 배포했습니다. 9주 전체에 걸쳐 샌드박스 생성의 99%가 성공했으며, 10월 말 이후에는 이 수치가 99.95%를 초과했습니다.

[![944K total sandboxes, 1,566 peak concurrent sandboxes, and a 99.95% success rate in the final run](https://www.radixark.com/images/daisyrose-27b/sandboxes.png)](https://www.radixark.com/images/daisyrose-27b/sandboxes.png)

프로젝트의 샌드박스에 사용한 클라우드 인프라 비용은 인기 있는 클라우드 샌드박스 제공업체를 통해 동일한 샌드박스를 실행하는 경우보다 약 80% 낮았습니다. 비용을 낮추는 한 가지 방법은 각 샌드박스의 메모리 할당량이 실제 사용량에 따라 늘어나고 줄어들도록 하는 것입니다. 이를 통해 더 많은 샌드박스가 각 머신을 공유할 수 있습니다.

## 모델을 사용해 보거나 직접 학습하기 {#section-6}

[DaisyRose-27B](https://huggingface.co/RadixArk/DaisyRose-27B)를 다운로드하고, [SGLang serving guide](https://docs.sglang.io/cookbook/autoregressive/RadixArk/DaisyRose-27B)를 따르거나, [Miles](https://miles.radixark.com/)를 사용해 직접 에이전트를 학습해 보세요.

DaisyRose-27B가 여러분의 작업에서 어떻게 성능을 발휘하는지, 그리고 어떤 부분에서 한계를 보이는지 듣고 싶습니다. 모델의 Community 탭에서 여러분의 설정과 결과를 공유해 주세요.

학습 데이터 구축에 긴밀히 협력해 준 Eigent AI와 Qwen3.8-27B를 만든 Qwen 팀에 감사드립니다. 또한 Miles 및 [SGLang](https://www.sglang.io/) 커뮤니티와 slime, AgentENV, Harbor를 비롯한 여러 오픈 소스 프로젝트에도 감사드립니다.

[RadixArk 원문](https://www.radixark.com/blog/daisyrose-27b)에서 각색했습니다.
