---
layout: post
title: "에이전트는 끝났다고 말했다. 데이터베이스는 동의하지 않았다."
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/NStwJm1AafDELVsU4KGwS.png
image: assets/images/blog/posts/2026-10-03-thinkingbox/thumbnail.png
authors:
  - user: microsoft
slug: "thinkingbox"
source_url: "https://huggingface.co/blog/microsoft/thinkingbox"
source_published_date: "2026-10-03"
source_published_at: "2026-10-03T22:56:48+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [The Agent Said It Was Done. The Database Disagreed.](https://huggingface.co/blog/microsoft/thinkingbox)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/microsoft/thinkingbox -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 에이전트는 끝났다고 말했다. 데이터베이스는 동의하지 않았다.

Microsoft ThinkingBox는 AI 에이전트를 생성한 문장이 아니라 남겨 놓은 레코드를 기준으로 평가한 다음, 이를 연속으로 20회 수행할 수 있는지 묻습니다. 이제 Hugging Face에서 사용할 수 있습니다.

[![Figure-1](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/Dzuml_9K2lfRq4hZi17EY.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/Dzuml_9K2lfRq4hZi17EY.png)

그림 1: ThinkingBox는 격리된 MCP 도구 세션에서 에이전트를 실행한 다음, 에이전트가 남긴 최종 백엔드 상태와 side effect를 평가합니다. [ThinkingBox paper](https://arxiv.org/abs/2608.19741)에서 가져왔습니다.

#### 이 글은 Microsoft와 Hugging Face가 공동으로 작성했으며, 공동 집필 및 검토에 참여한 Tommy Guy(Enderis AI 창립자, 전 Microsoft), Hugging Face의 Sergio Paniego, 그리고 전 인턴 Zhuochun Li(University of Pittsburgh), Ali Keramati(UC Irvine), Youngmin Ko(Northwestern)에게 특별히 감사드립니다.

한 고객이 문의를 보냅니다. 고객의 745달러짜리 주방 가전이 내슈빌 물류 센터에서 배송업체의 "exception" 상태에 멈춰 있으며, 예상 배송일보다 15일이 지났습니다.

AI 에이전트는 신중하게 작업합니다. 아홉 번의 도구 호출을 수행합니다. 주문을 가져오고, 배송 추적을 확인하고, 고객 프로필을 조회하고, 환불 정책을 두 번 검색하고, 티켓이 없는 것을 확인하고, 티켓을 하나 생성하고, 타임라인을 기록하고, 정책을 정확히 읽습니다. 고객의 계정 세그먼트는 실제로 배송 지연 보상 대상이 아닙니다.

그런 다음 티켓을 해결됨으로 닫고 이렇게 답합니다. “ 문의하신 내용이 해결되었는데, 제가 도와드릴 일이 더 있을까요? ”

두 가지가 잘못되었습니다. 배송업체의 exception이 아직 열려 있으므로, 필요한 최종 상태는 해결될 때까지 보류 상태였습니다. 그리고 고객은 자신이 실제로 질문한 내용에 대한 제대로 된 답변을 받지 못했습니다.

도구 호출을 확인하는 AI 평가자는 형식이 올바른 호출 아홉 개를 볼 것입니다. 에이전트가 데이터베이스에 기록했는지를 확인하는 평가자도 마찬가지입니다. 동의하지 않는 것은 데이터베이스입니다.

ThinkingBox가 측정하는 것은 바로 이 간극입니다. 507개의 상태 기반 비즈니스 워크플로를 대상으로 각각 다양한 LLM 모델에서 20회씩 실행하고, 에이전트의 최종 백엔드 상태와 side effect를 평가합니다. 이 글에서는 우리가 발견한 내용, 일관성에 드는 비용, 그리고 [OpenEnv](https://github.com/huggingface/OpenEnv/tree/main/envs/thinkingbox_env)를 통해 직접 벤치마크를 실행하는 방법을 다룹니다.

이 작업은 직접 실행해 볼 수 있습니다. 위의 예시는 벤치마크 태스크 [sandbox_external_retail_group1.py:test_case_ST003_006](https://github.com/microsoft/thinkingbox-data/blob/thinkingbox-bench-v1.0/dataset/test_case/sandbox_external_retail/sandbox_external_retail_group1.py#L984-L1223)에서 가져온 것이며, 실패하는 실행 가능한 검사는 단일 필드입니다. 티켓의 상태가 필요한 최종 상태인 hold가 아니라 solved로 되어 있습니다. 전체 trace는 [Appendix D.4, Case 3 of our paper](https://arxiv.org/pdf/2608.19741)에 있습니다.

목차

- [A tool call is not an outcome](#a-tool-call-is-not-an-outcome)

- [One success is not reliability](#one-success-is-not-reliability)

- [Can you depend on the model behind your agent?](#can-you-depend-on-the-model-behind-your-agent)

- [What consistency costs](#what-consistency-costs)

- [Failure signatures](#failure-signatures)

- [How it works](#how-it-works)

- [Run it yourself](#run-it-yourself)

- [Where this goes next](#where-this-goes-next)

결과를 읽기 전에 직접 시도해 보고 싶으신가요? [Run it yourself](#run-it-yourself) 섹션으로 건너뛰세요.

## 도구 호출은 결과가 아니다 {#section-1}

최종 응답과 유효한 도구 호출은 대리 지표일 뿐입니다. 에이전트는 올바른 것처럼 말하면서 잘못된 값을 남기거나, 잘못된 레코드를 변경하거나, 추가적인 side effect를 만들 수 있습니다. 이 문제를 결정하는 것은 에이전트가 남긴 레코드뿐입니다.

그 간극은 상당합니다. 12개 LLM 모델에서 유효한 시도 121,680건을 대상으로 공통 세트 절제 실험을 수행한 결과, 79,853건의 시도가 실행 가능한 검사를 통과하지 못했습니다. 이 실패 중 67.24%는 정상적으로 종료되었고, 상태를 변경하는 도구를 호출했으며, 최종 도구 오류도 보고하지 않았습니다. 그럼에도 실행 가능한 검사는 이들 중 77.61%에서 잘못된 필드 값을, 43.30%에서 의도하지 않은 추가 효과를, 25.36%에서 필요한 효과의 누락을 발견했습니다. 이러한 상태 검사 결과는 서로 중복될 수 있습니다.

trajectory는 주장입니다. 데이터베이스 상태는 증거입니다. 반복은 신뢰성 테스트입니다.

## 한 번의 성공은 신뢰성이 아니다 {#section-2}

환불을 한 번 정확히 처리하고 다음 네 번은 잘못 처리하는 에이전트는 제대로 작동하는 환불 에이전트가 아닙니다. 따라서 모든 태스크를 동일하게 초기화된 깨끗한 백엔드에서 독립적으로 20회 실행하고, 서로 다른 세 가지를 보고합니다.

표 1: 우리가 보고하는 세 가지 수치와 각각이 답하는 질문.

| Metric | 측정하는 항목 | 답하는 질문 |
| --- | --- | --- |
| pass@1 | 성공한 전체 시도의 비율 | 보통 어느 정도로 수행하는가? |
| pass@20 | 20번의 시도 중 최소 한 번 해결된 태스크의 비율 | 이 작업을 한 번이라도 할 수 있는가? 범위. |
| Observed 20/20 | 기록된 20번의 시도를 모두 통과한 태스크 | 항상 올바르게 수행할 수 있는가? |

이 글에서는 observed 20/20을 507개 태스크 중 20회 중 20회를 통과한 태스크의 실제 개수로 사용합니다. 추정량도, 평활화도 없습니다.

익숙한 관점에서 시작해 보겠습니다. 아래 표는 단일 시도 점수 추정치인 pass@1을 도메인별로 나누어 보여 줍니다. 이는 대부분의 리더보드가 게시하는 수치이며, 이것만 보면 일반적인 성능 순위처럼 읽힙니다.

표 2: 도메인별 ThinkingBox-Bench pass@1 (%). 각 모델은 모든 태스크에 대해 20회의 반복 시도로 평가됩니다. 굵게 표시된 항목은 그룹 선두이고, 밑줄은 2위입니다. 단일 시도 점수 추정치의 표준 오차는 [Table 4 in our ThinkingBox paper](https://arxiv.org/abs/2608.19741)에 제공되어 있습니다.

| Model | Retail (98) | Auto insurance (100) | Travel (104) | Neobank (104) | Consulting (101) | Overall, task-weighted (507) |
| --- | --- | --- | --- | --- | --- | --- |
| Proprietary models |  |  |  |  |  |  |
| Claude Opus 5.5 | 80.97 | 68.40 | 54.28 | 71.25 | 61.58 | 67.16 |
| Claude Opus 5 | 80.71 | 65.80 | 49.95 | 70.62 | 66.19 | 66.50 |
| GPT-5.4 | 76.33 | 62.65 | 68.12 | 65.34 | 54.60 | 65.36 |
| GPT-5.6 Sol | 67.65 | 65.30 | 60.34 | 59.09 | 57.52 | 61.91 |
| Claude Sonnet 4.6 | 72.35 | 54.40 | 58.94 | 56.39 | 54.31 | 59.19 |
| GPT-6 Astra | 71.73 | 46.55 | 55.87 | 60.87 | 56.83 | 58.31 |
| GPT-5.2 | 70.20 | 22.40 | 53.70 | 51.15 | 34.06 | 46.28 |
| Claude Opus 4.6 | 68.62 | 8.30 | 21.11 | 35.67 | 27.82 | 32.09 |
| o3-pro | 37.70 | 2.95 | 17.31 | 24.28 | 14.60 | 19.31 |
| Grok-4.3 | 43.93 | 2.60 | 15.14 | 1.78 | 9.55 | 14.38 |
| Open-weight models |  |  |  |  |  |  |
| Kimi-K3 | 82.24 | 50.80 | 61.83 | 41.35 | 51.63 | 57.37 |
| Qwen3.8-27B | 64.03 | 47.85 | 53.41 | 47.88 | 45.69 | 51.70 |
| DeepSeek-V4-Pro | 68.21 | 29.65 | 43.13 | 44.86 | 31.04 | 43.26 |
| Kimi-K2.6 | 53.72 | 24.50 | 39.52 | 33.65 | 37.33 | 37.66 |
| GLM-5.1 | 58.67 | 25.70 | 35.43 | 13.27 | 34.06 | 33.19 |
| Qwen3.6-27B | 43.11 | 29.00 | 46.39 | 27.84 | 18.37 | 32.94 |
| Qwen3.5-9B | 19.90 | 0.70 | 4.71 | 1.15 | 2.33 | 5.65 |
| Mistral-Large-3 | 11.28 | 1.30 | 8.99 | 1.15 | 0.74 | 4.66 |

Claude Opus 5.5가 67.16%로 전체 선두이며, Claude Opus 5보다 0.66%포인트 높습니다. Kimi-K3는 가장 강력한 open-weights 모델로, GPT-6-Astra보다 1%포인트 이내입니다. 도메인도 그만큼 중요합니다. Claude Opus 4.6은 retail에서 68.62%를 기록하지만 auto insurance에서는 8.30%에 그칩니다.

한 번의 좋은 실행은 모델이 작업을 수행할 수 있다는 것을 알려 줍니다. 하지만 다시 수행할지는 알려 주지 않습니다. 따라서 모든 태스크를 20회 실행하고, 그 점수 중 얼마나 유지되는지 확인해야 합니다.

[![Figure-2](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/DjaLNZKNeU--Xy2WFCvVX.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/DjaLNZKNeU--Xy2WFCvVX.png)

그림 2: 각 모델의 단일 시도 점수 중 20회 반복 후 유지되는 비율.

대부분의 pass@1 점수를 유지하는 모델은 세 개뿐입니다. GPT-6 Astra는 단일 시도 비율의 78%를 유지하고, Claude Opus 5.5와 Claude Opus 5는 각각 71%를 유지합니다. 반대쪽 끝에서는 GLM-5.1, Kimi-K2.6, DeepSeek-V4-Pro가 각각 약 8%만 유지합니다.

모델이 한 번 할 수 있는 일과 매번 하는 일 사이의 간극이 전체 이야기입니다.

## 에이전트 뒤의 모델을 믿고 맡길 수 있는가? {#section-3}

[![Figure-3](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/NONd2WGm4vtGpsVgs-CIq.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/NONd2WGm4vtGpsVgs-CIq.png)

그림 3: 범위와 일관성은 서로 다른 방향으로 움직입니다. 18개 모델 중 12개를 표시했으며, 가독성을 위해 pass@1이 33% 미만인 6개 모델은 제외했습니다.

Kimi-K3는 우리가 테스트한 모델 중 가장 폭넓은 범위를 보입니다. 벤치마크의 93.89%를 최소 한 번 해결합니다. 507개 태스크 중 476개입니다. 완전히 실패한 태스크는 31개뿐으로, 전체 모델 중 가장 적습니다. retail 워크플로에서는 pass@1 82.24%로 단독 선두이며, 모든 proprietary 모델보다 앞섭니다.

Kimi-K3는 일관성이 가장 낮은 모델 중 하나이기도 합니다. 507개 태스크 중 단 68개, 13.41%만 20번의 시도 모두에서 성공합니다.

Claude Opus 5는 반대입니다. 최소 한 번 해결한 태스크는 더 적지만(79.09%; 106개 태스크에서 완전히 실패), 매번 시도할 때마다 벤치마크의 47.53%를 완료합니다.

더 새로운 모델이라고 해서 이 문제가 해결되지는 않습니다. Claude Opus 5.5는 모든 시도 평균에서 Claude Opus 5보다 높은 67.16% 대 66.50%를 기록하고, 최소 한 번 해결하는 태스크도 더 많습니다. 그러나 20번의 시도 모두에서 통과한 태스크 수는 정확히 같습니다. 241개입니다. 헤드라인 정확도가 0.5%포인트 높아졌지만 신뢰성은 전혀 늘지 않았습니다.

- Kimi-K3는 Opus 5보다 최소 한 번 해결하는 태스크가 75개 더 많습니다.

- Opus 5는 Kimi-K3보다 일관되게 해결하는 태스크가 173개 더 많습니다.

실제 레코드에 영향을 미치는 작업을 위해 모델을 선택한다면 pass@20 열을 보는 것은 잘못된 선택입니다.

## 일관성에는 어떤 비용이 드는가 {#section-4}

성능 비교는 보통 점수에서 끝납니다. 하지만 배포하는 사람에게 중요한 질문은 성공한 작업 단위 하나에 얼마가 드는가입니다. 이를 성공한 태스크 시도당 비용으로 측정합니다. 태스크 시도라고 말하는 이유는 모든 벤치마크 태스크를 반복 실행하고 시도마다 비용이 발생하기 때문이며, 따라서 pass@1이 이에 대응하는 품질 분모입니다.

각 모델의 전체 507 × 20 캠페인에서 기록된 토큰 사용량을 가져와 [OpenRouter](https://openrouter.ai/)+에서 제공되는 할인 전 정가로 계산했습니다. 프로모션 할인은 되돌렸고, 양자화를 선언한 endpoint는 제외했습니다. 입력, 출력, 캐시 요금은 모두 모델별 하나의 provider endpoint에서 가져왔습니다.

그런 다음 한 번의 실행 비용을 성공한 시도 수로 나누었습니다.

성공한 태스크 시도당 비용 = 태스크당 한 번씩 총 507회 시도에 대한 추정 비용 ÷ (507 × pass@1)

이는 비교를 위한 효율성 지수이지 청구서가 아니며, production request 한 건을 서빙하는 가격도 아닙니다. 또한 일관성이 아니라 단일 성공에 대한 비용입니다. 이제 일관성의 비용을 계산합니다.

예시: GPT-5.4는 507회 시도(태스크당 한 번)에 43.49달러가 들고 pass@1은 65.36%이므로, 43.49달러 ÷ (507 × 0.6536) = 성공한 태스크 시도당 0.131달러입니다.

### Pareto 비용 프론티어

다른 어떤 모델도 더 저렴하면서 정확도가 같거나 높지 않다면 해당 모델은 프론티어에 있습니다. 세 모델이 이에 해당하며, 나머지 모든 모델은 최소 한 축에서 지배됩니다.

[![Figure-4](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/oGmXYrxoNPJfsQndxlcMD.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/oGmXYrxoNPJfsQndxlcMD.png)

그림 4: pass@1에 대한 성공한 태스크 시도당 비용. 고리로 표시된 점은 pareto 비용 프론티어 모델입니다.

프론티어에는 세 단계가 있습니다. GPT-5.6 Sol은 성공당 0.127달러로 비용이 가장 낮고, GPT-5.4는 성공당 0.004달러를 더 지불해 pass@1을 3.45%포인트 높입니다. Claude Opus 5.5는 성공당 0.276달러로 다시 1.80%포인트를 추가합니다. 세 모델 모두 더 저렴한 모델 중 pass@1이 같은 모델이 없기 때문에 비용 프론티어 선상에 남습니다.

Claude Opus 5가 가장 분명한 사례입니다. 성공한 시도당 0.475달러에 pass@1 66.50%인 이 모델은 0.276달러와 67.16%인 Claude Opus 5.5보다 비용이 더 높고 정확도는 더 낮습니다.

### 이제 일관성의 비용을 계산해 보자

성공당 비용은 저렴하면서 자주 맞히는 모델에 유리합니다. 매번 맞히는 모델에는 유리하지 않습니다. 따라서 신뢰할 수 있는 태스크당 비용도 계산합니다. 이는 전체 20회 실행 캠페인 비용을 20번의 시도 모두에서 통과한 태스크 수로 나눈 값입니다.

신뢰할 수 있는 태스크당 비용 = 507개 태스크를 20회 실행한 추정 비용 ÷ 20/20을 통과한 태스크 수

예시: GPT-6-Astra는 캠페인에 20 × 86.03달러 = 1,720.60달러가 들고 모든 시도에서 231개 태스크를 통과하므로, 1,720.60달러 ÷ 231 = 신뢰할 수 있는 태스크당 7.45달러입니다.

표 3: 관측된 20/20 태스크가 하나 이상 있는 모델 중 신뢰할 수 있는 태스크당 비용이 가장 낮은 9개 모델을 낮은 순서로 정렬했습니다. 추정 달러이며 실제 클라우드 청구액이 아닙니다.

| Model | 20/20을 통과한 태스크 | 추정 비용, 20회 실행 | 신뢰할 수 있는 태스크당 비용 |
| --- | --- | --- | --- |
| GPT-5.4 | 128 (25.25%) | $869.80 | $6.80 |
| GPT-6 Astra | 231 (45.56%) | $1,720.60 | $7.45 |
| Claude Opus 5.5 | 241 (47.53%) | $1,880.77 | $7.80 |
| GPT-5.6 Sol | 82 (16.17%) | $800.00 | $9.76 |
| Claude Opus 5 | 241 (47.53%) | $3,206.00 | $13.30 |
| Claude Sonnet 4.6 | 102 (20.12%) | $1,587.60 | $15.56 |
| GPT-5.2 | 44 (8.68%) | $878.00 | $19.95 |
| Kimi-K3 | 68 (13.41%) | $1,406.40 | $20.68 |
| Qwen3.8-27B | 38 (7.50%) | $925.80 | $24.36 |

이제 일관성 기준으로 순위를 매겨 보겠습니다. GPT-5.4가 6.80달러로 가장 저렴하지만, 기준을 충족하는 태스크는 128개뿐입니다. GPT-6 Astra는 7.45달러로 231개에 도달하고, Claude Opus 5.5는 7.80달러로 공동 최고인 241개에 도달합니다.

세 모델 중 어느 것도 다른 모델을 지배하지 않습니다. 신뢰할 수 있는 태스크를 하나 더 얻을 때마다 비용이 증가합니다. Claude Opus 5도 241개를 통과하지만 비용이 13.30달러이므로 Opus 5.5가 명백히 지배합니다. 단일 성공당 0.127달러로 가장 저렴한 GPT-5.6 Sol은 신뢰할 수 있는 태스크당 9.76달러가 듭니다. 정답을 얻는 가장 저렴한 방법이 신뢰할 수 있는 답을 얻는 가장 저렴한 방법은 아닙니다.

## 실패 시그니처 {#section-5}

각 실패 trace에 결정론적인 진단 시그니처를 하나씩 할당했으며, 핵심 내용은 실행 가능하다는 점입니다. 실패의 대략 5건 중 4건은 추론이 아니라 도구 처리 문제입니다. [Table 5 of our paper](https://arxiv.org/pdf/2608.19741)의 절제 연구 전체에서 다음과 같습니다.

| 실패 시그니처 | 실패 비율 |
| --- | --- |
| 도구 사용 | 79.9% |
| 잘못된 상태 업데이트 | 10.3% |
| 불완전한 사용자 해결 | 7.0% |
| 상태를 변경하는 작업 없음 | 2.9% |

이는 모델별 비율과 관찰 가능한 레이블의 비가중 평균이며, 고유한 인과적 설명이 아닙니다.

실제 패턴은 간단합니다. 에이전트는 보통 워크플로를 시도할 만큼은 진행하지만, 도구 오류, 실패한 사전 조건 또는 빈 조회 결과에서 복구하지 못합니다. 이는 모델 문제라기보다 먼저 재시도 및 오류 복구 문제입니다.

난이도도 도메인에 따라 달라집니다. 위 표 2에 나열된 모델 전체에서 retail의 평균 pass@1은 59.52%인 반면 auto insurance의 평균은 33.83%입니다.

이에 대해 할 일은 다음과 같습니다. 20/20 비율을 판정이 아니라 설계 입력으로 다루세요. 벤치마크가 평가하는 것과 동일한 신호를 production에서도 사용할 수 있습니다. 모델의 요약이 아니라 커밋하기 전에 최종 상태를 확인하세요.

재시도가 복구 가능한 오류를 대상으로 하도록 도구 및 시스템 오류를 분류하세요. 워크플로에 필요한 수준으로 도구 표면을 줄이세요. 그리고 저렴하게 되돌릴 수 없는 변경에는 사람의 승인을 요구하세요. 이 중 어떤 방법이 이 벤치마크에서 성능을 얼마나 높이는지는 아직 측정하지 않았습니다. 바로 이런 종류의 작업을 이제 이 환경에서 테스트할 수 있습니다.

## 작동 방식 {#section-6}

ThinkingBox는 에이전트 샌드박스이고, ThinkingBox-Bench는 에이전트를 평가하기 위한 데이터셋 벤치마크입니다. 이 글 상단의 다이어그램이 루프를 보여 줍니다. 각 부분이 하는 일은 다음과 같습니다.

[![Figure-1a](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/elbRGqTXW6PIqci7xcEJN.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/elbRGqTXW6PIqci7xcEJN.png)

그림 5: 위 그림 1의 패널 A에 있는 샌드박스 루프. 격리된 도구 세션, 최종 데이터베이스 상태, side effect, 실행 가능한 judge로 구성됩니다.

각 태스크는 시작 백엔드 상태, 사용자 목표, 사용 가능한 MCP 도구, 도메인 정책, 최종 상태에 대한 실행 가능한 검사를 정의합니다. 시뮬레이션된 사용자는 비공개 컨텍스트(예약 참조 번호, 선호 사항 또는 생년월일)를 보유하고 있으며, 요청받을 때만 이를 공개합니다.

모든 시도에는 새로 초기화된 상태의 격리된 MCP 세션이 할당됩니다. 동일한 태스크에 대한 두 시도는 데이터베이스 행이나 캐시된 도구 상태를 공유하지 않으며, 이것이 20회 시도 비교를 의미 있게 만듭니다.

마지막에는 side-effect extractor가 실제로 변경된 내용을 도출하고, 결정론적인 judge가 이를 필요한 최종 상태와 비교합니다. 올바른 결과를 만들어 내는 모든 trajectory는 허용하고, 잘못되었거나 누락되었거나 추가된 effect는 거부합니다. 깔끔한 데이터베이스 값이 없는 요구사항(“에이전트가 이것이 보장되지 않는다고 고지했는가?”)에는 좁은 범위의 이진 rubric 질문으로 의미를 처리합니다. 507개 태스크 중 477개는 상태만으로 평가하고, 30개는 응답 rubric을 추가합니다.

신뢰 경계: 모델은 태스크, 대화, 도구 스키마를 봅니다. Golden state, assertion, grading internals, credential은 evaluator 측에 남습니다.

## 직접 실행해 보기 {#section-7}

ThinkingBox는 이제 Hugging Face에서 [the harness](https://huggingface.co/docs/openenv/environments/thinkingbox) 및 [the dataset](https://huggingface.co/datasets/microsoft/ThinkingBox-Bench)로 제공됩니다. ThinkingBox-Bench는 이제 OpenEnv 인터페이스 뒤에서 실행되며, 완료된 각 episode는 이진 pass/fail reward를 반환합니다. 공개된 adapter는 평가용으로 설계되었고, 별도의 비벤치마크 시나리오는 training workflow에서 동일한 인터페이스를 사용할 수 있습니다.

### 시작하기 전에

Linux와 WSL에서 테스트되었으며, Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker가 필요합니다. 또한 pinned release의 [thinkingbox-data](https://github.com/microsoft/thinkingbox-data) checkout과 agent, simulated user, judge를 위한 model endpoint가 필요합니다. 하나의 endpoint가 세 역할을 모두 수행할 수 있으며, 이것이 시작하는 가장 간단한 방법입니다. OpenEnv image는 OpenEnv API만 시작하며, 나머지는 직접 실행해야 합니다.

### 설치

```
# 1. OpenEnv + the ThinkingBox environment
git clone https://github.com/huggingface/OpenEnv
cd OpenEnv
uv sync --project envs/thinkingbox_env --frozen
# 2. The executable benchmark, at the pinned release
git clone https://github.com/microsoft/thinkingbox-data
git -C thinkingbox-data checkout thinkingbox-bench-v1.0
# 3. The ThinkingBox CLI, which provides `tb`
uv tool install "thinkingbox @ git+https://github.com/microsoft/thinkingbox"
```


### Typesense 시작

두 번째 터미널에서 Typesense 30.1을 시작하고 health check를 기다립니다.

```
mkdir -p .typesense-data
docker run --rm -d --name thinkingbox-typesense \
  -p 8108:8108 \
  -v "$PWD/.typesense-data:/data" \
  typesense/typesense:30.1 \
  --data-dir /data --api-key=Fake --enable-cors
until curl -fsS http://127.0.0.1:8108/health; do sleep 1; done
```


### MCP 서버 시작

세 번째 터미널에서 Session Proxy와 MCP 서버를 시작합니다.

```
cd OpenEnv
tb mcp-start --host 127.0.0.1 --port 7111 \
  --servers "$PWD/thinkingbox-data/servers/servers.yaml"
curl -fsS http://127.0.0.1:7111/health
```


### OpenEnv 서버 시작

첫 번째 터미널로 돌아가 세 모델의 이름을 지정한 ThinkingBox YAML config를 사용해 OpenEnv 서버를 시작합니다([config guide](https://github.com/microsoft/thinkingbox/blob/main/docs/llm_endpoint_config.md)).

```
OPENENV_TB_CONFIG="$PWD/thinkingbox.yaml" \
uv run --project envs/thinkingbox_env --frozen server
```


### 준비 상태 확인

무엇이든 실행하기 전에 readiness를 확인하세요. 관찰 가능한 데이터, 구성, Session Proxy 검사가 통과할 때까지 503을 반환합니다. Typesense를 관찰하거나 모든 model endpoint를 실시간으로 probe할 수는 없으므로, 이를 별도로 확인하세요.

```
curl -sS http://127.0.0.1:8000/ready
```


### episode 점수 계산

이제 실제 episode의 점수를 계산합니다. [example_usage.py](https://github.com/huggingface/OpenEnv/blob/main/examples/thinkingbox/example_usage.py)는 reset 및 도구 목록만 수행합니다. agent action, effect, assertion에는 패키지된 evaluator를 사용하세요.

```
echo "- sandbox_external_retail_group1.py:test_case_ST002_001" > one_task.yaml
uv run --project envs/thinkingbox_env thinkingbox-eval \
  one_task.yaml \
  --config "$PWD/thinkingbox.yaml" \
  --output results.jsonl \
  --errors-output errors.jsonl \
  --repeat 1 --message-timeout 1800
```


OpenEnv adapter는 운영 실패를 errors sidecar에 기록하므로, 모델 결과와 조용히 섞이지 않고 다시 실행할 수 있습니다. canonical result는 이러한 시도를 해결하거나 명시적으로 설명해야 합니다. 우리는 시스템 오류를 실패한 trial로 계산했습니다.

실행은 pinned framework commit, pinned data release, bundle hash를 기준으로 제한되므로 canonical result는 단순히 주장되는 것이 아니라 검증 가능합니다.

## 앞으로의 방향 {#section-8}

이 작업에서 유용한 부분은 우리의 pass@1 리더보드가 아닙니다. 환경입니다.

실제 레코드에 영향을 미치는 에이전트를 평가한다면 다음을 수행하세요.

- 실패를 검사하세요. 정상적으로 종료했지만 여전히 실패한 실행을 찾아 데이터베이스에서 실제로 무엇이 변경되었는지 확인하세요. 이는 자체 eval이 무엇을 측정하는지 다시 생각하게 합니다.

- OpenEnv를 통해 자체 모델로 하나의 태스크를 재현하세요.

- 반복 metric을 보고하고 이를 정의하세요. 사용 사례가 정당화하는 어떤 k든 사용하되, best-of-k를 보고하는지 every-of-k를 보고하는지, 그리고 어떻게 계산했는지 밝히세요.

자세한 내용은 다음 링크에서 확인할 수 있습니다.

- Environment: [envs/thinkingbox_env](https://github.com/huggingface/OpenEnv/tree/main/envs/thinkingbox_env)

- OpenEnv: [https://huggingface.co/docs/openenv/environments/thinkingbox](https://huggingface.co/docs/openenv/environments/thinkingbox)

- Framework: [microsoft/thinkingbox](https://github.com/microsoft/thinkingbox) · [tutorial](https://github.com/microsoft/thinkingbox/blob/main/docs/tutorial.md)

- Benchmark: [microsoft/thinkingbox-data](https://github.com/microsoft/thinkingbox-data) · [v1.0 release](https://github.com/microsoft/thinkingbox-data/releases/tag/thinkingbox-bench-v1.0)

- Dataset viewer: [microsoft/ThinkingBox-Bench](https://huggingface.co/datasets/microsoft/ThinkingBox-Bench)

- Paper: [arXiv:2608.19741](https://arxiv.org/abs/2608.19741) 또는 [HF](https://huggingface.co/papers/2608.19741)

- RL training (coming soon): [microsoft/thinkingbox-training](https://github.com/microsoft/thinkingbox-training)

ThinkingBox 코드는 MIT 라이선스를 따르며, 벤치마크 데이터는 [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/)이고, OpenEnv 환경은 OpenEnv의 BSD-3-Clause에 따라 배포됩니다.

면책 조항: 공개 벤치마크의 모든 태스크는 synthetic reconstruction입니다. 워크플로와 정책은 실제 AI agentic enterprise 패턴을 기반으로 모델링되었지만, 고객은 실제 인물이 아닙니다.

ThinkingBox와 ThinkingBox-Bench는 Microsoft Copilot Studio 팀이 Toloka와 협력하고, Microsoft에서 인턴으로 근무했던 University of Pittsburgh, Northwestern University, Columbia University, UC Irvine의 협력자들과 함께 구축했습니다. 질문은 아래 댓글 섹션이나 [github](https://github.com/microsoft/thinkingbox/discussions)에서 남겨 주세요.

+ OpenRouter 비용 스냅샷은 2026년 9월 20일에 가져왔으며, Opus 5.5 가격은 Anthropic 사이트를 기준으로 했습니다.
