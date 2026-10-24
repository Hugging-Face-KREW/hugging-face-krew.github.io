---
layout: post
title: "데이터베이스 채점을 RL 보상으로 전환하기: ThinkingBox를 활용한 에이전트 사후 학습"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/oS8I8qx7iBS8EcNIXR3q9.png
image: assets/images/blog/posts/2026-10-22-thinkingbox-training/thumbnail.png
authors:
  - user: microsoft
slug: "thinkingbox-training"
source_url: "https://huggingface.co/blog/microsoft/thinkingbox-training"
source_published_date: "2026-10-22"
source_published_at: "2026-10-22T22:24:34+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Turning Database Grading into RL Rewards: Post-Training Agents with ThinkingBox](https://huggingface.co/blog/microsoft/thinkingbox-training)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/microsoft/thinkingbox-training -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 데이터베이스 채점을 RL 보상으로 전환하기: ThinkingBox를 활용한 에이전트 사후 학습

첫 번째 블로그 게시물에 이은 후속 글입니다. [The Agent Said It Was Done. The Database Disagreed.](https://huggingface.co/blog/microsoft/thinkingbox) ThinkingBox-Training을 이제 [GitHub](https://github.com/microsoft/thinkingbox-training)에서 사용할 수 있습니다.

최종 데이터베이스 상태를 기대 결과와 비교하여 에이전트의 작업을 평가하기 위해 ThinkingBox를 구축했습니다. 그런 다음 ThinkingBox-Bench 외부의 학습 작업에서 동일한 종류의 실행 가능한 신호를 보상으로 사용했습니다. Qwen3.8-27B의 평균 성공률은 51.70%에서 60.78%로 증가했으며, 50 GRPO steps 후 20번의 시도 모두에서 올바르게 완료된 작업 수는 38개에서 107개로 증가했습니다.

[![fig1-hero](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/KVaFjYjblWd15tC3M6S_0.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/KVaFjYjblWd15tC3M6S_0.png)

그림 1. ThinkingBox는 ThinkingBox-Bench 외부의 작업에서 터미널 데이터베이스 상태에 대한 실행 가능한 검사를 사후 학습 에이전트의 보상으로 전환합니다.

#### 이 글은 Microsoft와 Hugging Face가 공동으로 작성했습니다. 공동 집필 및 검토에 참여해 주신 Microsoft의 Mirco Milletari와 Xiang Jiang, Hugging Face의 Sergio Paniego, 그리고 Pittsburgh 대학교 출신의 전 인턴 Zhuochun Li에게 특별히 감사드립니다.

우리는 아직 끝나지 않았는데도 끝났다고 말하는 에이전트를 포착하기 위해 ThinkingBox를 구축했습니다.

ThinkingBox-Bench의 각 507개 작업(당사의 벤치마크 데이터셋)은 에이전트에 실제 백엔드와 도구 세트를 제공하고 작업하게 한 다음, 데이터베이스에서 실제로 무엇이 변경되었는지 확인합니다. [In the first post](https://huggingface.co/blog/microsoft/thinkingbox), 한 모델이 고객의 티켓을 해결됨으로 종료했지만, 그 배후의 운송업체 예외 처리는 여전히 열려 있는 사례를 포착했습니다. 유능해 보이는 도구 호출은 9번이었지만, 잘못된 데이터베이스 행은 하나였습니다.

그때 우리가 무엇을 구축했는지 깨달았습니다. 대부분의 작업에서 실행 가능한 검사는 최종 데이터베이스 상태로부터 직접 1 또는 0을 반환할 수 있습니다. 이는 단순한 채점기가 아닙니다. 보상 함수가 될 수 있습니다.+

그래서 벤치마크 외부의 ThinkingBox 형식 작업으로 학습 풀을 구성하고 실행 가능한 결과를 보상으로 사용했습니다. 문제는 에이전트 학습에 적합한 작업을 선택하는 것이었습니다. 대부분의 작업은 현재 모델이 이미 해결했거나 도저히 해결할 수 없는 작업이며, 어느 쪽도 모델에 아무것도 가르치지 못합니다.

난이도는 정책 상대적입니다. 어떤 작업은 현재 모델이 학습하기 전까지만 어렵습니다.

이를 오프라인 정책 상대적 난이도라고 부릅니다. 학습 전에 기본 정책을 사용하여 각 작업의 성공 확률을 한 번 추정하고, 실행 내내 선택된 학습 풀을 고정했습니다.

### 학습 후 어떤 일이 일어났는가

오프라인 정책 상대적 난이도를 사용해 학습 풀을 구성한 다음, GRPO로 Qwen3.8-27B를 50 steps 동안 파인튜닝했습니다. 다시 설명하면, 507개의 벤치마크 작업을 각각 20번 평가합니다. pass@1은 전체 10,140번의 시도에서 평균 성공률을 측정합니다. pass@20은 최소 한 번이라도 해결된 작업의 비율을 측정합니다. Observed 20/20은 기록된 모든 시도에서 통과한 작업의 비율을 측정합니다.

표 1. 오프라인으로 선택된 학습 풀에서 GRPO 50 steps 전후의 Qwen3.8-27B 결과.

| 지표 | 기본 모델 |   | RL 파인튜닝 | 변화 |
| --- | --- | --- | --- | --- |
| Observed 20/20 | 38 (7.50%) | → | 107 (21.10%) | +69개 작업 |
| pass@1 (성공한 시도) | 5,242 (51.70%) | → | 6,163 (60.78%) | +9.08pt |
| pass@20 (최소 한 번 해결된 작업) | 453 (89.35%) | → | 446 (87.97%) | −7개 작업 |

학습을 통해 반복 가능한 성공은 크게 증가했지만, 최소 한 번의 성공을 기준으로 한 커버리지는 소폭 감소했습니다. 이 글의 나머지 부분에서는 연습 작업을 선택하고, 보상 루프를 구축하며, 이러한 트레이드오프를 측정한 방법을 설명합니다.

이 개선은 하나의 작업에서 어떤 모습으로 나타날까요? 한 여행자가 객실 등급을 낮춰 달라고 요청하지만, 회사 정책상 먼저 관리자의 승인이 필요합니다. 학습 전에는 에이전트가 20번의 시도 중 15번에서 승인 없이 예약을 변경하고 환불을 처리했습니다. RL 파인튜닝 후에는 20번 모두에서 예약 및 결제 상태를 유지했습니다.

[![fig2-hotel-task](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/iu-NFQl-J3EOMe_KiME_8.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/iu-NFQl-J3EOMe_KiME_8.png)

그림 2. 기업 호텔 예약 작업 하나에 대해 모델별로 20번 시도한 결과. 기본 Qwen3.8-27B 모델은 필요한 관리자 승인 없이 15번 예약을 변경했습니다. RL 파인튜닝 모델은 20번 모두에서 예약 및 결제 상태를 유지했으며, 19번에서는 모든 예상 레코드가 일치했습니다.

## 목차 {#section-1}

* [벤치마크가 60초 만에 보여준 결과](#section-2)
* [결과: 신뢰성 열이 움직였다](#section-3)
* [이득이 발생한 영역](#section-4)
* [신뢰성의 비용](#section-5)
* [학습 루프](#section-6)
* [오프라인 정책 상대적 난이도](#section-7)
* [보상을 받는 조건](#section-8)
* [학습 스택 실행](#section-9)
* [다음 단계](#section-10)

## 벤치마크가 60초 만에 보여준 결과 {#section-2}

[ThinkingBox-Bench](https://huggingface.co/datasets/microsoft/ThinkingBox-Bench)은 에이전트가 남긴 데이터베이스 레코드를 기준으로 에이전트를 채점합니다. 507개 작업 각각은 실제 백엔드를 시작하고, 에이전트에 MCP 도구 세트와 비공개 컨텍스트를 가진 시뮬레이션된 LLM 사용자를 제공하며, 마지막에는 터미널 데이터베이스 상태에 대해 실행 가능한 검사를 수행합니다. 검사는 특정 도구 호출이 아니라 올바르게 기록되어야 하는 백엔드 레코드를 대상으로 합니다. 이를 통해 모델은 사람이 만든 해법을 그대로 “재발견”하기보다, 더 새롭고 최적이 아닐 수도 있지만 올바른 해결 경로를 찾아내도록 유도됩니다.

[In our first post](https://huggingface.co/blog/microsoft/thinkingbox) 우리는 18개 모델을 이 벤치마크에서 각각 20번 실행하고, 헤드라인 정확도가 숨기는 격차를 보고했습니다. Kimi-K3는 최소 한 번 93.89%의 작업을 해결했지만, 507개 작업 중 68개(13.41%)만 20번의 기록된 시도 모두에서 완료했습니다. Claude Opus 5는 그 반대였습니다. 커버리지는 더 좁았지만 일관성은 훨씬 높았습니다.

더 크거나 더 나은 모델이라고 해서 이 격차가 해소된 것도 아닙니다. Claude Opus 5.5는 pass@1에서 Claude Opus 5보다 0.66%포인트 높았지만, 20번의 모든 시도에서 통과한 작업 수는 정확히 동일한 241개였습니다. 헤드라인 정확도의 향상으로 신뢰성은 전혀 증가하지 않았습니다.

실제 프로덕션 데이터베이스 레코드에 영향을 주는 작업에 사용할 모델을 선택한다면 pass@1만 보고 판단해서는 안 됩니다. Observed 20/20은 모든 시도에서 성공하는 작업이 무엇인지 보여 주며, 이는 프로덕션 신뢰성에 필요한 조건에 더 가깝습니다.

실행 가능한 터미널 상태 판정은 학습 보상이기도 했습니다. 학습 루프를 자세히 살펴보기 전에, 봉인된 벤치마크에서 무엇이 달라졌는지 확인해 보겠습니다.

## 결과: 신뢰성 열이 움직였다 {#section-3}

Qwen3.8-27B를 학습한 후, 학습에 한 번도 사용하지 않은 507개 작업 벤치마크에서 선택된 step-50 체크포인트를 평가했습니다. 이 507개 작업은 분포 외 테스트 세트를 구성합니다. 해당 도메인, 데이터베이스, 그리고 대부분의 MCP 도구는 RL 학습 중에 접하지 않았습니다.

표 2. [our first blog's Table 2](https://huggingface.co/blog/microsoft/thinkingbox)에서 선택한 모델과 다양한 도메인 및 전체에 대한 당사의 파인튜닝 모델의 도메인별 ThinkingBox-Bench pass@1(%). Observed 20/20 수치는 507개 작업 전체 벤치마크를 기준으로 합니다. 각 모델은 507개 작업 모두에 대해 20회의 반복 시험으로 평가됩니다. 굵은 글씨는 해당 열의 1위, 밑줄은 2위를 나타냅니다.

| 모델 | 소매 | 자동차 보험 | 여행 | 네오뱅크 | 컨설팅 | 전체 | Observed 20/20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 80.97 | 68.40 | 54.28 | 71.25 | 61.58 | 67.16 | 241 (47.53%) |
| Claude Opus 5 | 80.71 | 65.80 | 49.95 | 70.62 | 66.19 | 66.50 | 241 (47.53%) |
| GPT-5.4 | 76.33 | 62.65 | 68.12 | 65.34 | 54.60 | 65.36 | 128 (25.25%) |
| GPT-5.6 Sol | 67.65 | 65.30 | 60.34 | 59.09 | 57.52 | 61.91 | 82 (16.17%) |
| Qwen3.8-27B RLFT | 67.91 | 56.45 | 62.69 | 62.40 | 54.51 | 60.78 | 107 (21.10%) |
| Claude Sonnet 4.6 | 72.35 | 54.40 | 58.94 | 56.39 | 54.31 | 59.19 | 102 (20.12%) |
| GPT-6 Astra | 71.73 | 46.55 | 55.87 | 60.87 | 56.83 | 58.31 | 231 (45.56%) |
| Kimi-K3 | 82.24 | 50.80 | 61.83 | 41.35 | 51.63 | 57.37 | 68 (13.41%) |
| Qwen3.8-27B | 64.03 | 47.85 | 53.41 | 47.88 | 45.69 | 51.70 | 38 (7.50%) |

핵심 결과는 표 1의 observed-20/20 행에 있습니다. 처음에는 모든 시도에서 ThinkingBox-Bench의 38개(7.5%)를 완료했던 27B 모델이 이제 모든 시도에서 107개(21.1%)를 완료하며, 거의 3배가 되었습니다. 비교를 위해 Kimi-K3의 observed 20/20은 68개, GPT-5.4는 128개, GPT-6 Astra는 231개이며, Claude Opus 5.5가 241개로 가장 높습니다.

평균 성공률과 반복 가능성은 서로 다른 순위를 만듭니다. 표 2를 보면 RL 파인튜닝된 Qwen3.8-27B는 GPT-5.6 Sol보다 pass@1 점수가 약간 낮았습니다(61.91% 대 60.78%). 그러나 기록된 20번의 모든 시도에서 통과한 작업은 82개 대 107개로 더 적었습니다. Claude Sonnet 4.6은 pass@1이 59.19%로 더 낮았지만, 매번 통과한 작업 수는 102개로 거의 같았습니다. 어느 지표 하나만으로는 전체 상황을 포착할 수 없습니다.

관찰된 패턴은 날카로워짐(sharpening)과 일치합니다. 반복 가능한 성공은 증가한 반면 최소 한 번의 커버리지는 소폭 감소했습니다. 이는 [prior work](https://arxiv.org/abs/2504.13837)과 일치합니다. 한 가지 설명은 확률 분포가 반복적으로 작동하는 궤적으로 이동하면서 일부 저확률 경로가 사라졌다는 것입니다.

전반적으로 모델은 더 폭넓어지지 않았습니다. 대신 더 반복 가능해졌습니다. 다음 질문은 이러한 이득이 어디에서 발생했는가입니다.

## 이득이 발생한 영역 {#section-4}

[![fig2-domains](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/Bu7GFUdUiJgQjHV_cbvU4.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/Bu7GFUdUiJgQjHV_cbvU4.png)

그림 3. 도메인별 기본 점수, RLFT 점수 및 백분율포인트 변화. 녹색 변화 셀은 증가를, 빨간색 셀은 감소를 나타내며, 파란색 음영은 각 지표 내 점수 수준을 나타냅니다.

다섯 도메인 모두에서 Pass@1과 observed 20/20이 증가했습니다. Pass@20은 혼재된 결과를 보였습니다. 네오뱅크와 컨설팅에서는 증가했지만, 소매, 여행, 자동차 보험에서는 감소했습니다. 자동차 보험은 가장 큰 폭의 커버리지 손실(−7.0포인트)을 보였고, 소매는 observed-20/20이 가장 크게 증가했습니다(+24.5포인트).

## 신뢰성의 비용 {#section-5}

[first post](https://huggingface.co/blog/microsoft/thinkingbox)은 간단한 수식으로 일관성의 비용을 계산했습니다. 전체 20회 실행 캠페인의 추정 비용을 모든 시도에서 통과한 작업 수로 나누는 방식입니다.

신뢰 가능한 작업당 비용 = 507 × 20회 캠페인의 추정 비용 ÷ 20/20을 통과한 작업 수

파인튜닝된 체크포인트의 측정된 토큰 사용량에 동일한 비용 계산을 적용하면 다음과 같습니다.

표 3. 기본 Qwen3.8-27B와 RL 파인튜닝된 Qwen3.8-27B의 평가 비용 및 신뢰성 추정치. 성공한 시도당 비용은 전체 추정 캠페인 비용(507 x 20)을 통과한 시도 수로 나눈 값입니다.

| 측정값 | 기본 모델 |   | RL 파인튜닝 | 변화 |
| --- | --- | --- | --- | --- |
| 캠페인 비용, 20회 실행 | $925.80 | → | $855.58 | −7.6% |
| 성공한 시도당 비용 | $0.177 | → | $0.139 | −21.4% |
| 20/20 통과 작업 | 38 | → | 107 | 2.8배 |
| 신뢰 가능한 작업당 비용 | $24.36 | → | $8.00 | −67.2% |

신뢰 가능한 작업당 $8.00로, 파인튜닝된 모델은 함께 비교한 결과에서 GPT-5.4($6.80), GPT-6 Astra($7.45), Claude Opus 5.5($7.80)에 가까워졌습니다.

#### 단일 시도 관점

그림 4는 성공한 시도당 비용을 pass@1에 대해 표시합니다.

[![fig3-cost-frontier](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/HPq2TNCT8yskCAGtKe86l.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/HPq2TNCT8yskCAGtKe86l.png)

그림 4. 황금색 화살표는 Qwen3.8-27B 기본 모델의 Pareto 비용 위치에서 RLFT(다이아몬드)로 이동하는 경로를 나타냅니다. pass@1은 51.70%에서 60.78%로 증가하는 반면, 성공한 시도당 추정 비용은 $0.177에서 $0.139로 감소합니다. 테두리가 있는 원은 Pareto 비용 프런티어를 나타냅니다.

이 이동으로 파인튜닝된 체크포인트는 GPT-5.6 Sol에 가까워졌지만, GPT-5.6 Sol은 여전히 더 저렴하고 pass@1도 약간 높습니다. [our paper](https://arxiv.org/abs/2608.19741)에 가격이 책정된 4개 모델은 표시된 범위를 벗어납니다.

달러 값은 청구서나 클라우드 비용이 아니라 비교를 위한 서빙 지표입니다. 파인튜닝된 체크포인트를 서빙하는 비용은 기본 모델과 동일합니다. 계산에는 모델 추론 비용이 포함되며, 학습 실행 비용과 학습 데이터셋 선별을 위한 정책 프로브 비용은 제외됩니다.

이 결과는 무엇이 변했는지를 보여 줍니다. 다음 절에서는 학습 루프와 올바른 작업을 선택하는 일이 왜 가장 중요한지 설명합니다.

## 학습 루프 {#section-6}

[![fig4-training-overview](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/ZaqyG8RtqCRsxGdxUQMfi.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/ZaqyG8RtqCRsxGdxUQMfi.png)

그림 5. 이제 맥락 속에서 본 6개 단계입니다. 후보 작업이 입력되고, 처음 3개 단계는 오프라인에서 한 번 실행되며, 마지막 3개 단계는 학습 실행의 각 그룹마다 반복됩니다. 학습된 체크포인트는 ThinkingBox-Bench에 대해 평가됩니다.

선택은 기본 정책 모델, 즉 이 경우 Qwen3.8 27B에 대해 오프라인으로 한 번 실행되며, 학습에서는 선별된 고정 풀을 사용합니다. 그림 5의 점선 황색 화살표는 우리가 택하지 않은 경로, 즉 학습 실행 중 난이도를 다시 프로브하는 경로를 나타냅니다.

모든 롤아웃은 텍스트 완성이 아니라 완전한 에피소드입니다. 각각 새로운 백엔드, 격리된 MCP 세션, 그리고 시뮬레이션된 사용자 역할의 GPT-5 Chat을 사용합니다. 대부분의 보상은 결정론적 터미널 상태 검사에서 나오며, 일부 좁은 루브릭 요구사항은 구성된 judge(GPT-5 Chat)를 호출할 수 있습니다.

학습과 평가에서는 Qwen의 출력을 동일하게 해석해야 합니다. 여기에는 추론 추출, 도구 호출, 인자 형식 지정, 렌더링, 격리된 작업 상태가 포함됩니다. 그렇지 않으면 보상은 평가된 에이전트와 다른 에이전트를 측정하게 됩니다. 공유 해석 코드는 [`trainer/core/tb_interpret.py`](https://github.com/microsoft/thinkingbox-training/blob/8c04db18da362357dc8a2d4ab187f44a8d63e20a/trainer/core/tb_interpret.py)에 있습니다.

참조 실행에서는 token-mean GRPO를 사용하여 187개 작업에서 전체 파라미터 RL 파인튜닝 Qwen3.8-27B를 실행했습니다. 각 업데이트는 24개의 H100에서 8개 롤아웃으로 구성된 18개 그룹, 즉 144개 에피소드를 생성했습니다. 학습은 파라미터 및 옵티마이저 오프로드를 사용해 동기식으로 실행되었으며, AdamW는 `1e-6` learning rate와 weight decay 0을 사용했습니다. 학습은 50 steps 동안 실행했습니다.

분산이 0인 그룹은 학습 중 [DAPO-style](https://arxiv.org/abs/2503.14476)로 대체했습니다. 시스템 오류는 별도로 처리했습니다. 백엔드 타임아웃은 잘못된 에이전트 결과가 아니므로, 영향을 받은 그룹은 실패로 점수를 매기는 대신 다시 채웠습니다.

이로써 유지된 작업을 어떻게 학습했는지 설명했습니다. 남은 질문은 어떤 작업이 학습 풀에 들어갔는가입니다.

## 오프라인 정책 상대적 난이도 {#section-7}

학습 작업 선택을 자세히 살펴보기 전에 한 가지 고지 사항을 전합니다.

ThinkingBox-Bench 작업은 RL 학습 풀에 포함하지 않았습니다. 벤치마크는 평가를 위해 보류했습니다.

ThinkingBox-Bench는 RL 학습 풀에 비해 분포 외에 해당합니다. 해당 도메인, 데이터베이스, 정책, 대부분의 도구는 RL 학습 중에 접하지 않았습니다. 그러나 두 세트는 ThinkingBox 프레임워크, 시뮬레이션된 사용자 장치, 실행 가능한 터미널 상태 채점, 일부 공통 도구 및 유사한 작업 스타일을 공유합니다. 이러한 공유 요소는 전이 주장의 범위를 제한합니다.

선택이 중요한 이유는 GRPO에 관한 한 가지 산술적 사실로 요약됩니다. 8개의 롤아웃이 모두 실패하면 보상 벡터는 `[0, 0, 0, 0, 0, 0, 0, 0]`입니다. 8개가 모두 성공하면 `[1, 1, 1, 1, 1, 1, 1, 1]`입니다. 두 경우 모두 그룹 내 중심화는 작업 보상 이점을 만들지 못합니다. 어떤 궤적도 다른 궤적보다 더 나은 결과를 내지 못했기 때문입니다.++

`[0, 0, 0, 1, 1, 1, 1, 1]`과 같은 혼합 그룹은 GRPO가 학습할 수 있는 대조를 제공합니다. 이러한 차이를 얻는 데는 상태를 유지하는 에이전트의 경우 큰 비용이 듭니다. 모든 롤아웃은 격리된 백엔드를 프로비저닝하고, 실제 도구 세션을 실행하며, 사용자를 연기할 별도의 모델을 호출합니다.

따라서 이 작업의 방향을 결정한 질문은 단순히 어떤 모델이나 옵티마이저를 사용할 것인가가 아니었습니다. 후보 작업 중 어떤 작업을 고정 학습 풀에 넣어야 하는가였습니다.

학습 풀을 선택하기 위해 posterior에 대해 오프라인 통과율 필터를 사용했습니다. 이 필터는 기본 정책에서 대조적인 터미널 보상을 만들어 낼 가능성이 높은 작업을 추정합니다.

그래서 프로브를 실행합니다. 각 후보 작업을 기본 정책 모델에 대해 반복 실행합니다. 규칙은 간단합니다. 모델이 항상 실패하는 작업은 아직 모델을 가르칠 가능성이 낮고, 항상 통과하는 작업은 이미 학습된 것이며, 가끔 해결하는 작업에서 결과의 대조가 생길 수 있습니다.

### 제한된 프로브로 난이도 추정하기

각 후보 작업을 n번 실행하여 k개의 성공을 얻습니다. Jeffreys posterior probability를 사용하여 잠재 성공 확률을 모델링합니다.

성공 확률 | n번의 프로브에서 k번 성공 ~ Beta(k + ½, n − k + ½)

[![fig5-policy-relative-difficulty](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/eR45y7Dl0VFFK3HinwDYg.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/eR45y7Dl0VFFK3HinwDYg.png)

그림 6. 오프라인 정책 상대적 난이도는 기본 정책을 반복 프로브하여 추정하며, 대조적인 보상을 만들어 낼 가능성이 높은 작업을 유지하는 데 사용됩니다.

단위 구간을 0.08, 0.40, 0.60, 0.90을 경계로 하는 5개 버킷으로 나눕니다. 이 경계는 이번 실행을 위한 하이퍼파라미터로 선택했습니다. 각 작업은 posterior 질량이 가장 큰 버킷에 할당됩니다.

- 거의 해결되지 않음 [0, 0.08): 제외

- 어려움 [0.08, 0.40): 유지

- 중간 [0.40, 0.60): 유지

- 자주 해결됨 [0.60, 0.90): 전체 파라미터 RLFT를 위해 유지

- 거의 항상 해결됨 [0.90, 1]: 제외

자세한 내용은 [Appendix C.4 of our ThinkingBox paper](https://arxiv.org/pdf/2608.19741)에 있습니다.

### 하나의 작업을 처음부터 끝까지

[![fig6-worked-example](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/IBBkNpYKQRK6jBLSusMXw.png)](https://cdn-uploads.huggingface.co/production/uploads/64b8491203124195cd795cad/IBBkNpYKQRK6jBLSusMXw.png)

그림 7. 하나의 작업을 6개 단계 전체에 걸쳐 확대한 모습입니다. GPT-5.4 선별, 기본 모델 프로브, 난이도 분류, 롤아웃 그룹, 터미널 상태 채점, GRPO 업데이트입니다.

그림 7은 학습 풀에 들어갈 수 있는 후보 작업 하나를 따라갑니다. Qwen3.8 27B 전체 파라미터 풀에서 GPT-5.4는 먼저 해결 가능성 선별기 역할을 했습니다. 그 다음 기본 정책은 프로브 중 일부만, 예를 들어 8번 중 3번만 성공하여 Beta(3.5, 5.5) posterior를 만들었습니다. 가장 큰 질량인 54.0%는 어려움 구간에 속했습니다.

학습을 위해 샘플링했을 때 이 작업은 혼합 보상 그룹 `[0, 0, 0, 0, 1, 1, 1, 1]`을 생성했습니다. 따라서 성공한 궤적에는 평균보다 높은 작업 advantage가, 실패한 궤적에는 평균보다 낮은 advantage가 주어져 GRPO가 학습할 대조를 제공했습니다. 다른 정책 모델이라면 동일한 작업을 다른 난이도 구간에 배치할 수 있습니다.

최종적으로 선별된 Qwen3.8-27B 학습 풀에는 187개 작업이 포함되었습니다.

## 보상을 받는 조건 {#section-8}

궤적 자체와는 아무런 관련이 없습니다. 보상은 작업 자체의 실행 가능한 판정입니다.

에피소드가 끝나면 샌드박스는 백엔드에서 실제로 변경된 사항을 추출하고, 터미널 상태에 대해 작업의 결정론적 검사를 실행합니다. 올바른 최종 상태를 만들어 내는 모든 궤적은 통과합니다. 잘못되었거나 누락되었거나 추가된 부작용은 실패합니다. 보상은 1 또는 0입니다.

벤치마크에서 데이터베이스 채점기는 “시험관”으로 볼 수 있습니다. 성공했다고 보고했지만 잘못된 레코드를 남긴 에이전트를 포착했기 때문입니다. 학습에서는 동일한 종류의 검사를 서로 분리된 다른 작업 세트에 대한 “교사”로 볼 수 있습니다.

두 가지 보상 경로, 하나의 이진 판정

- 실행 가능한 검사(대다수). 필요한 결과가 터미널 상태나 부작용으로 표현되는 경우 결정론적 테스트가 이를 승인하거나 거부하며, LLM judge 호출은 필요하지 않습니다.

- judge 지원 검사(소수). 데이터베이스 상태만으로 좁은 요구사항을 명확히 판단할 수 없는 경우, 구성된 LLM judge가 실행 가능한 테스트에서 사용하는 이진 루브릭 질문에 답합니다.

## 학습 스택 실행 {#section-9}

Qwen3.8-27B 학습에 사용한 데이터 선택, 롤아웃, 보상 및 Verl 통합 코드를 [microsoft/thinkingbox-training](https://github.com/microsoft/thinkingbox-training)에서 공개합니다. 저장소 README가 시작점입니다. 자세한 가이드는 다음과 같습니다.

- 작업 hydration, 정책 프로브, 벤치마크 제외, train/validation 분할은 [`docs/data-preparation.md`](https://github.com/microsoft/thinkingbox-training/blob/8c04db18da362357dc8a2d4ab187f44a8d63e20a/docs/data-preparation.md);

- 검증된 24-H100 Qwen3.8 구성은 [`docs/reference-q38-run.md`](https://github.com/microsoft/thinkingbox-training/blob/8c04db18da362357dc8a2d4ab187f44a8d63e20a/docs/reference-q38-run.md);

- 체크포인트 병합 및 ThinkingBox-Bench 507 x 20 실행은 [`docs/evaluation.md`](https://github.com/microsoft/thinkingbox-training/blob/8c04db18da362357dc8a2d4ab187f44a8d63e20a/docs/evaluation.md).

### 시작하기 전에

Python 3.12 및 [uv](https://docs.astral.sh/uv/) 환경의 Linux와 WSL에서 테스트했습니다. 호환되는 CUDA/NCCL 스택을 갖춘 NVIDIA GPU, 로컬 Qwen3.8-27B 체크포인트, 그리고 시뮬레이션된 사용자와 judge를 위한 OpenAI 호환 엔드포인트가 필요합니다. 검증된 참조 구성은 3 nodes x 8 H100 GPUs를 사용합니다. 더 작은 사용자 지정 구성은 모델 크기, 컨텍스트 길이, 롤아웃 동시성에 따라 달라집니다.

공개 샘플 작업 목록은 인터페이스를 실행해 봅니다. 이는 보고된 실행에 사용한 비공개 187개 작업 학습 풀이 아닙니다. 공개 벤치마크와 분리된 상태를 유지하는 승인된 학습 소스를 제공하세요.

### 설치

세 저장소를 나란히 배치하세요.

```
mkdir -p ~/thinkingbox-workspace
cd ~/thinkingbox-workspace

git clone https://github.com/microsoft/thinkingbox.git
git clone https://github.com/microsoft/thinkingbox-data.git
git clone --branch main https://github.com/microsoft/thinkingbox-training

git -C thinkingbox checkout 892964e7226044e5463ad188da119df838f4bec1
git -C thinkingbox-data checkout thinkingbox-bench-v1.0
```


### 검증된 환경 생성

공개된 스택은 Linux, Python 3.12, Verl 0.9.0 및 고정된 패키지 잠금 파일을 대상으로 합니다.

```
cd ~/thinkingbox-workspace/thinkingbox-training

uv venv --python 3.12
source .venv/bin/activate

uv sync --frozen --extra fast --extra tracking --extra eval-proxy
uv pip install -e ../thinkingbox
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/thinkingbox_tools
uv pip install --config-settings editable-mode=compat \
  -e ../thinkingbox-data/servers/tb_business_ops_servers_202606

python scripts/prepare_verl.py --dest .deps/verl --install
python scripts/verify_verl_install.py
uv pip check
```


마지막 두 명령은 검증된 패치 Verl 체크아웃과 손상된 요구사항이 없음을 보고해야 합니다. 이후 자동 패키지 동기화를 다시 실행하지 마세요. 패치된 소스 설치가 vanilla Verl wheel로 대체될 수 있습니다.

### 학습 데이터 준비

공개적으로 제공된 샘플로 시작하거나, ThinkingBox 형식으로 승인된 자체 데이터셋을 추가하세요. 그런 다음 저장소의 준비 스크립트를 실행합니다.

```
cd ~/thinkingbox-workspace/thinkingbox-training
source .venv/bin/activate
python data/prepare_data.py --help
```


스크립트는 정책 프로브를 집계하고, 오프라인 정책 상대적 난이도 필터와 벤치마크 제외를 적용하며, 선택된 `train.yaml` 및 `test.yaml` 파일을 작성합니다. [`docs/data-preparation.md`](https://github.com/microsoft/thinkingbox-training/blob/8c04db18da362357dc8a2d4ab187f44a8d63e20a/docs/data-preparation.md)에는 입력 필드, 로더 계약, 전체 명령이 설명되어 있습니다.

### 실행 구성 및 검사

[`docs/reference-q38-run.md`](https://github.com/microsoft/thinkingbox-training/blob/8c04db18da362357dc8a2d4ab187f44a8d63e20a/docs/reference-q38-run.md)에 따라 Typesense와 MCP 서버를 시작하고, 시뮬레이션된 사용자와 judge를 구성하며, 모델, 작업 목록, 체크포인트 및 토폴로지 변수를 설정하고, Ray 클러스터를 준비합니다. 자격 증명은 커밋된 파일이나 셸 기록이 아니라 런타임 비밀 주입으로 관리하세요.

학습 작업을 할당하기 전에 가이드의 dry-run 명령을 사용하세요.

```
DRY_RUN=true ./scripts/run_train.sh \
  trainer.total_training_steps=50 \
  trainer.save_freq=5
```


이 명령은 Ray를 시작하거나 GPU를 할당하지 않고 종료 코드 0으로 끝나야 합니다. 출력된 `python -m verl.trainer.main_ppo` 명령과 실행을 시작하기 전에 해석된 모든 오버라이드를 확인하세요.

### 학습

런처는 학습을 구성하지만 호스트를 프로비저닝하지는 않습니다. 참조 3노드 토폴로지에서는 모든 노드에 동일한 환경을 내보내고 `CHECKPOINT_DIR`을 동일한 절대 경로에 마운트한 다음 Ray를 시작하세요.

```
# head node
ROLE=head NODE_IP="$HEAD_IP" GPUS_PER_NODE=8 \
  ./scripts/start_ray.sh

# each worker node
ROLE=worker NODE_IP="$WORKER_IP" \
  HEAD_ADDRESS="$HEAD_IP:6379" GPUS_PER_NODE=8 \
  ./scripts/start_ray.sh
```


`ray status`이 24개의 GPU를 모두 보고하면 헤드 노드에서 한 번 실행하세요.

```
export RAY_ADDRESS=auto
export GPUS=0,1,2,3,4,5,6,7

./scripts/run_train.sh \
  trainer.project_name=thinkingbox-training \
  trainer.experiment_name=q38-full-rlft \
  trainer.total_training_steps=50 \
  trainer.save_freq=5
```


### OpenEnv를 통해 체크포인트 평가

[first ThinkingBox
post](https://huggingface.co/blog/microsoft/thinkingbox#run-it-yourself)에는 Typesense, MCP Session Proxy, OpenEnv 설정이 포함되어 있습니다. [`docs/evaluation.md`](https://github.com/microsoft/thinkingbox-training/blob/8c04db18da362357dc8a2d4ab187f44a8d63e20a/docs/evaluation.md)에 따라 학습된 체크포인트를 병합한 다음 vLLM으로 서빙합니다.

OpenEnv 저장소 루트에서 `MERGED_MODEL`을 병합된 모델 디렉터리로 설정하고 `THINKINGBOX_DATA`을 고정된 벤치마크 체크아웃으로 설정합니다. 과거 체크포인트 평가에서는 다음의 eight-A100 서빙 구성을 사용했습니다.

```
vllm serve "$MERGED_MODEL" \
  --served-model-name q38-rlft \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --language-model-only \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```


별도의 터미널에서 ThinkingBox Training 체크아웃에서 Qwen chat template용 OpenEnv 호환성 프록시를 시작합니다. 프록시는 포트 8014에서 수신하고 vLLM의 8001로 전달합니다.

```
cd ~/thinkingbox-workspace/thinkingbox-training
source .venv/bin/activate
bash scripts/launch_sysmerge.sh
```


평가하는 동안 계속 실행해 두세요. 다른 터미널에서 준비 상태를 확인합니다.

```
curl --fail http://127.0.0.1:8014/readyz
```


기존 `thinkingbox.yaml`에서 에이전트 항목을 업데이트하되, MCP, 시뮬레이션된 사용자 및 judge 구성은 유지하세요.

```
orchestrator:
  agent_model:
    type: aoai
    endpoint_url: http://127.0.0.1:8014/v1/chat/completions
    deployment: q38-rlft
    credential:
      type: api-key
      api_key: <your_api_key>
    is_reasoning: true
    temperature: 1.0
    top_k: 20
    top_p: 0.95
    max_completion_tokens: 8192
    timeout: 1800
```


첫 번째 게시물과 동일하게 OpenEnv를 시작하되, `OPENENV_TB_CONFIG`은 이 `thinkingbox.yaml`을 가리키고 `OPENENV_TB_DATASET`는 `$THINKINGBOX_DATA`을 가리키도록 합니다. `/ready`이 성공하면 507개 작업 × 20회 시도의 전체 프로토콜을 실행합니다.

```
uv run --project envs/thinkingbox_env --frozen \
  thinkingbox-eval \
  --config "$PWD/thinkingbox.yaml" \
  --base-url http://127.0.0.1:8000 \
  --dataset "$THINKINGBOX_DATA" \
  --output q38-rlft-full.jsonl \
  --errors-output q38-rlft-full-errors.jsonl \
  --repeat 20 \
  --coverage-profile full \
  --aggregate-output q38-rlft-full-aggregate.json \
  --message-timeout 1800
```


`full` 프로필은 집계 전에 예상되는 10,140개의 모든 작업/반복 키를 검증합니다.

## 다음 단계 {#section-10}

이 작업의 유용한 부분은 Qwen3.8-27B 체크포인트 하나가 아닙니다. 실행 가능한 결과를 보상으로 전환하는 학습 루프와 학습 데이터 큐레이션입니다.

실제 데이터베이스 레코드에 영향을 주는 에이전트를 학습하거나 평가한다면 다음을 따르세요.

- 학습 전에 프로브하세요. 자체 에이전트 작업 세트를 구성하고, 기본 정책을 n번 실행한 다음, 구간 히스토그램을 확인하세요. 우리의 경험상 팀은 이 분포를 검사하지 않는 경우가 많으며, 대개 예상보다 더 한쪽으로 치우쳐 있습니다.

- 패리티를 확인하세요. 동일하게 기록된 모델 출력을 학습 및 평가 파서에 재생하고, 양쪽이 동일한 도구 이름과 인자를 추출하는지 검증하세요. 그렇지 않으면 학습이 평가에서 다르게 해석하는 행동에 보상을 줄 수 있습니다.

- 커버리지와 반복 가능성을 모두 보고하세요. k번의 시도에서 최소 한 번 통과한 작업 수와 k번 모두 통과한 작업 수를 명시하세요. 가끔 해결책을 찾는 모델이 반드시 일관되게 이를 반복하는 모델인 것은 아닙니다.

자세한 내용은 다음 링크에서 확인할 수 있습니다.

- 환경: [`envs/thinkingbox_env`](https://github.com/huggingface/OpenEnv/tree/main/envs/thinkingbox_env)

- OpenEnv: [ThinkingBox environment documentation](https://huggingface.co/docs/openenv/environments/thinkingbox)

- 프레임워크: [microsoft/thinkingbox](https://github.com/microsoft/thinkingbox) · [tutorial](https://github.com/microsoft/thinkingbox/blob/main/docs/tutorial.md)

- 벤치마크: [microsoft/thinkingbox-data](https://github.com/microsoft/thinkingbox-data) · [v1.0 release](https://github.com/microsoft/thinkingbox-data/releases/tag/thinkingbox-bench-v1.0)

- 데이터셋 뷰어: [microsoft/ThinkingBox-Bench](https://huggingface.co/datasets/microsoft/ThinkingBox-Bench)

- 논문: [arXiv:2608.19741](https://arxiv.org/abs/2608.19741) 또는 [HF](https://huggingface.co/papers/2608.19741)

- RL 학습: [microsoft/thinkingbox-training](https://github.com/microsoft/thinkingbox-training)

- 블로그: [The Agent Said It Was Done. The Database Disagreed.](https://huggingface.co/blog/microsoft/thinkingbox)

### 논문 인용

```
@article{li2026thinkingbox,
  title   = {One Success Isn't Reliability: Thinkingbox, a Sandbox and Benchmark for Agents in Stateful Business Workflows},
  author  = {Li, Zhuochun and Ko, Youngmin and Keramati, Ali and Kundu, Tuhin and
             Tsai, Liang-Chun and Ferri, Nicola and Milletari, Mirco and Liu, Jiaxiang and
             Lopez Pelaez, Susana Palmaz and Wang, Yuepeng and Smolyakov, Vadim and
             Jiang, Xiang and Olafsson, Kjartan and Guy, Tommy},
  journal = {arXiv preprint arXiv:2608.19741},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.19741}
}
```


ThinkingBox 코드는 MIT 라이선스를 따르며, 벤치마크 데이터는 [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/)입니다.

고지 사항: 공개 벤치마크의 모든 작업은 합성적으로 재구성된 것입니다. 워크플로와 정책은 실제 엔터프라이즈 에이전트 패턴을 기반으로 모델링되었지만, 고객은 실제 인물이 아닙니다.

ThinkingBox, ThinkingBox-Bench 및 ThinkingBox-Training은 Microsoft Copilot Studio 팀이 Toloka와 협력하여 구축했으며, Microsoft에서 인턴으로 근무한 Pittsburgh 대학교, Northwestern University, Columbia University 및 UC Irvine의 협력자들이 함께 참여했습니다. 질문은 아래 댓글 섹션이나 [github](https://github.com/microsoft/thinkingbox/discussions), 또는 [firstnamelastname@microsoft.com](mailto:firstnamelastname@microsoft.com) 이메일로 보내 주세요.

+ 일부 작업에는 데이터베이스가 명확하게 포착할 수 없는 결과에 대한 좁은 이진 루브릭 요구사항도 포함됩니다. 이러한 검사는 구성된 LLM judge를 호출할 수 있습니다.

++ 이 0-gradient 문제에는 표준적인 배치 시간 해결책이 있습니다. 오버샘플링하고, 퇴화된 그룹을 버린 다음, 다시 채우는 것입니다. 전체 파라미터 실행은 [DAPO](https://arxiv.org/abs/2503.14476)과 유사하게 0-분산 그룹을 대체하여 정확히 이 작업을 수행했습니다. 그러나 다시 채우기를 하더라도 버리기 전에 낭비된 모든 롤아웃에 대해 전체 비용을 지불해야 합니다.
