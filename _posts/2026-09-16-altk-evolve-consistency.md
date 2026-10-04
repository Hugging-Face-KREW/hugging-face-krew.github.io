---
layout: post
title: "에이전트가 작업을 해냈다. 다시 해낼 수 있을까?"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/dI5J2sSc3TSprk9VB4EVJ.jpeg
image: assets/images/blog/posts/2026-09-15-altk-evolve-consistency/thumbnail.jpeg
authors:
  - user: ibm-research
slug: "altk-evolve-consistency"
source_url: "https://huggingface.co/blog/ibm-research/altk-evolve-consistency"
source_published_date: "2026-09-15"
source_published_at: "2026-09-15T16:00:44+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Your Agent Aced the Task. Will It Do It Again?](https://huggingface.co/blog/ibm-research/altk-evolve-consistency)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-research/altk-evolve-consistency -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 에이전트가 작업을 해냈다. 다시 해낼 수 있을까?

에이전트는 리허설에서는 제대로 작동하지만, 실제 데모에서는 다른 경로를 택해 같은 작업에 실패합니다.

무대 위에서는 난처한 일입니다. 프로덕션에서는 신뢰성 문제입니다. 한 번 성공한 워크플로가 다음에 사용자가 같은 요청을 했을 때 실패할 수 있기 때문입니다. 금융 거래를 조정하거나 계약에서 의무 조항을 확인하는 일처럼 미션 크리티컬한 작업에서는 치명적인 문제가 될 수 있습니다.

대부분의 벤치마크는 이 변동성을 평균값 뒤에 숨깁니다. AppWorld에서 GPT-4.1을 사용하는 ReAct 에이전트는 5회 반복 실행에서 평균 77.4%의 성공률을 보였습니다. 하지만 5회 모두 성공한 작업은 전체의 53.0%에 불과했습니다. 일관성 격차는 24.4포인트였습니다.

대부분의 벤치마크는 첫 번째 수치를 보고합니다. 우리는 두 번째 수치를 측정하고 개선하는 방법을 만들었습니다.

이전 글에서는 [ALTK-Evolve](https://huggingface.co/blog/ibm-research/altk-evolve)을 소개했습니다. 이는 에이전트의 과거 궤적을 재사용 가능한 가이드라인으로 바꾸고, 이를 자동으로 추출해 추론 시점에 다시 주입하는 시스템입니다. 이 시스템은 작업 성공률을 측정 가능하게 향상하지만, 당시 결과 역시 평균적인 경우만을 묻고 있었습니다. 이 글에서는 이 격차를 직접 겨냥하는 새로운 가이드라인 유형인 일관성 가이드라인을 소개합니다. 일관성 가이드라인은 Consistency Analyzer라고 부르는 진단 도구를 기반으로 [`altk-evolve`](https://github.com/AgentToolkit/altk-evolve)에 추가되었습니다.

TL;DR

- 정확도는 신뢰성 문제를 가립니다. 평균적으로 77.4%의 성공률을 보이는 ReAct 에이전트(GPT-4.1 on AppWorld `test_normal`)가 5회 반복 실행 모두에서 성공하는 작업은 53.0%에 불과합니다. 일관성 격차는 24.4포인트이며, 어려운 작업에서는 30포인트에 달합니다.

- 우리는 바로 이 문제를 위한 진단 도구를 만들었습니다. Consistency Analyzer는 에이전트가 기록한 자신의 궤적을 다시 샘플링해, 결과가 뒤집히기 쉬운 의사결정 지점을 찾습니다. 이는 모델이 다른 행동을 하기까지 토큰 하나만 더 샘플링하면 되었던 단계입니다. 하나의 트레이스와 ground truth 없이 동작하며, 작업을 처음부터 끝까지 다시 실행하는 대신 해당 트레이스의 각 의사결정 지점에서 한 번의 호출로 k개의 완료를 요청해 다시 샘플링합니다(k=5가 기본값).

- 이 진단을 가이드라인으로 바꾸면 격차가 절반으로 줄어듭니다. 평균 정확도에는 아무런 비용도 발생하지 않으면서 24.4pp에서 12.0pp로 감소합니다(동일 작업의 Pass⁵ +16.0pp, 유사 작업 +13.0pp).

- 전체 방법론과 평가는 [technical report on arXiv](https://arxiv.org/abs/2609.08832)에 있습니다.

## 거의 아무도 보고하지 않는 지표 {#section-1}

표준 에이전트 평가는 Mean@k를 보고합니다. 벤치마크를 k회 실행한 뒤 통과율의 평균을 내는 방식입니다. k=3인 경우가 많고, 때로는 1회만 실행합니다. 이는 모든 리더보드에 표시되는 수치이며, 실제로 "77% 정확하다"는 말이 의미하는 바입니다.

Mean@k는 "이 에이전트는 평균적으로 얼마나 좋은가?"라는 질문에 답합니다. 하지만 실제 사용자가 궁금해하는 질문에는 답하지 못합니다. "이 정확한 질문을 다시 했을 때도 여전히 잘할까?" 이에 답하려면 Pass^k가 필요합니다. Pass^k는 에이전트가 k회 실행 모두에서 성공한 작업의 비율입니다.

⚠️ Pass^k는 Pass@k가 아닙니다. 익숙한 Pass@k는 낙관적인 지표입니다. k번의 시도 중 최소 한 번 성공했는지를 묻으며, 검증하고 재시도할 수 있을 때 적절한 질문입니다. Pass^k는 그 비관적인 거울상입니다. 모든 시도가 성공해야 합니다. 같은 문자지만 정반대의 질문입니다. 항상 Pass^k ≤ Mean@k ≤ Pass@k입니다.

![Mean@5 vs. Pass^5 by task difficulty, GPT-4.1 on AppWorld test_normal, with the consistency gap called out in red](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/BdzmSgkon1dAtbOCWqmKF.png)

GPT-4.1을 기반으로 하는 ReAct 에이전트는 Mean@5 77.4%를 기록합니다. 이는 실제로 매우 높은 수치입니다. 하지만 Pass^5는 53.0%에 불과합니다. 벤치마크의 거의 4분의 1은 작업 자체에는 아무런 변화가 없는데도 에이전트가 때로는 해결하고 때로는 해결하지 못하는 작업으로 구성됩니다. 우리는 이 격차, 즉 Mean@k에서 Pass^k를 뺀 값을 일관성 격차라고 부릅니다.

이는 더 큰 모델로 해결할 수 있는 능력 문제가 아닙니다. 서로 직교하는 축의 문제입니다. 에이전트는 능력이 뛰어나면서 동시에 일관성이 없을 수 있습니다.

## 에이전트의 선택이 뒤집히는 이유: 날카로운 결정과 평평한 결정 {#section-2}

LLM 에이전트가 무언가를 결정할 때마다, 즉 어떤 API를 호출할지, 어떤 인자를 전달할지, 재시도할지 여부를 결정할 때마다 그 결정은 다음 토큰에 대한 확률 분포에서 나옵니다. 중요한 것은 그 분포의 형태입니다. 날카로운 분포는 대부분의 확률 질량을 하나의 토큰에 둡니다. 2순위 후보는 크게 뒤처지고, 실행할 때마다 같은 선택이 나옵니다. 평평한 분포는 서로 비슷한 확률을 가진 여러 토큰에 확률 질량을 분산하며, 어느 토큰이 승자가 될지는 동전 던지기에 가깝습니다.

분포의 형태가 결과를 바꾸는 데 필요한 노이즈의 크기를 결정합니다. 날카로운 분포는 견고합니다. GPU 부동소수점 연산의 비결합성, 요청 배치 처리 및 기타 플랫폼 측 효과가 수치를 조금씩 흔들 수는 있지만, 명확한 승자의 순서를 바꿀 만큼 크지는 않습니다. 평평한 분포는 바로 그런 흔들림에 취약합니다. 근소한 차이로 앞선 후보의 순서가 작은 섭동만으로도 바뀔 수 있습니다. 게다가 하나의 궤적은 수십 개의 결정을 연쇄적으로 연결하므로, 단계별로 선택이 뒤집힐 작은 확률이 누적되어 어떤 실행에서는 경로가 달라질 가능성이 커집니다. 24포인트의 격차는 여기서 발생합니다.

이 문제는 디코딩 설정을 바꿔도 사라지지 않습니다. Greedy decoding과 고정 시드는 분포를 토큰으로 변환하는 방식을 제어할 뿐이며, 분포 자체에 대해서는 아무것도 말해주지 않습니다. 호스팅 엔드포인트에서는 실행할 때마다 확률이 조금씩 달라지므로, temperature zero에서 같은 모델에 같은 프롬프트를 보내도 오늘은 근소한 차이의 승자를 한쪽으로 결정하고 내일은 다른 쪽으로 결정할 수 있습니다.

우리의 설정은 다음과 같습니다. ReAct 에이전트는 temperature 0.0에서 실행되므로, 위에서 설명한 변동성 중 어느 것도 일반적인 샘플링에 의한 것이 아닙니다.

## 진단한 다음, 수정하기 {#section-3}

이로써 문제는 탐색 문제로 바뀝니다. 주어진 궤적에서 어떤 단계가 평평했는지, 그리고 이를 파악한 뒤 무엇을 할 것인지 찾는 문제입니다.

일관성 가이드라인은 ALTK-Evolve의 기존 도구에 연결되는 2단계 파이프라인에서 생성되며, 무엇을 작성할지 결정하는 새로운 소스 신호를 사용합니다.

![consistency-guideline-pipeline](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/gdUd7l6Q8DzXCgzmchxi3.png)

1. 감지 — Consistency Analyzer. 
기록된 궤적 하나를 입력으로 받아 analyzer는 제어된 재샘플링을 통해 각 의사결정 단계를 재생하고, 해당 시점에서 모델의 출력이 실제로 얼마나 변하는지 측정합니다. 구체적으로는 의사결정 단계마다 모델을 한 번 추가로 호출하며, 오프라인에서 한 번만 수행합니다. 이 호출은 한 번에 k개의 완료를 샘플링하도록 sampling parameter를 설정해 실행합니다(k=5가 기본값). 이미 기록된 컨텍스트를 대상으로 재생하며, 새로운 도구 호출이나 새로운 환경 상호작용을 하지 않고 작업을 처음부터 끝까지 두 번째로 롤아웃하지도 않습니다. 그 결과 각 의사결정 단계에 대한 일관성 점수가 생성되고 scorecard에 기록되어, 다음 실행에서 정확히 어떤 결정이 뒤집힐 위험이 있는지 식별할 수 있습니다. 감지는 완전한 블랙박스 방식으로 이루어집니다. logits나 모델 내부 정보가 필요하지 않으며, 이미 보유한 트레이스 외에 별도의 계측도 필요하지 않습니다.

2. 생성 — 대상 지정 가이드라인. 표시된 각 단계는 표준 ALTK-Evolve 형식의 일관성 가이드라인 후보가 되므로, 기존 저장 및 검색 파이프라인에 그대로 연결됩니다. 다음은 AppWorld 작업 "How many activities are done in my bucket list as per my SimpleNote note?"의 궤적에서 GPT-4.1이 생성한 실제 예시입니다.

[Guideline 1] 노트 콘텐츠에서 체크박스 스타일 마커를 셀 때는 일반적인 부분 문자열 개수 세기보다 줄의 시작에 고정된 정규식 매치를 사용하세요. 노트 제목에는 범례 줄에 동일한 마커 기호가 반복되는 경우가 많습니다.

[Guideline 2] 노트 쿼리의 검색 결과를 사용할 때는 항상 여러 매치가 있는지 확인하고, 진행하기 전에 올바른 노트인지 확인하세요.

여기에는 작업에만 해당하는 사소한 정보가 없습니다. 문자열 개수 세기 버그와 검증되지 않은 검색 결과는 많은 AppWorld 작업에서 높은 불확실성을 보이는 의사결정 지점입니다. 이것이 핵심입니다. analyzer는 실패가 아니라 불안정성을 겨냥하므로, 에이전트가 이번에는 우연히 올바르게 처리했지만 다음에는 쉽게 틀릴 수 있는 단계를 포착합니다.

[Watch the 2-minute demo](https://www.youtube.com/watch?v=qlp7EzXe8Pg) — 에이전트를 병렬로 5회 실행하면 에이전트가 개수 세기 전략을 확신하지 못해 이 작업에서 3-2로 나뉘지만, 컨텍스트에 이 가이드라인을 넣고 다시 실행하면 5회 모두 같은 결과를 냅니다.

## 결과: 정확도를 잃지 않고 격차 줄이기 {#section-4}

AppWorld `test_normal`(168개 작업)에서 GPT-4.1을 사용하는 ReAct 에이전트를 평가했습니다. 작업별 단일 기준선 궤적에서 일관성 가이드라인을 생성하고, 새로 실행한 5회의 실행에서 이를 테스트했습니다.

![Pass⁵ lift from baseline to consistency guidelines, by task difficulty, GPT-4.1 on AppWorld test_normal](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/E_7HyAt5BSzZy9Pj0k3Wc.png)

![Mean@5 aggregate, baseline vs. consistency guidelines, same scale as the Pass⁵ chart above](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/jzxSU5MJ8Ajf21aEjANyB.png)

Mean@5 (%), 집계값 — 위의 Pass^5와 동일한 척도입니다.

일관성 격차는 대략 절반으로 줄었습니다. 집계 Pass^5는 53.0% → 69.0%로 상승했고 Mean@5는 77.4% → 81.0%로 상승했습니다. 그 결과 "능력이 있어 보이는 것"과 "믿고 맡길 수 있는 것" 사이의 격차가 24.4pp에서 12.0pp로 줄었습니다. 이전에 일관성이 없었던 작업 중 거의 3분의 1이 에이전트가 매번 통과하는 작업으로 바뀌었습니다.

중간 난이도와 어려운 난이도에서 가장 큰 향상이 나타났습니다. Medium은 +22.9pp(+상대적으로 44%), Hard는 +14.3pp(+상대적으로 45%)로, 상대적 기준으로는 사실상 비슷하며 절대값에서는 Medium이 앞섭니다. Easy는 개선 여지가 가장 적었음에도 +12.2pp 향상되었습니다. 이는 일관성 가이드라인이 설계된 목적을 그대로 수행한 결과입니다. 에이전트의 불확실성이 결과에 새어 들어가던 특정 의사결정 지점을 찾아 안정화한 것입니다.

Mean@5는 한 번도 하락하지 않았습니다. 평균 정확도를 유지하는 것은 있으면 좋은 조건이 아니라 엄격한 요구사항이었습니다. Mean@5를 희생해 Pass^5를 높이는 시스템은 신뢰성 문제를 해결하는 것이 아니라 단지 다른 곳으로 옮기는 것에 불과하기 때문입니다. 모든 난이도 수준에서 평균 정확도는 유지되거나 향상되었습니다.

### 가이드라인은 일반화됩니다 — 하나의 궤적만 패치하는 것이 아닙니다

같은 AppWorld 시나리오에서 관련은 있지만 다른 작업, 즉 가이드라인을 추출한 시나리오의 또 다른 변형에 적용했을 때도 일관성 가이드라인은 Pass^5를 +13.0pp 향상시켰습니다. 동일 작업에서의 수치보다 3포인트 낮을 뿐입니다. 한 번의 실행에서 도출한 가이드라인은 해당 실행만 패치하는 것이 아니라, 전이 가능한 무언가를 포착합니다.

더 강력한 증거는 더 약한 모델인 gpt-oss-120b에서 나옵니다. 동일 작업의 Pass^5는 훨씬 낮은 기준선에서 +6.0pp 상승했습니다(10.1% → 16.1%). 흥미롭게도 유사 작업에 대한 일반화 수치(+8.7 pp)는 동일 작업에서의 향상폭을 실제로 넘어섰습니다. 이는 가이드라인이 한 궤적의 세부 사항을 암기한 것이 아니라, 실제로 재사용 가능한 실패 패턴을 포착했음을 시사합니다.

## 에이전트를 배포한다면 {#section-5}

- Mean@k와 함께 Pass^k를 보고하세요. 평균값만으로는 신뢰할 수 있는 에이전트와 운이 좋았던 에이전트를 구분할 수 없습니다. k=3만 사용해도 자신에게 격차가 있다는 사실을 발견할 수 있습니다.

- 난이도가 높아질수록 격차가 커질 것으로 예상하세요. 가장 어려운 등급에서 단일 평균값은 가장 오해를 불러일으킵니다.

- 먼저 더 큰 모델을 선택하지 마세요. 일관성은 능력과 직교합니다. 더 강력한 모델은 Mean@k를 높이지만, 일관성 격차를 반드시 줄이지는 않습니다.

- 진단에는 grader나 실시간 재생이 필요하지 않습니다. 의사결정 단계마다 LLM을 한 번 추가로 호출하고(기본적으로 k=5개의 완료를 샘플링) 충분합니다. ground truth도, 환경에서 작업을 다시 실행하는 과정도 필요하지 않습니다. 따라서 프로덕션 트래픽에서 사용할 수 있습니다. 프로덕션에서는 작업을 처음부터 끝까지 한 번만이라도 재생할 수 없는 경우가 많기 때문입니다.

## 직접 사용해 보기 {#section-6}

[ALTK-Evolve](https://github.com/AgentToolkit/altk-evolve)을 사용해 보세요. 오픈 소스 저장소에는 이제 이 실험에서 사용한 Consistency Analyzer와 일관성 가이드라인 생성 기능이 포함되어 있습니다. 또는 [technical report on arXiv](https://arxiv.org/abs/2609.08832)에서 전체 방법론을 확인할 수 있습니다.

자신의 작업에서는 재현할 수 없는 정확도 수치가 익숙하게 들린다면, 그 이야기를 듣고 싶습니다. 에이전트에서 결과가 뒤집히기 쉬운 동작에 대한 구체적인 사례는 우리가 다음에 무엇을 만들지 결정하는 데 정확히 필요한 피드백입니다. [Open an issue or a discussion](https://github.com/AgentToolkit/altk-evolve).

## 부록: 지표 이해하기 {#section-7}

- Mean@k. 작업을 k회 실행하고 평균 통과율을 보고합니다. 대부분의 벤치마크가 "정확도"라고 부르는 값입니다.

- Pass^k. 에이전트가 독립적인 k회 실행 모두에서 성공한 작업의 비율입니다. 항상 ≤ Mean@k입니다. 사용자가 같은 쿼리를 두 번 실행했을 때 경험하는 값입니다.

- Pass@k k회 실행 중 최소 한 번 성공하는 경우입니다. 낙관적인 대응 지표이며 코드 생성 논문에서 흔히 사용됩니다.

- 일관성 격차. Mean@k − Pass^k이며, 퍼센트 포인트로 표시합니다.

## 연결된 산출물 / 참고 자료 {#section-8}

- ALTK-Evolve 오픈 소스 저장소 — [github.com/AgentToolkit/altk-evolve](https://github.com/AgentToolkit/altk-evolve)

- 기술 보고서 — [arXiv](https://arxiv.org/abs/2609.08832)

- 일관성 가이드라인 데모 —[2 min video](https://www.youtube.com/watch?v=qlp7EzXe8Pg)
