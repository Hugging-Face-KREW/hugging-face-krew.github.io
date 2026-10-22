---
layout: post
title: "ALTK-Evolve: frontier 모델 없이 고객 서비스 작업에서 학습된 메모리로 τ²-bench 리더보드 정상에 오르기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/rzXYJh7e7UdoYgCLwOZjH.jpeg
image: assets/images/blog/posts/2026-10-20-altk-evolve-bc/thumbnail.jpeg
authors:
  - user: ibm-research
slug: "altk-evolve-bc"
source_url: "https://huggingface.co/blog/ibm-research/altk-evolve-bc"
source_published_date: "2026-10-20"
source_published_at: "2026-10-20T15:19:48+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [ALTK-Evolve: How learned memory tops the τ²-bench leaderboard on customer service tasks without a frontier model](https://huggingface.co/blog/ibm-research/altk-evolve-bc)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-research/altk-evolve-bc -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# ALTK-Evolve: frontier 모델 없이 고객 서비스 작업에서 학습된 메모리로 τ²-bench 리더보드 정상에 오르기

ALTK-Evolve는 GPT-4.1을 사용해 플레이북을 학습하고 적용하며, 모델 가중치를 업데이트하지 않고도 고객 서비스 에이전트 벤치마크인 τ²-bench에서 리테일 출품작보다 뛰어난 성능을 냅니다.

## 학습된 메모리로 GPT-4.1을 사용하는 기본 ReAct 에이전트가 정상에 오르다 {#section-1}

오래된 비추론 모델인 GPT-4.1을 사용하면 기본 고객 서비스 에이전트가 τ²-bench 리테일에서 67.5%의 Pass^4에 도달합니다. 이는 작업의 67.5%에서 4번의 독립적인 시도 모두 작업을 해결한다는 의미입니다. 이 점수는 2026년 10월 12일 리더보드 스냅샷의 모든 리테일 출품작보다 높습니다. 이 에이전트는 벤치마크의 기본 ReAct 스캐폴드에서 실행되며, 시스템 프롬프트에 학습된 가이드라인이 추가됩니다.

[ALTK-Evolve](https://github.com/AgentToolkit/altk-evolve)는 GPT-4.1을 사용해 해당 플레이북을 구축합니다. 에이전트의 학습 작업 시도 결과를 분석하고, 모델을 사용해 성공과 실패를 짧은 행동 규칙으로 정리합니다. 예를 들어 고객을 인증하는 방법, 거래를 확인하는 방법, 반품을 처리하는 방법 등이 이에 해당합니다. 그런 다음 이러한 가이드라인을 에이전트의 프롬프트에 추가해 이후 대화에 사용합니다. 가이드라인 추출과 에이전트 모두 동일한 모델이 담당하며, 가중치 업데이트나 사람의 주석은 필요하지 않습니다.

저희의 [previous work](https://huggingface.co/blog/ibm-research/altk-evolve-sldd)에서는 코딩 에이전트를 대상으로 이 접근 방식을 시연했습니다. 여기서는 도메인 정책을 따르고, 고객의 요청을 이해하며, 올바른 도구 작업을 실행해야 성공할 수 있는 고객 서비스에 이를 적용해 테스트합니다.

## τ²-bench는 정책 제약이 있는 고객 서비스를 테스트한다 {#section-2}

[τ²-bench](https://github.com/sierra-research/tau2-bench)(Sierra 제공)는 실제 제약 조건에서 에이전트가 고객 서비스 대화를 얼마나 잘 처리하는지 측정합니다. 각 작업에서 에이전트는 목표를 가진 LLM 시뮬레이션 고객, 도메인 도구 세트(주문 조회, 환불 처리, 항공편 재예약), 그리고 에이전트가 따라야 하는 서면 정책 문서와 연결됩니다. 에이전트는 필요한 세부 정보를 이끌어 내고, 정책 범위 안에 머무르며, 올바른 작업을 수행해야 합니다. 이러한 작업은 대개 되돌릴 수 없습니다. 최종 데이터베이스 상태가 예상 결과와 일치할 때만 작업이 해결된 것으로 간주되며, 보상은 엄격한 이진 값입니다. 저희는 이 벤치마크의 두 고객 서비스 도메인인 리테일과 항공을 사용합니다.

τ²-bench 자체 구현의 기본 ReAct 에이전트를 사용해, 2단계 THINK → ACT 루프로 실행합니다. 에이전트 모델에는 `gpt-4.1`, 고객 시뮬레이터 모델에는 `gpt-5.5`을 사용하며, 작업당 4번 시도합니다. 변경하는 유일한 부분은 시스템 프롬프트에 추가하는 학습된 가이드라인 블록입니다. 나머지(정책, 도구, 스캐폴드)는 모두 τ²-bench의 기본 구성입니다. 구체적으로 학습된 가이드라인은 도메인 정책 뒤에 일반 Markdown 목록으로 추가됩니다.

```
You are a customer service agent that helps users according to the policy below.
Always follow the policy strictly. Use the provided tools when needed.

<policy>
# Retail agent policy

As a retail agent, you can help users:
- cancel or modify pending orders
- return or exchange delivered orders
- modify their default user address
…
</policy>

## Guidelines learned from past task attempts (follow these carefully):
- Authenticate the user before taking any order-related action. If email is unavailable, verify identity via full name and zip code using the find_user_id_by_name_zip API.
- Always confirm all intended changes (modify, cancel, return, exchange) with a clear, structured summary, and require explicit user approval before executing any database-updating action.
- When processing a return or exchange, communicate the refund with an itemized breakdown and clarify refund timing — don't wait for the user to ask.
- Take one action per step: never combine a tool call with a user-facing message in the same turn, and wait for the tool's result before replying.
- Before looking a user up, parse and clean their name — e.g. split a combined handle like `daiki_johnson_9523` into first/last name — and after a couple of failed lookups, state the exact format you need.
…
```


에이전트를 구동하는 동일한 모델인 `gpt-4.1`을 사용해 τ²-bench 학습 작업에서 에이전트의 궤적으로부터 가이드라인을 추출한 다음, 이를 변경하지 않고 보류된 테스트 세트에 적용합니다.

## 가이드라인은 역량과 신뢰성을 향상한다 {#section-3}

[prior post](https://huggingface.co/blog/ibm-research/altk-evolve-hmm)에서 서로 다른 모델과 작업에는 서로 다른 수준의 가이드가 필요하다는 점을 다뤘습니다. 이와 같은 관점에서 모델에 제공하는 가이드라인의 양도 다양하게 평가합니다.

각 가이드라인에는 지원 횟수가 있습니다. 이는 해당 가이드라인이 추출된 서로 다른 학습 작업의 수를 의미하므로, 지원 횟수가 높을수록 일회성 규칙이 아니라 더 많은 작업에 일반화된 규칙임을 뜻합니다. 세 가지 구성을 비교합니다. baseline(가이드라인 없음), sup3(지원 횟수 ≥ 3인 가이드라인만 유지 — 작고 높은 신뢰도의 핵심 세트), lossless(통합 후 유지되는 모든 가이드라인, 즉 지원 횟수 ≥ 1 유지)입니다.

표 1. 도메인별 가이드라인 구성과 유지된 가이드라인 수.

| 변형 | 컨텍스트에 포함된 가이드라인 | 리테일 | 항공 |
| --- | --- | --- | --- |
| baseline | 가이드라인 없음 | 0 | 0 |
| sup3 | 지원 횟수 `n ≥ 3` (높은 지원 횟수의 핵심 세트) | 31 | 14 |
| lossless | 모든 가이드라인 | 292 | 101 |

추적하는 지표는 다음과 같습니다. 모든 작업을 4번 시도하며, 해당 시도를 두 가지 방식으로 분석합니다.

- 평균 정확도(Mean Accuracy) — 4번의 시도에서 평균 성공률입니다. 이는 역량 축에 해당합니다. 에이전트가 얼마나 자주 작업을 올바르게 처리하는지를 나타냅니다.

- Pass^4 — 4번의 시도 모두 통과한 작업의 비율입니다. 이는 신뢰성 축을 측정하며, τ²-bench 리더보드가 보고하는 지표입니다. 여기서 중요한 이유는 환불을 4번 중 3번 처리하는 지원 에이전트도 실제 고객의 4분의 1에게는 실패하기 때문입니다. Pass^4는 평균적으로 잘하는지가 아니라 에이전트를 매번 신뢰할 수 있는지를 묻습니다.

[![Mean Accuracy and Pass^4 on the held-out test tasks, by variant, for Retail and Airline. Lossless lifts retail accuracy +4.4 pp and Pass^4 +10.0 pp; airline accuracy rises +7.5 pp with lossless while sup3 gives the best Pass^4 at +10.0 pp.](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/HtL2uTn98-xHkcTxKqkFY.png)](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/HtL2uTn98-xHkcTxKqkFY.png)

그림 1. 작업당 4번의 시도에서 `gpt-4.1`을 사용하는 에이전트의 보류된 리테일 및 항공 작업에 대한 평균 정확도와 Pass^4.

에이전트 모델: `gpt-4.1` (사용자 시뮬레이터: `gpt-5.5`).

표 2. 작업당 4번의 시도에서 `gpt-4.1`을 사용하는 보류된 에이전트 성능. 괄호 안은 baseline 대비 퍼센트포인트 변화입니다.

| 도메인(테스트 작업) | 구성 | 평균 정확도 | Pass^4 |
| --- | --- | --- | --- |
| 리테일 (40) | baseline | 78.1% | 57.5% |
|   | sup3 | 78.8% (+0.7) | 60.0% (+2.5) |
|   | lossless | 82.5% (+4.4) | 67.5% (+10.0) |
| 항공 (20) | baseline | 57.5% | 35.0% |
|   | sup3 | 60.0% (+2.5) | 45.0% (+10.0) |
|   | lossless | 65.0% (+7.5) | 40.0% (+5.0) |

평균 정확도는 두 도메인 모두에서 상승합니다. 리테일에서는 +4.4포인트(78.1 → 82.5), 항공에서는 +7.5포인트(57.5 → 65.0) 상승합니다. 신뢰성은 더 크게 향상됩니다. 엄격한 Pass^4는 리테일에서 +10포인트(57.5 → 67.5), 항공에서 +10포인트(35.0 → 45.0) 상승합니다. 그리고 이러한 향상은 더 긴 대화가 아니라 더 충분한 정보를 바탕으로 한 호출에서 비롯됩니다. 아래 [Better performance costs more tokens, not more turns](#better-performance-costs-more-tokens-not-more-turns)에서 대화 길이를 직접 살펴봅니다.

### 최적의 가이드 양은 도메인에 따라 다르다

두 도메인은 어느 정도의 가이드가 효과적인지에 대해 서로 다른 결과를 보입니다. 리테일에서는 모든 지표에서 많을수록 좋습니다. 작은 sup3 핵심 세트는 정확도에서 baseline을 거의 앞서지 못하며, 실질적인 상승을 얻으려면 전체 lossless 세트가 필요합니다. 항공에서는 신뢰성 측면에서 결과가 뒤집힙니다. lossless는 정확도에서 앞서지만(65% 대 60%), 더 작은 sup3 핵심 세트(가이드라인 14개)가 더 높은 신뢰성을 보여 Pass^4를 45%로 끌어올립니다. 이는 [delivery is a dial, not a constant](https://huggingface.co/blog/ibm-research/altk-evolve-hmm)라는 앞선 결과와도 일치합니다. 지원 횟수는 도메인별로 조정할 가치가 있는 조절 변수이지, 하드코딩할 값이 아닙니다.

## 더 나은 성능에는 더 많은 턴이 아니라 더 많은 토큰이 든다 {#section-4}

가이드라인에는 비용이 들지만, 그 비용은 예상할 수 있는 곳에 발생합니다. 더 길고 장황한 대화가 아니라 프롬프트 입력 토큰에 발생합니다.

[![Cost per conversation grouped by Retail and Airline. Lossless increases cost by 65% and 32%, respectively, over the no-guideline baseline.](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/MsqhICXsHtpJozVFZYEm7.png)](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/MsqhICXsHtpJozVFZYEm7.png)

그림 2. 전체 base split에서 도메인 및 가이드라인 구성별 대화당 총 LLM 비용.

표 3. 4번의 시도에 걸친 전체 base split에서 대화당 총 LLM 비용(에이전트 및 사용자 시뮬레이터). 낮을수록 좋습니다.

| 도메인 | 구성 | 대화당 비용 | baseline 대비 |
| --- | --- | --- | --- |
| 리테일 | baseline | $0.084 | — |
|   | sup3 | $0.095 | +12% |
|   | lossless | $0.139 | +65% |
| 항공 | baseline | $0.081 | — |
|   | sup3 | $0.097 | +19% |
|   | lossless | $0.107 | +32% |

에이전트는 더 오래 대화해서 목표에 도달할까요? 이를 명확히 답하기 위해 동일한 조건끼리 비교합니다. baseline과 가이드라인 변형 모두에서 해결된 작업 실행, 즉 동일한 시나리오와 시드를 사용한 진정한 "같은 작업을 올바르게 완료한" 집합을 비교하고, 양쪽의 어시스턴트 턴 수를 계산합니다.

표 4. 테스트 분할에서 두 설정 모두로 해결된 대응 실행의 대화당 어시스턴트 턴 수. 4번의 시도를 통합했습니다.

| 도메인 | 공통 집합 | 양쪽에서 해결됨 (n) | baseline 턴 수 | +가이드라인 | Δ |
| --- | --- | --- | --- | --- | --- |
| 리테일 | baseline ∩ sup3 | 108 | 10.40 | 10.35 | −0.05 |
|   | baseline ∩ lossless | 111 | 10.59 | 10.16 | −0.43 |
| 항공 | baseline ∩ sup3 | 38 | 8.97 | 8.53 | −0.45 |
|   | baseline ∩ lossless | 40 | 9.15 | 8.82 | −0.33 |

에이전트가 더 오래 대화하는 것은 아닙니다. 공통 집합에서 가이드라인을 사용하면 대화가 오히려 약간 짧아집니다. lossless는 두 도메인 모두에서 약 0.3~0.4턴을 줄이며(리테일 10.59 → 10.16, 항공 9.15 → 8.82), 어떤 변형도 대화를 더 길게 만들지 않습니다. 이는 비용을 넘어 중요한 의미를 가집니다. 각 턴은 순차적인 LLM 왕복이므로 턴 수가 적으면 엔드투엔드 지연 시간이 줄어듭니다. 즉 고객이 완료 결과를 더 빠르게 얻으며, 단순히 비용만 저렴해지는 것이 아닙니다. 실제 비용은 입력 측에만 발생합니다. 모든 LLM 호출에 포함되는 더 큰 시스템 프롬프트가 그것입니다. 리테일의 가이드라인 292개 프롬프트는 대화당 비용을 약 $0.084에서 약 $0.139로 높이는 반면, 항공의 더 작은 플레이북은 비용이 더 낮습니다. 또한 실제 운영 환경에서는 가이드라인 블록이 모든 호출에서 동일하므로 프롬프트 캐싱을 통해 이 비용의 상당 부분을 회수할 수 있습니다.

따라서 트레이드오프는 유리합니다. 입력 비용이 약간 더 높아지는 대신(실제로는 상당 부분을 캐싱할 수 있음), 에이전트는 더 높은 정확도, 더 높은 신뢰성, 더 짧은 완료 시간을 동시에 달성합니다. 이는 일반적인 품질과 지연 시간 사이의 트레이드오프가 아니라, 세 요소가 모두 함께 올바른 방향으로 움직이는 경우입니다.

## 학습된 가이드라인은 frontier 모델을 사용하는 에이전트도 향상한다 {#section-5}

지금까지는 오래된 비추론 모델인 `gpt-4.1`을 사용했습니다. 학습된 가이드라인이 frontier 모델을 사용하는 에이전트에는 추가할 것이 적을 수 있다는 합리적인 반론이 가능합니다. 즉 더 약한 모델의 부족한 부분만 보완하는 것일 수 있습니다. 그래서 이를 직접 테스트했습니다. 동일한 τ²-bench ReAct 스캐폴드에서 에이전트와 가이드라인 추출 모두에 `gpt-5.5`을 사용해 항공 도메인을 다시 실행하고, 에이전트 자체 실행 결과에서 가이드라인을 다시 추출했습니다. 모델에 따라 한 가지 세부 사항이 달라집니다. 이 설정에서 추출된 가이드라인 중 지원 횟수 ≥ 3에 도달한 것은 없었고(최대 지원 횟수는 2), 따라서 여기서 높은 지원 횟수의 핵심 세트는 sup3가 아니라 sup2입니다.

표 5. 작업당 4번의 시도에서 `gpt-5.5` 에이전트의 보류된 항공 성능.

| 항공 · `gpt-5.5` (보류된 테스트 작업 20개) | 정확도 | Pass^4 |
| --- | --- | --- |
| baseline (가이드라인 0개) | 72.5% | 55.0% |
| sup2 (가이드라인 103개) | 80.0% (+7.5) | 70.0% (+15.0) |
| lossless (가이드라인 146개) | 82.5% (+10.0) | 70.0% (+15.0) |

학습된 가이드라인은 이 설정도 향상합니다. 동일한 보류된 테스트 작업에서 가이드라인을 프롬프트에 추가하면 정확도가 +10포인트(72.5 → 82.5), 엄격한 Pass^4가 +15포인트(55.0 → 70.0) 상승합니다. 이는 동일한 작업에서 `gpt-4.1`을 사용했을 때 관찰한 신뢰성 향상(+10포인트)보다 크며, 훨씬 더 높은 기준점에서 출발한 결과입니다. 더 강력한 기반에서 시작한 에이전트라도 모델을 중심으로 가이드라인을 학습하면 향상될 수 있습니다. 또한 baseline의 신뢰성이 재실행마다 달라진 반면, lossless 구성은 정확히 동일한 결과를 재현했습니다. 이는 운이 좋았던 한 번의 결과가 아니라, 안정적으로 얻을 수 있는 향상입니다.

## ALTK-Evolve가 리더보드 비교에서 정상에 오르다 {#section-6}

출품작은 τ²-bench의 요구 사항에 따라 전체 base split(train ∪ test)에서 평가되며 Pass^4로 보고됩니다. 이는 보류된 테스트 분할이 아니라 전체 작업 세트를 포함하므로, 위에서 보고한 테스트 작업 수치와는 다릅니다. 실질적인 의미가 드러나는 지점은 바로 여기입니다. 프롬프트 수준의 메모리가 각 기반 모델을 사용하는 에이전트를 현재 frontier 모델 수준으로 끌어올리거나 그 이상으로 향상합니다.

- [Retail: #1](https://github.com/sierra-research/tau2-bench/pull/542). base split에서 학습된 가이드라인을 사용하는 에이전트는 Pass^4 = 67.5%에 도달해 모든 frontier 모델과 합성 τ²-bench 리테일 데이터로 미세 조정된 모델인 RAFT-30B-A3B(61.4%)를 앞섭니다. 기본 `gpt-4.1`과 프롬프트에 추가한 플레이북만 사용해 해당 미세 조정 시스템보다 +6.1포인트 높은 성능을 냅니다.

- [Airline: #1](https://github.com/sierra-research/tau2-bench/pull/551) using `gpt-5.5`, #8 using `gpt-4.1` 공개된 리더보드와 비교하면, 간결한 sup3 핵심 세트를 사용한 에이전트는 `gpt-4.1`에서 Pass^4 = 44%를 기록해 8위를 차지하며 GPT-5.2(추론 없음)를 앞서고 Claude Sonnet 4.5에 거의 근접합니다. 더 강력한 기반 모델로 교체하면 동일한 접근 방식이 전체 분야에서 정상에 오릅니다. `gpt-5.5`을 사용하는 에이전트는 82.0%에 도달해 모든 frontier 출품작(66~72%)을 넘어섭니다.

[![Saved Pass^4 leaderboard comparison for Retail and Airline, highlighting ALTK-Evolve runs at 67.5% in Retail and 82.0% and 44.0% in Airline.](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/iOmv6o938CgNfFlULP-VV.png)](https://cdn-uploads.huggingface.co/production/uploads/6435a1131860001f144239ea/iOmv6o938CgNfFlULP-VV.png)

그림 3. ALTK-Evolve 실행 결과와 2026년 10월 12일 τ²-bench 리더보드 스냅샷 간의 Pass^4 비교.

두 도메인에서 메시지는 동일합니다. 모델을 중심으로 가이드라인을 학습하는 방식은 서로 다른 기반 모델에서 작동합니다. 리테일에서는 오래된 비추론 모델을 사용하는 에이전트를 frontier 모델 수준으로 끌어올리고, 항공에서는 frontier 모델을 사용하는 에이전트를 정상에 올립니다.

## 에이전트는 가중치 업데이트 없이도 향상될 수 있다 {#section-7}

벤치마크 데이터로 모델을 미세 조정하는 일은 비용이 많이 들고 취약하며 다음 도메인으로 전이되지 않습니다. 여기서 ALTK-Evolve는 두 도메인 모두에서 기본 모델의 기본 에이전트를 향상하며, 리테일에서는 미세 조정된 시스템과 같거나 더 나은 성능을 냅니다. 이때 에이전트 자체 실행에서 자동으로 추출한 프롬프트 수준의 가이드라인만 사용하며, 사람의 주석도 테스트 세트 누수도 없습니다.

### 링크:

- 이전 게시물 
[Thinking of ACE? We Can Do It with Fewer Tokens](https://huggingface.co/blog/ibm-research/altk-evolve-sldd)
[How much memory does your agent actually need?](https://huggingface.co/blog/ibm-research/altk-evolve-hmm)
[Your Agent Aced the Task. Will It Do It Again?](https://huggingface.co/blog/ibm-research/altk-evolve-consistency)

- ALTK-Evolve 라이브러리 — [https://github.com/AgentToolkit/altk-evolve](https://github.com/AgentToolkit/altk-evolve)

- τ²-bench — [https://github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)

- 리테일 리더보드 출품 — [τ²-bench PR #542](https://github.com/sierra-research/tau2-bench/pull/542)

- 항공 리더보드 출품 - [τ²-bench PR #551](https://github.com/sierra-research/tau2-bench/pull/551)
