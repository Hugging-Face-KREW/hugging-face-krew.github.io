---
layout: post
title: "누구를 위한 안전성인가? 주제 전체가 아니라 올바른 하위 집합을 거부하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/HuauFbdznYW4fNQh8j4Tp.png
image: assets/images/blog/posts/2026-09-08-safety-for-whom/thumbnail.png
authors:
  - user: MultiverseComputingCAI
slug: "safety-for-whom"
source_url: "https://huggingface.co/blog/MultiverseComputingCAI/safety-for-whom"
source_published_date: "2026-09-08"
source_published_at: "2026-09-08T14:23:07+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Safety for Whom? Refusing the Right Subset of a Topic, Not the Whole Topic](https://huggingface.co/blog/MultiverseComputingCAI/safety-for-whom)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/MultiverseComputingCAI/safety-for-whom -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 누구를 위한 안전성인가? 주제 전체가 아니라 올바른 하위 집합을 거부하기

대부분의 안전성 정렬 작업은 유해성을 주제의 속성으로 다룬다. 프롬프트가 무기, 사기, 자해와 같은 일반적인 범주에 속하기 때문에 안전하지 않다고 보고, [LlamaGuard-3](https://huggingface.co/meta-llama/Llama-Guard-3-8B)와 같은 가드 모델은 바로 이런 주제 수준의 분류 체계를 인코딩한다. 그런 다음 [XSTest](https://arxiv.org/abs/2308.01263) 및 [OR-Bench](https://arxiv.org/abs/2405.20947)와 같은 벤치마크는 이로 인해 발생하는 실패 양상, 즉 위험해 보이는 단어가 포함되어 있다는 이유로 안전한 프롬프트를 거부하는 모델을 점검하며, 거부 보정 연구는 그 수치를 다시 낮추려 한다.

실제 배포 환경은 주제 수준의 관점에 좀처럼 들어맞지 않는다. 동일한 기본 모델이 범용 어시스턴트, 교육용 제품, 엔터프라이즈 시스템 또는 공공 부문 서비스에 맞게 조정될 수 있으며, 각 환경은 동일한 주제 안에서도 서로 다른 경계를 필요로 한다. 시민 교육 튜터와 공공 부문 어시스턴트는 같은 모델을 공유하면서도 정치에 대해서는 정반대의 동작을 요구할 수 있다. 둘 다 선거에 관한 사실 질문에는 답해야 하지만, 특정 대상을 겨냥한 정치적 조작을 작성해 달라는 요청은 한쪽만 거부해야 할 수 있다. 주제 수준의 가드는 이러한 구분을 표현할 수 없다. 예를 들어 LlamaGuard-3는 선거를 "선거 제도와 절차에 관한 사실적으로 부정확한 정보"로만 다루며, 설득과 조작을 제외하는 동시에 배포 환경에서 계속 답해야 하는 사실 질문도 제외한다.

최신 논문인 [Safety for Whom? Boundary-Aware Self-Distillation for Controlled LLM Safety Refusal](https://huggingface.co/papers/2609.04482)에서는 이처럼 더 좁은 문제를 직접 연구한다. 핵심 질문은 주제 전체를 거부해야 하는지가 아니라, 주어진 배포 정책과 양립할 수 없는 주제의 하위 집합이 무엇인지, 그리고 그 경계를 기준으로 모델을 어떻게 학습하고 측정할 것인지이다.

## 좁은 경계의 안전성 {#section-1}

우리는 이 설정을 주제 전체, 즉 실험에서 사용한 모든 정치 프롬프트가 배포 환경에서 거부하려는 유해 대상 하위 집합을 포함하는 것으로 형식화한다. 의도된 정책은 정치에 관한 모든 내용을 거부하는 것이 아니라, 유해한 하위 집합은 거부하면서 나머지 무해한 여집합에는 계속 답하는 것이다. 이상적인 동작은 뚜렷한 단계 형태를 이룬다. 하위 집합 안에서는 거부하고, 주제 내 다른 모든 영역에서는 답해야 한다.

[![Narrow-boundary safety. The topic universe of political prompts contains a smaller subset that the deployment should refuse, while the benign complement should still be answered. The ideal refusal is a sharp step, but a trained model's refusal probability is smoother and can overshoot into benign territory near the boundary.](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/g_Xngr05EUhlIBNT-mG8s.png)](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/g_Xngr05EUhlIBNT-mG8s.png)

좁은 경계 설정. 배포 환경에서는 정치에 관한 모든 내용을 거부하는 대신, 조작이나 특정 대상을 겨냥한 설득을 요청하는 정치 프롬프트만 거부하면서 다른 정치 프롬프트에는 답해야 할 수 있다. 학습된 모델의 거부 동작은 이상적인 구분보다 더 완만하며, 경계 주변의 무해한 영역까지 번질 수 있다. 출처: 논문 Figure 1.

학습된 모델은 이러한 뚜렷한 단계를 결코 그대로 학습하지 않는다. 모델은 목표를 근사하는 거부 확률을 학습하며, 유해한 하위 집합 내부의 거부를 높이는 cross-entropy 학습이 무해한 여집합으로 거부를 밀어낼 수도 있다. 따라서 실제 문제는 유해한 프롬프트에 대한 거부를 높이는 것뿐 아니라 경계 자체의 동작을 형성하는 것이다. 우리는 동일한 주제 앵커를 공유하고 의도만 다른 프롬프트 쌍으로 이 경계를 구체화한다. 한쪽은 거부해야 하고 다른 한쪽은 답해야 한다.

우리는 정치적 설득을 테스트베드로 사용한다. 조작적 설득은 실제 피해를 일으킬 수 있는 반면 사실에 기반한 정치 정보는 정당하기 때문이다. 이는 주제 수준의 거부가 지나치게 뭉뚱그려지는 정확한 사례다.

## 자체 생성 안전성 튜닝이 무너지는 지점 {#section-2}

여기서 학습 데이터를 구축하는 자연스러운 방법은 자체 생성이다. 대상 모델을 각 유해 프롬프트에 대해 거부하도록 유도하고, 가드 모델이 진정한 거부로 검증한 추적 결과만 남기는 방식이다. 이는 [ThinkSafe](https://arxiv.org/abs/2601.23143)와 같은 방법의 기반이 되는 레시피이며, 우리는 이를 정치 프롬프트에 적용하고 구성 요소별로 측정하는 기준선으로 채택한다. 문제를 주제가 아니라 경계로 바라보면 이러한 표준 파이프라인의 세 가지 약점이 드러난다.

첫 번째는 커버리지 공백이다. 한 번의 유도 시도가 항상 허용되는 거부를 생성하는 것은 아니며, 그런 프롬프트는 학습 세트에서 조용히 삭제된다. 감사한 데이터 풀에서 단일 시도 생성은 프롬프트의 19.88%, 즉 8,009개를 삭제하며, 이렇게 실패한 프롬프트가 가장 어려운 예시일 가능성도 충분하다. 우리는 이를 버리는 대신 복구한다. 점진적으로 더 강한 유도로 동일한 프롬프트를 재샘플링하는 단계적 재시도 전략을 사용하면 남은 실패율은 0.20%, 즉 79개 프롬프트까지 낮아진다. 커버리지 복구를 통해 단순한 파이프라인이라면 수천 개를 버렸을 유해 학습 프롬프트 40,293개를 남길 수 있다.

두 번째는 부작용 반응이다. 안전성 튜닝은 표면적으로 위험해 보이는 무해한 프롬프트에 대해 거짓 거부를 만들어내는 경향이 있다. 이를 보완하기 위해 18개 의미 유형에 걸친 검증된 표면적 위험 무해 프롬프트 11,955개를 포함하는 분포 내 무해 데이터를 구축한다. 이를 통해 모델은 평가 시점에만 위험해 보이는 표현을 접하는 것이 아니라 학습 중에도 위험해 보이는 표현을 포함한 안전한 프롬프트를 보게 된다.

세 번째는 일반적인 유해·무해 분할로는 경계의 형태를 전혀 측정할 수 없다는 점이다. 모델은 인접한 허용 프롬프트까지 거부를 확장하는 것만으로 유해 거부율을 높일 수 있으며, 주제 수준의 지표는 이를 개선으로 판단한다. 양쪽 각각 1,539개로 구성된 보류 유해·무해 쌍을 사용하면 경계의 양쪽을 직접 측정할 수 있다.

## 트레이드오프와 그 안에 숨은 함정 {#section-3}

정치적 거부 데이터에 대한 학습은 명백한 의미에서 효과가 있다. [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)에서 커버리지를 확대한 모델은 분포 내 정치 거부율을 9.47%에서 84.75%로 높이며, 전이 효과도 나타난다. LlamaGuard-3가 평가한 세 가지 광범위한 유해성 벤치마크 [HarmBench](https://www.harmbench.org), StrongREJECT 및 [WildJailbreak](https://arxiv.org/abs/2406.18510)에서 평균 안전하지 않은 응답률은 가장 강한 구성에서 26.26%에서 0.14%로 감소한다.

이 수치만 따로 제시하면 완벽한 승리처럼 보인다. 하지만 그렇지 않다. 같은 체크포인트에서 XSTest의 과잉 거부율은 2.00%에서 74.00%로 상승한다. 유해 응답률이 가장 낮은 구성은 명백히 안전한 프롬프트의 거의 4분의 3도 거부한다. 이는 더 안전한 모델이 아니라 무차별적인 거부 기계이며, 무해한 측면을 측정하지 않으면 이를 알 수 없다. 핵심 메시지는 다음과 같다. 데이터 구성에 따라 체크포인트가 안전성과 과잉 거부 사이의 공간에서 어느 위치에 놓이는지가 결정되므로, 두 축을 함께 보고해야 한다.

우리의 두 가지 데이터 구성 요소는 안전성 향상을 포기하지 않으면서 과잉 거부 수치를 다시 낮춘다. 외부에서 채택한 순응 응답을 대상 모델 자체가 생성하고 검증한 응답으로 대체하면, 단일 시도 생성에서 XSTest 과잉 거부율이 15.20%에서 5.20%로 낮아지며 유해성 비용은 크지 않다. 그리고 유해·무해 경계 쌍이 가장 정밀한 역할을 한다.

[![Pairwise boundary data reduces over-refusal at the boundary. Adding the benign side of the boundary pairs drops comply-side over-refusal from roughly 0.49 down to 0.03 to 0.08, while harmful-side refusal falls only slightly, from about 0.92 to 0.88.](https://cdn-uploads.huggingface.co/production/uploads/68db932961906f42259438b7/6W8oCYwy3TnSfdaBsH74C.png)](https://cdn-uploads.huggingface.co/production/uploads/68db932961906f42259438b7/6W8oCYwy3TnSfdaBsH74C.png)

왼쪽: 보류된 경계에서 응답해야 하는 측면의 과잉 거부율. 낮을수록 좋다. 무해 경계 데이터를 사용한 실행(PB)은 0.03~0.08까지 낮아지지만, 사용하지 않으면 수치가 0.49에 가까워진다. 오른쪽: 유해한 측면의 거부율. 높을수록 좋으며, 이 수치는 약간만 감소한다. 출처: 논문 Figure 6.

구체적으로 무해 경계 데이터를 추가하면 보류된 쌍에서 응답해야 하는 측면의 과잉 거부율이 32.94%에서 4.16%로 감소한다. 유해한 측면의 거부율은 91.88%에서 87.72%로 소폭 하락하는 데 그친다. 다시 말해 경계 주변의 거짓 거부 대부분이 사라지는 동안 진정한 거부는 거의 모두 유지된다. 실제 recall 비용은 존재하지만 작고 측정 가능하다. 바로 이것이 핵심이다. 양쪽을 모두 측정해야만 이 비용을 의도적으로 조정할 수 있다.

## 달라지는 점 {#section-4}

실무적 결론은 안전성 튜닝을 유해 거부율만으로 평가해서는 안 된다는 것이다. 더 많이 거부하는 모델이 자동으로 더 안전한 것은 아니며, 좁은 경계에서는 유해 프롬프트에 대한 거부를 높이는 동일한 조치가 바로 옆의 정당한 프롬프트에서 모델을 조용히 쓸모없게 만들 수 있다. 이러한 트레이드오프를 제어하는 것은 학습 데이터의 구성, 커버리지 복구, 분포 내 보완, 경계 쌍이며, 수치가 의미를 가지려면 의도한 경계의 양쪽을 모두 평가해야 한다.

이 연구는 광범위한 주제 범주 수준이 아니라 실제 배포 환경이 중요하게 여기는 수준에서 모델 동작을 제어하고 측정할 수 있도록 하는 [Multiverse Computing's](https://multiversecomputing.com) 연구의 일부다. 동일한 생성 파이프라인은 정치 이외의 다른 주제에도 확장할 수 있으며, 논문에서는 위 결과를 뒷받침하는 전체 데이터 구성 절제 실험 결과를 보고한다.

커버리지 복구 전략, 유해 cross-entropy와 무해 forward-KL 보존을 분리하는 loss routing, 전체 보류 경계 평가를 포함한 기술적 세부 사항을 모두 확인하고 싶은가? 전체 논문을 읽거나, 여러분의 모델에 대한 배포별 안전성에 관해 논의하려면 우리 팀에 문의하라.
