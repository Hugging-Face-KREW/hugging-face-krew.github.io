---
layout: post
title: "물리학자처럼 LLM 가지치기하기: 블록 제거를 Ising 최적화 문제로 보기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/693c2a4eb0871ba57155b4ed/q4gZV9ENtfFwge7H3Q7YJ.png
image: assets/images/blog/posts/2026-09-21-pruning-llms-like-a-physicist-block-removal-as-an/thumbnail.png
authors:
  - user: MultiverseComputingCAI
slug: "pruning-llms-like-a-physicist-block-removal-as-an"
source_url: "https://huggingface.co/blog/MultiverseComputingCAI/pruning-llms-like-a-physicist-block-removal-as-an"
source_published_date: "2026-09-21"
source_published_at: "2026-09-21T13:44:34+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Pruning LLMs Like a Physicist: Block Removal as an Ising Optimization Problem](https://huggingface.co/blog/MultiverseComputingCAI/pruning-llms-like-a-physicist-block-removal-as-an)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/MultiverseComputingCAI/pruning-llms-like-a-physicist-block-removal-as-an -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 물리학자처럼 LLM 가지치기하기: 블록 제거를 Ising 최적화 문제로 보기

대규모 언어 모델을 더 빠르게 만드는 가장 저렴한 방법 중 하나는 동시에 가장 투박한 방법이기도 합니다. 바로 transformer 블록 전체를 삭제하는 것입니다. 모델이 실제로 더 짧아지기 때문에 [block removal](https://huggingface.co/papers/2602.00161)(depth pruning이라고도 함)는 메모리 절감 효과에 더해 예측 가능한 추론 속도 향상을 제공하며, 양자화, 저랭크 압축 및 기타 기법과도 깔끔하게 결합됩니다. 어려운 부분은 어떤 블록을 제거할지 결정하는 것입니다. 잘못된 블록을 제거하면 모델이 무너지고, 어떤 블록 하나를 제거했을 때의 효과는 함께 제거하는 다른 블록에 따라 달라지므로 선택들이 서로 상호작용합니다. 따라서 이는 순위 매기기 문제가 아니라 조합 문제이며, 상호작용하는 이진 변수를 가진 조합 문제는 바로 스핀 시스템의 물리학으로 설명하기에 적합한 문제입니다.

저희의 최신 논문 [LLM Compression by Block Removal with Constrained Binary Optimization](https://huggingface.co/papers/2602.00161)에서는 이러한 대응 관계를 그대로 활용합니다. 블록 선택을 제약 이진 최적화(CBO) 문제로 재정식화하여, 모든 스핀 사이에 상호작용이 있고 "up" 스핀의 수가 고정된 무질서 스핀 시스템인 Ising glass에 직접 매핑합니다. 이 스핀 시스템의 에너지는 가지치기된 모델이 실제 벤치마크에서 얼마나 좋은 점수를 낼지를 나타내는 강력하고 저렴한 proxy로 밝혀졌습니다. 따라서 어떤 구성도 실제로 벤치마크하지 않고도 방대한 수의 후보 구성을 순위화할 수 있으며, 어려운 인스턴스는 Multiverse에서 다른 문제에 사용하는 것과 동일한 고전적 및 양자 영감 솔버에 맡길 수 있습니다. 깊은 압축 영역에서 그 효과는 큽니다. Llama-3.3-70B-Instruct를 50% 압축했을 때, 가장 성능이 좋은 경쟁 블록 제거 방법보다 MMLU에서 거의 23 percentage points 높은 성능을 얻었습니다.

## 블록 선택이 다체 문제인 이유 {#section-1}

기존의 대부분의 블록 제거 방법은 각 블록을 개별적으로 평가한 다음, magnitude, sensitivity 또는 "block influence" 휴리스틱을 사용해 중요도가 가장 낮아 보이는 블록을 제거합니다. 물리학의 관점에서 보면 이는 mean-field 방법입니다. mean-field 이론이 스핀의 이웃을 하나의 평균장으로 대체하는 것처럼, 각 블록의 기여가 다른 블록과 독립적이라고 가정하기 때문입니다. 이와 유사한 지름길로, 연속된 하나의 블록 구간만 제거하는 방법도 있습니다. 이 방식은 문제의 크기를 작게 유지하지만 탐색 공간의 대부분을 포기하게 만듭니다.

문제는 실제 자석의 스핀들이 그렇지 않은 것처럼 블록도 서로 독립적이지 않다는 점입니다. 블록 20을 제거했을 때 모델이 받는 영향은 블록 19 또는 블록 24도 함께 제거했는지에 따라 달라집니다. 이는 두 결정 사이의 상호작용, 즉 coupling입니다. 모델이 더 깊어지고 이질적으로 변할수록 이러한 coupling을 무시하면 품질을 놓치게 되며, 특히 한 번에 많은 블록을 제거하려 할 때 더욱 그렇습니다. 실제로 필요한 것은 블록 간 상호작용을 고려하면서 블록 조합을 탐색하는 것이지만, 조합의 수는 지수적으로 증가하므로 완전 탐색은 가망 없어 보입니다. 이는 바로 쌍별 coupling을 가진 지수적으로 큰 구성 공간, 즉 통계물리학의 도구가 진가를 발휘하는 영역입니다.

## 아이디어: 블록 선택을 에너지 최소화 문제로 바꾸기 {#section-2}

각 transformer 블록에 이진 변수를 할당합니다. 0은 블록을 유지한다는 뜻이고 1은 제거한다는 뜻으로, 아래쪽 또는 위쪽을 가리킬 수 있는 스핀과 같습니다. 그런 다음 이 변수에 대한 모델 loss의 2차 Taylor 전개를 수행하면 (근사) Hessian 행렬이 생성됩니다. Hessian의 대각 성분은 각 블록이 단독으로 얼마나 중요한지를 나타내고, 비대각 성분은 블록 간의 쌍별 coupling, 즉 mean-field 방법이 버리는 다체 물리학을 정확히 나타냅니다.

이 재정식화를 통해 "어떤 블록을 제거해야 하는가?"라는 질문은 명확한 최적화 문제로 바뀝니다. 즉, 정확히 M개의 블록을 N개 블록 중에서 제거한다는 제약 아래, 제거했을 때 에너지를 최소화하는 M개 블록의 집합을 찾는 것입니다 `xᵀH⁰x`. 수학적으로 이는 제약 이진 최적화 문제이고, 물리적으로는 보존되는 자화량을 가진(제거되는 블록의 수가 고정된 총 스핀의 역할을 하는) 전방향 coupling 스핀 시스템인 Ising glass입니다. 저희가 확립한 핵심 특성은 이 에너지가 downstream 품질에 대한 강력한 proxy라는 점입니다. 스핀 시스템의 저에너지 상태는 성능이 높은 가지치기 모델에 대응합니다. 에너지 최소화와 벤치마크 점수 최대화가 동일한 탐색이 되는 것입니다.

[![Sketch of the method: block removal is cast as a constrained binary optimization / Ising problem whose low-energy states correspond to high-performing pruned models.](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/zD5BW0R8lXV9fmVyap9Jb.png)](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/zD5BW0R8lXV9fmVyap9Jb.png)

블록 선택은 제약 이진 최적화 문제가 되며, 이는 Ising glass의 저에너지 상태를 찾는 것과 같습니다. 각 해는 N개 블록 중 어떤 M개를 삭제할지를 나타냅니다. 오른쪽: Hessian을 구축하기 위해 각 블록의 residual path에 삽입하는 coupling 변수 α. 출처: 논문 Figure 1.

이 방법이 실용적인 이유는 비용에 있습니다. 즉, 모든 coupling을 포함하는 Hessian을 작은 calibration 데이터셋에 대한 forward 및 backward pass를 통해 단 한 번만 계산합니다. 그 이후에는 후보 구성의 평가가 단 한 번의 저렴한 에너지 계산으로 끝나며, 실제 모델을 실행할 필요는 물론 벤치마크할 필요도 없습니다. 또한 coupling은 압축 목표에 의존하지 않으므로 동일한 Hessian을 재사용하여 다양한 M 값에 대한 문제를 풀 수 있습니다.

## 문제 풀기: 가능할 때는 정확하게, 불가능할 때는 양자 또는 양자 영감 방식으로 {#section-3}

대부분의 모델에서 구성 공간은 크지만 여전히 확인할 수 있는 수준입니다. 한 번의 에너지 계산이 매우 저렴하기 때문에 단일 GPU에서 완전 탐색을 수행하여 최대 수백억 개의 스핀 구성을 확인할 수 있습니다. 수백만 개를 처리하는 데는 수 초가 걸리며, 여기서 다룬 가장 어려우면서도 처리 가능한 경우인 Llama-3.3-70B의 80개 블록 중 8개를 제거하는 문제(약 290억 개 구성)는 대략 이틀이 걸렸습니다.

그 이상에서는 정확한 접근법이 무너집니다. 이때 문제를 Ising glass로 정식화한 것이 두 번째로 효과를 발휘합니다. 동등한 QUBO 형식(제약을 penalty term에 흡수한 형식)으로 바꾸면, 정확히 동일한 작업을 이 Hamiltonian 클래스에 맞게 구축된 고도로 최적화된 고전적·양자·양자 영감 솔버에 넘길 수 있습니다. 여기에는 quantum annealing, QAOA, tabu search 및 특화된 branch-and-bound를 위한 도구가 포함됩니다. 저희는 오픈 소스 tabu solver가 완전 탐색으로 검증할 수 있는 가장 어려운 경우에도 수 초 안에 안정적으로 최저 에너지 상태에 도달한다는 것을 확인했습니다. 따라서 구성 열거가 불가능한 모델로도 방법을 확장할 수 있으며, Multiverse의 전문 영역에 해당하는 솔버를 활용할 수 있습니다.

여기에는 미묘하지만 중요한 점이 있으며, 이는 일반적인 최적화 방식과는 반대입니다. 일반적으로 CBO 또는 annealing solver는 진정한 ground state를 찾는지에 따라 평가됩니다. 하지만 저희에게 ground state는 실제로 필요하지 않습니다. 필요한 것은 좋은 저에너지 상태 몇 개를 빠르게 생성하는 방법이며, 이는 훨씬 낮은 기준입니다. 그렇기 때문에 가벼운 솔버가 저희에게 매우 잘 작동하고, 여러 솔버를 실행할 여유도 생깁니다.

## 전체 저에너지 스펙트럼이 중요한 이유 {#section-4}

에너지는 품질에 대한 강력한 proxy이지만 완벽하지는 않으므로, 에너지가 가장 낮은 단일 상태가 항상 최상의 모델인 것은 아닙니다. 이는 결함이 아니라 오히려 장점으로 드러납니다. Hamiltonian을 구성하고 나면 ground state와 낮은 에너지의 excited state를 읽어내는 데 드는 비용이 사실상 없기 때문에, 하나의 취약한 답이 아니라 시도해 볼 만한 고품질 가지치기 후보의 스펙트럼을 얻을 수 있습니다. ground state뿐 아니라 excited state를 탐색하는 것은 그 자체로 활발히 연구되는 물리학 분야이며, 여기서 실무자들이 실제로 필요로 하는 것과도 잘 대응합니다.

구체적인 예를 들어 보겠습니다. Llama-3.1-8B-Instruct에서 32개 블록 중 16개를 제거하는 경우, 상위 상태 대부분은 기존 연구가 예상하듯 모델 후반부에 있는 블록을 제거합니다. 그러나 17번째 excited state는 모델 앞부분에 가까운 블록을 제거하는 구성을 처음으로 제안하며, 이 구성은 가벼운 재학습 후 여러 벤치마크에서 ground state를 능가합니다. 이는 최상의 가지치기가 중간 또는 후반부 블록으로 이루어진 하나의 연속된 덩어리라는 일반적인 가정을 직접 반박하며, 문제의 전체 다체 구조를 고려하는 것이 왜 효과적인지를 보여줍니다.

[![Block-removal map and benchmark scores for the ground state versus the 17th excited state of Llama-3.1-8B-Instruct at 16/32 blocks removed.](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/n1GCTwe9Ulg53TAdgqjAf.png)](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/n1GCTwe9Ulg53TAdgqjAf.png)

왼쪽: 에너지가 가장 낮은 20개 상태 각각이 제거하는 블록(빨간색 = 제거됨). 오른쪽: 초기 블록을 제거하는 17번째 excited state는 재학습 후 여러 벤치마크에서 ground state를 능가합니다. 최상의 모델은 ground state가 아니라 excited state입니다. 출처: 논문 Figure 2.

## 결과 {#section-5}

Llama-3.1-8B-Instruct, Qwen3-14B 및 Llama-3.3-70B-Instruct 전반에서 저희 방법(CBO)은 최신 블록 제거 baseline과 대등하거나 더 뛰어난 성능을 보이며, 압축이 공격적으로 이루어질수록 그 격차가 커집니다.

가장 분명한 성과는 재학습 없이 평가한 Llama-3.3-70B-Instruct의 깊은 압축에서 나타납니다. 80개 중 최대 24개 블록을 제거했을 때 CBO는 block influence와 대략 대등합니다. 그러나 32/80 및 40/80에서는 확실하게 앞서며, 가장 깊은 설정에서 MMLU 기준 거의 23점의 우위를 보입니다. 이 설정에서는 테스트한 모든 벤치마크에서 baseline을 능가합니다. Qwen3-14B에서 40개 중 12개를 제거했을 때 CBO는 MMLU에서 약 10점 앞섭니다. 더 가벼운 압축에서는 두 방법의 성능이 비슷한데, 이는 예상된 결과입니다. 깊게 잘라낼수록 coupling이 가장 큰 영향을 미치기 때문입니다.

| Llama-3.3-70B-Instruct, 재학습 없음 | 제거된 블록 | MMLU |
| --- | --- | --- |
| Original | 0 | 82.2 |
| CBO (ours) | 32 / 80 | 76.6 |
| Block influence | 32 / 80 | 59.3 |
| CBO (ours) | 40 / 80 | 76.9 |
| Block influence | 40 / 80 | 54.0 |

40/80(깊이 기준 50%)에서 CBO는 MMLU를 77에 가깝게 유지하는 반면, 가장 강력한 baseline은 50점대 중반까지 하락합니다. 출처: 논문 Table 2.

## dense transformer를 넘어 일반화하기 {#section-6}

서로 다른 블록 유형이 교차 배치되는 현대의 이질적 아키텍처에서는 블록 제거가 훨씬 어려워지지만, Ising 정식화는 이에 영향을 받지 않습니다. 각 site에 어떤 종류의 블록이 있든 coupling은 coupling이기 때문입니다. 이를 검증하기 위해 NVIDIA-Nemotron-3-Nano-30B-A3B-FP8에 이 방법을 적용했습니다. 이 모델은 Mamba2, attention 및 mixture-of-experts(MoE) 레이어가 불균일한 패턴으로 교차 배치된 hybrid 모델이며, 재학습은 수행하지 않았습니다.

저희의 정식화는 homogeneous stack을 가정하지 않으므로 그대로 이전할 수 있습니다. MoE 레이어 2–3개 또는 attention 레이어 2개를 제거했을 때, CBO는 AIME25와 GPQA에서 block influence를 능가하는 구성을 찾습니다. 결과는 또한 이러한 hybrid 모델의 redundancy가 실제로 존재하지만 고르게 분포되어 있지는 않다는 점을 확인해 줍니다. 일부 expert 레이어는 다른 레이어보다 훨씬 더 쉽게 제거할 수 있으며, coupling된 구성 공간을 탐색하는 이 방법이 좋은 제거 지점을 찾아냅니다. 여기에서도 dense 모델에서 나타난 패턴이 유지됩니다. 최상의 구성은 ground state가 아니라 excited state인 경우가 많습니다.

## 이것이 Multiverse Computing에 적합한 이유 {#section-7}

복잡한 machine-learning 문제를 Ising Hamiltonian으로 재구성한 다음, 물리학을 위해 구축된 고전적 및 양자 영감 최적화 도구로 해결하는 것은 Multiverse의 전문 영역에 정확히 들어맞습니다. 이는 저희 compression stack 전반을 관통하는 것과 같은 접근 방식입니다. 또한 블록 제거는 quantization, 저랭크/SVD 압축, width pruning 및 knowledge-distillation 기반 healing을 비롯한 stack의 다른 요소들과 결합할 수 있으므로, 이들과 경쟁하는 대신 더 큰 pipeline에 자연스럽게 편입됩니다.

Taylor 전개 유도, QUBO 매핑, solver 벤치마크, calibration 데이터셋 ablation 및 전체 결과 표를 포함한 자세한 기술 내용을 확인하고 싶으신가요? [Hugging Face](https://huggingface.co/papers/2602.00161)에서 전체 논문을 읽거나, 여러분의 모델에 이를 적용하는 방법을 논의하기 위해 저희 팀에 문의해 주세요. 코드는 [github.com/CompactifAI/Block_removal_through_constrained_binary_optimization](https://github.com/CompactifAI/Block_removal_through_constrained_binary_optimization)에 오픈 소스로 공개되어 있습니다.
