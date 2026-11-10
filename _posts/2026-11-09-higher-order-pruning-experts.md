---
layout: post
title: "전문가에게는 파트너가 있습니다. 이들을 함께 유지할 HOPE가 있습니다"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6ab2f79bd8000a811c648a32/xZT8tg4be3bgZIx2EQS_m.png
image: assets/images/blog/posts/2026-11-09-higher-order-pruning-experts/thumbnail.png
authors:
  - user: amazon
slug: "higher-order-pruning-experts"
source_url: "https://huggingface.co/blog/amazon/higher-order-pruning-experts"
source_published_date: "2026-11-09"
source_published_at: "2026-11-09T12:34:29+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Your Experts Have Partners. There's HOPE for Keeping Them Together](https://huggingface.co/blog/amazon/higher-order-pruning-experts)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/amazon/higher-order-pruning-experts -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 전문가에게는 파트너가 있습니다. 이들을 함께 유지할 HOPE가 있습니다

## Mixture-of-Experts 언어 모델에서 전문가의 고차 가지치기 {#section-1}

### 소개

Mixture-of-Experts(MoE) 모델은 최첨단 LLM을 수천억 개의 파라미터 규모로 확장할 수 있게 해 주었지만, 근본적이고 비용이 큰 문제를 동반합니다. 특정 토큰에서 대다수의 전문가가 유휴 상태임에도 추론 시 모든 전문가 MLP가 메모리를 차지해야 한다는 점입니다. 예를 들어 Qwen3.5-122B는 48개의 MoE 레이어 각각에 256개의 전문가를 가지며, 약 250GB에 달하는 파라미터 메모리 사용량의 95%가 이 전문가 MLP만으로 구성됩니다. 그러나 특정 토큰마다 실제로 사용되는 전문가는 각 레이어에서 8개뿐입니다.

전문가 가지치기는 이러한 메모리 병목을 해결하는 가장 직접적인 방법 중 하나입니다. 즉, 유용성이 가장 낮은 전문가를 영구적으로 삭제하는 것입니다. 가지치기는 전문가를 동적으로 로드하거나 오프로딩할 때 발생해 실용적이지 않게 만드는 지연 시간 문제를 피하고, 학습이 필요하지 않으며, [양자화와 함께 사용할 수 있습니다](https://arxiv.org/abs/2510.13999). 이러한 매력적인 이점과 아이디어의 단순성 때문에 전문가 가지치기가 MoE를 축소하는 대표적인 방법이 된 이유를 쉽게 이해할 수 있습니다. 어려운 점은 어떤 전문가를 제거해도 가장 안전한지 결정하는 데 있습니다.

기존 가지치기 방법은 각 전문가의 점수를 독립적으로 계산합니다. 즉, 각 전문가에 스칼라 중요도 점수를 할당하고(캘리브레이션 데이터셋을 기반으로), 각 레이어에서 중요도가 가장 낮은 전문가를 독립적으로 제거합니다. 각 전문가의 기여가 다른 모든 전문가와 독립적이라고 가정하기 때문에 이러한 방법을 1차 방법이라고 합니다. 실제로 전문가들은 협력 그룹을 형성합니다. 자체 실험에서 함께 활성화되는 전문가의 일관된 클러스터를 관찰했으며, 독립적인 연구에서도 [전문가 협력이 MoE의 안정적인 특성이고](https://arxiv.org/abs/2505.20225) [모델 성능에 중요하다는 사실이 확인되었습니다](https://arxiv.org/abs/2510.16411). 이러한 상호작용을 무시하면 특히 공격적인 비율로 가지치기할 때 최적보다 낮은 결과를 얻게 됩니다.

[최신 논문](https://arxiv.org/abs/2609.18916)에서는 쌍별 전문가 상호작용을 고려하는 2차 방법인 HOPE(Higher-Order Pruning of Experts)를 도출합니다. 50% 가지치기에서 HOPE는 모든 방법 중 가장 높은 평균 순위를 달성했으며, 차순위 가지치기 방법보다 에이전트 코딩 작업에서 최대 +6.1%의 향상을 보였습니다.

이 글에서는 HOPE의 작동 방식과 기존 방법보다 뛰어난 지점을 설명하고, 여러분의 모델에 HOPE를 사용하는 방법을 단계별로 살펴봅니다. MoE 아키텍처가 익숙하지 않은 독자는 [Transformers의 Mixture of Experts(MoEs)](https://huggingface.co/blog/moe-transformers)를 참고하세요.

### 전문가는 혼자 작동하지 않습니다

표준 MoE에서는 각 레이어에 EEE개의 전문가가 있고, 각 입력 토큰 xxx은 독립적으로 KKK개의 전문가로 라우팅됩니다(K=8 K = 8 K=8은 일반적인 선택입니다). 라우팅은 단일 레이어에서 수행되며, 이 레이어는 각 전문가 kkk에 대해 소프트맥스로 정규화된 게이트 가중치 gk(x)g_{k}(x)gk​(x)를 생성합니다. 그런 다음 게이트 가중치가 가장 높은 KKK개의 전문가를 선택해 활성 전문가 집합 T(x)⊂{1,..,E}T(x) \\subset\\{1,..,E\\}T(x)⊂{1,..,E}를 구성합니다. 이후 레이어의 출력은 ∑k∈T(x)gk(x)fk(x)\\sum\\limits_{k\\in T(x)}g_{k}(x)f_{k}(x)k∈T(x)∑​gk​(x)fk​(x)가 되며, 여기서 fk(x)f_{k}(x)fk​(x)는 토큰 xxx에 대한 전문가 kkk의 출력입니다.

![pruning_cartoon](https://cdn-uploads.huggingface.co/production/uploads/6ab2f79bd8000a811c648a32/ZTLrZSA4pDn81OVRHtCXq.png)

왼쪽: 각 토큰이 상위 K개 전문가로 라우팅되는 일반적인 MoE 레이어. 오른쪽: 1차 방법은 스칼라 중요도 점수를 기준으로 전문가의 순위를 매기고 삭제하지만, 최적의 가지치기 결정은 전문가 간 상호작용의 강도도 고려해야 합니다.

중요한 점은 선택된 집합 T(x)T(x)T(x)의 전문가들이 서로 독립적이지 않다는 것입니다. 전문가들은 함께 일관되게 공동 활성화되는 안정적인 협력 그룹을 형성합니다. 이는 [기존 문헌에서도 관찰된 현상이며](https://arxiv.org/abs/2505.20225), 전문가 공동 선택 패턴은 학습 초기에 나타나고 이후 안정화됩니다.

이는 MoE 레이어에서 전문가를 가지치기할 때 이러한 전문가 간의 고차 효과를 고려해야 한다는 뜻입니다. 전문가 A만 보면 제거해도 괜찮아 보일 수 있지만, 전문가 A와 자주 협력하고 A에 의존하는 전문가 B가 매우 중요하다면 A를 제거하는 것만으로 협력 그룹이 해체됩니다. 1차 방법은 A와 B의 점수를 পৃথ পৃথ 계산하기 때문에 이를 전혀 포착하지 못합니다.

이 효과는 높은 가지치기 비율에서 누적됩니다. 전문가를 최대 50%까지 제거하면 많은 협력 연결을 끊을 위험이 있습니다. 2차 가지치기가 가장 큰 이점을 제공하는 지점이 바로 여기입니다.

### HOPE가 가지치기 집합을 식별하는 방법

(HOPE 목적 함수의 전체 도출 과정은 [해당 논문의 부록 A](https://arxiv.org/abs/2609.18916)를 참고하세요.)

일부 MoE 레이어에서 PPP개의 전문가 집합을 삭제하면 모든 토큰 xxx에 대해 선택되는 전문가 집합 T(x)T(x)T(x)가 바뀔 수 있습니다. 선택되었을 전문가 중 가지치기된 전문가는 그다음으로 게이트 가중치가 높은 전문가로 대체됩니다. 그 결과 발생하는 레이어 출력의 변화가 가지치기 오류입니다. 우리의 목표는 캘리브레이션 집합의 토큰 xxx에 대한 가지치기 오류를 최소화하는, 미리 정한 가지치기 예산 ∣P∣\\vert P\\vert∣P∣을 갖는 가지치기 집합 PPP를 찾는 것입니다.

안타깝게도 PPP에 대해 가지치기 오류를 직접 최소화하는 것은 계산 불가능합니다. 가지치기 오류는 대체 전문가의 정체성에 의존하고, 대체 전문가의 정체성은 다시 가지치기 집합 PPP 자체에 의존하기 때문입니다. HOPE는 가지치기 오류의 지배적인 성분을 엄밀하게 상계하는 계산 가능한 대리 목적 함수 ZZZ를 정의하여 이 문제를 우회합니다.

Z=[∑j∈P∩T(x)gj(x)∥fj(x)∥2]2≥true errorZ = \\bigl[\\sum\\limits_{j \\in P\\cap T(x)} g_{j}(x)\\Vert f_j(x)\\Vert_{2}\\bigr]^{2} \\geq \\text{true error}Z=[j∈P∩T(x)∑​gj​(x)∥fj​(x)∥2​]2≥true error

ZZZ는 제곱합이므로 이를 전개하면 전문가 쌍 사이에 gi(x)gj(x)∥fi(x)∥2∥fj(x)∥2g_{i}(x)g_{j}(x)\\Vert f_{i}(x)\\Vert_{2}\\Vert f_{j}(x)\\Vert_{2}gi​(x)gj​(x)∥fi​(x)∥2​∥fj​(x)∥2​와 같은 교차항이 자연스럽게 생성됩니다. 이러한 항이 1차 방법에서 버리는 상호작용 항입니다.

최적의 가지치기 집합을 찾기 위해 HOPE는 캘리브레이션 집합에 대해 E[Z]\\mathbb{E}[Z]E[Z]를 최소화하며, 이는 다음과 같은 이차 계획법으로 다시 정식화할 수 있습니다.

P∗=arg min⁡p  p⊤Fps.t.p∈{0,1}E,∑k=1Epk=∣P∣P^{*} = \\argmin_{p}\\; p^{\\top}Fp \\quad \\text{s.t.} \\quad p \\in \\{0,1\\}^{E}, \\sum\\limits_{k=1}^{E}p_{k} = \\vert P\\vertP∗=pargmin​p⊤Fps.t.p∈{0,1}E,k=1∑E​pk​=∣P∣

여기서 F∈RE×EF\\in\\mathbb{R}^{E\\times E}F∈RE×E는 다음과 같이 정의됩니다.

Fi,j=1∣Xi,j∣∑x∈Xi,jgi(x)gj(x)∥fi(x)∥2∥fj(x)∥2whereXi,j={x:i,j∈T(x)}F_{i,j} = \\frac{1}{\\vert \\mathcal{X}_{i,j}\\vert} \\sum\\limits_{x\\in\\mathcal{X}_{i,j}} g_{i}(x)g_{j}(x)\\Vert f_{i}(x)\\Vert_{2}\\Vert f_{j}(x)\\Vert_{2}\\quad\\text{where}\\quad \\mathcal{X}_{i,j} = \\{x: i,j \\in T(x)\\}Fi,j​=∣Xi,j​∣1​x∈Xi,j​∑​gi​(x)gj​(x)∥fi​(x)∥2​∥fj​(x)∥2​whereXi,j​={x:i,j∈T(x)}

이 F-행렬은 캘리브레이션 데이터셋에서 수집되며, 연속 완화를 사용해 결과 이차 계획법(QP)을 푼 뒤(이진 해로 반올림하여) 최적의 가지치기 집합을 얻습니다. 각 항 Fi,jF_{i,j}Fi,j​는 두 전문가가 모두 공동 활성화된 토큰에 대해 전문가 iii와 jjj의 게이트 가중 출력 곱의 평균을 나타냅니다. 대각선 Fk,kF_{k,k}Fk,k​가 개별 전문가의 중요도를 측정하는 반면, 비대각선 항(Fi,jF_{i,j}Fi,j​, i≠ji\\neq ji=j)은 쌍별 협력 기여를 측정합니다.

실제로 FFF는 어떤 형태일까요? FFF는 레이어에 따라 달라지는 전문가 협력 구조를 보여 줍니다.

![fmatrix](https://cdn-uploads.huggingface.co/production/uploads/6ab2f79bd8000a811c648a32/xyjeUqqFm2FtQtKVdZhwq.png)

세 개의 대표 레이어(초기, 중간, 후기)에서 GLM-4.5-Air(레이어당 128개 전문가, 총 106B 파라미터)의 상호작용 행렬 F를 시각화한 것입니다. 블록 구조가 드러나도록 전문가를 클러스터링했습니다. 대각선에 나타나는 밝은 블록은 쌍별 상호작용 점수가 높은 전문가 그룹을 의미합니다.

실용적인 비용 측면에서 HOPE는 다른 전문가 가지치기 방법과 동일한 캘리브레이션 패스가 필요합니다(캘리브레이션 데이터에 대한 한 번의 순전파). 추가되는 작업은 이차 계획법을 푸는 것뿐입니다. 이 문제는 비볼록이지만 표준 QP 솔버를 사용해 효율적으로 풀 수 있습니다(레이어당 1~2초). 캘리브레이션에 걸리는 시간이 지배적이므로, 전체 추가 오버헤드는 1차 방법과 비교해 무시할 수 있는 수준입니다.

#### REAP과의 연관성

특히 HOPE는 최첨단 1차 전문가 가지치기 방법인 [Router-weighted Expert Activation Pruning(REAP)](https://arxiv.org/abs/2510.13999)과 밀접한 관련이 있습니다. HOPE의 F-행렬은 대각선에 개별 전문가 중요도 점수를 인코딩하며, 이 대각선은 REAP이 전문가의 가지치기 순위를 독립적으로 정할 때 사용하는 점수를 사실상 복원합니다(전문가 기여의 분산에 의해서만 차이가 납니다). FFF의 대각선은 REAP의 전문가 순위를 거의 완벽하게 복원합니다(Spearman ρ=0.988\\rho = 0.988ρ=0.988). 따라서 HOPE가 REAP보다 얻는 모든 향상은 FFF-행렬의 비대각선 항에 있는 2차 상호작용 항에서 비롯됩니다.

### 결과

HOPE를 전문가 가지치기 방법으로 평가하기 위해 3개 모델(Qwen3.5-122B, Qwen3.5-35B, GLM-4.5-Air)과 2개 캘리브레이션 데이터셋(Evol-CodeAlpaca 및 SWE-Bench Verified trajectories)을 사용했습니다. 가지치기 결과 모델을 테스트하기 위해 Tulu3 suite와 SWE-Bench Pro(에이전트 코딩)를 사용했으며, 10%에서 50%에 이르는 가지치기 비율에서 HOPE를 4개의 1차 기준선([REAP](https://arxiv.org/abs/2510.13999), [MAN](https://arxiv.org/abs/2606.15716), [EAN](https://arxiv.org/abs/2504.05586), Frequency)과 비교했습니다.

![results](https://cdn-uploads.huggingface.co/production/uploads/6ab2f79bd8000a811c648a32/5ArwjL0FoYK6nW1eazSzF.png)

HOPE는 모든 조건에서 가장 높은 종합 순위를 달성했으며 공격적인 가지치기 비율에서 우세했습니다. 왼쪽: 모든 모델, 캘리브레이션 집합 및 벤치마크에서의 평균 순위(낮을수록 좋음). 오른쪽: Qwen3.5-35B에서의 SWE-Bench Pro 성능.

모든 조건에서 HOPE는 가장 높은 평균 순위(5개 방법 중 2.07)를 달성했으며, 1위 성적 비율(조건의 39%)과 2위 이내 성적 비율(67%)도 가장 높았습니다. 일대일 비교에서 HOPE는 대다수 조건에서 다른 모든 기준선보다 뛰어났습니다.

예상대로 HOPE의 이점은 더 높은 가지치기 비율에서 커집니다. 더 많은 전문가를 제거할수록 더 많은 협력 연결이 끊어지고 FFF의 비대각선 항이 더 중요해집니다. 50% 가지치기에서 HOPE의 평균 순위는 1.58로 향상되었습니다(차순위 방법인 REAP은 2.42). 이는 메모리 절감량이 더 긴 컨텍스트, 더 깊은 추론, 더 많은 도구 호출을 수용할 만큼 충분히 큰, 배포와 가장 관련성이 높은 설정이기도 합니다.

HOPE의 이점은 모델이 실제 GitHub 이슈를 자율적으로 해결해야 하는 에이전트 코딩 벤치마크인 SWE-Bench Pro에서 가장 두드러집니다. 이러한 문제는 여러 턴에 걸쳐 구성적으로 다양한 작업(예: 코딩, 도구 호출, 추론 등)을 포함하며 다양한 전문가 조합을 활용할 가능성이 높습니다. 즉, 협력 구조를 보존하는 것이 가장 중요한 환경입니다. Qwen3.5-35B-A3B를 50% 가지치기했을 때 HOPE는 SWE-Bench Pro에서 42.3을 달성해 가지치기하지 않은 모델의 41.8과 비슷한 성능을 보였습니다(REAP은 36.2로 하락하여 13% 회귀).

특히 가지치기된 모델은 때때로 가지치기하지 않은 기준선과 비슷하거나 더 높은 성능을 보일 수 있습니다. 이는 관련성이 낮은 전문가를 제거하면 대상 작업에서 노이즈가 줄어들기 때문일 가능성이 큽니다. 이러한 효과는 HOPE에만 해당하지 않으며 [여러 가지치기 방법에서 관찰되었습니다](https://arxiv.org/abs/2504.06792). 또한 실험 결과, 어떤 가지치기 방법을 사용하든(HOPE 포함) 코드 중심 캘리브레이션이 비코딩 벤치마크의 성능을 심각하게 저하시키지는 않았고, HOPE는 캘리브레이션 집합과 관계없이 Tulu3 suite 전반에서 우수한 성능을 보였습니다.

#### HOPE는 더 많은 협력 전문가를 유지합니다

![exp_interaction_breakdown](https://cdn-uploads.huggingface.co/production/uploads/6ab2f79bd8000a811c648a32/S2fm99InMXuhOUFqEbI_2.png)

HOPE와 REAP의 결정이 일치하지 않는 전문가를 기준으로 나누어, 유지된 전문가와의 협력 강도(F-행렬 값으로 측정)의 평균을 나타낸 것입니다.

HOPE의 이점은 전체 상호작용 행렬 FFF를 대상으로 최적화하는 데서 비롯됩니다. 이것이 실제 가지치기 결정에 어떤 영향을 미치는지 이해하기 위해 HOPE와 REAP의 결정이 다른 전문가를 살펴보았습니다. 의견이 갈린 각 전문가에 대해 다른 유지된 전문가와의 평균 F-값, 즉 협력 점수를 계산했습니다. HOPE만 유지한 전문가는 REAP만 유지한 전문가보다 일관되게 더 높은 협력 점수를 보였습니다. 다시 말해 QP 최적화 과정에서 HOPE는 개별 점수가 높은 일부 전문가를 희생하고 협력 클러스터를 보존하는 쪽을 선택합니다. 벤치마크 성능의 향상(위에 제시됨)은 이러한 절충이 유리하다는 점을 입증합니다.

### HOPE 실행 방법

![pipeline](https://cdn-uploads.huggingface.co/production/uploads/6ab2f79bd8000a811c648a32/b8EewrlBWbGA0DlrBKa2E.png)

HOPE를 실행할 수 있는 기본 [패키지](https://github.com/awslabs/HOPE)를 만들었습니다. 단계는 세 가지입니다. 1) 캘리브레이션, 2) solve, 3) 가지치기. 이 패키지는 Qwen3, Qwen3-Next, Qwen3.5 및 GLM-4.5에서 테스트되었습니다.

#### 1단계: 캘리브레이션

```
hope calibrate \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --prompts calibration_prompts.txt \
    --out-path observations.h5
```


캘리브레이션 단계에서는 에이전트 롤아웃이든 단일 턴 트레이스든 캘리브레이션 데이터에 대해 한 번의 순전파를 실행합니다. 이 과정에서는 HuggingFace `transformers`를 사용해 각 MoE 레이어의 순전파에 연결합니다. 각 토큰에 대해 선택된 전문가, 게이트 가중치, 출력 노름 및 쌍별 공동 활성화 통계를 기록합니다. 이 단계의 출력은 HDF5 파일에 저장되는 F-행렬입니다.

#### 2단계: QP 풀기

```
hope solve \
    --obs-path observations.h5 \
    --prune-frac 0.50 \
    --out-path pruneset.json
```


이 명령은 F-행렬을 구체화하고 연속 완화를 통해 이진 이차 계획법을 풉니다. 레이어당 몇 초가 걸리며, 출력은 가지치기 집합을 포함하는 JSON 파일입니다. 기술적으로 이것이 HOPE 알고리즘의 최종 출력입니다.

#### 3단계: 가지치기

```
hope prune \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --pruneset-path pruneset.json \
    --out-path ./Qwen3.5-35B-A3B-pruned-50
```


파이프라인을 완료하려면 이 명령을 실행해 가지치기 집합으로부터 더 작은 모델 체크포인트를 실제로 생성합니다. 가지치기된 모델은 디스크와 메모리에서 더 작은 표준 HuggingFace 체크포인트입니다(vLLM 또는 SGLang 모델 정의를 변경할 필요가 없습니다).

더 간결한 Python API도 공개했습니다.

```
from hope.calibrate import calibrate
from hope.solve import solve
from hope.prune import prune_model

calibrate("Qwen/Qwen3.5-35B-A3B", calibration_prompts, "observations.h5")
solve("observations.h5", prune_frac=0.50, out_path="pruneset.json")
prune_model("Qwen/Qwen3.5-35B-A3B", "pruneset.json", "./Qwen3.5-35B-A3B-pruned-50")
```


### HOPE의 한계

HOPE의 이점은 더 높은 가지치기 비율(40~50%)에서 가장 두드러집니다. 낮은 가지치기 비율(10~20%)에서는 REAP 및 MAN과 같은 1차 방법이 경쟁력 있는 경우도 있으며 HOPE의 이점은 줄어듭니다. 또한 대부분의 전문가 가지치기 알고리즘과 마찬가지로 HOPE는 각 레이어를 독립적으로 가지치기하며 레이어 간 상호작용은 고려하지 않습니다. 논문에서는 레이어 간 HOPE를 위한 이론적 프레임워크를 제시했지만 결과는 아직 초기 단계입니다. 마지막으로 HOPE는 쌍별 상호작용을 포착하지만, 더 높은 차수의 상호작용(3개 이상의 전문가 조합)이 추가적인 신호를 제공할 수도 있습니다. 다만 이러한 상호작용은 조합적으로 추정하기 어렵습니다.

### 추가 정보

자세한 내용은 해당 논문과 코드베이스를 참고하세요.

논문: [https://arxiv.org/abs/2609.18916](https://arxiv.org/abs/2609.18916)

코드: [https://github.com/awslabs/HOPE](https://github.com/awslabs/HOPE)

HOPE를 차세대 하이브리드 아키텍처에 적용하는 방법은 [해당 repository](https://github.com/awslabs/hybrid-model-factory/)를 참고하세요.

인용:

```
@article{tseng2026hope,
    title={Higher-Order Pruning of Experts in Mixture-of-Experts Language Models},
    author={Tseng, Alex M. and Kaul, Prannay and Zancato, Luca and Xia, Wei and Soatto, Stefano},
    journal={arXiv preprint arXiv:2609.18916},
    year={2026},
}
```
