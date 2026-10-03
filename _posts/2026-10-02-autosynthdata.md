---
layout: post
title: "AutoSynthData: 엔터프라이즈 에이전트를 위한 학습 데이터 생성"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/68642da9885da181fde1ad70/8kSzlHLOdu5hchCtPnK8D.png
image: assets/images/blog/posts/2026-10-02-autosynthdata/thumbnail.png
authors:
  - user: ServiceNow-AI
slug: "autosynthdata"
source_url: "https://huggingface.co/blog/ServiceNow-AI/autosynthdata"
source_published_date: "2026-10-02"
source_published_at: "2026-10-02T04:01:31+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [AutoSynthData: Generating Training Data for Enterprise Agents](https://huggingface.co/blog/ServiceNow-AI/autosynthdata)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ServiceNow-AI/autosynthdata -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# AutoSynthData: 엔터프라이즈 에이전트를 위한 학습 데이터 생성

[![AutoSynthData_thumbnail_1200x648 (2)](https://cdn-uploads.huggingface.co/production/uploads/68642da9885da181fde1ad70/mlI168havBLGXY2qbd4ud.png)](https://cdn-uploads.huggingface.co/production/uploads/68642da9885da181fde1ad70/mlI168havBLGXY2qbd4ud.png)

엔터프라이즈에는 자체 환경에서 제대로 작동하는 에이전트가 필요합니다. 이러한 에이전트에게 요청하는 작업은 사용하는 시스템, 따르는 규칙, 데이터 상태에 따라 결정됩니다. 모델이 전반적으로 뛰어난 역량을 갖추고 있어도 특정 환경에서는 어려움을 겪을 수 있습니다. 예를 들어 제대로 처리하지 못하는 워크플로, 잘못 사용하는 도구 조합, 지키지 못하는 제약 등이 있을 수 있습니다. 엔터프라이즈가 개선해야 하는 부분은 바로 이러한 약점입니다.

어려운 점은 이러한 약점을 학습 데이터로 전환하는 것입니다. 개별 실패는 무언가를 알려주지만, 모델을 학습하려면 동일한 역량을 다양한 상황에서 발휘하게 하는 새로운 작업이 많이 필요합니다. 또한 이러한 작업은 해당 환경에서 실제로 완료할 수 있어야 하고, 누군가 실제로 요청할 법한 업무와 유사해야 하며, 에이전트의 성공 여부를 신뢰성 있게 확인할 방법도 있어야 합니다.

ServiceNow CoreAI에서는 이러한 역량 격차를 학습 데이터로 전환하기 위해 AutoSynthData를 구축했습니다. AutoSynthData는 대상 모델의 실패와 더 강력한 교사의 성공을 활용해 모델이 다음에 학습해야 할 내용을 결정한 다음, 해당 역량을 발휘하게 하는 새로운 작업을 생성하고 검증합니다. 모델이 개선되면 커리큘럼은 모델이 여전히 어려워하는 부분으로 이동합니다. 이 파이프라인은 EnterpriseOps Gym([Malay et al., 2026](https://arxiv.org/abs/2603.13594))과 [released dataset](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym)을 사용해 설명합니다. 먼저 에이전트가 작동하는 환경과 학습에 유용한 작업의 조건을 살펴보겠습니다.

## 유용한 에이전트 작업이란 무엇인가? {#section-1}

에이전트 환경은 에이전트가 작동하는 세계를 정의합니다. 여기에는 에이전트가 관찰하고 수정할 수 있는 상태, 호출할 수 있는 도구와 API, 그리고 에이전트의 행동으로 인해 발생하는 상태 전이가 포함됩니다.

작업은 이 환경 안에서 구체화됩니다. 다음과 같은 추상화를 사용합니다.

```
task = (system specification, user prompt, verifier)
```


### 시스템 사양

시스템 사양은 시스템 지침, 환경 정책, 그리고 해당하는 경우 시드된 데이터베이스 상태나 지식 문서 집합과 같은 작업별 초기화를 포함하여 에이전트가 작동하는 제약을 정의합니다.

사양은 환경의 도구, 상태 및 지원되는 행동과 호환되어야 합니다. 지침은 명확해야 하며, 오로지 난이도를 인위적으로 높이기 위해 도입된 자의적인 제약은 피해야 합니다.

### 에이전트 대상 작업

사용자 프롬프트는 사용자가 에이전트에게 달성하도록 요구하는 내용과 사용자 수준의 제약을 지정합니다. 생성된 작업은 세 가지 속성을 충족해야 합니다.

실행 가능성. 현재 환경에서 시스템 사양을 준수하면서 사용자 프롬프트를 만족하는 궤적이 하나 이상 존재해야 합니다. 이를 통해 사용할 수 없는 도구, 접근할 수 없는 지식, 불가능한 상태 전이 또는 정책상 금지된 행동에 의존하는 작업을 배제합니다.

현실성. 사용자 프롬프트는 대상 환경에서 사용자가 실제로 요청할 법한 내용과 유사해야 합니다. 일반적으로 실행 가능한 행동의 공간은 현실적인 워크플로의 공간보다 훨씬 큽니다.

난이도. 학습을 위해 작업은 현재 에이전트의 약점을 드러내야 합니다. 이미 안정적으로 해결되는 작업은 새로운 학습 신호를 거의 제공하지 않습니다. 따라서 유용한 영역은 실행 가능하고 현실적이지만 아직 일관되게 해결되지 않는 작업입니다.

### 검증기

검증기는 결과 궤적이 작업을 성공적으로 완료했는지 판단합니다. 검증기는 세 가지 속성을 충족해야 합니다.

일관성. 사용자 프롬프트, 시스템 사양 및 작업별 환경 상태와 일치해야 합니다.

건전성. 작업을 충족하지 못하거나 관련 제약을 위반하는 궤적을 거부해야 합니다.

완전성. 특정 하나의 참조 궤적을 인코딩하는 대신 유효한 해를 수용해야 합니다.

이러한 속성은 학습 중에 직접적인 영향을 미칩니다. 느슨한 검증기는 잘못된 행동에 보상을 줄 수 있고, 지나치게 제한적인 검증기는 유효한 해에 불이익을 줄 수 있습니다.

## 개요 {#section-2}

환경과 대상 모델이 주어지면 AutoSynthData는 시스템 사양, 사용자 프롬프트 및 검증기로 구성된 학습 작업을 생성합니다. 생성된 작업은 환경에 기반하며 현재 모델에 유용한 학습 신호를 제공하도록 선택됩니다.

[![figure-01](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/sM50frTSx9Q7Yyh6UNcGz.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/sM50frTSx9Q7Yyh6UNcGz.png)

AutoSynthData는 먼저 진단 작업을 사용해 환경에서 대상 모델을 평가하고, 모델이 완료하는 데 어려움을 겪는 작업의 패턴을 식별합니다. 더 강력한 교사는 이러한 작업 중 어떤 작업을 해결할 수 있는지, 성공적인 행동은 어떤 모습인지 파악하는 데 도움을 줍니다. AutoSynthData는 그 결과로 얻은 역량 격차를 실행 가능한 새로운 작업으로 전환하고, 환경에서 각 작업을 확인하며, 승인된 샘플을 사후 학습에 사용합니다. 업데이트된 모델을 평가하면 남아 있는 격차가 드러나고 다음 생성 단계를 이끌 수 있습니다.

[![figure-02](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/huMlVCBcc-IseCIk5uEIJ.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/huMlVCBcc-IseCIk5uEIJ.png)

## 모델의 실패에서 커리큘럼으로 {#section-3}

AutoSynthData는 대상 환경에서 수행한 평가 실행을 사용해 모델이 다음에 학습해야 할 내용을 식별합니다. EnterpriseOps Gym 실험에서는 대상 모델과 더 강력한 교사를 모두 평가 작업에 실행합니다. 이러한 실행을 분석해 다음을 식별합니다.

- 테스트 중인 역량;

- 관련된 도구와 워크플로 구조;

- 대상 모델이 실패하는 지점과 교사가 성공하는 방식;

- 올바른 최종 상태가 충족해야 하는 속성;

- 테스트 중인 역량을 유지하면서 달라질 수 있는 차원.

이러한 결과를 정제해 비식별화된 역량 사양 카드로 만듭니다. 평가 작업은 모델이 무엇을 학습해야 하는지 안내하지만, 생성기는 원래의 프롬프트, 엔터티, 궤적 또는 검증기 세부 정보를 받지 않습니다. 생성기는 카드를 받아 서로 다른 프롬프트, 상태 및 해결 경로를 가진 새로운 작업을 만드는 데 사용합니다.

[![figure-03](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/ZwShYC2FDYFXU9yR9yz1t.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/ZwShYC2FDYFXU9yR9yz1t.png)

## 작업 생성 및 확장 {#section-4}

역량 격차를 식별하면 무엇을 가르쳐야 하는지는 알 수 있지만, 학습에는 해당 역량을 발휘하게 하는 다양하고 많은 작업이 필요합니다. AutoSynthData는 사양 카드를 사용해 이러한 작업을 생성합니다.

대상 모델이 다음 워크플로를 요구하는 작업에서 어려움을 겪는다고 가정해 보겠습니다.

[![figure-04](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/UCTRFFZ7ttq6JEXZr5_Qi.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/UCTRFFZ7ttq6JEXZr5_Qi.png)

생성기는 이 워크플로를 실행하는 새로운 작업을 만들면서 엔터티, 초기 환경 상태, 워크플로 구성, 도구 조합, 표현 방식 및 난이도를 변경합니다. 그런 다음 더 강력한 교사가 각 작업에 대해 성공적인 궤적을 시연합니다. 지도 미세 조정(SFT)의 경우 이러한 시연을 통해 대상 모델은 새로운 상황에서 해당 역량을 적용하는 방법을 학습합니다.

AutoSynthData는 두 단계로 데이터셋을 구축합니다. 먼저 핵심 샘플을 생성하고 검증한 다음, 이를 새로운 변형으로 확장합니다.

### Target

Target 단계에서는 역량 사양으로부터 핵심 학습 샘플 집합을 만듭니다. 워커는 독립적인 작업을 병렬로 생성하고, 작업을 마치면 새로운 target을 가져옵니다. 각 후보는 승인되기 전에 검증, 실행, 솔버 평가 및 수리 과정을 거칩니다. 그 결과 대상 모델이 학습해야 할 내용을 중심으로 검토가 완료된 예제 배치가 만들어집니다.

### Multiply

Multiply 단계에서는 승인된 target 샘플의 새로운 변형을 만들어 데이터셋을 확장합니다. 각 변형은 자체 사용자 요청, 환경 상태, 엔터티 구성, 참조 궤적 및 검증기를 가지며 동일한 검증 및 실행 검사를 통과해야 합니다. Multiply된 샘플은 다른 Multiply 샘플의 시드가 될 수 없습니다. 이를 통해 확장을 검토가 완료된 target 집합에 고정하고 세대 간 드리프트를 제한합니다.

### 구현 세부 사항

두 단계를 모두 지원하기 위해 AutoSynthData는 생성 제어와 환경별 실행을 분리합니다. 공유 컨트롤러는 생성, 품질 관리, 커버리지 및 데이터셋 구축을 조정하고, 어댑터는 환경 실행, 작업 및 상태 관리, 참조 재생, 결정론적 검증, 솔버 실행 및 작업 프로파일링을 처리합니다.

병렬 target 생성과 Multiply를 함께 사용하면 학습 규모의 데이터셋을 구축할 수 있습니다. 그러나 그 유용성은 모든 후보에 적용되는 검사에 달려 있습니다. 작업은 실행 가능해야 하고, 해는 작동해야 하며, 검증기는 성공과 실패를 구분해야 합니다.

[![figure-05](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/zUTtl1ILl-fTtA-0RZdgV.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/zUTtl1ILl-fTtA-0RZdgV.png)

## 고품질 합성 데이터에는 생성 이상의 것이 필요하다 {#section-5}

그럴듯한 요청을 생성하는 것만으로는 유용한 학습 데이터를 만들 수 없습니다. 작업이 대상 환경에서 불가능할 수도 있고, 참조 해가 실행될 때 실패할 수도 있으며, 검증기가 잘못된 최종 상태에 보상을 줄 수도 있습니다. AutoSynthData는 작업을 학습에 승인하기 전에 이러한 속성을 확인합니다.

AutoSynthData는 두 수준에서 품질을 검토합니다. 개별 후보는 검증을 통과해야 하고, 배치는 유용한 커버리지와 다양성을 제공해야 합니다.

### 샘플 수준의 검증 및 수리

각 후보는 학습 데이터셋에 들어가기 전에 품질 관리 루프를 통과해야 합니다. 먼저 솔버 평가를 통해 난이도를 측정합니다. 여기서 사용한 구성에서는 대상 모델이 세 번의 시도 중 최대 한 번만 해결하고, 더 강력한 솔버가 세 번 중 최소 두 번 해결하는 작업을 선호합니다. 후보는 양성 및 음성 검증과 제한된 수리 과정도 거칩니다.

[![figure-06](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/r3nXamYF6OEI2ezj7MX78.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/r3nXamYF6OEI2ezj7MX78.png)

#### 양성 검증

양성 게이트의 질문은 다음과 같습니다. 의도한 해가 생성된 작업을 해결하는가?

파이프라인은 대상 환경에서 참조 궤적을 실행하고, 그 결과 상태를 후보의 검증기와 대조합니다. 이를 통해 프롬프트, 초기 상태, 해 및 성공 기준 사이의 불일치를 찾아냅니다.

#### 음성 검증

음성 게이트의 질문은 다음과 같습니다. 관련된 잘못된 결과는 실패하는가?

예를 들어 예상 결과의 일부를 변경한 뒤 해당 상태가 더 이상 검증을 통과하지 못하는지 확인할 수 있습니다. 이를 통해 의도한 행동을 요구하지 않고도 성공을 부여하는 취약한 검증기를 포착합니다.

#### 비평 및 수리

실패한 후보는 폐기되기 전에 비평가에게 전달됩니다. 비평가는 샘플과 그 실패를 조사하면서 일관되지 않은 상태, 불가능한 워크플로, 잘못된 작업 구성, 잘못된 참조 궤적, 취약한 검증기 로직 또는 의도한 역량과의 불일치를 찾습니다. 비평가의 분석 결과는 재시도 횟수를 고정된 한도로 제한하면서 수리를 안내합니다.

```
candidate
↓
failure
↓
critique / diagnosis
↓
targeted repair
↓
run the gates again
↓
accept or retry
```


수리된 작업은 관련 검사를 다시 통과해야 합니다. 진단 결과는 생성을 처음부터 다시 시작할 필요 없이 기존 후보를 수리하도록 안내합니다.

이러한 검사를 통과하면 샘플은 학습에 사용될 자격을 얻지만, 개별적으로 유효한 샘플도 반복적이거나 불균형한 데이터셋을 구성할 수 있습니다. 따라서 AutoSynthData는 배치 수준에서도 생성을 검토합니다.

### 배치 수준 검토

배치는 일부 쉬운 작업군을 과도하게 대표하거나, 특정 역량을 놓치거나, 낮은 수율의 패턴에 너무 많은 생성 노력을 투입한 결과를 반영할 수 있습니다.

메타 검토는 각 배치에서 승인된 샘플, 거부된 샘플 및 생성 동작을 조사합니다. 다음과 같은 질문을 던집니다.

- 어떤 작업군이 과도하게 대표되며, 어떤 역량 차원이 빠져 있는가?

- 동일한 종류의 예제가 반복해서 나타나는가?

- 특정 target이 계속 생성에 실패하는가?

- 비평에서 체계적인 문제가 나타나는가?

- 다음 배치를 위해 어떤 지침을 변경해야 하는가?

컨트롤러는 승인된 데이터셋의 커버리지를 추적하고, 과도하게 대표되는 영역에서는 생성을 줄이며, 격차가 있는 부분에 더 많은 작업을 할당합니다. 특정 영역에서 계속해서 품질이 낮은 후보가 생성되면 비평과 메타 검토를 통해 생성 전략을 변경합니다. 이러한 조정은 사용 가능한 생성 예산과 데이터셋 크기 요구사항 내에서 유용한 학습 신호, 작업 품질, 커버리지, 다양성 및 낮은 중복성을 균형 있게 유지합니다.

이러한 피드백 루프는 개별 작업과 그 작업으로 구성된 데이터셋을 모두 개선합니다. 샘플 수준 검사는 후보 수리를 안내하고, 배치 수준 검토는 향후 생성을 안내합니다.

## 학습 프런티어 확장 {#section-6}

모델이 개선되면 유용한 학습 분포도 변합니다. AutoSynthData는 합성 데이터 생성을 대상 모델의 역량 경계 근처에 있는 작업을 탐색하는 과정으로 봅니다. 약점을 드러낼 만큼 어렵지만, 교사가 신뢰할 수 있는 시연을 제공할 만큼은 해결 가능한 작업을 찾는 것입니다.

사후 학습 후에는 동일한 환경에서 업데이트된 모델을 평가합니다. 이제 안정적으로 해결하는 작업은 다음 학습 라운드에서 덜 유용하고, 지속적인 실패는 여전히 주의가 필요한 역량을 가리킵니다. 이러한 결과는 다음 생성 라운드를 안내할 수 있습니다.

[![figure-07](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/ONKQXJ1u9buDLEa-AuXMo.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/ONKQXJ1u9buDLEa-AuXMo.png)

실험은 SFT에 초점을 맞추고 있지만, 동일한 메커니즘은 강화 학습(RL)에도 적용할 수 있습니다. 현재 정책에 도전하고 신뢰할 수 있는 학습 신호를 제공하는 작업을 생성하고, 학습한 다음, 업데이트된 정책에 맞춰 생성 대상을 이동하는 방식입니다. 앞으로 SFT를 넘어 이러한 이동형 난이도 보정 프런티어를 테스트할 계획입니다.

## EnterpriseOps Gym 실험 {#section-7}

이 접근 방식이 상태를 유지하는 엔터프라이즈 환경의 작업에서 모델을 개선하는지 테스트하기 위해 EnterpriseOps Gym을 사용합니다. Gym의 Hybrid 및 ITSM 환경에서 학습 작업을 생성하고, 승인된 샘플로 대상 모델을 미세 조정한 뒤, 그 결과로 생성된 체크포인트를 평가합니다.

### Hybrid

[Gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)을 대상 모델로, [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)을 교사로 사용해 EnterpriseOps Gym의 Hybrid 도메인에서 파이프라인을 테스트했습니다.

AutoSynthData는 약 18시간 동안 2,000개의 합성 학습 샘플을 생성했습니다. 이 데이터셋으로 Gemma를 미세 조정하고, 그 결과로 생성된 체크포인트를 benchmark에서 평가했습니다. 가장 성능이 좋은 체크포인트는 epoch 5였습니다.

### Hybrid 결과

합성 SFT 체크포인트는 평균 Pass@1을 7.2%포인트, 상대적으로 35% 향상시키고, 검증기 성공률을 63.01%에서 68.55%로 높입니다. Gemma와 reference model 사이에 있던 원래 Pass@1 격차의 59%를 해소합니다.

[![figure-08](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/D2CdcykJInh82mYAMGz3I.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/D2CdcykJInh82mYAMGz3I.png)

학습 작업은 역량 사양으로부터 새롭게 생성되었으며, 생성기는 원래 평가 작업을 받지 않았습니다. 이 결과는 이 실험에 사용된 환경인 EnterpriseOps Gym Hybrid에서 개선이 이루어졌음을 보여줍니다.

### ITSM

또한 EnterpriseOps Gym의 ITSM 도메인에 AutoSynthData를 적용했습니다. Gemma-4-26B-A4B-it을 대상 모델로 사용하고 [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)을 교사로 사용했습니다. AutoSynthData는 66시간 동안 1,994개의 합성 학습 샘플을 생성했습니다. 이후에 설명하는 Hybrid 실행보다 생성에 더 오래 걸린 주된 이유는 ITSM 실행에서 더 큰 교사 모델을 사용했고, 처리량을 개선한 파이프라인 최적화보다 먼저 수행되었기 때문입니다.

ITSM에서 합성 SFT는 평균 Pass@1을 18.77%에서 27.18%로 높였으며, 이는 이 접근 방식이 두 번째 도메인에서도 성능을 개선한다는 것을 보여줍니다.

[![figure-09](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/3G6bJhA0JKZMJdiwfeHYs.png)](https://cdn-uploads.huggingface.co/production/uploads/66a41f3c1d52bffc13c285a5/3G6bJhA0JKZMJdiwfeHYs.png)

## 루프 닫기 {#section-8}

학습에 가장 유용한 작업은 환경과 그 환경에서 작동하는 모델 모두에 따라 달라집니다. AutoSynthData는 모델의 실패를 활용해 무엇을 생성할지 선택하고, 환경을 기준으로 새로운 작업을 검증하며, 해당 작업을 사후 학습에 사용할 수 있도록 합니다. EnterpriseOps Gym의 결과는 통제된 환경에서 이 접근 방식의 가치를 보여줍니다. 모델이 변화하면 동일한 프로세스를 통해 남아 있는 격차에 집중할 수 있습니다.
