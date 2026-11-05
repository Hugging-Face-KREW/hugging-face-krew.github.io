---
layout: post
title: "StateBench: 긴 문서에서 언어 모델이 상태 변화를 추적하는 방식 테스트"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/638e39b249de7ae552d977b5/5g9HfQLFFaQE0fslWrxh0.png
image: assets/images/blog/posts/2026-11-03-statebench/thumbnail.png
authors:
  - user: allenai
slug: "statebench"
source_url: "https://huggingface.co/blog/allenai/statebench"
source_published_date: "2026-11-03"
source_published_at: "2026-11-03T17:15:21+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [StateBench: Testing how language models track state changes over long documents](https://huggingface.co/blog/allenai/statebench)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/allenai/statebench -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# StateBench: 긴 문서에서 언어 모델이 상태 변화를 추적하는 방식 테스트

📄 [기술 보고서](https://allenai.org/papers/statebench) | 🤗 [데이터](https://huggingface.co/datasets/allenai/state-bench) | 💻 [코드](https://github.com/allenai/state-bench)

[![StateBench illustration showing colored paths changing positions across a sequence.](https://www.datocms-assets.com/64837/1792006215-statebench-draft-blog-google-docs-image-1.png)](https://www.datocms-assets.com/64837/1792006215-statebench-draft-blog-google-docs-image-1.png)

[StateBench](https://github.com/allenai/state-bench)를 소개합니다. StateBench는 긴 문맥에서 상태를 추적하는 능력, 즉 긴 문서에서 엔터티가 변화함에 따라 언어 모델이 해당 엔터티에 대한 정보를 얼마나 잘 추적하는지를 평가하는 벤치마크입니다.

상태 추적은 변수에 값이 할당되고 재할당되는 코드를 읽는 작업부터, 등장인물이 물리적으로 이동하고 물건이 다른 사람의 손으로 넘어가는 이야기를 따라가는 작업까지 다양한 과제의 기반이 됩니다. 예를 들어 추리 소설에서 살인 사건이 일어나기 직전에 “John이 Mary에게 칼을 건넸다”라고 쓰여 있다면, 각 인물의 상태가 바뀌었다는 것을 아는 것이 중요합니다. 하지만 모든 상태 추적 문제가 같은 방식으로 만들어지는 것은 아닙니다. 어떤 문제는 최신 업데이트만 찾으면 되지만, 더 어려운 문제는 변화의 연쇄를 따라가야 하며, 한 단계를 놓치면 최종 답이 틀릴 수 있습니다. 예를 들어 더 어려운 추리 상태 추적 문제는 John이 자신의 배낭 안에 있던 물건을 Mary에게 건네는 경우입니다. 이때 Edward가 John의 배낭에 칼을 넣었다는 사실이 앞서 알려져 있었다고 해 보겠습니다.

Transformers는 긴 의존성 연쇄를 포함하는 어려운 상태 추적 변형 문제에서 이론적으로 어떤 한계가 있는지 잘 알려져 있으며, 이는 실행 중인 메모리를 계속 전달하는 Gated DeltaNet과 같은 순환 및 하이브리드 아키텍처에 대한 동기를 제공합니다. 기존 상태 추적 테스트는 가장 어려운 사례에 집중하는 경향이 있지만, 이러한 설계가 더 일반화됨에 따라 연구자들은 각 아키텍처가 어떤 종류의 상태 추적을 처리할 수 있는지 알아야 합니다. 우리는 가장 어려운 종류의 상태 추적뿐 아니라 더 단순하고 잠재적으로 더 흔한 종류에서도 아키텍처 간 차이를 이해하고자 합니다.

이를 위해 서로 다른 복잡도의 상태 추적 문제를 포괄하는 StateBench 벤치마크를 개발하고, 이를 사용해 서로 다른 아키텍처를 가진 언어 모델 간의 차이를 평가했습니다. 이러한 과제에 직접 학습된 경우 순환 및 하이브리드 모델은 transformers보다 훨씬 더 멀리 일반화할 수 있었습니다. 반면 Gemini 3.8 Flash, Claude Opus 5.5, GPT-6 Sol and Luna를 포함한 프런티어 모델은 입력이 길어질수록 더 어려운 형태의 상태 추적에서 여전히 무너졌습니다.

## 상태 추적의 네 가지 수준 {#section-1}

StateBench는 점점 더 복잡한 상태 업데이트 연산을 허용하는 네 가지 난이도 수준으로 구성됩니다. 프로그래머라면 이러한 상태 추적 복잡도 수준을 서로 다른 유형의 변수 할당 연산을 허용하는 것으로 이해할 수 있습니다.

- R-trivial 문제는 불변 할당을 사용합니다. 변수의 값은 절대 바뀌지 않으므로 모델은 처음 설정된 값을 찾아내기만 하면 됩니다. 코드로 표현하면 let x = 4;와 같고, 일상생활에서는 생년월일처럼 변하지 않고 고정된 값에 해당합니다.

- Aperiodic 문제는 값에 의한 재할당을 허용합니다. 모델은 가장 최근의 값을 찾아야 하며 이전 값은 무시할 수 있습니다. 코드로 표현하면 x = 4 다음에 나중에 x = 2가 나오는 경우이고, 일상생활에서는 최신 설정이 현재 온도를 결정하는 온도 조절기와 같습니다.

- Soluble 문제는 자기 참조 업데이트를 추가합니다. 각 업데이트는 현재 값을 기반으로 하므로 모델은 최신 할당을 찾는 데 그치지 않고 변화를 계속 전달해야 합니다. 코드로 표현하면 x = x + 5;와 같고, 일상생활에서는 각 점수가 누적 합계에 더해지는 게임 점수와 같습니다.

- General finite 문제는 참조에 의한 할당을 추가합니다. 한 변수가 다른 변수의 현재 값을 취할 수 있으므로 업데이트 사이에 의존성이 생깁니다. 코드로 표현하면 x = y 또는 두 변수의 교환과 같습니다. 마술사가 다섯 개의 컵 중 하나 아래에 공을 숨긴 다음 컵의 위치를 반복해서 바꾼다고 상상해 보세요. 공을 찾으려면 그러한 교환으로 공의 위치가 어떻게 바뀌는지 계속 추적해야 합니다. 이는 다섯 개 객체를 연속적인 교환이 어떻게 재배열하는지 추적하는 S₅ 상태 추적 문제를 보여줍니다. 이 문제는 네 가지 수준 중 가장 어려운 수준에 속하며, [상태 추적의 한계에 관한 이전 연구](https://arxiv.org/html/2404.08819v2#S3.SS1)에서 논의되었습니다.

이러한 수준은 직관적으로 변수 할당의 관점에서 이해할 수 있지만, 상태 추적에 관한 이론적으로 동기가 부여된 구분이기도 합니다. 이는 유한 모노이드에 대한 단어 문제로 볼 수 있는 상태 추적 문제를 원자적 기본 요소로 분해하는 대수학의 한 분야인 Krohn–Rhodes 이론에 기반합니다. 계층 구조의 각 수준은 서로 다른 기본 요소 집합으로 구성할 수 있는 상태 추적 문제의 한 부류에 해당합니다.

네 가지 수준을 공정하게 비교하기 위해, 일반적인 유한 변수 할당 시퀀스에서 시작해 더 쉬운 세 가지 버전을 도출하여 난이도별로 하나씩, 총 네 개의 비교 가능한 StateBench 문제를 만듭니다. 네 문제 모두 변수와 서로 다른 값의 개수가 같고, 할당이 시퀀스에 나타나는 지점도 같습니다. 달라지는 것은 할당의 유형뿐입니다.

## 우리가 발견한 것 {#section-2}

### 사전 학습된 모델

StateBench의 r-trivial, aperiodic, general finite 수준에서 다양한 오픈 사전 학습 모델을 테스트하여 상태 추적이 어디에서 무너지는지, 그리고 서로 다른 아키텍처가 이론적으로 예측된 지점에서 실패하는지를 살펴보았습니다. 표준적인 이론적 가정에 따르면 transformers는 r-trivial을 안정적으로 처리할 수 있고 soluble 및 aperiodic도 어느 정도 처리할 수 있지만, general 문제에서는 실패할 것으로 예측됩니다. 반면 더 유연한 메모리 업데이트를 사용하는 Gated DeltaNet은 원칙적으로 네 가지 수준을 모두 처리할 수 있습니다.

먼저 학습 데이터와 문맥 길이가 모두 문서화된 오픈 사전 학습 모델을 살펴보았습니다. 여기에는 70억 개 파라미터 모델인 Olmo 3(transformer)와 Olmo Hybrid(Gated DeltaNet 레이어를 추가)가 포함되었으며, 2억 7,500만 개 및 27억 개 파라미터를 가진 더 작은 모델도 포함되었습니다. 이 모델들은 모두 동일한 데이터로 사전 학습되었고, 학습 중 64,000 토큰보다 긴 문맥을 본 모델은 없었습니다. 또한 대부분의 모델을 이 길이를 훨씬 넘어서는 입력으로 테스트했습니다.

각 모델의 점수를 산출했을 때 거의 모든 모델이 r-trivial 문제에서는 좋은 성능을 보였지만, 입력이 길어질수록 aperiodic 문제에서 정확도가 떨어졌고 general finite 문제에서는 무작위 추측 수준에 머물렀습니다. 여기서는 문제 형식이 중요했습니다. 2억 7,500만 개 파라미터의 hybrid 및 Gated DeltaNet 모델은 RULER 스타일의 r-trivial 문제에서는 최고 성능에 가까운 점수를 기록했지만, 동일한 문제를 스프레드시트 편집 형식으로 제시했을 때는 무작위 추측 수준에 가까웠습니다. 이러한 차이는 StateBench가 동일한 근본 문제를 여러 형식으로 테스트하는 이유 중 하나입니다.

[![Accuracy of pretrained transformer, hybrid, and Gated DeltaNet models versus input length in tokens; soluble results are not reported.](https://www.datocms-assets.com/64837/1792006353-statebench-draft-blog-google-docs-image-2.png)](https://www.datocms-assets.com/64837/1792006353-statebench-draft-blog-google-docs-image-2.png)

### 프런티어 모델

그다음 네 가지 프런티어 모델인 Gemini 3.8 Flash, Claude Opus 5.5, GPT-6 Sol and Luna를 테스트했습니다. 각 모델에 변수 할당 시퀀스를 읽고 모든 변수의 최종 값을 생성하도록 요청했습니다.

네 모델 모두 r-trivial 및 aperiodic 문제에서는 좋은 성능을 보였습니다. 하지만 soluble 및 general finite 문제에서는 입력이 길어질수록 정확도가 급격히 떨어졌고, 가장 긴 general finite 문제에서는 13% 이하로 하락했습니다. 가능한 요인 중 하나는 민감도입니다. 이러한 더 어려운 수준에서는 최종 답에 영향을 줄 수 있는 단계가 훨씬 많아지므로, 모델이 업데이트 하나를 놓치고도 결과를 틀리지 않을 여지가 줄어듭니다.

[![Accuracy of Gemini 3.8 Flash, Claude Opus 5.5, GPT-6 Sol, and GPT-6 Luna across the four StateBench levels versus input length in tokens.](https://www.datocms-assets.com/64837/1792006435-statebench-draft-blog-google-docs-image-3.png)](https://www.datocms-assets.com/64837/1792006435-statebench-draft-blog-google-docs-image-3.png)

## 과제에 직접 학습된 모델 {#section-3}

다음으로 더 어려운 수준에서 모델이 겪는 어려움이 아키텍처 자체의 한계를 반영하는지, 아니면 사전 학습만으로는 모델이 이러한 문제를 해결하는 방법을 배우지 못하는지를 조사했습니다. 특정 과제에 대한 직접 학습은 더 유리한 설정으로, 상태 추적 문제가 모델을 사전 학습하는 데 사용된 데이터보다는 underlying architecture와 optimizer에서 비롯되는 정도를 보여줍니다.

StateBench를 사용해 2억 7,500만 개 파라미터 모델 여러 개를 처음부터 학습했습니다. 두 개의 transformer 변형, 두 개의 hybrid, 서로 다른 메모리 업데이트 제약을 적용한 두 개의 Gated DeltaNet 변형이었습니다. 모델은 최대 256개 할당으로 이루어진 시퀀스에서 학습되었고, 최대 64,000개 할당 또는 학습 시 본 최대 길이의 250배에 해당하는 시퀀스에서 테스트되었습니다.

시퀀스가 길어지면서 아키텍처 간 차이가 나타나기 시작했습니다. Gated DeltaNet 및 hybrid 모델은 테스트한 가장 긴 시퀀스인 수만 개의 할당에 이르기까지 r-trivial, aperiodic, soluble 문제에서 거의 완벽하게 일반화했습니다. Transformers는 더 쉬운 수준을 학습했지만 시퀀스가 길어지면서 성능이 저하되었고, general finite 문제는 전혀 학습하지 못했습니다. 가장 어려운 수준에서는 더 유연한 메모리 업데이트를 사용하는 Gated DeltaNet 변형이 가장 멀리 일반화했으며, 학습 중 본 가장 긴 시퀀스 길이의 약 10배에 이르기까지 거의 완벽한 성능을 유지했습니다.

Gated DeltaNet 및 hybrid 모델에서는 이러한 차이가 아키텍처가 아니라 학습을 가리킵니다. 직접 학습했을 때 이 모델들은 사전 학습된 모델에서는 처리하지 못했던 상태 추적 수준을 처리하는 방법을 배웠습니다. Transformers는 달랐습니다. 직접 학습했을 때조차 가장 어려운 수준을 학습할 수 없었습니다.

[![Accuracy of transformer, hybrid, and Gated DeltaNet models trained directly on StateBench across all four levels versus input length in assignments.](https://www.datocms-assets.com/64837/1792006493-statebench-draft-blog-google-docs-image-4.png)](https://www.datocms-assets.com/64837/1792006493-statebench-draft-blog-google-docs-image-4.png)

## 미해결 질문과 공개 {#section-4}

아직 해결되지 않은 질문도 있습니다. 새로운 증명에 따르면, 더 제한적인 메모리 업데이트(음이 아닌 고유값)를 사용하는 Gated DeltaNet은 반올림을 포함한 논문의 고정 정밀도 산술 아래에서 모든 aperiodic 과제와 특정 periodic 과제를 해결할 수 있습니다. 예를 들어 반복 입력이 짝수 번 발생했는지 홀수 번 발생했는지를 추적하는 과제가 이에 해당합니다. 계층 구조에서 정확히 어느 위치에 놓이는지는 여전히 미해결 상태이며, 모든 soluble 과제를 해결할 수 있는지, 수치 정밀도 또는 attention 레이어와의 결합이 그 능력에 어떤 영향을 미치는지도 밝혀지지 않았습니다. 또한 StateBench는 고정된 수의 변수를 가정하지만, 실제 과제에서는 시간이 지남에 따라 새로운 엔터티가 도입되는 경우가 많습니다. 예를 들어 긴 대화가 진행되면서 새로운 사람과 사물이 등장할 수 있고, 모델은 이들 역시 추적하기 시작해야 합니다. 현재 프레임워크는 아직 이러한 설정을 다루지 않습니다.

새로운 형태의 순환 및 하이브리드 아키텍처가 등장함에 따라, 각 아키텍처가 어떤 종류의 상태 추적을 처리할 수 있는지 이해하는 일은 미래 시스템의 개발과 학습 방식을 정하는 데 도움이 될 수 있습니다. 우리는 StateBench가 모델이 어디에서 무너지는지 더 세밀하게 파악하고, 아직 부족한 능력을 위한 학습 및 평가 데이터를 구축하는 방법을 연구자들에게 제공하기를 바랍니다. 더 근본적으로는 현재 모델보다 더 강건한 상태 추적 능력을 갖춘 새로운 아키텍처를 개발하는 연구자들에게 StateBench가 북극성 역할을 하기를 바랍니다.
