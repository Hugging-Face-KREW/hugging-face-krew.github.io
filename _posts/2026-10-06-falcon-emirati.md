---
layout: post
title: "Falcon-Emirati: 언어 모델이 방언과 문화, 뉘앙스를 학습할 때"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/MJHJMYW3O4CkLvXvn7DOp.png
image: assets/images/blog/posts/2026-10-06-falcon-emirati/thumbnail.png
authors:
  - user: tiiuae
slug: "falcon-emirati"
source_url: "https://huggingface.co/blog/tiiuae/falcon-emirati"
source_published_date: "2026-10-06"
source_published_at: "2026-10-06T06:44:39+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Falcon-Emirati: When an LLM Learns the Dialect, the Culture, and the Nuance](https://huggingface.co/blog/tiiuae/falcon-emirati)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/tiiuae/falcon-emirati -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Falcon-Emirati: 언어 모델이 방언과 문화, 뉘앙스를 학습할 때

[![image](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/Wq7IUqy86uY6JcIpgCwJx.png)](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/Wq7IUqy86uY6JcIpgCwJx.png)

[Try Falcon Emirati →](https://chat.falconllm.tii.ae/?model=Falcon-Emirati-7B)

아랍어는 하나의 이름 아래 공존하는 언어군에 가깝습니다. Modern Standard Arabic은 뉴스나 교과서에서 읽는 언어지만, 사람들이 실제로 서로 대화할 때 사용하는 방식과는 거리가 있는 경우가 많습니다. UAE에서는 일상적인 대화와 유머, 협상, 이야기가 고유한 어휘와 리듬을 지니고 문화가 긴밀하게 얽힌 걸프 방언인 Emirati Arabic으로 이루어집니다. Emirati 시, 특히 나바티 시와 속담 및 짧은 일화에는 단어 하나하나를 그대로 옮겨 읽는 방식으로는 보존되지 않는 의미가 담겨 있습니다. MSA만 아는 모델은 Emirati 문장의 모든 단어를 번역하고도 그 문장이 실제로 무엇을 의미하는지는 놓칠 수 있습니다.

Falcon-Emirati-7B는 바로 이 간극을 메우기 위해 만들어졌습니다. Falcon-H1-Arabic을 기반으로 한 방언 특화 모델로, 원어민처럼 Emirati Arabic을 이해하고 생성하는 것을 목표로 합니다. 여기에는 어휘와 어조, 그리고 그 표현 뒤에 있는 문화적 맥락이 모두 포함됩니다.

## Falcon-H1-Arabic을 기반으로 구축 {#section-1}

처음부터 시작한 것은 아닙니다. Falcon-Emirati-7B는 올해 초 이미 아랍어에 대한 새로운 벤치마크를 세운 아랍어 모델 제품군인 Falcon-H1-Arabic을 기반으로 구축되었습니다. Falcon-H1-Arabic은 Falcon-H1 하이브리드 아키텍처를 사용합니다. 모든 블록 내부에서 State Space Models (Mamba)와 Transformer attention이 병렬로 실행되고, 각 블록의 projection 전에 출력이 결합됩니다. 이 조합은 긴 시퀀스에서 Mamba의 선형 시간 효율성을 제공하면서도 장거리 의존성에는 attention의 정밀도를 유지합니다. 이는 형태론적으로 풍부한 언어인 아랍어에서 중요합니다. 이 제품군은 세 가지 규모(3B, 7B, 34B parameters)로 구성되며, 최대 128K 및 256K tokens의 context window를 지원합니다. 또한 English와 multilingual data뿐 아니라 MSA 및 다양한 아랍어 방언(Gulf, Levantine, Egyptian, Maghrebi)을 폭넓게 혼합해 이미 학습되어 있었습니다.

이는 강력한 출발점이 되었습니다. 아랍어를 폭넓게 이해하고, 긴 context를 잘 처리하며, 어느 정도의 방언 노출도 이미 갖춘 모델이었기 때문입니다. Falcon-Emirati-7B는 이 기반을 바탕으로 Emirati 방언, 어휘, 문법, 문화 지식에 특별히 초점을 맞춥니다. 아무리 뛰어난 일반 아랍어 모델이라도 스스로 습득하기 어려운 부분들입니다.

Falcon-Emirati-7B는 특히 7B variant를 기반으로 구축했습니다. 이 제품군에서 균형이 가장 좋은 지점입니다. 방언 적응에 필요한 뉘앙스를 유지할 만큼 크면서도 학습과 추론을 모두 실용적인 수준으로 유지할 수 있을 만큼 작습니다. 34B model은 품질을 조금 더 높일 가능성이 있지만, 방언 특화 chat model에 필요한 학습 및 서빙 비용을 고려하면 합리적이지 않습니다. 반면 3B model은 우리가 목표로 한 문화적·언어적 이해의 깊이를 확보할 여지가 충분하지 않습니다. 7B는 품질과 학습 및 추론 비용 사이에서 가장 좋은 균형을 제공했습니다.

## 방언 적응이 어려운 이유 {#section-2}

일반 아랍어 모델을 Emirati 방언 전문가로 바꾸는 일은 처음부터 base model을 구축하는 것보다 작은 작업처럼 들립니다. 하지만 그렇지 않습니다. 실제로 어렵게 만드는 요인은 다음과 같습니다.

- Emirati는 대부분 구어 방언입니다. 온라인에서 MSA는 물론 다른 걸프 및 레반트 방언에 비해서도 글로 나타나는 경우가 훨씬 적기 때문에, 학습에 사용할 raw text 자체가 많지 않습니다.

- 의미가 비문자적인 경우가 많습니다. 관용구와 속담, 시적 표현은 표면적인 어휘가 아니라 공유된 문화적 맥락에 의존합니다.

- 확립된 playbook이 없습니다. 방언 데이터가 얼마나 있어야 충분한지, MSA 및 일반 아랍어와 어떻게 혼합해야 하는지, 방언을 습득하는 데 어떤 학습 단계(continued pre-training, SFT 또는 preference optimization)가 가장 중요한지에 대한 잘 정립된 방법이 없습니다.

마지막 항목은 우리의 작업 방식에 큰 영향을 주었습니다. Falcon-Emirati-7B 구축의 상당 부분은 시행착오로 이루어졌습니다. 서로 다른 데이터 혼합, 학습 단계, supervision 전략을 시험하고, 실제로 유의미한 변화를 만들어내는 요소가 무엇인지 파악하기 위해 사람의 판단과 벤치마크 점수를 모두 활용했습니다.

## 데이터 접근 방식 {#section-3}

Falcon-H1-Arabic의 pretraining 위에 전용 Emirati 데이터 pipeline을 구축하고, 서로 보완적인 세 가지 출처를 활용했습니다.

### 1. 실제 Emirati 방언 웹 데이터

MSA에서 번역하거나 음역한 것이 아니라, 방언으로 자연스럽게 작성된 Emirati 웹사이트와 포럼의 콘텐츠를 수집하고 선별했습니다. 이 데이터에서 실제 기준을 얻었습니다. Emirati 사람들이 온라인에서 실제로 어떻게 쓰고 말하는지, 일상적인 표현과 구어적 표현, 그리고 실제 사용에서 나타나는 Emirati와 MSA 사이의 자연스러운 오가는 방식을 확인할 수 있었습니다.

### 2. Emirati 문화와 정체성에 관한 MSA 데이터

방언 텍스트와 함께 Emirati 문화와 유산, 언어를 구체적으로 다루는 MSA 자료도 포함했습니다. 여기에는 지역 관습과 가치관, 역사, 사회 규범에 관한 글과 참고 자료가 포함되며, Emirati 사람들이 어떻게 인식되고 고정관념화되는지도 다룹니다. 이 자료가 모델에 방언으로 글을 쓰는 방법을 가르치는 것은 아니지만, 유산과 예절, 원어민이라면 자연스럽게 알고 있는 맥락 등 Emirati 관련 주제가 등장했을 때 무엇에 대해 이야기하는지 이해하도록 가르칩니다.

### 3. 용어집과 스타일 규칙으로 안내한 Synthetic Data

실제 방언 텍스트만으로는 chat model이 일상적으로 처리해야 하는 다양한 주제를 충분히 다룰 수 없었습니다. 그래서 부족한 부분을 채우기 위해 대량의 synthetic Emirati-dialect data를 생성했습니다. 생성 모델이 단순히 "Gulf-ish" Arabic으로 즉흥적으로 작성하도록 내버려 둔 것은 아닙니다. Emirati 어휘와 문법을 위해 특별히 구축한 엄격한 규칙과 용어집, 사전을 사용해 제약을 걸었습니다. 이러한 안전장치 덕분에 synthetic output이 실제로 방언을 사용하는 사람에게도 진정한 Emirati처럼 읽히도록 할 수 있었습니다. 그렇지 않았다면 문법적으로는 올바르지만 방언 사용자에게 어색하게 들리는 결과가 나왔을 것입니다.

## 적절한 적응 방법 찾기 {#section-4}

MSA에서 방언으로 적응하는 표준 방법이 없었기 때문에, 학습 전략 자체를 실험적으로 찾아야 할 대상으로 보았습니다. 방언 데이터를 얼마나, 학습의 어느 단계에 주입할지, 모델이 synthetic pattern에 과적합하지 않도록 실제 crawl data와 synthetic data의 균형을 어떻게 맞출지, 표면적으로 유창한 데 그치지 않고 문화적 기반을 유지하기 위해 MSA 문화 맥락이 얼마나 필요한지를 ablation으로 검증했습니다. 각 단계에서 automatic scoring과 원어민 검토를 함께 활용했습니다. 자동 metric만으로는 자연스러움과 어조, 문화적 적합성을 충분히 포착할 수 없기 때문입니다.

## 평가 방법론 {#section-5}

학습 전반에 걸쳐 서로 보완적인 두 가지 접근 방식으로 진행 상황을 추적했습니다.

### 원어민의 수동 평가

Emirati 원어민이 모델의 output을 직접 검토했습니다. 답이 맞는지만 판단한 것이 아니라 자연스러움과 어조, 문화적 적절성처럼 답이 실제로 자연스럽게 들리는지도 평가했습니다. 이런 요소는 벤치마크 점수만으로는 알 수 없지만 원어민의 귀에는 즉시 포착됩니다.

### Alyah에 대한 자동 평가

정량적 추적을 위해 [Alyah](https://huggingface.co/datasets/tiiuae/alyah-emirati-benchmark) (الياه, "북극성")를 사용했습니다. 이는 Arabic LLM의 Emirati 방언 역량을 평가하기 위해 우리와 커뮤니티가 특별히 공개한 벤치마크입니다. Alyah는 원어민 Emirati speakers로부터 수동 수집한 1,173개 sample로 구성된 완전한 native multiple-choice benchmark입니다. 일상적인 인사와 예절부터 비유적 언어, 유산 지식, Emirati poetry까지 다양한 category를 포함하며, 방언과 문화가 가장 중요하고 일반적인 Arabic model이 어려움을 겪는 영역을 다룹니다. Alyah의 구축 방식과 category breakdown에 관한 전체 내용은 [benchmark blog post](https://huggingface.co/blog/tiiuae/emirati-benchmarks)에서 확인할 수 있으며, base model family에 대한 배경 정보는 [Falcon-H1-Arabic announcement](https://huggingface.co/blog/tiiuae/falcon-h1-arabic)에 있습니다.

## 결과 {#section-6}

Falcon-Emirati-7B는 Alyah에서 84.83%를 기록해, 비교한 다른 모든 Arabic 및 multilingual model을 앞섰습니다. 여기에는 Falcon-Emirati-7B보다 몇 배나 큰 여러 모델도 포함됩니다. 아래 차트는 Alyah leaderboard의 주요 instruction-tuned model을 대표하는 집합과 비교했을 때 Falcon-Emirati-7B의 위치를 보여줍니다.

[![Falcon-Emirati-7B vs. leading Arabic and multilingual models on Alyah](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/UL3Uy6LFKWEj5fALZ0Sg4.png)](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/UL3Uy6LFKWEj5fALZ0Sg4.png)

Alyah accuracy (%), instruction-tuned models. Falcon-Emirati-7B는 Falcon-H1-Arabic family 위에 구축되었으므로 이 비교에서는 Falcon-H1-Arabic family models를 제외했습니다.

## 결과가 알려주는 것 {#section-7}

이 비교에서는 몇 가지가 두드러집니다. 규모만으로는 방언 역량을 얻을 수 없습니다. 여기에서 가장 큰 multilingual model 중 일부가 더 작고 방언에 특화된 모델보다 훨씬 낮은 점수를 기록한 것은 Emirati 능력이 규모의 부수적인 결과로 습득되는 것이 아니라 의도적으로 학습되어야 한다는 점을 보여줍니다. 가장 좋은 성능을 보이는 모델은 처음부터 Arabic-native 또는 Arabic-focused인 경우가 많았습니다. 이는 우리의 ablation에서 확인한 결과와도 일치합니다. 일반적인 Arabic 및 방언 coverage는 필요한 출발점이지만, 특히 poetry와 heritage knowledge, 그리고 language-and-dialect category 자체처럼 Alyah에서 가장 어려운 부분의 나머지 격차를 줄이려면 여전히 방언에 특화된 작업이 필요합니다.

이는 Alyah benchmark 공개에서 전반적으로 확인된 결과와도 일치합니다. 강력한 모델조차 진정한 방언적·문화적 맥락이 담긴 콘텐츠로 들어가면 실제 성능 저하를 보입니다. 더 큰 모델을 사용하는 것만으로는 이 격차가 저절로 해소되지 않습니다. 해당 방언을 위해 특별히 구축된 데이터와 평가가 필요합니다.

## 객관식 평가를 넘어: LLM-as-Judge 평가 {#section-8}

Multiple-choice accuracy는 모델이 네 가지 선택지 중 정답을 인식할 수 있는지를 알려줍니다. 하지만 누군가 모델과 대화할 때 모델이 실제로 스스로 Emirati Arabic을 생성할 수 있는지는 알려주지 않습니다. 그래서 Alyah와 함께 두 번째 평가를 진행했습니다. 동일한 1,173개 Alyah 질문에 대해 open-ended generation을 수행하고, Alyah leaderboard에서 가장 강력한 경쟁 모델로 선택한 다섯 모델인 Falcon-Emirati-7B, ALLaM-7B-Instruct-preview, gemma-3-27b-it, Jais-2-8B-Chat, Fanar-2-27B-Instruct를 대상으로 LLM judge (Gemini 3.7 Flash)가 점수를 매겼습니다.

judge는 각 답변을 두 가지 별도 차원으로 평가했습니다. 내용이 올바른지, 그리고 이와 독립적으로 답변이 MSA가 아니라 실제로 Emirati 방언으로 생성되었는지를 평가했습니다. judge의 등급 평가인 partial-credit score와 더 엄격한 pass/fail version을 모두 보고하며, 각 모델이 답변하는 대신 얼마나 자주 abstain했는지도 제시합니다.

[![LLM-judged correctness on open-ended Emirati questions](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/uc4Lj0T23SxjZVaV1w5bU.png)](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/uc4Lj0T23SxjZVaV1w5bU.png)

1,173개 Alyah 질문에 대한 LLM-judged correctness, open-ended generation, Gemini 3.7 as judge.

[![LLM-judged dialect fidelity](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/Q1ArWubn7hwUco5ldt7Xo.png)](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/Q1ArWubn7hwUco5ldt7Xo.png)

동일한 질문에 대한 LLM-judged dialect fidelity: 답변이 실제로 Emirati로 생성되는가, 아니면 모델이 MSA를 기본값으로 사용하는가?

Falcon-Emirati-7B는 correctness에서 앞서지만, 진정한 격차는 두 번째 차트에서 나타납니다. dialect fidelity에서 Falcon-Emirati-7B는 0.52(partial credit)를 기록한 반면 ALLaM은 0.05, gemma-3-27b-it은 0.03, Jais-2-8B-Chat은 0.02, Fanar-2-27B-Instruct는 사실상 0.00을 기록했습니다. 이는 작은 우위가 아니라 낮은 점수대와 비교하면 거의 두 자릿수 배에 가까운 차이입니다. 실제로는 다른 모델들도 정답을 알고 있는 경우가 많지만, Emirati로 직접 질문받았을 때조차 기본적으로 Modern Standard Arabic으로 답합니다. Falcon-Emirati-7B는 다섯 모델 중 질문받은 방언으로 안정적으로 답하는 유일한 모델입니다.

Fanar-2-27B-Instruct는 두 번째 이유에서도 두드러집니다. 다른 어떤 모델보다 훨씬 자주 abstain하며, 26.2%의 경우 답변을 거부했습니다. 비교 대상의 다른 모든 모델은 5% 미만이었습니다. 여기에 다섯 모델 중 가장 낮은 correctness score인 0.27(partial credit)을 함께 고려하면, 단순히 잘못된 register로 답하는 모델이 아니라 Emirati-specific content에 참여하려는 의지도 능력도 모두 낮은 모델임을 시사합니다.

[![Dialect fidelity by category](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/sklbQUT3s8CXmcse-BPln.png)](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/sklbQUT3s8CXmcse-BPln.png)

Alyah category별 dialect fidelity, partial credit. Falcon-Emirati-7B만 일관되게 Emirati로 전환하며, 다른 모델은 거의 모든 category에서 MSA를 유지합니다.

dialect fidelity를 category별로 나누어 보면 패턴이 더욱 명확해집니다. 일상적인 인사부터 poetry까지 Alyah의 모든 category에서 이러한 경향이 나타납니다. 이는 소수의 질문 유형을 위해 학습한 좁은 요령이 아니라는 점을 시사합니다. Emirati가 기대되는 register일 때 모델이 기본적으로 선택하는 register 자체가 전반적으로 바뀐 것입니다. 경쟁 모델이 상대적으로 더 나은 성능을 보이는 유일한 영역인 Greetings & Daily Expressions는 Emirati와 MSA가 가장 많이 겹치는 category이기도 합니다. 따라서 dedicated dialect training이 없어도 일반적인 Arabic model이 우연히 올바르게 들리기 가장 쉬운 영역입니다.

### Category별 Pairwise Comparison

같은 질문을 세 번째 관점에서 살펴보기 위해 head-to-head pairwise judging을 진행했습니다. 모든 Alyah 질문에 대해 judge(Gemini 3.7 Flash)에게 Falcon-Emirati-7B의 답변과 경쟁 모델의 답변을 나란히 보여주고, 어느 것이 어느 모델의 답변인지 알 수 없도록 한 뒤 더 나은 답변을 선택하게 했습니다. 아래 radar chart는 Jais-2-8B-Chat, ALLaM-7B-Instruct-preview, Fanar-2-27B-Instruct라는 세 경쟁 모델과 비교한 category별 win rate를 보여줍니다.

[![pairwise_combined](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/ruWwB7xCdfiETIgD4wmyy.png)](https://cdn-uploads.huggingface.co/production/uploads/659bc8a7b0f43ed69f0b2300/ruWwB7xCdfiETIgD4wmyy.png)

Category별 pairwise win rate. Falcon-Emirati-7B와 Jais-2-8B-Chat, ALLaM-7B-Instruct-preview, Fanar-2-27B-Instruct를 Gemini 3.7이 head-to-head로 평가했습니다.

Falcon-Emirati-7B는 세 경쟁 모델 모두를 상대로 대부분의 category에서 승리했으며, 방언 및 문화적 유창성에 가장 크게 의존하는 category에서는 큰 차이로 앞섰습니다. Jais-2-8B-Chat과 비교했을 때 격차가 가장 큰 영역은 Poetry & Creative Expression(0.69 대 0.31)과 Language & Dialect(0.62 대 0.38)였습니다. ALLaM-7B-Instruct-preview와 비교해도 동일한 두 category에서 가장 큰 차이가 나타났습니다. Poetry & Creative Expression(0.66 대 0.34)과 Language & Dialect(0.58 대 0.42)입니다. Fanar-2-27B-Instruct와의 비교에서는 세 대결 중 가장 큰 격차가 나타났으며, Falcon-Emirati-7B는 모든 category에서 승리했습니다. 특히 Poetry & Creative Expression(0.88 대 0.12)과 Religious & Social Sensitivity(0.80 대 0.20)에서 가장 높은 차이를 보였습니다.

경쟁 모델이 대등한 성능을 보이는 유일한 category는 Greetings & Daily Expressions입니다. Falcon-Emirati-7B는 이 영역에서 Jais-2-8B-Chat에 근소하게 패배했고(0.46 대 0.54), ALLaM-7B-Instruct-preview와는 0.50으로 동률이었습니다. 다만 Fanar-2-27B-Instruct에 대해서는 뚜렷하게 승리했습니다(0.70 대 0.30). 이는 앞서 살펴본 dialect-fidelity breakdown과 대체로 일치합니다. 인사는 Emirati와 MSA가 가장 많이 겹치는 category이므로, dedicated dialect training이 없어도 일반적인 Arabic model이 원어민처럼 들리기 가장 쉬운 영역입니다. 방언의 특징이 더 뚜렷한 모든 영역, 즉 poetry와 figurative language, heritage knowledge에서는 우리가 테스트한 모든 경쟁 모델을 상대로 Falcon-Emirati-7B의 우위가 명확하게 유지됩니다.

## Emirati 문화 이해 {#section-9}

언어는 대화 뒤에 있는 문화를 이해하는 일이기도 합니다. Arabic의 문화적 이해를 평가하는 벤치마크인 [ArabCulture-Dialogue](https://huggingface.co/datasets/Almheiri/ArabCulture-Dialogue)의 UAE portion에서 Falcon-Emirati-7B를 평가했습니다. multiple-choice task에서 모델은 세 가지 선택지 중 문화적으로 가장 적절한 답변을 선택합니다. 자세한 내용은 [research paper](https://aclanthology.org/2026.acl-long.963/)에서 확인할 수 있습니다.

위치 정보의 양을 달리하면서 동일한 283개의 UAE scenario를 Emirati Arabic과 Modern Standard Arabic으로 각각 제시해 네 모델 모두를 테스트했습니다.

[![Cultural understanding accuracy: Falcon-Emirati-7B 85.57%, ALLaM 83.39%, Jais-2 73.79%, Fanar-2 71.50%.](https://cdn-uploads.huggingface.co/production/uploads/65b79f63d919aa79555d17e1/78dgJaRrFACpGTBJkSuTE.png)](https://cdn-uploads.huggingface.co/production/uploads/65b79f63d919aa79555d17e1/78dgJaRrFACpGTBJkSuTE.png)

두 언어 변종과 모든 위치 설정에 걸쳐 평균을 낸 UAE multiple-choice task의 overall accuracy입니다. 이는 우리의 evaluation results입니다.

Falcon-Emirati-7B는 85.57%를 기록해 테스트한 네 모델 중 가장 높은 점수를 받았습니다. ALLaM-7B(83.39%), Jais-2-8B(73.79%), Fanar-2-27B(71.50%)를 앞선 결과입니다. 이는 Emirati 대화에서 문화적으로 적절한 응답을 인식하는 Falcon-Emirati-7B의 능력을 보여줍니다.

## 실제 사용 예시 {#section-10}

수치만으로는 이야기의 일부만 알 수 있으므로, 아래에는 실시간 interactive comparison을 준비했습니다. 실제 Emirati prompt 다섯 개를 Falcon-Emirati-7B와 Fanar-2-27B-Instruct, ALLaM-7B-Instruct-preview에 나란히 입력했습니다. Swipe하거나 화살표를 사용해 prompt 사이를 이동하고, 응답을 확장해 전체 내용을 읽을 수 있습니다.

Interactive comparison: 다섯 Emirati prompt에 대한 Falcon-Emirati-7B와 Fanar-2-27B-Instruct, ALLaM-7B-Instruct-preview의 비교입니다. Eid 인사, 지역 역사, poetry, heritage knowledge를 다룹니다. Output은 생성된 그대로 표시되며 오류가 포함될 수 있습니다. Highlighting은 우리 모델을 식별하기 위한 것이며 사실성 평가를 의미하지 않습니다.

## Falcon-Emirati-7B 사용해 보기 {#section-11}

Falcon-Emirati-7B는 우리의 chat platform에서 사용할 수 있습니다. 🚀 지금 사용해 보세요: [https://chat.falconllm.tii.ae/?model=Falcon-Emirati-7B](https://chat.falconllm.tii.ae/?model=Falcon-Emirati-7B)

## Responsible AI와 한계 {#section-12}

모든 언어 모델과 마찬가지로 Falcon-Emirati-7B는 학습 데이터의 편향을 반영할 수 있으며, 특히 희귀한 표현이나 매우 지역적인 references, 또는 데이터가 충분하지 않았던 edge case에서는 때때로 잘못된 답을 낼 수 있습니다. 방언과 문화적 뉘앙스는 경우에 따라 주관적이며, 원어민조차 항상 "올바른" 답에 동의하는 것은 아닙니다. 민감하거나 공식적이거나 중대한 용도의 작업에 모델을 사용하기 전에 구체적인 사용 사례에 맞춰 평가할 것을 권장합니다. 또한 앞으로 모델과 Alyah를 모두 개선할 수 있도록 Emirati community의 피드백을 진심으로 환영합니다.

## 감사의 말 {#section-13}

compute infrastructure를 지속적으로 지원해 준 Mikhail Lubinets와 Matthieu Berjon, 그리고 Falcon Chat app을 통해 모델을 이용할 수 있도록 도움을 준 Jatin Mittal에게 진심으로 감사드립니다.

## 인용 {#section-14}

```
@misc{falcon_emirati_7b_2026,
  title = {Falcon-Emirati-7B: A Dialect-Specialized Arabic LLM for the Emirati Dialect},
  author = {Shaikha Alsuwaidi, Omar Alkaabi, Maitha Alhammadi, Hamza Alobeidli, Ahmed Alzubaidi, Mohammed Alyafeai, Leen AlQadi, Basma Boussaha, Hakim Hacid}, 
  organization = {Technology Innovation Institute},
  year = {2026}
}
```
