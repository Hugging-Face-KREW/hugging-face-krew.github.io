---
layout: post
title: "하나의 모델 패밀리, 두 개의 금메달급 결과: IOI와 IMO를 위한 Nemotron 미세 조정"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/67b8b0096c3182e96bf6cea1/zB6BNaB-3tiCCsnOo7gik.png
image: assets/images/blog/posts/2026-10-07-nemotron-ioi-and-imo-2026/thumbnail.png
authors:
  - user: nvidia
slug: "nemotron-ioi-and-imo-2026"
source_url: "https://huggingface.co/blog/nvidia/nemotron-ioi-and-imo-2026"
source_published_date: "2026-10-07"
source_published_at: "2026-10-07T12:45:31+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [One Model Family, Two Gold-Level Results: Fine-Tuning Nemotron for IOI and IMO](https://huggingface.co/blog/nvidia/nemotron-ioi-and-imo-2026)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/nemotron-ioi-and-imo-2026 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 하나의 모델 패밀리, 두 개의 금메달급 결과: IOI와 IMO를 위한 Nemotron 미세 조정

국제정보올림피아드(IOI)와 국제수학올림피아드(IMO)는 서로 다른 능력을 평가합니다. IOI에서는 엄격한 시간 및 제출 제한 아래에서 비공개 테스트를 통과하는 알고리즘과 코드를 요구합니다. IMO에서는 엄밀한 자연어 증명을 요구합니다. 어느 대회에서든 성공하기는 어렵습니다. 두 대회 모두에서 성공했다는 것은 더 폭넓은 역량을 시사합니다.

최근 결과는 Nemotron이 세계적 수준의 전문 모델을 구축하기 위한 강력하고 적응력 높은 파운데이션임을 보여줍니다. Nemotron 3에서 시작해, 각 팀은 지도 미세 조정(SFT), 강화 학습(RL), 피드백 기반 추론을 사용하여 [IMO 2026](https://arxiv.org/abs/2609.10712)와 [IOI 2026](https://arxiv.org/abs/2609.02849) 모두에서 금메달 수준에 도달한 시스템을 만들었습니다.

[![fig_ioi_imo_gold_results_headline](https://cdn-uploads.huggingface.co/production/uploads/67b8b0096c3182e96bf6cea1/LqM9etFUGTjJT116M7Hq8.png)](https://cdn-uploads.huggingface.co/production/uploads/67b8b0096c3182e96bf6cea1/LqM9etFUGTjJT116M7Hq8.png)

| 대회 | Nemotron 전문화 | 결과 |
| --- | --- | --- |
| IOI 2026 | SFT와 GenCorrect를 적용한 Nemotron-3-Ultra-CC | 535.4/600점으로, 361.12의 금메달 기준과 인간 최고 점수인 498.27을 상회 |
| IMO 2026 | 생성-검증-개선 시스템에 사용된 Nemotron 3 Ultra 일반, SFT 및 RL 체크포인트 | 30/42점으로, 공식 금메달 기준인 29점을 상회 |

IOI 결과는 인간 참가자와 동일한 시간, 인터넷 액세스 및 제출 제약 아래에서 실제 대회 방식으로 사전에 수행한 결과입니다. 이는 비공식 비감독 벤치마크였으며 공식 IOI 순위에는 포함되지 않았습니다. IMO 시스템이 제출한 증명은 공식 IMO 채점관이 채점했습니다.

## 재사용 가능한 전문화 레시피 {#section-1}

"미세 조정하기 쉽다"는 것은 체크포인트를 학습 가능하게 만드는 것 이상을 의미해야 합니다. 유능한 파운데이션 모델을 명확하고 재사용 가능한 레시피로 까다로운 도메인에 맞게 조정할 수 있어야 한다는 뜻입니다.

두 프로젝트에서 이 레시피는 네 부분으로 구성되었습니다.

- 강력한 Nemotron 기반 모델로 시작합니다.

- 도메인별 문제와 고품질 추론 트레이스를 선별합니다.

- SFT와 필요에 따른 RL 같은 표준 사후 학습 방법을 적용합니다.

- 후보 답변을 생성하고, 평가하고, 개선하는 추론 루프와 전문 모델을 결합합니다.

학습과 추론 실행에는 상당한 자원이 필요했지만, 기본 접근 방식은 익숙하고 재현 가능합니다. 각 과제마다 새로운 파운데이션 모델을 구축할 필요는 없었습니다. 과제에 맞게 Nemotron을 전문화했습니다.

## 일반적인 코딩 능력에서 IOI 금메달까지 {#section-2}

경쟁 프로그래밍을 위해 22,000개의 문제를 선별하고 합성 추론 트레이스를 생성하여 두 개의 전문 모델을 학습했습니다. 총 300억 개의 파라미터와 30억 개의 활성 파라미터를 갖는 Nemotron-3-Nano-CC에는 SFT와 RL을 모두 적용했습니다. 총 5,500억 개의 파라미터와 550억 개의 활성 파라미터를 갖는 Nemotron-3-Ultra-CC에는 SFT를 적용했습니다.

IOI 2025에서의 진행 과정은 전문화의 가치를 명확하게 보여줍니다. Nano는 사후 학습 전 130점에서 SFT 후 280점, RL 후 291점으로 향상되었습니다. 반복적인 생성-평가-개선 전략인 GenCorrect를 적용하자 468점에 도달하여 438.3점의 금메달 기준을 넘었습니다. Ultra-CC는 동일한 테스트 시점 전략으로 502점을 기록했습니다.

[![fig_main_capability_progression_previous_style](https://cdn-uploads.huggingface.co/production/uploads/67b8b0096c3182e96bf6cea1/GptKxXbYlSsm2ksIJqZzE.png)](https://cdn-uploads.huggingface.co/production/uploads/67b8b0096c3182e96bf6cea1/GptKxXbYlSsm2ksIJqZzE.png)

이 실험은 또한 모든 규모에서 적응 방식이 동일할 필요는 없다는 점을 보여주었습니다. Nano에서는 SFT가 대부분의 성능 향상을 이끌었고, RL은 더 작지만 일관된 추가 향상을 가져왔습니다. 더 강력한 Ultra 모델에서는 한 번의 SFT epoch만으로도 IOI, ICPC, LiveCodeBench Pro 전반에서 완전한 사후 학습을 거친 Nano 모델을 능가하기에 충분했습니다. 이 결과를 바탕으로 IOI 2026에 사용된 대회 특화 Ultra-CC 시스템을 구성했으며, 이 시스템은 600점 만점에 535.4점을 기록했습니다.

## Nemotron에게 증명하고, 검증하고, 수정하는 방법 가르치기 {#section-3}

IMO 프로젝트에서는 같은 아이디어를 올림피아드 수학에 적용했습니다. Nemotron 3 Ultra에서 시작해 한 전문 모델은 SFT로, 다른 전문 모델은 RL로 학습했습니다.

SFT 코퍼스에는 15,818개의 고유한 증명 문제에 걸쳐 품질 필터링을 거친 414,890개의 예제가 포함되었습니다. 이 데이터는 최종 답변만 가르친 것이 아닙니다. 증명 생성, 개선, 검증, 메타 검증을 다루었기 때문에 모델은 논증을 구성하고, 빈틈을 식별하고, 비판에 대응하고, 증명이 완전한지 판단하는 방법을 학습했습니다. RL 모델은 모델의 능력 프런티어에 가까운 9,597개의 증명 문제를 선별하여 학습했습니다.

두 사후 학습 체크포인트는 모두 개발 실험에서 일반 공개 모델을 능가했습니다. SFT 체크포인트는 첫 번째 탐색 라운드에서 가장 강력했고, RL 체크포인트는 전체적으로 단일 체크포인트 기준 최고의 결과를 달성했습니다. 두 체크포인트의 강점은 상호 보완적이었기 때문에, 최종 시스템에서는 일반 모델과 함께 두 전문 모델을 모두 사용했습니다.

각 IMO 문제에 대해 모델은 후보 증명을 생성하고, 점수를 매기고, 비평을 작성하고, 가장 유망한 시도를 개선했습니다. 별도의 고연산 단계에서 최종 제출물을 선택했습니다. 전체 시스템은 형식 증명기, 외부 도구 또는 인터넷 액세스 없이 자연어로 작동했습니다. 총 42점 중 30점을 획득했으며, 6개 문제 중 4개에서 만점을 받아 공식 금메달 기준을 넘었습니다.

[![image](https://cdn-uploads.huggingface.co/production/uploads/67b8b0096c3182e96bf6cea1/Wo0kpoA9qTLPaxWv48DnL.png)](https://cdn-uploads.huggingface.co/production/uploads/67b8b0096c3182e96bf6cea1/Wo0kpoA9qTLPaxWv48DnL.png)

## 미세 조정과 테스트 시점 연산의 결합 {#section-4}

앞서 소개한 [IOI 2025 Hugging Face post](https://huggingface.co/blog/nvidia/ioi-gold-medal-with-open-weight)에서는 테스트 시점 연산이 오픈 웨이트 모델을 금메달 수준의 성능으로 끌어올리는 방법을 보여주었습니다. 이번 결과는 중요한 요소를 하나 더 제시합니다. 더 나은 전문화는 추론 시스템에 더 나은 후보, 더 나은 비평가, 더 나은 개선 결과를 제공합니다.

IOI에서는 GenCorrect가 여러 피드백 라운드에 걸쳐 미세 조정의 향상을 더 크게 확대했습니다. IMO에서는 하나의 체크포인트에서 더 많은 샘플을 단순히 추출하는 것보다 상호 보완적인 SFT 및 RL 체크포인트를 사용하는 편이 더 효과적이었습니다. 두 경우 모두 유능한 전문 모델과 탐색하고, 검증하고, 개선할 수 있는 시스템을 결합했을 때 최상의 결과가 나왔습니다.

이 차이는 중요합니다. 메달은 미세 조정만으로 만들어진 것이 아니며, 무차별 샘플링만으로 만들어진 것도 아닙니다. 모델, 데이터, 추론 루프를 함께 설계한 결과였습니다.

## Hugging Face의 오픈 모델, 데이터 및 레시피 {#section-5}

우리는 이러한 결과가 대회 너머에서도 유용하게 활용되기를 바랍니다. [Nemotron Labs IMO 2026 collection](https://huggingface.co/collections/nvidia/nemotron-labs-imo-2026)에는 SFT 및 RL 체크포인트, 두 개의 학습 데이터셋, 그리고 올림피아드 수준의 문제 200개로 구성된 새로운 벤치마크인 Nemotron-IMO-Bench가 함께 제공됩니다. [IMO paper](https://arxiv.org/abs/2609.10712)에서는 학습 접근 방식과 생성-검증-개선 시스템을 설명하며, [NeMo-Skills repository](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes/nemotron-imo-tts)에는 IMO 추론 파이프라인, 프롬프트, 제출된 증명, 재현 가능한 빠른 시작 가이드가 포함되어 있습니다. 경쟁 프로그래밍을 위한 [Nemotron-3-Ultra-CC model](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Competitive-Coding-550B-A55B-NVFP4)은 Hugging Face에서 사용할 수 있으며, [IOI paper](https://arxiv.org/abs/2609.02849)에서는 학습 레시피와 GenCorrect 방법론을 제공합니다. IOI 평가 및 추론 파이프라인도 [NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills)에서 사용할 수 있습니다.

IMO와 IOI는 단순한 아이디어에 대해 이례적으로 까다로운 증거를 함께 제공합니다. Nemotron은 세계적 수준의 도메인 전문 모델로 미세 조정할 수 있으며, 이를 투명한 추론 워크플로와 결합하면 인간 경쟁의 최전선에 있는 문제를 해결할 수 있습니다.

Hugging Face 커뮤니티가 다음에는 무엇을 만들어낼지 기대됩니다.
