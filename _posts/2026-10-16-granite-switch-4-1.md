---
layout: post
title: "Granite Switch 4.1: 하나의 엔드포인트에서 제공하는 대화, 의사 결정, 음성"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/662836b742709fe826510597/gltB0o--PV_Jh0D3aDm8-.png
image: assets/images/blog/posts/2026-10-15-granite-switch-4-1/thumbnail.png
authors:
  - user: ibm-granite
slug: "granite-switch-4-1"
source_url: "https://huggingface.co/blog/ibm-granite/granite-switch-4-1"
source_published_date: "2026-10-15"
source_published_at: "2026-10-15T20:34:01+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Granite Switch 4.1: Chat, Decisions, and Speech from One Endpoint](https://huggingface.co/blog/ibm-granite/granite-switch-4-1)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-granite/granite-switch-4-1 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Granite Switch 4.1: 하나의 엔드포인트에서 제공하는 대화, 의사 결정, 음성

지난봄에 소개된 [Granite Switch](https://github.com/generative-computing/granite-switch) 프로젝트는 모델 개발에 모듈식 접근 방식을 제공합니다. 이제 Decision adapter와 OpenASR 리더보드에서 가장 빠른 ASR 모델을 탑재한 Granite Switch 4.1 모델은 하나의 호스팅 엔드포인트에서 자연어 응답, 일반 및 전문 의사 결정, 음성-텍스트 변환 작업 사이를 효율적으로 전환할 수 있습니다.

[![granite_switch_adapter_swap](https://cdn-uploads.huggingface.co/production/uploads/662836b742709fe826510597/LHBbw-MVVQgnYcpDK_DWC.png)](https://cdn-uploads.huggingface.co/production/uploads/662836b742709fe826510597/LHBbw-MVVQgnYcpDK_DWC.png)

구체적으로 [Granite-Switch-4.1-8B](https://huggingface.co/ibm-granite/granite-switch-4.1-8b) 및 [-30B](https://huggingface.co/ibm-granite/granite-switch-4.1-30b) 모델은 다음을 포함하도록 구성되어 있습니다.

- 범용 의사 결정을 위한 새로운 [Decision aLoRA](https://huggingface.co/ibm-granite/granitelib-core-r1.0/blob/main/decision/README.md)(예: v1/systemone),

- 안전 관련 의사 결정 작업에서 평균적으로 Jev 1.13, Microsoft-Decision-1, GPT-Luna-Decisions, Clef-Flash보다 우수한 특화 [Guardian aLoRA](https://huggingface.co/ibm-granite/granitelib-guardian-r1.0/blob/main/guardian-core/README.md), 그리고

- 전사 작업에 사용할 수 있도록 내장된 [Granite-Speech-5.0-Turbo-CTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)—[OpenASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)에서 가장 빠른 ASR 모델.

이 세 가지 기술은(기반이 되는 base model과 함께) 하나의 모델로 구성되며, GPU에서 base model만 단독으로 사용할 때와 거의 동일한 양의 VRAM을 차지합니다. 즉, 더 적은 수의 GPU로 더 많은 기능을 사용할 수 있습니다.

또한 Switch adapter는 캐시 효율적인 [Activated LoRAs](https://research.ibm.com/blog/inference-friendly-aloras-lora)로 구축되었기 때문에 base model 생성에서 만들어진 KV cache를 이후의 모든 adapter 기반 의사 결정에 재사용할 수 있습니다. low rank adapter 추론 속도를 높이기 위한 우리 팀의 [recent vLLM contributions](https://github.com/generative-computing/granite-switch/blob/main/docs/SWITCH_KERNELS.md) 및 함께 배치된 ASR encoder와 결합하면 Granite Switch 4.1 모델은 채팅 요청, 의사 결정 요청, 음성 요청 사이를 매우 빠른 속도로 전환할 수 있습니다.

[Speech-to-Decisions HF space](https://huggingface.co/spaces/ibm-granite/speech-to-decisions)에서 granite-switch-4.1-8b가 오디오를 전사하고, 요약을 생성하며, 유해 콘텐츠를 선별하고, 사용자가 정의한 15개의 다른 플래그를 테스트하는 모습을 확인해 보세요. 이 모든 작업은 하나의 엔드포인트에서 실시간으로 실행됩니다.

## Granite Libraries: OG 의사 결정 모델 {#section-1}

의사 결정 모델은 새로운 개념이 아닙니다. 이전에 출시한 [Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries)에는 특화된 “decision model” adapter와 기타 특화 작업 adapter로 구성된 repository가 포함되어 있으며, 이를 맞춤형 Granite Switch 모델로 조합할 수 있습니다. Jev 및 다른 폐쇄형 의사 결정 모델이 범용성에 초점을 맞추는 반면, 우리는 다수의 특화 granitelib adapter가 특화 분류 작업에서 범용 모델보다 우수한 성능을 낼 수 있다는 사실을 확인했습니다.

[Compare granitelib adapters](https://huggingface.co/spaces/ibm-granite/granitelib-evals) 특화 분류 작업에서 평가했을 때 범용 의사 결정 모델과 비교해

adapter는 base model의 KV cache를 공유하므로, 범용 의사 결정 adapter와 함께 전문 adapter를 추가하는 데 드는 비용은 거의 없습니다. 범용 adapter를 통해 폭넓은 범위를 다루고, 관심 있는 작업에서는 전문 adapter를 통해 더 높은 정확도를 얻을 수 있습니다.

Granite-Switch-4.1-30B(동결된 base Granite + 특화 aLoRA 4개)와 Jev-27B(동결된 base Qwen model + 범용 decision LoRA 1개)의 “Race”에서 Granite-Switch-4.1-30B는 경쟁 모델보다 실제 환경의 5단계 RAG workflow를 훨씬 빠르게 완료할 수 있었습니다.

## 자신만의 Granite Switch 모델 조합 및 확장 {#section-2}

Granite-Switch-4.1-8B와 -30B는 “batteries included” 방식으로 출시됩니다. 이는 유용한 adapter를 미리 정해진 세트로 컴파일한 Granite Switch 모델입니다. 하지만 반드시 저희 버전을 사용할 필요는 없습니다. 누구나 [compose their own Granite Switch model](https://generative-computing.github.io/granite-switch/adapter_catalog.html)하여 Granite Libraries의 adapter를 필요에 맞게 조합할 수 있습니다.

앞으로 커뮤니티가 맞춤형 및 타사 adapter를 사용해 Granite Switch 모델을 조합하고 확장할 수 있는 엄청난 기회가 있다고 생각합니다. 저희는 [Mellea](https://docs.mellea.ai/advanced/lora-and-alora-adapters)에서 adapter를 학습하기 위한 Adapter SDK를 다른 튜토리얼 및 Granite Switch 지원과 함께 출시할 계획입니다. 이러한 adapter는 의사 결정 기반 workflow를 보강하는 특화 분류 adapter일 수도 있고, 특화 언어 작업을 위한 adapter일 수도 있습니다. 맞춤형 adapter를 사용해 3B parameter 버전의 Granite이 클래식 게임 Doom을 플레이하도록 가르칠 수도 있습니다.

Jev와 마찬가지로 저희의 맞춤형 3B parameter Granite Switch 모델은 Doom 게임을 탐색하기 위해 찰나의 순간에 의사 결정을 내렸지만, Jev가 보고한 초당 10회의 의사 결정과 비교해 초당 60~100회의 의사 결정을 내릴 수 있었습니다.[1](#footnote-1) Jev와 달리 이 Granite Switch 모델은 플레이하는 동안 사용자의 질문에 답하고 상황을 설명할 수도 있습니다.

## 어디서나 사용 가능 {#section-3}

8b와 30b 크기로 제공되는 Granite Switch 4.1은 workload가 실행되는 어디에든 배치할 수 있습니다. 이 모델은 플러그인과 함께 vllm에서 실행되며(아래의 Granite Switch repo 참고), `llama.cpp` 및 downstream 플랫폼에서 native inference를 지원합니다. 따라서 어디에서든 사용 중인 `/v1/systemone` API 호출을 모델로 원활하게 대체할 수 있고, 기존 chat 호출도 모두 공유 prefix cache를 사용해 처리할 수 있습니다.

## 시작을 위한 리소스 {#section-4}

- [Granite Switch 4.1 Model Cards](https://huggingface.co/collections/ibm-granite/granite-switch)

- [Speech-to-Decisions Demo](https://huggingface.co/spaces/ibm-granite/speech-to-decisions)

- [Granite Switch GitHub Repo](https://github.com/generative-computing/granite-switch)

- [Original Granite Switch and Granite Libraries Release Blog](https://research.ibm.com/blog/granite-libraries-project-switch)

- [Colab](https://colab.research.google.com/github/generative-computing/granite-switch/blob/main/tutorials/notebooks/alora_vs_lora_race.ipynb)에서 자신만의 aLoRA v LoRA race를 만들어 보세요.

각주

1 [https://typesafe.ai/blog/introducing-system-one-models-and-jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
