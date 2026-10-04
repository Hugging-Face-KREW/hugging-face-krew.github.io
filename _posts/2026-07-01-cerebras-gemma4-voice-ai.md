---
layout: post
title: "Hugging Face와 Cerebras가 Gemma 4를 실시간 음성 AI로 선보입니다"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-07-01-cerebras-gemma4-voice-ai/thumbnail.png
authors:
  - user: A-Mahla
  - user: andito
  - user: lvwerra
  - user: vyassaurabh
slug: "cerebras-gemma4-voice-ai"
source_url: "https://huggingface.co/blog/cerebras-gemma4-voice-ai"
source_published_date: "2026-07-01"
source_published_at: "2026-07-01T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Hugging Face and Cerebras bring Gemma 4 to real-time voice AI](https://huggingface.co/blog/cerebras-gemma4-voice-ai)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/cerebras-gemma4-voice-ai -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Hugging Face와 Cerebras가 Gemma 4를 실시간 음성 AI로 선보입니다

<video controls width="100%">
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/cerebras-gemma4-voice-ai/s2s-space-demo.mp4" type="video/mp4">
</video>

음성 AI의 경우 지연은 중요한 매개변수입니다. 개발자들은 모델 품질에서 상당한 진전을 이뤘지만, 사용자 경험은 여전히 응답 시간에 의해 좌우되는 경우가 많습니다. Hugging Face와 Cerebras가 그 경험을 바꾸고 있습니다. 오늘은 개방적이고 모듈형인 음성 AI 아키텍처와 업계 최고 수준의 추론 속도가 결합될 때 가능해지는 것을 시연합니다.

그 결과는 훨씬 더 자연스럽게 느껴지는 음성-음성 대화 경험입니다. AI의 응답을 기다리는 대신, 대화는 인간 간의 상호작용에서 사용자가 기대하는 반응 속도로 흐릅니다.

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/5.38.0/gradio.js"></script>

<gradio-app theme_mode="light" space="smolagents/hf-realtime-voice"></gradio-app>

## 아키텍처: 개방형 계층형 음성-음성 스택 {#section-1}

데모는 실시간 음성-음성 파이프라인으로 구축되었습니다. 시스템의 각 부분은 모듈식이며 개방적이고 대체 가능하여, 개발자가 다양한 어시스턴트, 로봇, 제품 또는 연구 프로젝트에 맞게 스택을 쉽게 조정할 수 있습니다.

이는 완전히 개방된 음성-음성 루프를 만듭니다:

```text
Speech input
  -> speech recognition with Nvidia's Parakeet
  -> Gemma 4 VLM inference on Cerebras
  -> text-to-speech with Alibaba's Qwen3TTS
  -> spoken response
```


이 아키텍처는 오픈 소스 AI 생태계의 강점을 한데 모읍니다: 빠른 추론을 위한 Cerebras, 언어 모델을 위한 Google DeepMind의 Gemma 4 31B, 텍스트 음성 변환을 위한 Qwen. 개발자들은 모든 층을 검사, 수정, 확장할 수 있습니다

## Cerebras와 Hugging Face 파트너십 {#section-2}

오늘날 일부 프로덕션 시스템은 합리적인 중앙값 지연을 보이지만, P95에서 여전히 다수의 초에 걸친 지연을 경험합니다. 도구 호출이나 다중 모달 단계가 여러 차례의 턴을 필요로 할 때 이러한 지연은 더욱 두드러집니다.

Cerebras는 스택에서 가장 중요한 병목 중 하나인 언어 모델의 응답 시간을 해결하는 데 도움을 줍니다. 추론을 현저히 더 빠르고 안정적으로 만들어, 나머지 Hugging Face 파이프라인이 빛나도록 합니다.

그 안정성은 특히 롱테일에서 중요합니다. 많은 시스템이 합리적인 중앙값 응답 시간을 제공할 수 있지만, 간헐적으로 발생하는 느린 응답은 대화를 여전히 불안정하게 만듭니다.

## 실제 세계의 상호작용을 위한 설계 {#section-3}

이 동일한 Hugging Face의 음성-음성 파이프라인은 이미 Reachy Mini 로봇에 동력을 제공하고 있으며, 야생에서 9,000대가 넘는 로봇이 작동 중입니다. 로봇, 음성 비서, 구현된 AI의 경우 반응성은 미적 개선이 아닙니다. 그것이 상호작용을 살아있게 만드는 핵심 요소입니다.

따라서 Cerebras를 사용하는 동기는 단순한 비용 절감이 아닙니다. 낮은 지연 시간, 예측 가능한 성능, 그리고 대규모에서도 자연스럽게 느껴지는 실시간 경험을 만들어낼 수 있는 능력입니다.

이 협력은 AI의 미래가 개방적이면서도 성능이 뛰어날 것이라는 공동의 믿음을 반영합니다. 오픈 소스 모델, 개방형 인프라, 그리고 획기적인 추론 속도가 함께 차세대 대화형 AI의 토대를 형성합니다.

개발자들이 데모를 탐색하고, 코드를 실험하며, 실시간 음성 AI의 차후 발전 방향을 함께 만들어 가도록 초대합니다.

데모: [Hugging Face Space](https://huggingface.co/spaces/smolagents/hf-realtime-voice)

저장소: [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)
