---
layout: post
title: "Granite Switch 4.1.1 — 소프트웨어를 업데이트하듯 모델을 업데이트하세요"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/662836b742709fe826510597/az6afuuoAkyiMImufSePe.png
image: assets/images/blog/posts/2026-10-29-granite-switch-4-1-1/thumbnail.png
authors:
  - user: ibm-granite
slug: "granite-switch-4-1-1"
source_url: "https://huggingface.co/blog/ibm-granite/granite-switch-4-1-1"
source_published_date: "2026-10-29"
source_published_at: "2026-10-29T13:37:07+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Granite Switch 4.1.1 — Update models like you update software](https://huggingface.co/blog/ibm-granite/granite-switch-4-1-1)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-granite/granite-switch-4-1-1 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Granite Switch 4.1.1 — 소프트웨어를 업데이트하듯 모델을 업데이트하세요

[![granite_switch_adapter_swap](https://cdn-uploads.huggingface.co/production/uploads/662836b742709fe826510597/F5YZJX3QZ8HbecaJnHe4Q.png)](https://cdn-uploads.huggingface.co/production/uploads/662836b742709fe826510597/F5YZJX3QZ8HbecaJnHe4Q.png)

일주일 조금 전, [Granite Switch 4.1](https://huggingface.co/blog/ibm-granite/granite-switch-4-1)을 출시했습니다. 이를 통해 모듈형 Switch 아키텍처로 특화된 캐시 효율적 어댑터를 구성해 단일 endpoint에 새로운 전문성을 추가할 수 있음을 보여드렸습니다.

[Granite Switch 4.1.1](https://huggingface.co/ibm-granite/granite-switch-4.1-30b)은 patch release입니다. 버전 번호는 소프트웨어에서 의미하는 그대로입니다. 4.1과 완전히 하위 호환되며, 마이그레이션이 필요 없고, 내부적으로 점진적인 개선이 이루어졌습니다.

- [Updated Granite Speech ASR model](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) — 10개 언어를 지원하는 다국어 음성-텍스트 변환으로, 네이티브 구두점과 대문자 표기를 제공합니다. 따라서 별도의 서식 처리 없이도 전사 결과를 downstream 모델에 바로 사용할 수 있습니다.

- [granitelib-core](https://huggingface.co/ibm-granite/granitelib-core-r1.0)의 업데이트된 Decision 어댑터 — Jev Decision Index 테스트셋에서 전반적으로 향상된 성능을 제공합니다.

이러한 새로운 기능도 기대되지만, 특히 주목할 점은 우리 팀이 이 업데이트를 출시할 수 있었던 속도와 효율성입니다. 새로운 ASR 모델과 Decision 어댑터를 학습한 것 외에는 아무것도 다시 계산하지 않았습니다. base model은 변경되지 않았습니다. 원하는 최신 어댑터를 조합해 배포하면 됩니다.

이 패턴은 IBM이 제공하는 모델의 patch release를 넘어 훨씬 더 다양한 방식으로 활용될 수 있습니다. 현재 개발자와 타사 제공업체가 직접 어댑터를 구축할 수 있도록 튜토리얼과 Adapter SDK를 준비하고 있습니다. 버전을 반복해서 개선하고, 기능을 테스트하며, 소프트웨어를 출시하듯 업데이트를 제공할 수 있습니다. 곧 더 자세히 소개하겠습니다.

그때까지 [latest models](https://huggingface.co/collections/ibm-granite/granite-switch)을 테스트하고, [Speech-to-Decisions](https://huggingface.co/spaces/ibm-granite/speech-to-decisions) 데모를 지원하는 새로운 다국어 및 구두점 기능을 사용해 보세요.
