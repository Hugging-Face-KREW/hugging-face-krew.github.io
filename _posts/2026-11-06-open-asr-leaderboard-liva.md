---
layout: post
title: "이집트 및 나이지리아 영어, Open ASR Leaderboard에 합류"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/open-asr-leaderboard-liva/thumbnail.png
image: assets/images/blog/posts/2026-11-06-open-asr-leaderboard-liva/thumbnail.png
authors:
  - user: bezzam
  - user: ashleymo2025
slug: "open-asr-leaderboard-liva"
source_url: "https://huggingface.co/blog/open-asr-leaderboard-liva"
source_published_date: "2026-11-06"
source_published_at: "2026-11-06T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Egyptian and Nigerian English join the Open ASR Leaderboard](https://huggingface.co/blog/open-asr-leaderboard-liva)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/open-asr-leaderboard-liva -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 이집트 및 나이지리아 영어, Open ASR Leaderboard에 합류

몇 달 전, [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)에 비공개 데이터를 추가했습니다. Appen과 DataOceanAI에서 제공한 11개의 split으로, 5개 accent의 대본 낭독 및 대화 음성을 다룹니다([블로그](https://huggingface.co/blog/open-asr-leaderboard-private-data)). 커뮤니티의 반응은 매우 좋았으며, 많은 분들이 **신뢰할 수 있는 평가를 위해 held-out 세트가 중요하다는 점을 강조했습니다**. 그 후 Voice Arena와 함께 인도 accent 영어를 위한 비공개 split을 하나 더 추가했습니다([블로그](https://huggingface.co/blog/open-asr-leaderboard-global-south)).

새로운 협업을 통해 **이집트** 🇪🇬 및 **나이지리아** 🇳🇬라는 두 가지 새로운 accent의 비공개 데이터를 더 추가할 수 있게 되어, [Liva AI](https://huggingface.co/liva-ai)와의 협업을 발표하게 되어 기쁩니다.

| 데이터셋 | Accent | 기간 [h] | 스타일 | 전사 |
| --- | --- | --- |  --- | --- |
| Liva AI Conversational EG | Egyptian | 1.34 | 대화형, 즉흥적 | 구두점과 대소문자가 적용되며, 비유창성이 포함됨. |
| Liva AI Conversational NG | Nigerian | 1.55 | 대화형, 즉흥적 | 구두점과 대소문자가 적용되며, 비유창성이 포함됨. |

이는 ASR 평가에서 대표성을 높이고(더 많은 accent와 언어 지원), 신뢰도를 높이기 위한 지속적인 노력의 일환입니다(벤치맥싱을 최소화하기 위한 새로운 비공개 데이터와 새로운 데이터 제공자). 전 세계를 위한 ASR 성능을 커뮤니티가 계속해서 향상해 나가는 모습을 보게 되어 기쁩니다 🤗

## 비공개 탭에서 더욱 풍부해진 accent 분석 {#section-1}

이러한 새로운 accent가 추가됨에 따라 이제 [Kachru의 세 원](https://en.wikipedia.org/wiki/World_Englishes#Kachru%27s_Three_Circles_of_English)을 기반으로 더욱 풍부한 accent 분석을 제공할 수 있게 되었습니다(이 아이디어를 [리더보드](https://huggingface.co/spaces/HumeAI/asr-leaderboard)에 적용한 Hume AI에 감사를 전합니다). 이제 Open ASR Leaderboard는 다음과 같은 정보를 보고합니다.

1. US vs. non-US (*UK, Australia, Canada, India, Nigeria, Egypt*)
2. Inner Circle (*US, UK, Australia, Canada*) vs. Outer & Expanding Circles (*India, Nigeria, Egypt*)

더 많은 accent가 추가되면 Outer와 Expanding Circles를 분리하겠습니다.

<div align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/private_data_accent_breakdown.png" width="1024px" alt="thumbnail" />
</div>

## Liva AI 소개 {#section-2}

Liva AI는 실제 사람들이 사용하는 소비자 제품을 통해 멀티모달 상호작용 데이터를 수집합니다. Liva는 데이터 수집 환경과 배포를 모두 소유하고 있기 때문에 도메인, 산업, 언어, 모달리티 및 상호작용 유형 전반에서 특정 모델의 격차를 중심으로 제품을 설계할 수 있습니다. 이를 통해 인위적으로 유도되거나 작업 기반으로 수집된 상호작용이 아니라, 실제 제품 사용을 통해 나타나는 자연스럽고 맥락이 풍부한 행동에 접근할 수 있습니다.
