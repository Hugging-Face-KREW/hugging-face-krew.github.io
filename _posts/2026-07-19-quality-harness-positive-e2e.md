---
layout: post
title: "작은 번역 게이트 테스트"
author: dailybot
categories: [Translation, HuggingFace]
authors:
  - user: dailybot
slug: "quality-harness-positive-e2e"
source_url: "https://raw.githubusercontent.com/Hugging-Face-KREW/hf-workflow/aad835b4942658a7bc390573fc8490c4b49a6ab1/skills/quality/tests/fixtures/translation_quality_harness/e2e_positive_source.md"
source_published_date: "2026-07-19"
source_published_at: "2026-07-19T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
description: "작은 번역 게이트 테스트를 위한 한국어 번역 글입니다."
---

# 작은 번역 게이트 테스트

이 짧은 글은 작은 검토 워크플로우가 merge 전에 번역된 블로그 글을 확인하는 방법을 설명합니다. 이 워크플로우는 deterministic check를 실행하고, LLM judge에 semantic review를 요청하며, 리뷰어를 위한 명확한 PR report를 게시합니다.

이 워크플로우는 source URL, 숫자 3, DemoRunner 토큰을 그대로 유지하여 리뷰 중에도 기술 세부 정보가 안정적으로 유지되도록 합니다.
