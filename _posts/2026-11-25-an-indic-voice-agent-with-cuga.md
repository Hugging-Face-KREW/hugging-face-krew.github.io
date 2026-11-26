---
layout: post
title: CUGA를 활용한 인도 지역 언어 음성 에이전트
author: dailybot
categories:
- Translation
- HuggingFace
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/67fe070a4e8cc28ba346f53c/-QNcrxSDbjI8VOCBlhi-C.png
image: assets/images/blog/posts/2026-11-25-an-indic-voice-agent-with-cuga/thumbnail.png
authors:
- user: ibm-research
slug: an-indic-voice-agent-with-cuga
source_url: https://huggingface.co/blog/ibm-research/an-indic-voice-agent-with-cuga
source_published_date: '2026-11-25'
source_published_at: '2026-11-25T10:22:09+00:00'
locale: ko
translation_status: draft
translator: openai
description: '* TOC {:toc} <!--toc--> _이 글은 Hugging Face 블로그의 [An Indic Voice Agent
  with CUGA](https://huggingface.co/blog/ibm-research/an-indic-voice-agent-with-cuga)를
  한국어로 번역한 글입니다_.'
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [An Indic Voice Agent with CUGA](https://huggingface.co/blog/ibm-research/an-indic-voice-agent-with-cuga)를 한국어로 번역한 글입니다_.

<!-- Source: https://huggingface.co/blog/ibm-research/an-indic-voice-agent-with-cuga -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# CUGA를 활용한 인도 지역 언어 음성 에이전트

가장 유용한 지식은 대부분 영어로 작성되고 디지털화되어 있습니다. 하지만 그 지식이 필요한 사람들은 대부분 힌디어, 타밀어, 텔루구어, 구자라트어, 마라티어 등 자신의 지역 언어로 말하고 생각합니다. 많은 사람들은 입력하는 것보다 말하는 것을 선호합니다.

이 간극을 좁히기 위해 [CUGA](https://cuga.dev)를 에이전트 하니스로 사용하는 작은 음성 에이전트를 만들었습니다. 이 글에서는 사용 사례와 그 과정에서 얻은 교훈을 설명합니다. [CUGA Apps](https://huggingface.co/spaces/ibm-research/cuga-apps)에서 직접 사용해 볼 수 있습니다.

## 사용 사례: 농업 도우미 {#section-1}

한 농부가 자신의 언어로 큰 소리 내어 질문합니다. 좋은 씨앗과 나쁜 씨앗을 어떻게 구분하나요? 에이전트는 다음과 같이 동작합니다.

- 음성을 듣고 텍스트로 변환합니다.

- 영어로 작성된 154페이지의 농업 핸드북을 검색합니다.

- 질문을 받은 언어로, 핸드북에 적힌 내용만 사용해 두 문장에서 네 문장으로 답변합니다.

- 답변을 음성으로 들려주고 출처 페이지를 표시하여 답변을 확인할 수 있게 합니다.

누군가 크리켓이나 코딩에 관해 질문하면 내용을 지어내는 대신 정중하게 다시 농업에 관한 질문으로 유도합니다.

## CUGA의 역할 {#section-2}

[CUGA](https://cuga.dev)(ConfigUrable Generalist Agent)는 IBM Research의 오픈 소스 에이전트 하니스입니다. 모델 연결, 도구 관리, 상태 및 가드레일을 처리하므로, 에이전트 루프 대신 도구와 프롬프트를 작성하면 됩니다. 여기서는 작은 웹 앱 내부에 임베드된 SDK로 사용하여 다음 세 가지를 처리합니다.

- 하나의 에이전트, 어떤 모델이든 사용 가능. 설정 하나로 watsonx와 OpenAI 사이를 전환할 수 있습니다.

- 지식 엔진. PDF를 파싱하고, 청크로 나누고, 임베딩한 다음 `agent.knowledge.search()`을 사용해 로컬에서 검색합니다.

- 대화 스레드. 각 질문마다 새로운 스레드를 생성하므로 한 사용자의 언어나 컨텍스트가 다른 사용자의 대화로 유출되지 않습니다.

핵심 설계 아이디어는 간단합니다. CUGA는 텍스트를 입력받아 텍스트를 출력하는 두뇌입니다. 음성은 그 주변에 추가되는 얇은 두 계층입니다.

```
audio ──▶ speech-to-text ──▶ knowledge.search ──▶ CUGA (grounded answer) ──▶ text-to-speech ──▶ audio
```


에이전트는 오직 텍스트만 보기 때문에 먼저 일반 텍스트로 구축하고 테스트한 다음 귀와 입을 추가할 수 있습니다. 에이전트를 수정하지 않고 음성 제공업체를 교체할 수도 있습니다. 여기서는 두 가지를 지원합니다. 하나는 사용자의 브라우저에서 직접 호출하는 Sarvam AI의 클라우드 음성 API이고, 다른 하나는 자체 서버에서 실행하는 오픈 소스 모델(SraVaani는 음성-텍스트 변환, Svara는 텍스트-음성 변환)입니다.

### 어려운 부분은 음성이 아닙니다

음성을 추가하는 일은 쉽습니다. 어려운 부분은 다국어 검색입니다. 힌디어 질문으로 영어 문단을 찾아야 하기 때문입니다.

검색은 텍스트를 벡터로 변환한 다음 가장 가까운 벡터를 찾는 방식으로 작동합니다. CUGA의 기본 임베딩 모델은 영어만 지원하므로 힌디어를 의미가 거의 담기지 않은 벡터로 변환합니다. 이 모델을 사용하면 힌디어나 타밀어 질문에 대해 여전히 다섯 개의 문단을 검색하지만, 엉뚱한 문단이 반환됩니다. 주어진 내용만 바탕으로 답하라는 지시를 정확히 따른 모델은 조용히 "핸드북에 해당 내용이 없습니다"라고 답합니다. 모든 계층이 올바르게 동작하므로 오류도 경고도 없습니다.

해결 방법은 다국어 임베딩 모델(여기서는 `intfloat/multilingual-e5-large`을 사용)을 사용하는 것입니다. 그러면 언어가 무엇이든 같은 의미가 같은 위치에 놓입니다. 개념적으로는 간단하지만, 이를 CUGA에 연결하려면 약간의 우회 작업이 필요했습니다(튜토리얼에서 설명합니다). 그 이후에는 하나의 영어 문서로 모든 언어를 지원할 수 있으며, 어떤 내용도 번역본을 따로 마련할 필요가 없습니다.

저희가 드리는 조언은 다음과 같습니다. 프롬프트 하나를 조정하기 전에 영어가 아닌 언어로 에이전트에 질문을 하나 던지고 올바른 페이지를 찾는지 확인하세요.

### 자연스럽게 들리게 만들고 안전하게 유지하기

예상보다 큰 영향을 미친 몇 가지 작은 결정이 있었습니다.

- 귀로 듣는 것을 기준으로 작성합니다. Markdown, 글머리 기호, 인용 태그, 그리고 "millet (बाजरा)"와 같은 괄호 안 번역은 소리 내어 읽으면 모두 어색하게 들립니다. 프롬프트에서 모델이 이러한 요소를 생성하지 않도록 요청하고, 음성 변환 전에 간단한 포매터로 남아 있는 요소를 제거합니다.

- 가드레일을 저렴하게 유지합니다. 간단한 입력 검사를 통해 명백히 주제에서 벗어난 질문을 잡아냅니다. 하나가 통과하면 모델은 `OFF_TOPIC`으로 답변을 시작하고, 앱이 정중한 안내 문구로 바꿉니다. 두 번째 LLM 호출이 없으므로 음성 경로가 빠르게 유지됩니다.

- 모든 답변을 근거에 기반하게 합니다. 모델에는 검색된 문단만 제공하고 외부 지식을 사용하지 말라고 지시하며, 페이지 출처를 사용자에게 표시합니다.

- 키를 서버에 두지 않습니다. Sarvam 옵션을 사용하면 브라우저가 Sarvam과 직접 통신합니다. 서버는 텍스트만 받고 사용자의 키는 볼 수 없습니다.

## 사용해 보기 {#section-3}

작동하는 모습을 보기 위해 별도로 설정할 필요는 없습니다. Hugging Face Spaces의 CUGA Apps에서 음성 에이전트를 열고, 자신의 언어로 농업에 관한 질문을 한 다음 답변을 들어 보세요.

[Hugging Face의 CUGA Apps →](https://huggingface.co/spaces/ibm-research/cuga-apps)

직접 만들어 보고 싶으신가요? 단계별 튜토리얼인 [Build an Indic Voice Agent with CUGA](https://huggingface.co/blog/ibm-research/build-an-indic-voice-agent-with-cuga)에서는 빈 폴더에서 시작해 각 계층을 차례로 추가합니다.

## 농업을 넘어: Bharat를 위한 확장 {#section-4}

여기서 사용한 방식은 농업에만 해당되지 않습니다. 핸드북을 정부 제도 문서, 건강 관련 안내문 또는 법률 가이드로 바꾸면 다른 도우미를 만들 수 있습니다. 음성 제공업체를 바꾸면 에이전트를 다시 작성하지 않고도 배포 모델을 변경할 수 있습니다. 예를 들어 속도를 위해 클라우드를 사용하거나 데이터 주권을 위해 온프레미스를 사용할 수 있습니다.

음성 우선의 다국어 도우미는 영어 전용 텍스트 화면으로는 충분히 지원받지 못했던 사람들에게 디지털 서비스가 도달할 수 있게 하는 방법입니다. 사람들이 확인할 수 있는 출처에 답변을 연결하고, 개방적이며 감사 가능한 하니스 위에 이를 구축해야 신뢰를 얻을 수 있습니다.

이 접근 방식은 IBM Bharat Core를 통해 저희가 구축하는 것과 정확히 맞닿아 있습니다. Bharat Core는 IBM Sovereign Core를 기반으로 구축된 최적화된 AI 스택으로, 인도 생태계를 위해 특별히 설계되었으며 로컬 인프라에서 Indic AI 워크로드를 안전하게 실행합니다. 음성 에이전트를 프로토타입에서 안전한 엔터프라이즈 애플리케이션(예: 현지 언어 기반 KYC 시스템 또는 다국어 정부 봇)으로 확장할 때는 데이터 주권에 대한 더 큰 통제력이 필요합니다. CUGA의 유연한 에이전트 하니스와 Bharat Core의 즉시 배포 가능한 Indic 커넥터 및 미세 조정 모델을 결합하면, 개발자는 인도의 고유한 언어적 다양성에 맞춘 완전한 규정 준수형 주권 AI 솔루션을 구축할 수 있습니다.

인도를 대상으로 대규모 서비스를 구축하고 있다면 [IBM Bharat Core](https://www.ibmbharatcore.info/)에서 진행 중인 작업을 확인해 보세요.

## CUGA에 대해 더 알아보기 {#section-5}

CUGA가 처음이신가요? 다음 자료부터 시작해 보세요.

- [cuga.dev](https://cuga.dev): CUGA 홈페이지입니다.

- [GitHub의 CUGA](https://github.com/cuga-project/cuga-agent): 오픈 소스 하니스입니다(Apache 2.0). Star를 누르거나, 이슈를 등록하거나, 좋은 첫 번째 이슈를 골라 참여해 보세요.

- [CUGA 문서](https://docs.cuga.dev): 설정, SDK, 지식, 정책 및 구성에 관한 문서입니다.

- [GitHub의 CUGA Apps 갤러리](https://github.com/cuga-project/cuga-apps): Hugging Face Space에 있는 앱의 소스입니다.

브라우저에서 사용해 볼 수 있는 Spaces:

- [CUGA Agent](https://huggingface.co/spaces/ibm-research/cuga-agent): 데모 CRM 시나리오에서 실행되는 범용 에이전트입니다.

- [CUGA Apps](https://huggingface.co/spaces/ibm-research/cuga-apps): 하나의 대시보드에서 작동하는 CUGA 앱을 실행합니다.

- [CUGA Policy Playground](https://huggingface.co/spaces/ibm-research/cuga-policy-playground): 나란히 배치된 두 개의 라이브 에이전트로, 하나에는 정책이 적용되고 다른 하나에는 적용되지 않습니다.

Hugging Face 블로그에서 더 많은 CUGA 게시물을 확인해 보세요.

- [Hugging Face의 CUGA: Configurable AI Agents의 대중화](https://huggingface.co/blog/ibm-research/cuga-on-hugging-face): CUGA와 구성 가능한 추론 모드를 소개하는 글입니다.

- [CUGA를 사용해 실제 에이전트 앱 구축하기: 경량 하니스에서 작동하는 24개의 예제](https://huggingface.co/blog/ibm-research/cuga-apps): 이 음성 에이전트가 속한 앱 갤러리입니다.

- [같은 모델, 다른 하니스: CUGA, Hermes, Deep Agents — CUGA는 언제 선택해야 할까?](https://huggingface.co/blog/ibm-research/cuga-harness): 모델이 같을 때 하니스가 얼마나 중요한지 설명합니다.

