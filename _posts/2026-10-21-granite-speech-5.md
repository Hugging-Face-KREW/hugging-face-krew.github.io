---
layout: post
title: "Granite Speech 5.0: 더 정확하고 빠른 전사, 번역 및 귀속"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/653fb52cbfd890805461d691/M_JJyIOBOWVzHXdy7CVVG.png
image: assets/images/blog/posts/2026-10-21-granite-speech-5/thumbnail.png
authors:
  - user: ibm-granite
slug: "granite-speech-5"
source_url: "https://huggingface.co/blog/ibm-granite/granite-speech-5"
source_published_date: "2026-10-21"
source_published_at: "2026-10-21T14:58:18+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Granite Speech 5.0: Transcribe, Translate, and Attribute more Accurately and Faster](https://huggingface.co/blog/ibm-granite/granite-speech-5)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-granite/granite-speech-5 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Granite Speech 5.0: 더 정확하고 빠른 전사, 번역 및 귀속

IBM의 Granite Speech 컬렉션에 대한 세 가지 업데이트를 공유하게 되어 기쁩니다. 새로운 음성 언어 모델인 [granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b)과 커뮤니티에서 이미 사용 중인 두 모델의 개정 버전인 [granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) 및 [granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)를 선보입니다. 이 모델들은 엔터프라이즈 음성 작업의 전체 범위를 아우릅니다. 높은 정확도의 전사와 번역, 엣지 환경에서의 초고속 전사, 그리고 무엇을 말했는지만큼 누가 언제 말했는지가 중요한 녹음 처리까지 지원합니다.

| 모델 | 상태 | 새로운 기능 | 적합한 용도 |
| --- | --- | --- | --- |
| `granite-speech-5.0-2b` | 신규 | 더 정확하고 잡음 및 잔향에 강건한 자동 음성 인식(ASR), 언어 ID 프롬프트, 2,000개 용어 키워드 편향, 향상된 장문 번역 | 고정확도 다국어 전사 및 번역 |
| `granite-speech-5.0-470m-turboctc` | 업데이트 | 다국어 지원, 네이티브 구두점 및 대문자 표기, 잡음과 잔향에 대한 향상된 강건성 | 대용량 및 온디바이스 전사 |
| `granite-speech-4.1-2b-plus` | 업데이트 | 향상된 화자 귀속, 타임스탬프, 키워드 편향 및 장시간 음성 성능 | 회의, 통화 및 기타 다화자 녹음 |

## Granite Speech 5.0 2B: 더 정확하고, 더 강건하며, 더 폭넓은 제어 {#section-1}

`granite-speech-5.0-2b`는 동일한 파라미터 수를 유지하면서 `granite-speech-4.1-2b`의 후속 모델로 출시되었습니다. 영어, 프랑스어, 독일어, 스페인어, 포르투갈어 및 일본어를 전사하며, 해당 언어들과 영어 간 번역을 지원하고 영어에서 이탈리아어 및 만다린으로의 번역도 지원합니다. 일본어 ASR, 키워드 편향 및 번역을 위해 구축한 합성 데이터와 390,000시간의 공개 오디오 데이터로 학습되었습니다.

4.1 2B Speech와 비교해 새로워진 점은 다음과 같습니다:

- 더 정확하고 강건한 영어 ASR: 영어 단문 벤치마크에서 더 높은 정확도를 보이며, 잡음, 잔향 및 원거리 마이크 음성을 더 잘 처리합니다.

- 언어 ID 프롬프트: 프롬프트에 입력 언어를 지정해 다국어 전사를 유도할 수 있습니다.

- 향상된 장문 및 대화형 번역: 짧은 낭독 문장뿐 아니라 더 자연스러운 음성에서도 긴 구간 동안 번역 성능을 유지합니다.

- 최대 2,000개 용어의 키워드 목록: 지원되는 모든 언어에서 제품명, 사람 이름 및 약어를 중심으로 전사와 번역을 유도할 수 있습니다.

- 요청에 따른 구두점 및 대문자 표기: 프롬프트에서 형식이 지정된 출력을 요청할 수 있으며, 대문자 표기 F1은 테스트 세트 전반에서 89.7~99.5 범위입니다.

이 모델은 Apache 2.0 라이선스로 제공되며, Transformers, vLLM, llama.cpp 및 Apple Silicon용 mlx-audio를 지원합니다.

[![granite-speech-5.0-2b average English WER compared with top open-weight ASR models](https://cdn-uploads.huggingface.co/production/uploads/653fb52cbfd890805461d691/87yRtLq4X2CwMZrDhv0L_.png)](https://cdn-uploads.huggingface.co/production/uploads/653fb52cbfd890805461d691/87yRtLq4X2CwMZrDhv0L_.png)

차트 1: granite-speech-5.0-2b는 평균 영어 WER(5.06%) 기준 상위 오픈 웨이트 ASR 모델 중 2위를 차지합니다. 낮을수록 좋습니다. 결과는 2026년 10월 21일 기준 공식 결과이며, Open ASR Leaderboard의 공개 영어 단문 테스트 세트에 대한 평균입니다.

## Granite Speech 5.0 Turbo CTC: 이제 다국어를 지원하고 더 강건하며 더욱 빠르게 {#section-2}

처음 [introduced Turbo CTC in August](https://huggingface.co/blog/ibm-granite/granite-speech-5-0-470m-turboctc)했을 때 이 모델은 속도를 위해 설계된 영어 전용 470M 파라미터 인코더였습니다. 이번 개정판은 동일한 규모를 유지하면서 사용자들이 가장 많이 요청한 세 가지 기능을 추가했습니다:

- 다국어 전사: 영어, 일본어, 스페인어, 독일어, 프랑스어, 이탈리아어 및 포르투갈어.

- 네이티브 구두점 및 대문자 표기: 별도의 구두점 또는 truecasing 단계 없이 바로 읽을 수 있는 형태로 전사 결과가 출력됩니다.

- 어려운 음향 환경에서 향상된 강건성: 잔향이 있는 오디오를 명시적으로 대상으로 하는 학습을 통해 Turbo CTC가 울림이 있는 공간과 원거리 녹음에서 더 강건하게 작동합니다.

속도도 더 빨라졌습니다. 단일 Conformer 블록을 컴파일하고 전체 스택에서 그래프를 재사용함으로써 워밍업 시간을 단축했으며, Open ASR Leaderboard에서 Turbo CTC의 성능을 19,870 RTFx까지 높였습니다. 이는 출시 당시 약 12,600 RTFx에서 향상된 수치로, 동일한 정확도 수준에서 약 1.6배 더 빠릅니다.

차트 2: Open ASR Leaderboard의 평균 WER 대 RTFx(영어 단문, 공개 테스트 세트 9개, 2026년 10월 기준). WER은 낮을수록, RTFx는 높을수록 좋으며, RTFx는 로그 스케일입니다.

## Granite Speech 4.1 2B Plus: 누가 무엇을 언제 말했는지 더 정확하게 {#section-3}

업데이트된 `granite-speech-4.1-2b-plus`는 엔터프라이즈에서 실제로 접하는 녹음, 즉 여러 화자가 장시간 참여하는 회의, 지원 통화 및 인터뷰에 맞게 조정되었습니다. 아키텍처, 프롬프트 형식 및 API가 변경되지 않았기 때문에 4.1이라는 이름을 유지하며, 향상된 가중치를 적용한 현재 모델의 드롭인 교체 모델입니다. 이번 개정판은 다음 네 가지 영역을 개선합니다:

- 화자 귀속: 누가 말하고 있는지 더 정확하게 라벨링합니다. 단어 화자 다이어리제이션 오류율(WDER)은 7%, 화자 귀속 WER(cpWER)은 전체적으로 7% 감소합니다.

- 타임스탬프: 평균 타임스탬프 이동(AAS)을 절반 이상 줄여 자막, 검색 및 검토에서 더 정확한 타이밍을 제공합니다.

- 장시간 음성 성능: 장문 테스트 세트의 WER이 24% 감소합니다.

- 키워드 편향: 20개 용어의 키워드 목록을 사용하면 편향된 키워드의 오류가 45% 감소합니다.

| 기능 | 지표 | 이전 릴리스 | 업데이트된 모델 |
| --- | --- | --- | --- |
| 타임스탬프, 모든 음성 | AAS (평균 타임스탬프 이동) | 210.86 | 98.54 |
| 타임스탬프, 짧은 음성 | AAS (평균 타임스탬프 이동) | 29.27 | 26.20 |
| 키워드 편향 | 편향된 단어 WER, 20개 용어 목록 | 11.79 | 6.48 |
| 장문 음성 | WER, 장문 테스트 세트 | 13.53 | 10.22 |
| 화자 귀속 | WDER | 4.40 | 4.10 |
| 화자 귀속 전사 | cpWER | 30.50 | 28.48 |

표 1: 모든 지표는 오류율이므로 낮을수록 좋습니다. 더 자세한 결과는 [model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)를 참조하세요.

## 시작하기 {#section-4}

어떤 모델을 선택해야 할지 잘 모르겠나요? 다음과 같이 시작해 보세요:

- 높은 전사 정확도, 번역 또는 키워드 편향이 필요한가요? [granite-speech-5.0-2b](https://huggingface.co/ibm-granite/granite-speech-5.0-2b)를 사용하세요.

- 원시 속도 또는 온디바이스 전사가 필요한가요? [granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)를 사용하고, [WebGPU streaming demo](/blog/ibm-granite/DEMO_URL)를 통해 브라우저에서 사용해 보세요.

- 회의 또는 다화자 통화를 처리하나요? [granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus)를 사용하세요.

이 모델들은 모두 Apache 2.0 라이선스로 공개되며 Transformers를 네이티브로 지원합니다. 전체 평가 결과, 아키텍처 세부 정보, 학습 데이터 및 사용 예시는 각 모델 카드에서 확인할 수 있습니다. 추가적인 위험 탐지가 필요한 프로덕션 배포에는 [Granite Guardian](https://huggingface.co/collections/ibm-granite/granite-guardian)를 함께 사용할 것을 권장합니다.

직접 사용해 보고 어떻게 생각하는지 알려 주세요!
