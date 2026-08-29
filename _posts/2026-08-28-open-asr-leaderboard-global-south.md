---
layout: post
title: "Open ASR Leaderboard에 최초의 글로벌 사우스 언어가 추가되다"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/open-asr-leaderboard-global-south/thumbnail.png
image: assets/images/blog/posts/2026-08-28-open-asr-leaderboard-global-south/thumbnail.png
authors:
  - user: bezzam
  - user: Shobhitbanga
slug: "open-asr-leaderboard-global-south"
source_url: "https://huggingface.co/blog/open-asr-leaderboard-global-south"
source_published_date: "2026-08-28"
source_published_at: "2026-08-28T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [The Open ASR Leaderboard Adds Its First Global South Language](https://huggingface.co/blog/open-asr-leaderboard-global-south)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/open-asr-leaderboard-global-south -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Open ASR Leaderboard에 최초의 글로벌 사우스 언어가 추가되다

*Voice Arena와 Hugging Face가 Hindi 및 Indian English를 위한 오픈 ASR 평가를 출시하기 위해 협력합니다*

벤치마크는 무엇이 만들어질지를 결정합니다. [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)에서 높은 점수를 받은 모델은 채택되고 개선되는 반면, 리더보드가 측정하지 않는 역량은 개선되지 않는 경향이 있습니다. 최근 리더보드 관련 작업의 상당 부분은 평가 지표를 더 신뢰할 수 있게 만드는 데 집중되어 왔습니다.

1. [Held-out private splits](https://huggingface.co/blog/open-asr-leaderboard-private-data).
2. 모델이 오디오만 듣고 전사하는 것이 아니라 참조 전사를 얼마나 재현하는지 정량화하기 위한 [Benchmark-fitting analysis](https://huggingface.co/blog/asr-benchmark-optimization).
3. 올바른 예측/변형이 불이익을 받지 않도록 정규화기의 격차를 해소하기.

이 모든 작업은 하나의 수치(WER)를 조작하기 어렵게 만듭니다. 하지만 여전히 하나의 수치입니다. 오랜 기간 이어진 연구에 따르면 ASR 오류율은 이를 사용하는 사람들 사이에 고르게 분포하지 않습니다. [Racial disparities in automated speech recognition](https://www.pnas.org/doi/10.1073/pnas.1915768117)은 상용 시스템이 백인 화자보다 흑인 화자에게 대략 두 배 더 나쁜 성능을 보인다는 사실을 발견했으며, [Quantifying Bias in Automatic Speech Recognition](https://huggingface.co/papers/2103.15122)은 성별, 연령, 억양에 따른 추가적인 차이를 발견했습니다. 이 중 어느 것도 리더보드에는 드러나지 않으며, 이는 리더보드가 이를 숨기기 때문이 아닙니다. 리더보드가 실행하는 테스트 세트는 발화된 내용을 기록할 뿐, 누가 말했는지에 대해서는 거의 아무것도 기록하지 않기 때문입니다.

이 격차를 해결하기 위해 Open ASR Leaderboard에 두 개의 평가 세트를 도입합니다: [Monsoon en-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN) 및 [Monsoon hi-IN](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN). **5억 명이 넘는 사람들이 사용하는** Hindi는 현재 유럽 언어만 다루는 다국어 탭에 추가되는 최초의 Indic language입니다. 각 세트는 공개 스플릿으로 출시되어 자체 점수 산출이 가능하며, 벤치마크별 최적화를 제한하기 위해 비공개 스플릿은 공개하지 않습니다. 네 개의 스플릿은 화자가 서로 겹치지 않으며, 총 4,888명의 화자로 구성되고 각 화자에 대해 12개의 화자 속성이 기록됩니다.

## 수집 설계 {#section-1}

테스트 세트는 변화를 주어 수집한 축을 따라서만 실패 모드를 드러낼 수 있습니다. 대부분의 벤치마크는 쉽게 구할 수 있었던 오디오로 구성됩니다. Monsoon은 지리, 연령, 성별, 어휘, 기기, 음향 환경, 발화 유형, 발화 속도, 동일한 오디오에 대해 유효한 전사가 여러 개 존재하는지 여부라는 아홉 가지 축에서 변화를 주도록 설계되었습니다. 각각은 집계된 WER이 평균적으로는 맞지만 특정 집단에 대해서는 틀릴 수 있는 방식입니다.

![nine axes of variation in the Monsoon collection](https://storage.googleapis.com/research_team_data/blog_figures/nine-axes-gray.png)

수집 방법은 이러한 설계에서 자연스럽게 도출됩니다.

- **지리적 범위**는 적은 지역에서 더 긴 세션을 녹음하는 대신 수백 개의 지구에서 모집하여 확보했습니다.
- **기기와 음향 조건**은 조용한 방에서 제공된 하드웨어를 사용하는 대신 참여자가 자신의 휴대전화와 네트워크를 사용해 실내와 실외에서 녹음하여 확보했습니다.
- **어휘, 발화 유형, 발화 속도**는 프롬프트에서 비롯됩니다. 일상적인 주제가 참여자를 의견 제시, 이견 표현, 서술, 회상으로 이끌도록 하며, 이 과정에서 고유명사, 숫자, 준비되지 않은 표현이 등장합니다.
- **연령과 성별**은 화자별로 기록하고 검증했습니다.
- **유효한 전사가 여러 개 존재하는 것**은 오디오가 아니라 참조 전사의 속성이며, 이후 절에서 다룹니다.

## 데이터셋 구성 {#section-2}

하나의 파이프라인을 통해 수집한 네 개의 스플릿과 두 개의 언어입니다.

| 세트 | 언어 | 길이 | 화자 수 | 클립 길이(평균 / 중앙값) | 남/여 | 지구 수 | 주/연방 직할지 | 기기 수 | 스타일 | 전사 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| [Monsoon en-IN public](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN) | Indian English | 5.62 h | 1,444 | 9.6s / 10.4s | 50/50 | 428 | 24/6 | 556 | 대화형, 즉흥적 | 정규화됨, 비유창성 포함 |
| Monsoon en-IN private | Indian English | 5.58 h | 1,405 | 9.6s / 10.4s | 45/55 | 420 | 24/6 | 560 | 대화형, 즉흥적 | 정규화됨, 비유창성 포함 |
| [Monsoon hi-IN public](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN) | Hindi | 1.33 h | 468 | 6.4s / 5.0s | 54/46 | 202 | 11/3 | 315 | 대화형, 즉흥적 | lattice(허용되는 철자 변형) |
| Monsoon hi-IN private | Hindi | 4.47 h | 1,571 | 6.6s / 5.3s | 55/45 | 295 | 12/3 | 582 | 대화형, 즉흥적 | lattice(허용되는 철자 변형) |

데이터는 대본 없이 진행된 양방향 즉흥 대화에서 가져왔으며, 각 클립에는 한 명의 화자만 포함되도록 단일 채널에서 분할했습니다. 표에 보고된 필드 외에도 각 클립에는 직업, 학력, 결혼 여부, 소득 구간, 휴대전화 브랜드, 현재 도시, 현재 지구에서 거주한 연수가 기록되어 있습니다.

공개 Indian English 스플릿에서 가져온 다섯 개의 클립과 각 클립에 포함된 메타데이터입니다.

29세 여성, West Tripura, Tripura. 학생, samsung SM-G781B.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/41512950_850_59_859_68.wav"></audio>

32세 여성, Satna, Madhya Pradesh. 무직, samsung SM-E146B.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/04646786_238_95_252_63.wav"></audio>

22세 남성, Rohtas, Bihar. 학생, motorola moto g54 5G.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/19508278_139_86_149_79.wav"></audio>

27세 여성, Warangal, Telangana. 무직, vivo V2247.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/41114168_788_37_795_21.wav"></audio>

57세 남성, Puducherry. 민간 기업 근무, Xiaomi M2006C3LI.

<audio controls src="https://storage.googleapis.com/delivery_team_data/sampled_chunks/english/21485437_227_25_237_45.wav"></audio>

English 세트는 표준 문자열 참조를 사용하며, 리더보드의 정규화기는 대부분의 철자 변형을 하나로 통합합니다. Hindi에는 이러한 변형이 훨씬 많고, 정규화기로 해결할 수 없습니다. 변형이 두 관례 사이의 고정된 매핑이 아니기 때문입니다. 따라서 Hindi 세트는 lattice를 제공합니다. 전사의 각 구간마다 정답으로 허용되는 철자 목록이 포함됩니다.

## 화자 범위 {#section-3}

Monsoon은 시간으로 측정하면 작지만 화자 수로 측정하면 큽니다. 이것이 설계이며, 대부분의 가치가 집중되는 지점이기도 합니다.

**위의 필드로 나타나는 범위를 넘어선 화자 집중도와 다양성.**

| | [Monsoon hi-IN public](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-hi-IN) | Monsoon hi-IN private | [Monsoon en-IN public](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN) | Monsoon en-IN private |
| :--- | :--- | :--- | :--- | :--- |
| 화자당 세그먼트 수(평균) | 1.61 | 1.56 | 1.46 | 1.48 |
| 단일 세그먼트만 보유한 화자 수 | 261 | 994 | 956 | 924 |
| 화자당 오디오 길이(중앙값) | 8.34 s | 8.28 s | 12.36 s | 12.39 s |
| 상위 10명의 화자가 차지하는 비율 | 6.8% | 3.1% | 2.8% | 2.9% |
| 현재 도시 수 | 289 | 814 | 641 | 584 |
| 기기 제조업체 수 | 18 | 25 | 23 | 20 |

세 가지 특성이 나타나며, 각각은 데이터의 양이 아니라 분산에 대한 주장입니다.

- **어떤 한 목소리도 점수를 좌우하지 않습니다:** 가장 많은 기여를 한 10명이 전체 길이에서 차지하는 비율은 2.8%에서 6.8% 사이이며, 전체 화자의 절반 이상은 정확히 한 번만 등장합니다. Monsoon의 결과는 장시간 녹음된 소수의 화자가 아니라 수백 개의 서로 다른 목소리에 대한 평균입니다. 비슷한 길이의 테스트 세트는 보통 반대 방식으로 구성됩니다.

- **지역이나 휴대전화도 점수를 좌우하지 않습니다:** Indian English 공개 세트는 30개 주 및 연방 직할지에 걸친 428개의 고유 지구를 포함합니다. Hindi 세트는 Hindi belt 언어이므로 더 좁게 집중되어 있지만, 여전히 202개 및 295개의 지구를 아우릅니다. 녹음에는 315개에서 582개의 서로 다른 기기 모델이 사용되었으며, 어떤 하위 집합에서도 단일 모델이 세그먼트의 2.1%를 초과하지 않습니다. 표준화된 하드웨어로 수집한 말뭉치는 하나의 마이크 응답에 과적합되지만, 이 데이터셋은 그럴 수 없습니다.

- **여기서 Indian English는 하나의 억양이 아닙니다:** 이는 한 지역의 English가 아니라 전국에서 사용되는 English입니다. 여섯 개 구역이 모두 포함됩니다. 공개 세트에서는 세그먼트의 35%가 남부 화자, 18%가 동부, 18%가 중부, 16%가 북부, 11%가 서부 화자의 기여입니다. 이러한 분포에서 비롯되는 억양 변이는 단정하는 대신 메타데이터에 기록됩니다.

### 메타데이터 필드

Monsoon은 세그먼트당 18개의 열을 제공하며, 이 중 12개가 메타데이터입니다. 대부분의 공개 ASR 테스트 세트가 식별자, 전사, 길이만 제공하는 것과 대조적입니다. 인구통계학적 필드는 완전하거나 거의 완전하며, 참여자들은 이러한 사용에 동의했습니다.

| 그룹 | 필드 |
| :--- | :--- |
| 세그먼트 | `id`, `audio`, `audio_length_s`, `language` |
| 참조 | `lattice` (Hindi) 또는 `text` (Indian English) |
| 화자 | `speaker_id`, `gender`, `date_of_birth` |
| 배경 | `occupation`, `educational_background`, `marital_status`, `income` |
| 지리 | `native_district`, `native_state`, `current_city`, `years_spent_in_current_district` |
| 녹음 | `device_manufacturer`, `device_model` |

두 언어는 서로 다른 지리적 분포를 보이며, 이 분포는 유의미한 정보를 제공합니다. Hindi 세트는 Hindi belt에 집중되어 있으며, 화자의 약 40%를 Uttar Pradesh가 차지합니다. 이는 인구 비례로 표본을 추출한 Hindi 말뭉치에서 예상되는 모습입니다. Indian English 세트의 분포는 훨씬 평평합니다. 어떤 주도 13%를 초과하지 않으며, 화자의 3분의 1은 규모가 가장 큰 8개 주 밖에서 왔습니다. 공개 및 비공개 절반은 두 언어 모두에서 매우 유사한 분포를 보입니다.

인도의 주 경계는 언어적 기준에 따라 설정되었으므로 지구와 주는 실제 억양 신호를 포함합니다. 따라서 이 필드들은 요약으로 없애지 않고 공개됩니다. 인도 ASR에 대해서는 이러한 유형의 분석이 대규모로 보고된 바 있습니다. 지구별 오류율은 대략 4%에서 44%까지 분포하며, 대표성이 낮은 지역은 Hindi belt와 대도시에 크게 뒤처집니다. 또한 오디오 품질, 발화 속도, 발화 지속 시간, 성별, 연령, 기기별 세분화도 이루어졌습니다. 이러한 실행은 비공개 벤치마크에서 수행되었습니다. Monsoon은 공개 리더보드 테스트 세트에서 동일한 종류의 분석을 가능하게 합니다.

## 수집 및 품질 관리 {#section-4}

![monsoon_collection_and_annotation_pipeline](https://storage.googleapis.com/research_team_data/blog_figures/flowchart%201.png)

폭넓은 지리적 범위를 확보하려면 소수의 화자로부터 더 긴 세션을 수집하는 대신 수백 개의 지구에서 참여자를 모집해야 합니다. 이러한 규모의 분산 모집은 소규모 수집에서는 나타나지 않는 실패 모드를 유발합니다. 참여자가 과제를 조작하거나, 실제 발화로 제출해야 할 오디오를 재생해 녹음하거나, 부주의하게 주석을 다는 경우가 그 예입니다. 각각의 문제는 명시적인 검사를 통해 처리합니다.

- **모집 및 녹음:** 참여자는 Voice Arena 커뮤니티를 통해 모집되었습니다. Voice Arena는 전 세계 디지털 플랫폼으로, 음성 말뭉치에서 거의 다루지 않는 농촌 및 준도시 지구까지 도달합니다. 이후 두 명씩 짝을 이루어 지정된 일상 주제에 관해 P2P 인터페이스를 통해 양방향 채널로 대화를 녹음했습니다. 참여자는 자신의 휴대전화와 네트워크를 사용했습니다. 그중 상당수는 불안정한 대역폭을 사용하는 저가형 기기였으며, 이러한 조건을 필터링하지 않고 공개 오디오에 포함한 이유도 여기에 있습니다. 참여 희망자는 녹음 권한을 받기 전에 언어 능력 선별 검사를 완료했으며, 보상을 받고 학습 및 배포에 사용하는 데 대한 사전 동의를 제공했습니다. 화자별 지속 시간 상한은 각 언어 화자의 인구 규모와 지리적 분포에 맞게 조정하여, 소수의 활발한 참여자가 특정 언어나 지역을 지배하지 못하도록 했습니다. 이 세트의 화자 중 절반 이상은 정확히 하나의 세그먼트만 제공합니다.

- **발화 유도:** 대규모로 즉흥 발화를 유도하는 일은 그 자체로 어렵습니다. 구조화된 안내가 없으면 참여자는 짧고 내용이 빈약한 응답을 하는 경향이 있기 때문입니다. 따라서 각 대화는 개방형 서술 단서로 시작하고, 여행, 의료, 농업, 교육, 디지털 서비스 등을 포함하는 영역의 후속 질문을 점진적으로 공개했습니다. 이를 통해 대화를 대본 없이도 긴 설명으로 이끌었습니다. 후보 주제는 대규모 언어 모델로 생성한 뒤 모국어 화자인 언어학자가 검토하고 현지화했습니다.

- **품질 관리:** 모든 녹음은 전사에 앞서 일련의 통과 검사를 거쳤습니다. 30개 이상의 언어에 걸친 사람 주석 데이터로 학습한 언어 식별 모델을 사용해 발화 언어가 지정된 언어와 일치하는지 확인했습니다. 화자의 성별은 전용 분류기를 사용해 자기 보고 레이블과 대조했으며, 자기 보고를 대체하는 것이 아니라 이를 뒷받침하는 용도로 적용했습니다. 추가 모델은 진짜 즉흥 대화와 사전 녹음 또는 재생된 오디오를 구분했습니다. 신호 대 잡음비 추정으로 명료도를 저해하는 수준으로 품질이 저하된 녹음을 제거했지만, 자연스러운 환경 소음은 의도적으로 보존하여 실제 환경에서 수집된 발화의 음향적 현실성을 유지했습니다. 검사를 통과한 녹음은 음성 활동 감지로 분할했으며, 2초간 연속된 무음 지점 또는 다음에 감지되는 무음 지점에서 닫히는 15초의 소프트 상한에서 분할했습니다. 분할은 채널별로 독립적으로 적용했으므로 모든 세그먼트는 단일 화자이자 단일 채널입니다. 이후 세그먼트는 DNSMOS P.808 검사를 통과했습니다.

- **전사:** 참조 전사는 사람이 작성합니다. 첫 초안은 도메인 내 데이터로 학습한 내부 ASR 모델이 생성했습니다. 이 모델들은 어떤 공개 리더보드에도 등장하지 않으므로, 이 세트에서 평가되는 시스템은 자신이 평가받는 참조 전사 작성에 기여하지 않았습니다. 이후의 모든 단계는 모국어 화자인 언어학자가 5단계 프로토콜에 따라 수행했습니다. 이 프로토콜은 엄격한 업무 분리를 기반으로 하며, 각 수정 라운드 뒤에는 다른 주석자가 독립적으로 검증하는 라운드가 이어지므로 어떤 언어학자도 자신의 결과를 직접 감사하지 않습니다. 먼저 한 언어학자가 음향 신호를 기준으로 세그먼트별 초안을 수정하고, 두 번째 언어학자가 이를 다시 검증하며 남은 의견 불일치를 표시합니다. 이후 단계에서는 새로운 주석자가 이 과정을 반복하면서 모호한 음성 실현, 코드 스위칭 경계, 고유명사, 철자 변형 간의 표기 일관성을 점진적으로 해결합니다. 숫자는 단어로 작성하여 전사가 실제 발화와 직접 대응하도록 했습니다. 최종 단계에서도 표시가 남은 세그먼트는 수록 전에 재전사하도록 돌려보냈습니다. 주석자의 행동은 전 과정에서 자동으로 모니터링했으며, 대상 문자 체계에 없는 문자가 포함된 제출물, 부자연스러운 문자 또는 단어 반복, 비정상적으로 적거나 많은 편집 횟수를 표시했습니다.

## 지역적 변이 {#section-5}

이하에서는 공개 Indian English 스플릿을 대상으로 수행한 한 가지 예를 통해 메타데이터로 어떤 평가가 가능한지 보여줍니다. 이는 이 세트가 제공하고자 존재하는 핵심 발견이 아니라, 모든 클립에 화자 정보가 포함될 때 무엇을 답할 수 있게 되는지를 보여주는 예시입니다.

리더보드의 8개 모델은 이 세트에서 4.81에서 4.99 WER 사이의 점수를 기록합니다. 최고와 최저의 차이는 0.18포인트로, 5시간의 데이터가 구분할 수 있는 범위 안에 있습니다. 말뭉치 전체에서 순위를 매기면 이들은 동일한 모델입니다.

화자를 지역별로 묶으면 다른 이야기가 나타납니다. 각 화자의 출신 지구를 인도 주를 분류하는 내무부의 구역 위원회 기준에 따라 상위 구역으로 집계하면, 표본이 충분한 5개 구역이 됩니다. [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)의 차이는 이들 사이에서 0.46포인트입니다. 말뭉치 전체에서는 [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)보다 14/100포인트 뒤처졌지만, 구역별 차이는 1.68포인트로, 중부 구역에서는 4.38, 동부 구역에서는 6.06을 기록합니다. 리더보드에서는 구분할 수 없는 두 시스템이 화자의 출신 지역에 따라 정확도가 얼마나 달라지는지에서는 거의 네 배의 차이를 보입니다.

![corpus wer against zone range](https://storage.googleapis.com/research_team_data/csv_files/corpus_wer_versus_zone_range_coloured_by_zone.png)

어느 구역이 가장 어려운지도 고정되어 있지 않습니다. [ibm-granite/granite-speech-3.3-2b](https://huggingface.co/ibm-granite/granite-speech-3.3-2b)은 북부에서, [microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF)은 남부에서, [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)는 동부에서 가장 낮은 성능을 보입니다. 단순히 특정 지역의 전사가 더 어렵다면 모든 모델이 같은 방식으로 구역 순위를 매겨야 합니다. 하지만 그렇지 않으며, 이는 오디오보다는 모델 자체를 가리킵니다.

지역은 기록된 12개 속성 중 하나이며, 위의 구역은 428개 지구를 거칠게 집계한 것입니다. 동일한 분석은 연령, 학력, 직업, 휴대전화에도 적용할 수 있으며, 공개된 파일에는 이를 재현하는 데 필요한 모든 정보가 포함되어 있습니다. 발화된 내용만 기록하는 테스트 세트에서는 이 중 어느 것도 사용할 수 없습니다.

## Hindi의 표기 변이 {#section-6}

English의 표기 변이는 한정적입니다. 영국식과 미국식 철자, 구두점, 대소문자, 숫자와 단어의 차이 등은 정규화기가 대부분 하나의 형식으로 매핑할 수 있으며, 리더보드의 정규화기도 그렇게 합니다. Hindi는 같은 방식으로 한정되지 않습니다. 일상적인 발화에는 코드 믹싱이 많이 나타나고, English에서 유래한 단어에는 정착된 Devanagari 철자가 없으며, 합성어는 선호에 따라 붙여 쓰거나 띄어 씁니다. 하나의 구절에 유효한 표기 형식이 10개 이상 존재할 수 있으며, 이를 하나로 통합할 고정된 매핑도 없습니다. 매핑할 표준적인 한쪽 형식이 없기 때문입니다.

![english_normaliser](https://storage.googleapis.com/research_team_data/blog_figures/flowchart%202.png)

단일 참조로 평가하면 WER은 주석자가 우연히 선택한 철자를 시스템이 생성했을 때 그 시스템에 보상을 줍니다. 오디오를 동일하게 잘 인식한 두 시스템도 표기만으로 몇 포인트씩 차이가 날 수 있습니다.

따라서 Hindi 세트는 lattice를 제공합니다. 전사의 각 구간마다 정답으로 허용되는 표기 형식의 집합이 포함됩니다. 이를 구축하는 일은 수작업입니다. 동일한 오디오에 대한 여러 ASR 전사에서 후보 변형을 추출하고 언어 모델로 확장한 다음, 모국어 화자인 언어학자가 해당 발화에 유효한 형식을 결정하고 나머지를 제거합니다. 이를 통해 실제 발화 내용과 일치하는 형식만 허용됩니다.

![hindi_oiwer_scoring](https://storage.googleapis.com/research_team_data/blog_figures/flowchart%203a.png)

따라서 Hindi에는 WER 대신 AI4Bharat가 도입한 [Orthographically-Informed Word Error Rate (OIWER)](https://huggingface.co/papers/2603.00941)를 보고합니다. 각 구간에서 가설을 허용된 집합과 정렬하므로, 허용된 형식은 무엇이든 정답으로 처리되고 실제 인식 오류에 대해서만 오류가 부과됩니다.

효과를 정량화하기 위해 동일한 가설을 두 번 평가했습니다. 각 lattice를 구간별 첫 번째 변형으로 평탄화하면 일반적인 벤치마크가 제공하는 것과 같은 단일 문자열 참조가 생성됩니다. 허용된 변형 중 어느 것이든 동일하게 사용할 수 있으며, 어떤 변형을 선택하느냐에 따라 다른 참조가 만들어집니다. 평탄화된 참조와 비교해 평가하면 모든 시스템의 오류율이 상승하지만, 상승 폭은 균일하지 않습니다. 그 결과 순위가 바뀝니다. 아래 그림은 두 참조 사이에서 순서가 뒤집히는 두 쌍의 시스템을 보여줍니다. 단일 참조에서는 시스템이 주석자의 표기를 재현한 것에 대해 부분적으로 보상을 받지만, lattice는 인식만 평가합니다.

![WER against OIWER on Monsoon hi](https://storage.googleapis.com/research_team_data/csv_files/hindi_oiwer_flips.png)

또한 모든 사람이 이 세트의 결과를 직접 재현할 수 있도록 구현체 [voi-oiwer](https://pypi.org/project/voi-oiwer)를 오픈 소스로 공개합니다.

## 평가받는 방법 {#section-7}

비공개 스플릿의 경우 Open ASR Leaderboard에 모델을 등록하면 Hugging Face 팀이 평가를 실행합니다. 이전과 마찬가지로 리더보드에 모델을 추가하는 절차는 [Open ASR Leaderboard GitHub](https://github.com/huggingface/open_asr_leaderboard)에서 진행됩니다.

1. pull request를 열면 [model checklist](https://github.com/huggingface/open_asr_leaderboard/blob/main/.github/PULL_REQUEST_TEMPLATE.md#new-model-checklist)가 표시됩니다. 이전과 마찬가지로 공개 데이터셋에서의 결과를 보고해야 합니다.
2. 공개 세트에서 결과를 검증하고 비공개 세트의 지표를 계산합니다.
3. 저희가 얻은 결과를 확인합니다.

Indian-English는 [main leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)에 `Voice Arena Monsoon`로 추가되며, 선택적으로 활성화하는 토글이 아니라 기본 열 집합에 포함됩니다. 따라서 모든 모델의 대표 Average WER에 반영됩니다. 비공개 스플릿은 [Appen and DataoceanAI data](https://huggingface.co/blog/open-asr-leaderboard-private-data)와 함께 집계된 `Private (conversational)` 열에 반영됩니다. 공개 및 비공개 Hindi는 [Multilingual tab](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)에 표시됩니다. 이 열에서는 선택한 모든 언어를 지원하는 모델만 순위에 포함되므로, 동일한 조건에서 비교할 수 있습니다. 또는 "Language dataset breakdown" 드롭다운 메뉴에서 "Hindi"를 선택할 수 있습니다.

## 앞으로의 과제 {#section-8}

Hindi는 일반적인 문제를 선명하게 보여주는 사례입니다. 하나 이상의 방식으로 표기되고, 벤치마크가 표본으로 포함하지 않은 사람들이 사용하며, 이 글에서 설명한 두 가지 실패를 모두 지니는 언어는 무엇이든 같은 문제를 겪습니다. 이 세트들이 문제를 해결하는 것은 아닙니다. 대신 문제를 볼 수 있는 방법을 추가합니다. 즉, 해당 분야가 이미 주목하고 있는 리더보드의 테스트 세트에 각 화자와 각 참조 전사에 대한 충분한 정보가 담기므로, 두 시스템 간 차이를 하나의 수치 속에서 사라지게 하지 않고 누가 말했는지, 그리고 어떻게 표기하는지에 따라 추적할 수 있게 합니다.

이 네 개의 세트는 Global South를 위한 Voice Arena의 광범위한 데이터셋 이니셔티브인 Monsoon의 일부입니다.
