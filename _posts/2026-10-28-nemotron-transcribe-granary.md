---
layout: post
title: 'NVIDIA Nemotron 3.5 Transcribe and Granary v2: 프롬프트로 지시하고 조정할 수 있는 지능형 다국어
  전사'
author: dailybot
categories:
- Translation
- HuggingFace
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/3vFxCEUXz_xJVrTBr4hNr.jpeg
image: assets/images/blog/posts/2026-10-28-nemotron-transcribe-granary/thumbnail.jpeg
authors:
- user: nvidia
slug: nemotron-transcribe-granary
source_url: https://huggingface.co/blog/nvidia/nemotron-transcribe-granary
source_published_date: '2026-10-28'
source_published_at: '2026-10-28T14:58:17+00:00'
locale: ko
translation_status: draft
translator: openai
description: '* TOC {:toc} <!--toc--> _이 글은 Hugging Face 블로그의 [NVIDIA Nemotron 3.5
  Transcribe and Granary v2: Intelligent Multilingual Transcription You Can Prompt
  and Adapt](https://h...'
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [NVIDIA Nemotron 3.5 Transcribe and Granary v2: Intelligent Multilingual Transcription You Can Prompt and Adapt](https://huggingface.co/blog/nvidia/nemotron-transcribe-granary)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/nemotron-transcribe-granary -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# NVIDIA Nemotron 3.5 Transcribe and Granary v2: 프롬프트로 지시하고 조정할 수 있는 지능형 다국어 전사

## 다국어 전사, 화자 식별, 번역, 오디오 Q&A를 하나의 오픈 모델로 제공하며, 리더보드에서 최고 수준의 결과를 달성하고, 모델을 조정할 수 있는 오픈 Granary v2 데이터 파이프라인과 파인튜닝 레시피까지 지원 {#section-1}

전사문은 모든 발화 단어를 담고 있어도 누가 무엇을 말했는지, 어떻게 번역할지, 도메인별 전문 용어를 어떻게 표기할지에 대해 사용자나 애플리케이션에 여전히 의문을 남길 수 있습니다. 다국어 통화, 빠르게 진행되는 회의, 의료 받아쓰기는 각각 번역, 화자 레이블, 정확한 의학 용어 표기처럼 서로 다른 출력을 필요로 합니다.

NVIDIA Nemotron 3.5 Transcribe는 녹음된 음성을 유용한 텍스트로 변환하는 오픈 웨이트 프롬프트 가능 모델로, 이제 [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) 및 [ModelScope](https://modelscope.ai/collections/nv-community/Nemotron-35-Transcribe)에서 사용할 수 있습니다. 하나의 체크포인트에서 전사, 화자 식별(다이어리제이션), 번역, 오디오 질의응답을 제공합니다. 40개 언어를 전사하고, 그중 31개 언어를 영어와 상호 번역하며, 13개 언어에서 화자를 식별합니다. 최대 1시간 길이의 녹음을 처리하고 최대 8명의 화자를 레이블링할 수 있습니다. 개발자는 자연어로 출력 결과를 안내하고, 오디오를 처리할 위치를 선택하며, 도메인 어휘와 억양에 맞게 인식을 조정할 수 있습니다.

Transcribe는 영어 기준 [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)에서 오픈 웨이트 모델 중 1위를 차지했으며, [Artificial Analysis](https://artificialanalysis.ai/speech-to-text/non-streaming?error-rate=word-error-rate-vs-active-size)의 정확도–활성 파라미터 프런티어에 위치합니다. 화자 식별에서는 [David AI’s 21-language diarization leaderboard](https://research.withdavid.ai/blog/dai-asr-i18n#leaderboard)에서 가장 낮은 오류율을 기록했고 [VoiceArena’s 14-system comparison](https://voicearena.com/diarization-bench/methodology)에서 1위를 차지했습니다. 양자화 버전도 곧 출시될 예정이므로, RTX Spark, DGX Spark 또는 Jetson AGX Thor에서 Transcribe를 로컬로 실행하거나 Blackwell 데이터센터 GPU에서 규모를 확장할 수 있습니다.

Granary v2의 다국어 데이터([Hugging Face](https://huggingface.co/datasets/nvidia/Granary-v2) 및 [ModelScope](https://modelscope.ai/datasets/nv-community/Granary-v2)에서 제공)와 재사용 가능한 큐레이션 파이프라인은 [NeMo Speech fine-tuning tools](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/speechlm2/finetuning/README.md)와 함께 프롬프트 수준의 제어를 학습 단계까지 확장합니다. 개발자는 필요한 표기 형식을 정의하고 애플리케이션에서 접하는 음성에 맞게 모델을 조정할 수 있습니다. [medical case study](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/finetuning/docs/medical-asr-synthetic-data.md)에서 파인튜닝을 수행한 결과, 보류 화자에 대한 단어 오류율(WER)이 최적 프롬프트 대비 상대적으로 45.8% 감소했습니다.

# 자연어로 전사와 분석 안내하기

동일한 체크포인트라도 지시에 따라 하나의 회의 녹음에서 서로 다른 출력을 생성할 수 있습니다.

- “이 녹음을 전사하고 각 화자를 레이블링하세요.”

- “이 오디오를 영어로 번역하세요.”

- “이 회의에서 어떤 결정이 내려졌나요?”

아래 영상에서 모델이 작동하는 모습을 확인해 보세요. 영상에서는 빠르고 서로 겹치는 대사로 유명한 1940년 스크루볼 코미디 영화 His Girl Friday의 대표적인 장면을 처리합니다.

Nemotron 3.5 Transcribe가 His Girl Friday(1940)의 장면에 자막을 생성하고, 겹쳐서 말하는 대화를 포함해 화자를 레이블링합니다.

익숙하지 않은 용어라면 학습 전에 문맥을 활용해 보세요. 예를 들어 의료 환경에서는 다음과 같이 지시할 수 있습니다.

`Transcribe this medical discussion in the original language. Possible terms: HbA1c, metformin. Use these terms only when supported by the audio. Preserve standard medical spelling and write numbers in conventional written form.`

[model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)에는 오디오 질문에 대한 추론을 포함한 권장 프리셋과 전체 프롬프트 가이드, 작업별 언어 지원 매트릭스가 제공됩니다. 작업에 따라 지원 언어가 다르며, 오디오 질의응답은 영어로 학습되었습니다. 단어 수준 타임스탬프도 지원됩니다. 모든 음성 모델과 마찬가지로 이름, 화자 전환, 답변은 녹음과 대조해 일부 확인하는 것이 좋습니다.

실시간 수업에서 오래 남는 학습 자료까지. 앱에서 실시간 대화 기능을 제공하는 실시간 음성 및 비디오 플랫폼 Agora는 강의 데모에서 단어 수준 타임스탬프를 사용했습니다. Nemotron 3.5 Transcribe가 녹화된 수업을 리플레이와 함께 표시되는 전사문으로 변환해 학생들이 무엇이 언제 말해졌는지 확인할 수 있도록 한 것입니다. [Agora’s blog](https://www.agora.io/en/blog/how-live-teaching-can-become-a-lasting-learning-resource/)에서 구축 방법을 자세히 설명합니다.

# 전사부터 화자 식별까지 이어지는 리더보드 성과

오픈 웨이트 모델 가운데 Transcribe는 무엇이 말해졌는지에 대한 평가에서 앞서며, 누가 말했는지에 대해서는 두 화자 식별 벤치마크에서 모두 단독 1위를 차지합니다.

Hugging Face Open ASR Leaderboard. 이 리더보드는 공개 테스트 세트에서 단어 오류율(WER)을 기준으로 전사 정확도를 측정하며, 세 가지 트랙으로 구성됩니다.

- 영어: 평균 WER 5.33%로 오픈 웨이트 모델 중 1위(그림 1).

- 다국어: 7개 언어에서 평균 WER 4.62%로, 모델 카드의 Open ASR 비교에 포함된 평가 오픈 웨이트 모델 중 가장 낮음.

- 장문: 평균 WER 10.98%로, 해당 비교에서도 가장 낮음.

[![Hugging Face Open ASR English Open (2) Leaderboard table showing model rankings on the Hugging Face English Open ASR leaderboard, with columns for rank, model, average WER, RTFx, and dataset scores. nvidia/Nemotron-3.5-Transcribe is ranked first among open-weight models with an average Word Error Rate (WER) of 5.33% and an RTFx of 462.09.](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/vGIg17AxweCAnacvnPOet.png)](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/vGIg17AxweCAnacvnPOet.png)

그림 1. Nemotron 3.5 Transcribe는 API 모델을 제외한 [Hugging Face Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)(영어)에서 평균 WER 5.33%로 오픈 웨이트 모델 중 1위를 차지합니다. WER이 낮을수록 전사 오류가 적다는 의미입니다.

Artificial Analysis Speech-to-Text Leaderboard. Artificial Analysis는 자체 단어 오류율 지수인 AA-WER를 기준으로 음성-텍스트 모델의 순위를 독립적으로 산정합니다. Nemotron 3.5 Transcribe는 AA-WER 2.8%로 오픈 웨이트 모델 중 공동 1위를 기록했습니다. 활성 파라미터는 약 4B로, 공동 1위를 차지한 모델의 24B보다 적습니다. AA-WER 지수의 50%를 구성하는 보류 음성 에이전트 음성 세트 AA-AgentTalk에서 Transcribe는 WER 1.8%를 기록했으며, 다음으로 우수한 오픈 웨이트 모델은 2.3%였습니다. Transcribe는 음성 인코더를 하이브리드 Mamba–Transformer mixture-of-experts 언어 모델과 결합하므로, 30B 파라미터 중 토큰당 약 4B만 활성화됩니다. 따라서 [Artificial Analysis](https://artificialanalysis.ai/speech-to-text/non-streaming)의 활성 파라미터 대비 정확도 오픈 웨이트 파레토 프런티어에 위치합니다. AA의 전체 모델 차트에서는 널리 사용되는 여러 독점 음성 API보다 높은 순위를 기록합니다.

[![ArtificialAnalysis-WER vs Model Size (Open Weights) (22 Oct '26)Scatter chart from Artificial Analysis plotting word error rate (AA-WER, non-streaming) against active parameters on a log scale for open-weight speech models. Nemotron 3.5 Transcribe sits on the Pareto frontier at about 4B active parameters and 2.80% WER.](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/a7N1ycYypBeM7WKuUphE7.png)](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/a7N1ycYypBeM7WKuUphE7.png)

그림 2. 약 6분의 1에 해당하는 활성 파라미터로 [Leading AA accuracy](https://artificialanalysis.ai/speech-to-text/non-streaming): Nemotron 3.5 Transcribe는 약 4B의 활성 파라미터로 2.80% AA-WER를 달성합니다.

언어와 겹침을 아우르는 화자 식별. Sortformer 화자 신호는 음향 특징과 함께 언어 모델에 전달되므로, Transcribe는 이후 단계에서 단어에 화자를 연결하는 방식에만 의존하지 않고 생성 중에 각 발화 차례를 태그할 수 있습니다.

David AI DAI-ASR-I18N Leaderboard. [David AI’s DAI-ASR-I18N combined transcription-and-speaker-attribution track](https://research.withdavid.ai/blog/dai-asr-i18n#leaderboard)에서 Transcribe는 평가된 오픈 웨이트 모델 중 1위이며 전체 2위를 차지합니다. 21개 언어에서 누가 언제 말했는지를 평가하는 이 벤치마크의 다이어리제이션 트랙에서, 테스트된 14개 시스템 중 가장 낮은 다이어리제이션 오류율을 기록했으며 NVIDIA의 전용 [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) 모델보다 근소하게 앞섰습니다.

[![DavidAI_Nemotron_3.5_Transcribe_Diarization_Accuracy Bar chart from David AI’s DAI-ASR-I18N benchmark ranking diarization systems by overall diarization error rate (DER) across 21 languages, where lower is better. Nemotron 3.5 Transcribe ranks first at 18.26%, followed by Nemotron 3 Diarization at 18.38%; the remaining open-weight and proprietary systems range from 22.85% to 37.55%](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/zwwsN64zfjYDvTOhQapp5.png)](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/zwwsN64zfjYDvTOhQapp5.png)

그림 3. NVIDIA 모델이 [David AI’s multilingual diarization leaderboard](https://research.withdavid.ai/blog/dai-asr-i18n#leaderboard)의 상위 두 자리를 차지합니다. Nemotron 3.5 Transcribe는 21개 언어에서 전체 다이어리제이션 오류율(DER) 18.26%로 가장 낮은 수치를 기록했고, [Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)가 18.38%로 뒤를 이었습니다. DER은 화자 시간 오류를 측정하며, 낮을수록 좋습니다.

VoiceArena Diarization-Bench. [VoiceArena’s English full-session benchmark](https://voicearena.com/diarization-bench)에서 Transcribe는 14개 시스템 중 연결 최소 순열 WER(cpWER)가 14%로 가장 낮았으며, 대면 및 온라인 환경 모두에서 1위를 차지했습니다. 또한 삭제된 단어 수도 가장 적었습니다. 삭제율은 기준 전사 단어의 8.32%로, 중앙값 약 31%보다 낮았습니다.

[![VoiceArena_Track2_cpWER_ranking_1Bar chart from VoiceArena’s Diarization-Bench v1 ranking 14 diarization-plus-ASR systems on English full sessions with overlapping speech by concatenated minimum-permutation word error rate (cpWER), where lower is better. NVIDIA Nemotron 3.5 Transcribe ranks first at 32.4%, followed by NVIDIA Multitalker Parakeet \+ Nemotron 3 Diarization at 34.2%; the remaining systems range from 34.7% to 71.9%](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/kU1uIXD-hl4ikaalIYFfI.png)](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/kU1uIXD-hl4ikaalIYFfI.png)

그림 4. NVIDIA 모델이 [VoiceArena’s 14-system comparison](https://voicearena.com/diarization-bench)에서 상위 두 자리를 차지합니다. Nemotron 3.5 Transcribe가 32.4% cpWER로 선두를 차지했고, NVIDIA Multitalker Parakeet with Nemotron 3 Diarization이 34.2%로 뒤를 이었습니다. Diarization-Bench v1은 영어 전체 세션, 겹치는 발화, 2~8명의 화자를 다룹니다. cpWER은 화자 레이블을 매칭한 후 단어 오류를 계산하며, 낮을수록 좋습니다.

추가 평가. Ocular AI의 Converse-STT 평가에서 Transcribe는 최고 오픈 모델로 선정되었습니다. Josh Talks의 [Voice of India](https://voiceofindia.ai) 리더보드는 인도 언어 전반에서 Transcribe의 결과를 보고합니다.

고처리량 전사. 벤치마킹에서 Transcribe는 출시 예정인 NVFP4 양자화 버전을 사용해 NVIDIA RTX PRO 6000 Blackwell Server Edition GPU 한 장에서 100시간 분량의 오디오를 3분 이내에 처리했습니다. 결과는 높은 동시성으로 30초 클립을 처리해 측정했습니다.

현재 [vLLM](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/vllm_serving.rst)로 제공되는 체크포인트를 서빙할 수 있습니다. 양자화 버전과 함께 [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)를 사용한 로컬 배포도 곧 지원될 예정입니다. Transcribe는 실시간 스트리밍이 아니라 녹음된 오디오를 대상으로 설계되었습니다.

특수한 녹음은 도메인 어휘와 전사 규칙에 맞춘 조정이 필요할 수 있습니다. Granary v2와 NeMo Speech 파인튜닝이 이를 지원하며, Transcribe의 기반 데이터부터 시작합니다.

# Transcribe의 기반이 되는 오픈 데이터 큐레이션 파이프라인, Granary v2

프롬프트 가능한 음성 모델에는 명확한 형식 지정 목표를 갖춘 학습 데이터가 필요합니다. [Granary v2](https://huggingface.co/datasets/nvidia/Granary-v2)는 Hugging Face 및 [ModelScope](https://modelscope.ai/collections/nv-community/Nemotron-Speech)의 [Nemotron speech collection](https://huggingface.co/collections/nvidia/nemotron-speech)에 포함되어 있습니다. [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)로 구축된 이 재사용 가능한 파이프라인은 Nemotron 3.5 Transcribe 학습에 사용된 다국어 음성을 준비하며, 정렬된 전사 변형, 작업 레이블, 검사 가능한 품질 메타데이터를 생성합니다. 공개 매니페스트는 48개 언어와 16개 소스 코퍼스에 걸쳐 463,062시간을 포함하며, [Granary v1](https://huggingface.co/datasets/nvidia/Granary)의 25개 언어에서 증가한 수치입니다. 이 수치는 데이터셋 릴리스를 설명하는 것이며, Transcribe의 전체 학습 혼합 데이터나 샘플링 가중치를 의미하지는 않습니다.

Granary v2는 각 녹음에 대해 세 가지 전사 형식을 생성합니다(표 1). 이는 개발자가 Transcribe에 프롬프트로 요청할 수 있는 verbatim 및 clean 스타일과 동일합니다. 전사 외에도 파이프라인은 짧은 오디오 캡션, 차용어의 코드 스위칭 복원, 음성 기반 Q&A 쌍, 문맥 기반 ASR 바이어싱 용어, 자동 품질 점수가 포함된 번역을 생성합니다.

| 학습 목표 | 예시 전사 |
| --- | --- |
| 텍스트 정규화, verbatim | I paid, uh, twenty-five euros. |
| 역 텍스트 정규화, 비유창성 포함 | I paid, uh, €25. |
| 역 텍스트 정규화, 비유창성 제외 | I paid €25. |

표 1. 동일한 발화에 대한 Granary v2 학습 목표.

파이프라인은 재사용 가능한 네 단계로 구성됩니다.

- 1. Audio Preparation (Metadata): 원시 오디오 또는 장시간 오디오를 세그먼트로 나누고, 전체 녹음에서 화자를 추적하며, 음향 이벤트와 언어에 태그를 지정합니다. Granary v2의 소스는 이미 세그먼트로 나뉘어 있었으므로 이번 릴리스에서는 이 단계를 생략했습니다.

- 2. Transcription (ASR): 각 언어를 주요 ASR 모델과 복구 ASR 모델로 라우팅하고, 주요 모델이 환각 검사를 통과하지 못하면 복구 모델로 전환합니다. 사람이 작성한 참조 전사문을 활용해 신뢰할 수 있는 표기를 유지하면서 구어체 형식과 비유창성을 추가합니다.

- 3. Task Label Generation (Text): 전사 변형을 생성한 뒤 문자 비율, 환각 패턴, 모델 또는 사람 참조와의 일치 여부를 검사합니다. 승인된 레코드에만 문맥 기반 ASR 용어와 번역을 추가합니다.

- 4. High Fidelity (HiFi): 잡음이 많거나 클리핑이 발생했거나 여러 화자가 포함된 세그먼트를 필터링해 ASR 이외의 용도에 사용할 고품질 녹음의 소규모 세트를 유지합니다. VoxPopuli 데이터셋의 경우 354k시간을 22개 유럽 언어에 걸친 111k시간으로 줄였습니다.

[![fig6-granary-v2-pipeline-redesign Diagram of the Granary v2 curation pipeline following one roughly 12-minute sample from raw audio to a training target. The Metadata pipeline segments the audio with voice activity detection and diarization and identifies the language. The ASR pipeline transcribes a segment with a primary and a fallback model and keeps the best prediction, “Like, Granary-v2 is, uh, making every word matter.” The Text pipeline derives tn_raw, itn_raw, and itn_clean transcripts, plus translations, a caption, a speech Q&A pair, and context-ASR terms. The training view packs these into an instruction pair that maps “Transcribe without disfluencies” to “Granary-v2 is making every word matter.”](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/yORas7atvTqY3x-4BGaCb.png)](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/yORas7atvTqY3x-4BGaCb.png)

그림 6. Granary v2 데이터 큐레이션 파이프라인. 하나의 샘플이 원시 오디오에서 Metadata, ASR, Text 단계를 거쳐 학습 예제로 변환되는 과정을 보여줍니다.

개발자는 Ray 및 [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) 백엔드를 사용하는 [NeMo Curator pipeline](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/audio/granary_v2)를 자체 녹음이나 사용 권한이 있는 공개 코퍼스에서 실행하거나, [Granary v2 public manifests](https://huggingface.co/datasets/nvidia/Granary-v2)를 기반으로 구축할 수 있습니다. 매니페스트에는 위에서 설명한 레이블과 함께 세 가지 전사 형식이 tn_raw, itn_raw, itn_clean으로 포함되며, 엔터티 힌트, 생성된 지시 쌍, 출처 정보, 레코드별 품질 플래그도 포함됩니다. 소스 오디오는 각 코퍼스의 원래 라이선스에 따라 별도로 다운로드하고 릴리스에 게시된 약관을 따르세요.

# 중요한 어휘에 맞게 인식 조정하기

언어, 문자 체계, 전사 형식을 지정하는 개발 세트 프롬프트부터 시작하세요. 아래 사례 연구에서는 프롬프트에 영어와 라틴 알파벳을 명시해 전체 벤치마크 WER을 30.8%에서 23.8%로 낮췄습니다. 남아 있는 도메인 오류에 대해서는 NeMo Speech 파인튜닝 워크플로가 대표 음성과 명시적인 표기 목표를 결합합니다.

[NeMo Speech medical ASR case study](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/finetuning/docs/medical-asr-synthetic-data.md)는 인도 의사와 의대생의 녹음으로 구성된 [Eka Care’s open benchmark](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset)에서 이러한 진행 과정을 보여줍니다. 이 데이터셋에는 3,619개의 클립이 있으며 별도의 학습 분할은 지정되지 않았습니다. 이 연구는 생성된 의료 텍스트와 공개 의학 카탈로그로 조정 데이터를 만든 다음, 약 153시간의 합성 음성, Common Voice의 억양이 있는 영어 29시간, MultiMed의 의료 녹음 63시간을 결합합니다.

[![fig7-eka-care-62pct-rebrand Bar chart titled “62% fewer transcription errors” for the Eka Care medical ASR benchmark of Indian medical speech. On the full test set, word error rate falls from 30.8% out of the box to 23.8% with a better prompt that names English and the Latin alphabet, and to 11.8% after fine-tuning. Side labels note about 14 GPU-hours on one NVIDIA GB300 and about 153 hours of synthetic speech plus 92 hours of public accented and medical speech](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/Dq54q2P9bYJwWZ7prznAT.png)](https://cdn-uploads.huggingface.co/production/uploads/68127936de624fb6f57d9989/Dq54q2P9bYJwWZ7prznAT.png)

그림 7. Eka Care의 인도 의료 음성 벤치마크 전체 테스트 세트에서 더 나은 프롬프트를 사용하면 WER이 30.8%에서 23.8%로 감소하고, 파인튜닝 후에는 11.8%까지 낮아집니다. 이는 기본 모델보다 전사 오류가 약 62% 적은 수치입니다. 전체 세트에는 프롬프트와 체크포인트를 선택하는 데 사용된 855개의 튜닝 클립이 포함되어 있으며, 아래의 보류 결과가 더 엄격한 테스트입니다.

합성 생성은 표기 목표와 음성 표현을 분리합니다. “HbA1c”는 레이블로 사용하고 “H B A one C”는 합성기로 전달합니다. 여러 TTS 음성을 사용해 오디오의 다양성을 넓히고, 실제 녹음으로 억양과 녹음 환경을 추가합니다. 학습 텍스트는 벤치마크 참조와 정확히 일치하는지, 8단어 연속 구간을 공유하는지 기준으로 필터링합니다.

Granary v2의 명시적 목표와 마찬가지로 학습 예제에서는 숫자, 단위 간격, 회사명 확장, 브랜드명 하이픈 표기를 다양하게 바꾸고, 지시문으로 필요한 규칙을 지정합니다. 일부 프롬프트에는 거의 맞지만 틀린 방해 용어가 포함된 용어 목록이 있어, 모델은 오디오가 뒷받침할 때만 특정 표기를 사용하도록 학습합니다. 최종 시스템은 하나의 고정 규칙 프롬프트와 한 번의 모델 실행을 사용하며, 검색이나 전사 후처리는 수행하지 않습니다.

이 레시피는 음성 인코더와 프로젝션을 학습하고 LoRA를 사용해 언어 모델을 조정합니다. 두 번의 실행에서는 서로 다른 데이터 혼합을 사용하며, 어댑터를 병합한 후 선택된 체크포인트를 균등하게 평균 내 하나의 최종 모델을 만듭니다. 체크포인트 선택은 학습 손실만이 아니라 디코딩한 개발 세트 결과를 사용합니다. 텍스트 생성부터 평가까지 전체 조정 과정에는 NVIDIA GB300 한 장에서 약 14 GPU-시간이 걸렸습니다.

## 보류 화자를 대상으로 측정한 도메인 성능 향상 {#section-2}

이 연구는 벤치마크를 화자 기준으로 나누어 프롬프트와 체크포인트 선택에 사용한 855개의 튜닝 클립과 최종 시스템별로 한 번씩 디코딩한 2,764개의 보류 클립으로 구성했습니다. Eka Care의 [KARMA scoring pipeline](https://github.com/eka-care/KARMA-OpenMedEvalKit)를 사용했을 때 파인튜닝으로 보류 WER이 최적 프롬프트 기준선의 22.44%에서 12.17%로 감소했습니다. 이는 상대적으로 45.8% 감소한 수치이며, 새로운 화자에게도 성능 향상이 일반화된다는 가장 명확한 근거입니다. 키워드 WER은 9.2%에서 6.5%로, 의미 WER은 10.0%에서 7.6%로 낮아졌습니다. 동일한 보류 화자와 평가기를 기준으로, 조정된 모델의 WER 12.17%는 오픈 의료 전문 모델(13.79%)과 널리 사용되는 범용 ASR 모델(14.92%)보다 낮았습니다.

일반 영어에서는 LibriSpeech test-clean WER이 1.54%에서 1.69%로 소폭 상승했습니다. 의료 전문 모델은 브랜드 의약품과 진단명에 대한 정확한 일치에서는 여전히 앞섰습니다. 이 연구는 임상의가 브랜드명을 말하는 실제 녹음을 다음 단계로 제시합니다. 자체 도메인에 적용할 때는 체크포인트 선택을 위해 실제 레이블이 지정된 클립을 최소 100개 확보하고, 일반 음성 성능 유지와 함께 애플리케이션이 의존하는 엔터티 유형을 평가하세요.

이 사례 연구에는 각 단계를 재현하는 명령이 포함되어 있습니다. [fine-tuning tutorials](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/speechlm2/finetuning/README.md)에는 학습, 내보내기, 평가를 위한 레시피, 리소스 요구 사항, 도구가 포함되며 새로운 언어와 코드 스위칭 ASR도 다룹니다. 함께 제공되는 [agent skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/finetune-speech-llm/SKILL.md)를 사용하면 코딩 에이전트가 기준선 선택과 데이터 검사부터 학습 검증, 어댑터 병합, 평가까지 워크플로를 실행할 수 있습니다.

ModelScope를 사용하면 개발자가 특수한 멀티모달 애플리케이션을 위해 [ms-swift framework](https://github.com/modelscope/ms-swift)를 통해 Nemotron 3.5 Transcribe를 파인튜닝할 수도 있습니다.

# Nemotron 3.5 Transcribe와 Granary v2 시작하기

[Hugging Face demo](https://huggingface.co/spaces/nvidia/nemotron-transcribe) 또는 [ModelScope Studios](https://modelscope.ai/studios/nv-community/Nemotron-3.5-Transcribe)에서 Nemotron 3.5 Transcribe를 사용해 본 다음, [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) 또는 [ModelScope](https://modelscope.ai/collections/nv-community/Nemotron-35-Transcribe)에서 체크포인트를 다운로드하세요. 자체 호스팅 전사를 시작하려면 [vLLM](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/vllm_serving.rst) 또는 [SGLang-Omni](https://sgl-project.github.io/sglang-omni/cookbook/nemotron_transcribe.html)를 이용하세요. 두 환경 모두 출시 첫날부터 지원을 제공합니다.

자체 호스팅을 생략하려면 [Baseten](https://www.baseten.co/library/nvidia-nemotron-35-transcribe/), [DeepInfra](https://deepinfra.com/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) 또는 Simplismart에서 호스팅 엔드포인트를 사용하세요.

도메인에 맞게 조정하려면 [Hugging Face](https://huggingface.co/datasets/nvidia/Granary-v2) 또는 [ModelScope](https://modelscope.ai/datasets/nv-community/Granary-v2)의 Granary v2 매니페스트와 [medical ASR case study](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speechlm2/finetuning/docs/medical-asr-synthetic-data.md)부터 시작하거나, [fine-tuning agent skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/finetune-speech-llm/SKILL.md)를 사용해 코딩 에이전트가 워크플로를 실행하도록 하세요.

모델 및 데이터셋

- 모델 체크포인트: [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B), [ModelScope](https://modelscope.ai/collections/nv-community/Nemotron-35-Transcribe)의 NVIDIA Nemotron 3.5 Transcribe | [Hugging Face](https://huggingface.co/spaces/nvidia/nemotron-transcribe), [ModelScope](https://modelscope.ai/studios/nv-community/Nemotron-3.5-Transcribe)의 인터랙티브 데모

- 호스팅 엔드포인트: [Baseten](https://www.baseten.co/library/nvidia-nemotron-35-transcribe/), [DeepInfra](https://deepinfra.com/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)

- 데이터셋 매니페스트: [Hugging Face,](https://huggingface.co/datasets/nvidia/Granary-v2) [ModelScope](https://modelscope.ai/datasets/nv-community/Granary-v2)의 Granary v2 매니페스트

- High Fidelity 하위 집합: [Hugging Face,](https://huggingface.co/datasets/nvidia/Granary-HiFi) [ModelScope](https://modelscope.ai/datasets/nv-community/Granary-HiFi)의 매니페스트 | [Space on Hugging Face](https://huggingface.co/spaces/nvidia/Granary-HiFi)

리포지터리 및 튜토리얼: [NVIDIA NeMo Speech GitHub](https://github.com/NVIDIA-NeMo/Speech) | [NeMo Curator Granary v2 Pipeline](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/audio/granary_v2) | [vLLM Deployment Guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/speechlm2/vllm_serving.rst) | [Speech Fine-Tuning Guide](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/speechlm2/finetuning/README.md) | [Fine-Tuning Agent Skill](https://github.com/NVIDIA-NeMo/Speech/blob/main/.claude/skills/finetune-speech-llm/SKILL.md)

파트너 블로그: [Agora](https://www.agora.io/en/blog/how-live-teaching-can-become-a-lasting-learning-resource/) | [Inflection AI](https://inflection.ai/blog/building-inflection-ais-multi-speaker-capabilities-with-nvidia-nemotron-speech-open-models) | [VoiceArena](https://voicearena.com/research/diarization-bench/diarization-asr-track-cpwer-results) | [Simplismart](https://simplismart.ai/blog/nvidia-nemotron-3-5-transcribe) | [DeepInfra](https://deepinfra.com/blog/nvidia-nemotron-3-5-transcribe-release)

로컬 AI: [What’s new in local AI on NVIDIA RTX PCs, laptops, and DGX Spark](https://blogs.nvidia.com/blog/local-ai-rtx-spark-laptops-updates-models-agents-games)
