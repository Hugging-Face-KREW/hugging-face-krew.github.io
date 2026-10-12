---
layout: post
title: "음성 인식에서 벤치마크 최적화 측정"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/asr-benchmark-optimization/thumbnail.png
image: assets/images/blog/posts/2026-08-21-asr-benchmark-optimization/thumbnail.png
authors:
  - user: tlebryk02
slug: "asr-benchmark-optimization"
source_url: "https://huggingface.co/blog/asr-benchmark-optimization"
source_published_date: "2026-08-21"
source_published_at: "2026-08-21T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Measuring benchmark optimization in speech recognition](https://huggingface.co/blog/asr-benchmark-optimization)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/asr-benchmark-optimization -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 음성 인식에서 벤치마크 최적화 측정

공개된 음성 인식 벤치마크는 점점 더 모델이 인간 수준의 성능을 보이고 있다는 신호를 시사한다. 그러나 이러한 점수는 모델이 실제 세계에서 어떻게 작동하는지 항상 반영하지 않는다. 공개 벤치마크가 열려 있고 널리 사용되기 때문에, 모델이 테스트 자체에 맞춰 최적화될 수도 있다. 그들의 점수는 벤치마크 특유의 패턴을 학습했기 때문이 아니라 근본적인 과제를 더 잘 수행하게 되었기 때문일 수 있다.

전통적인 벤치마크가 음성 시스템을 실세계에서 신뢰할 수 있게 만드는 조건과 특성을 충분히 반영하지 못하기 때문입니다. 그래서 우리는 최근 [Real World VoiceEQ](https://huggingface.co/spaces/HumeAI/rw-voice-eq)의 보류 평가셋, [Open-ASR Leaderboard](https://huggingface.co/blog/open-asr-leaderboard-private-data) 및 [Far-field ASR Leaderboard](https://huggingface.co/spaces/treble-technologies/ffasr)를 도입했습니다: 실제 사용에서 더 중요한 것을 더 많이 측정하기 위함입니다.

하지만 더 넓은 측정만으로 문제를 해결하는 것은 아니다. 이 현상은 때때로 벤치마크 최적화 또는 "benchmaxxing"이라고 불리며 머신러닝 분야에서 자주 논의되지만, 음성 인식에서 이를 측정하는 데는 어려움이 있었다.

최신 연구는 이를 정량화하는 데 도움이 되는 세 가지 테스트를 도입했다. 우리는 널리 사용되는 11종의 오픈 소스 ASR 모델을 평가했고, 여러 고득점 시스템이 [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) 영어 및 [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr)(clean, other) 데이터셋의 벤치마크 트랜스크립트를 재현했다는 것을 발견했다 — 오디오가 이를 반박했고, 관련 단어가 제거되었거나, 오디오가 두 가지 서로 다른 표기를 균등하게 지지했을 때도 말이다.

일부 경우 모델은 말한 내용뿐 아니라 테스트 중인 벤치마크를 가리키는 미묘한 음향 신호에도 의존하는 것으로 보였다. 그 결과 점수는 일반적으로 음성을 더 충실히 전사하는 능력에 대해 과대평가될 수 있다.

## 참조 불일치( VoxPopuli 사례 연구) {#section-1}

VoxPopuli는 다수의 전사 오류를 포함하는 것으로 알려져 있다(그래서 Artificial Analysis가 [cleaned version](https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA)를 발표했다). 우리의 합의 불일치 프로브는 선도하는 ASR 모델들이 이러한 오류에 직면했을 때 어떤 일이 벌어지는지 테스트한다: *오디오가 실제로 말하는 내용을 정확히 전사하는가, 아니면 벤치마크의 잘못된 참조 전사를 재생산하는가?*

이를 대규모로 테스트하기 위해, 음소 오류율(PER)이 낮은 독립적 모델들로 구성된 앙상블을 사용한다. PER은 작성된 전사가 오디오의 소리와 얼마나 가깝게 일치하는지 측정하여 모델이 들은 것을 얼마나 충실히 전사하는지에 대한 유용한 대리 지표가 된다. 앙상블의 결과는 모델들이 벤치마크의 참조 전사와 만장일치로 동의하지 않는 사례를 표시하는 데 사용할 수 있다. 그런 표시된 사례들 중 일부를 인간 주석과 비교하여 수정된 전사를 검증한다.

예를 들어, VoxPopuli 클립 중 하나는 "Thank you, Mr. President" 구를 소리로 포함하지만 참조 전사는 "Thank you"를 생략한다. 우리가 테스트한 11개 모델 중 여섯 개는 벤치마크의 잘못된 전사를 재현했다 — 오디오와 모순되더라도 "예상된" 정답을 제시했다. 실제 클립에서도 형식은 같은 패턴을 따른다: "Thank you"를 생략한 모델은 또한 벤치마크의 구두점 스타일을 재현하고, "Mr"에 마침표가 없고, 들리는 구절을 포함하는 모델은 보통 "Mr."에 마침표를 찍는 경향이 있다.

유럽 의회의 녹음이나 일반 음성으로 새로 수집된 음성을 제시하면 이 행동은 종종 약해지거나 사라지는 경우가 많다. 아래 샘플에서는 한 모델을 제외한 모든 모델이 새 의회 녹음의 클론에 대해 음성에 충실한 전사로 되돌아간다. 이는 모델이 벤치마크의 구성원을 식별하는 데 도움이 되는 음향 단서에 반응하여 오디오와 모순되더라도 기대된 전사를 생성한다는 것을 시사한다.

이 클립의 참조 전사는 "Mr President, I have another complaint about this procedure, which is that it is not secret." 이 아래의 세 클립의 오디오도 실제로는 같은 내용을 말하며 앞에 들리는 "Thank you,"—클론은 그 참된 문장의 음성 합성 버전이므로 세 클립 모두 예의 표현이 들린다. 초록 하이라이트와 ✅ 표시는 들리는 "Thank you"를 포함하는 전사를; 빨간 하이라이트와 ❌ 표시는 벤치마크의 잘못된 생략을 재현하는 전사를 나타낸다. 모든 전사는 정규화되기 전의 원시 모델 출력으로, 대소문자 및 구두점은 생성된 그대로 보존되며, 일부 모델의 소문자 출력도 포함된다.

**원본 VoxPopuli 녹음**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/1287_real.wav"></audio>

**같은 화자의 음성 클론**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/1287_clone_same_speaker.wav"></audio>

**모델 학습 컷오프 이후에 녹음된 의회 연설자 클론**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/1287_clone_ep_fresh.wav"></audio>

| 모델 | 실제 클립 | 동일 화자 클론 | ep-fresh 클론 |
| --- | --- | --- | --- |
| [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr President…</span> |
| [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#dcfce7">✅ Thank you Mr. President…</span> |
| [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | <span style="background-color:#fee2e2">❌ mr president…</span> | <span style="background-color:#fee2e2">❌ mr president…</span> | <span style="background-color:#dcfce7">✅ thank you mr president…</span> |
| [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#fee2e2">❌ Mr President…</span> |
| [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | <span style="background-color:#fee2e2">❌ Mr President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> |
| [bosonai/higgs-audio-v3-8b-stt-v2](https://huggingface.co/bosonai/higgs-audio-v3-8b-stt-v2) | <span style="background-color:#fee2e2">❌ mr president…</span> | <span style="background-color:#fee2e2">❌ mr president…</span> | <span style="background-color:#dcfce7">✅ thank you mr president…</span> |
| [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mister President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mister President…</span> |
| [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> |
| [moonshotai/Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) | <span style="background-color:#dcfce7">✅ Thank you, mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, mr. President…</span> |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> | <span style="background-color:#dcfce7">✅ Thank you, Mr. President…</span> |
| [moonshine-ai/moonshine-streaming-medium](https://huggingface.co/moonshine-ai/moonshine-streaming-medium) | <span style="background-color:#dcfce7">✅ thank you mr president…</span> | <span style="background-color:#dcfce7">✅ thank you mr president…</span> | <span style="background-color:#dcfce7">✅ thank you mr president…</span> |
| **11개 중 예의를 제외** | **6** | **5** | **1** |

Parakeet은 실제 클립에서 벤치마크를 재현하는 것과 같은 화자 클론에서 정확하게 재현하는 것 사이를 오가는 유일한 모델이다. Phi-4는 ep-fresh 클론에서 여전히 예의를 생략하는 유일한 모델이다. 대신 의회 녹음과 연결되지 않은 일반 TTS 보이스로 문장을 재합성하면 모든 열한 모델이 예의를 회복한다.

이 결과는 이 문제가 널리 퍼져 있으며 의미가 있음을 시사한다. 우리의 방법론은 분석한 VoxPopuli 테스트 클립의 40%에서 참조 오류 가능성을 표시했고, 전체 참조 단어의 약 3%에 영향을 미쳤다.

벤치마크 최적화 현상을 보인 모델은 잘못된 참조 전사를 18–30%의 시점에서 재현했다. 아래 산점도는 x축에 VoxPopuli의 단어 오류율(WER)을 두고 각 모델이 합의 수정 대신 벤치마크의 잘못된 참조를 재현하는 비율을 비교한다. 가장 낮은 WER를 보이는 모델들—따라서 가장 강하게 보고되는 벤치마크 성능—도 이러한 오류를 재현할 가능성이 가장 높다.

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/wer_vs_badref.png" width="800px" alt="Scatterplot comparing VoxPopuli WER to the rate at which each model reproduces the benchmark's incorrect reference transcript." />
</div>

## 마스킹된 엔티티 검색 {#section-2}

합의 불일치 탐침을 바탕으로, 테스트 데이터셋의 오디오 샘플에서 숫자를 의도적으로 침묵시키고 모델이 들리는 것을 전사하도록 한다. 숫자는 오디오에서 문자 그대로 없기 때문에, 모델은 아무 숫자도 출력하지 말아야 하며, 텍스트의 정확한 숫자까지도 출력하지 않아야 한다.

이 숫자들 중 일부는 부분적으로 예측 가능하지만(모델이 예측하기는 어렵다), 다른 숫자는 꽤 놀랍다. 아래 클립은 두 탐침을 모두 결합하여, 모델이 참조 전사 오류를 재현하는 방식과 한 모델이 침묵된 연도(2011)조차 자동완성하는 사례를 모두 보여준다. 각 모델의 아래 행에서:

- 초록 하이라이트에 취소선은 모델이 정확히 재현하지 않은 참조 전사를 나타낸다(음향에 충실함);
- 초록 하이라이트에 <u>밑줄</u>은 참조의 잘못된 표현을 대체한 올바르고 음향에 충실한 삽입을 표시;
- 빨간 하이라이트(일반 텍스트)는 참조 전사의 잘못되고 음향에 의해 뒷받침되지 않는 내용을 재현한다: 예를 들어 "Mr President"를 유지하고, 음성에서 말하는 "one thousand six hundred" 대신 "more than 1 amendments"를 쓰며, 침묵된 연도 "2011"을 제공하거나, "plenary"로 끝내는 것 등.

**2011 초안 예산(마스킹된 숫자)**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/2011_draft_budget.wav"></audio>

| 참조 | <span style="background-color:#fee2e2">Mr President,</span> in the Committee on Budgets, we voted on more than <span style="background-color:#fee2e2">1</span> amendments to the <span style="background-color:#fee2e2">2011</span> draft budget … voted in the <span style="background-color:#fee2e2">plenary</span>. |
| --- | --- |
| What the audio says | In the Committee on Budgets, we voted on more than <span style="background-color:#dcfce7"><u>one thousand six hundred</u></span> amendments to the ⟨silenced⟩ draft budget … voted in the … |
| [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | <span style="background-color:#fee2e2">Mr President,</span> in the Committee on Budgets we voted on more than <span style="background-color:#fee2e2">1</span> amendments to the <span style="background-color:#fee2e2">2011</span> draft budget … voted in the <span style="background-color:#fee2e2">plenary</span>. |
| [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | <span style="background-color:#fee2e2">Mr President,</span> in the Committee on Budgets we voted on more than <span style="background-color:#fee2e2">one</span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | <span style="background-color:#dcfce7">~~Mr President~~</span> in the committee on budgets we voted on more than <span style="background-color:#dcfce7"><u>one thousand six hundred</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted on in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | <span style="background-color:#dcfce7">~~Mr President~~</span> In the Committee on Budgets we voted on more than <span style="background-color:#fee2e2">1</span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted on in the <span style="background-color:#fee2e2">plenary</span>. |
| [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | <span style="background-color:#dcfce7">~~Mr President~~</span> In the Committee on Budgets we voted on more than <span style="background-color:#fee2e2">one</span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted in the Protestants. |
| [bosonai/higgs-audio-v3-8b-stt-v2](https://huggingface.co/bosonai/higgs-audio-v3-8b-stt-v2) | <span style="background-color:#dcfce7">~~Mr President~~</span> in the committee on budgets we voted on more than <span style="background-color:#dcfce7"><u>one thousand six hundred</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | <span style="background-color:#dcfce7">~~Mr President~~</span> In the Committee on Budgets, we voted on more than <span style="background-color:#dcfce7"><u>1,600</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | <span style="background-color:#dcfce7">~~Mr President~~</span> In the Committee on Budgets, we voted on more than <span style="background-color:#dcfce7"><u>1,600</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| [moonshotai/Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) | <span style="background-color:#dcfce7">~~Mr President~~</span> <span style="background-color:#dcfce7"><u>Ah</u></span> in the committee on budgets we voted on more than <span style="background-color:#dcfce7"><u>one thousand six hundred</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | <span style="background-color:#dcfce7">~~Mr President~~</span> In the Committee on Budgets, we voted on more than <span style="background-color:#dcfce7"><u>1,600</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |
| [moonshine-ai/moonshine-streaming-medium](https://huggingface.co/moonshine-ai/moonshine-streaming-medium) | <span style="background-color:#dcfce7">~~Mr President~~</span> in the committee on budgets we voted on more than <span style="background-color:#dcfce7"><u>one thousand six hundred</u></span> amendments to the <span style="background-color:#dcfce7">~~2011~~</span> draft budget … voted in the <span style="background-color:#dcfce7">~~plenary~~</span> |

복구율은 공개 벤치마크에서 가장 높고 보류 평가셋 또는 최근 수집된 오디오(ep-fresh 및 libri-fresh 아래)에서는 낮았다. LibriSpeech에서 가장 강력한 벤치마크 성능을 보인 일부 모델은 숫자가 제거되었음에도 대략 30–40%의 예시에서 마스킹된 숫자를 재현했다. 여러 모델의 신규 수집 데이터에서 그 효과가 약화되었으며, 이는 벤치마크 연관 오디오가 단순한 텍스트 자동완성뿐만 아니라 참조를 회복하는 데 도움을 주었음을 시사한다.

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/masking_freshpairs.png" width="700px" alt="Recovery rate of masked numbers on public benchmarks versus freshly collected held-out audio." />
</div>

## 정자 표기 전환 {#section-3}

우리의 정자 표기 전환 탐침은 모델이 음향에서 명확하지 않더라도 벤치마크의 참조 전사에서 사용된 정확한 철자를 재현하는지 여부를 테스트한다. 정자 표기 대안은 의미적으로 같고 음성적으로도 동일한 단어들이 서로 다르게 표기되는 경우를 말한다(1 vs one, Mr. vs mister, John vs Jon, Honor vs Honour 등). 이론적으로 모델은 한 철자를 일관되게 선호하거나, 평균적으로 무작위로 번갈아가야 한다. 각 벤치마크의 참조 전사에 맞춰 특정 철자를 사용하도록 체계적으로 바뀌는 경우가 있다면, 테스트가 어떤 철자 표기를 벤치마크가 기대하는지 모델이 파악하고 있는 것을 시사한다.

**전사: "I URGED ON THE BOYS THAT WHATEVER HAPPENED WE SHOULD NOT SHOOT ANY ONE" — 모델이 "any one"을 사용하는 경우: 6/11, "anyone"을 사용하는 경우: 5/11**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/any_one.wav"></audio>

**전사: "CAMOUFLAGE WAS NOT A WORD THE CAPTAIN OR ANYONE ELSE OF HIS TIME YET UNDERSTOOD" — 모델이 "any one"을 사용하는 경우: 2/11, "anyone"을 사용하는 경우: 9/11**

<audio controls src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/anyone.wav"></audio>

리브리스피치 내에서 오래된 띄어쓰기 규칙을 포함하는 하나의 *데이터셋 내(intra-dataset)* 전환을 테스트한다: 일부 참조 전사는 "any one"을 사용하고, 다른 것은 "anyone"을 사용한다. 주어진 변형의 최소 정확도를 측정하며 이를 "전환율"이라고 부른다. 한 변형만 사용하는 모델은 0% 전환율을 가지며, 임의로 선택하는 모델은 50%의 전환율을 기대한다. 모든 테스트 샘플에서 어떤 변형을 사용할지 아는 모델은 100%의 전환율을 달성한다.

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/pair_spacing_sorted.png" width="700px" alt="Switch rate for the &quot;any one&quot; vs &quot;anyone&quot; spacing convention, sorted by model." />
</div>

## 인터-데이터셋 전환 {#section-4}

다수의 모델이 50%의 무작위 선택 기준치를 넘었으며, 일부는 대략 90%의 전환 정확도에 도달했다. **이것은 모델이 오디오 샘플이 어느 데이터셋에서 왔는지 식별하고 벤치마크가 기대하는 철자 표기를 선택할 수 있다는 것을 시사한다, 두 형태의 발음이 동일하게 들리지만 말이다.**

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/pair_mister_sorted.png" width="700px" alt="Switch rate for the &quot;Mr.&quot; vs &quot;Mister&quot; convention across VoxPopuli and LibriSpeech, sorted by model." />
</div>

## 전환 위치 파악 {#section-5}

이러한 행동이 공개 벤치마크를 넘어 일반화되는지 확인하기 위해, 모델의 학습 컷오프 이후 같은 소스 도메인에서 새로 수집된 데이터도 수집했다: VoxPopuli의 최근 유럽 의회 녹음과 LibriSpeech의 새롭게 활동하는 LibriVox 내레이터의 녹음이다. 그러나 같은 도메인에서 최근 수집된 데이터를 제시하면 많은 모델이 참조 전사와 일치하지 않고 더 음성에 충실한 전사로 돌아가 버린다.

다른 개입도 같은 결론을 가리킨다. 오디오에 존재하지만 참조 전사에서 생략된 구절은 모델에 오디오를 번역하라고 요청하거나 관련 프레임에 주의가 제한될 때 다시 나타날 수 있다. 주변 벤치마크 맥락을 제거하거나 일반 대화 음성을 추가하는 것도 충실한 전사를 복원시킬 수 있다. VoxPopuli 오디오를 추가하면 반대 효과가 있을 수 있어, 그렇지 않게 충실한 합성 또는 채굴된 샘플이 벤치마크 참조와 더 잘 일치하게 만들 수 있다.

<div align="center">
  <img src="https://huggingface.co/datasets/HumeAI/hf-assets/resolve/main/blog/asr-benchmark-optimization/steer_input_level_full.png" width="800px" alt="Effect of steering the amount of surrounding benchmark-associated audio context on transcription behavior." />
</div>

**함께 보면, 모델은 문자 그대로의 말소리를 충실히 전사할 수 있지만, 오디오를 따를지 벤치마크별 전사 정책을 따를지 결정하기 위해 주변 음향 맥락을 사용한다는 결론에 이른다.**

## 결론 {#section-6}

우리의 발견은 두 개의 주요 오픈 소스 데이터셋에서 일부 모델이 데이터셋 연관 음향 단서를 감지하고 그에 따라 전사 동작을 조정할 수 있음을 시사한다. 구체적으로, 모델은 오디오에 없지만 참조 전사에 존재하는 단어를 재현하거나, 침묵된 숫자를 더 높은 비율로 회복하거나, 특정 벤치마크가 기대하는 표기 형태를 선택하기 위해 주변 음향 맥락을 사용할 수 있다.

모델을 선택하는 사람들에게 이 발견은 RW-Voice-EQ Bench와 Open ASR Leaderboard가 하듯 완전히 보류된 평가 셋을 사용하는 것의 중요성과, 하나의 공개 벤치마크에서의 단어 오류율만을 보는 관행 너머를 살피는 것의 중요성을 강조한다. 이를 위해 [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)에 "벤치마크 적합(Benchmark fitting)" 탭이 추가되었으며, 위의 분석 중 두 가지를 모든 모델에 대해 포함한다: (1) VoxPopuli의 참조 오류율 측정 및 (2) 모든 공개 데이터셋에서의 정자 표기 전환. 관련 스크립트는 [GitHub](https://github.com/huggingface/open_asr_leaderboard/tree/main/benchmark_fitting) 및 [un-normalized model outputs](https://huggingface.co/buckets/hf-audio/asr_leaderboard_h200)에서도 오픈 소스로 제공된다.

또한 우리의 발견은 벤치마크 개발자들이 단순한 독립적이고 동일분포하는 테스트 분할보다는 시간적, 화자 또는 기타 메타데이터 기반의 분리를 피할 것을 시사한다. 학습 데이터 및 모델 선발 절차에 대한 더 큰 투명성은 연구자들이 이러한 행동이 어떻게 발생하는지 이해하는 데 도움이 될 것이다.

공개 벤치마크는 여전히 가치가 있다: 투명하고, 재현 가능하며, 실행이 쉽고 연구 커뮤니티가 잘 이해한다. 그러나 새로운 오디오에 일반화되지 않는 벤치마크 특유의 이득과 실제 전서 개선을 구분할 수 있을 때 가장 유용하다.

자세한 정보는 저희의 [full report](https://huggingface.co/papers/2608.19936)를 읽어보시길 권장합니다.
