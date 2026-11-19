---
layout: post
title: "Rumble-2-Flash: 40개 이상의 언어를 지원하는 오픈 0.5B 음성 모델"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/rumble-2-flash/thumbnail.png
image: assets/images/blog/posts/2026-11-18-rumble-2-flash/thumbnail.png
authors:
  - user: reson8-thomas
slug: "rumble-2-flash"
source_url: "https://huggingface.co/blog/rumble-2-flash"
source_published_date: "2026-11-18"
source_published_at: "2026-11-18T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Rumble-2-Flash: an open 0.5B speech model for 40+ languages](https://huggingface.co/blog/rumble-2-flash)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/rumble-2-flash -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Rumble-2-Flash: 40개 이상의 언어를 지원하는 오픈 0.5B 음성 모델

오늘 Apache 2.0 라이선스로 [Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash)를 오픈 소스로 공개합니다. 40개 이상의 언어를 지원하는 0.5B 파라미터 음성 인식 모델로, 이보다 3~60배 큰 모델들과 경쟁할 수 있습니다.

- **어디서나 실행**: 🤗 Transformers와 vLLM을 기본 지원하며, 사용자 정의 코드가 필요하지 않음
- **40개 이상의 언어**: 자동 언어 감지
- **출력 제어**: 숫자를 숫자 표기 또는 단어로 출력, 필러, 축어적 반복, 언어 강제 지정
- **극도로 빠름**: 약 0.5초 만에 1시간 분량의 오디오를 처리하며, 전체 배치에서 7,000 RTFx 초과
- **실제 전화 통화에서도 견고함**: 302시간 규모의 콜센터 벤치마크에서 NVIDIA의 Nemotron 3.5 Transcribe (30B)와 대등한 성능

## 빠른 시작 {#section-1}

<details>
<summary>Transcribe with 🤗 Transformers</summary>

```bash
uv pip install torch transformers accelerate
```


```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

model_id = "Reson8/Rumble-2-Flash"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

inputs = processor.apply_transcription_request(
    audio="https://hf.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    language="English",  # omit to detect automatically
).to(model.device, dtype=model.dtype)

output = model.generate(**inputs, max_new_tokens=512)
generated = output[:, inputs["input_ids"].shape[-1]:]
print(processor.decode(generated, return_format="transcription_only")[0])  # return_format="parsed" if detecting language
```


`audio`는 로컬 경로, URL, NumPy 배열 또는 이러한 항목의 목록을 입력으로 받습니다.

</details>

## 어떻게 만들었나 {#section-2}

Rumble-2-Flash는 출시 즉시 지원을 위해 Qwen3-ASR 아키텍처를 사용합니다. 스택에서 Transformers 또는 vLLM을 실행한다면, 별도의 포크된 런타임이나 `trust_remote_code` 없이 Rumble-2-Flash를 실행할 수 있습니다.

정확도가 떨어지기 전에 모델을 얼마나 작게 만들 수 있는지 알아보기 위해, 더 큰 프로덕션 모델인 Rumble-2를 온폴리시 증류 방식으로 Rumble-2-Flash에 증류했습니다. 학생 모델은 교사가 작성한 전사문으로 학습하지 않습니다. 학생 모델이 직접 오디오를 전사하고, 교사 모델은 해당 전사문을 토큰 단위로 평가합니다. 즉, 각 위치에서 다음에 나와야 할 토큰에 대한 자체 확률을 제시하고, 학생 모델은 이를 맞추도록 학습합니다. 학생 모델은 자체 출력에서 학습하므로, 클립당 하나의 점수가 아니라 모든 토큰에 대한 피드백을 통해 자신이 실제로 만드는 오류를 정확히 교정받습니다.

온폴리시 증류에는 일반적으로 교사 모델과 학생 모델이 동일한 토크나이저를 사용해야 하지만, 두 모델은 그렇지 않습니다. Qwen3-ASR을 선택한다는 것은 Qwen3-ASR의 어휘도 사용한다는 뜻이기 때문입니다. [GOLD](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)는 이러한 제약을 없앱니다. 두 토크나이저가 텍스트를 서로 다르게 분할하는 경우 토큰을 병합하여 두 모델이 동일한 텍스트를 기준으로 비교되도록 합니다. 두 어휘에 모두 존재하는 토큰은 직접 비교하고, 나머지 토큰은 정렬된 확률을 기준으로 비교합니다.

<figure class="image text-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/gold.png" alt="On-policy distillation with GOLD: the student transcribes, the teacher scores every token, and GOLD aligns the two tokenizers">
  <img class="hidden dark:block" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/gold-dark.png" alt="On-policy distillation with GOLD: the student transcribes, the teacher scores every token, and GOLD aligns the two tokenizers">
  <figcaption>On-policy distillation with GOLD</figcaption>
</figure>

## 결과 {#section-3}

### 모델 크기에 비해 높은 정확도

Open ASR Leaderboard의 [다국어](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard?languages=Reson8%2FRumble-2-Flash&proprietary=false) 트랙에서 Rumble-2-Flash는 중국어, 독일어, 프랑스어, 이탈리아어, 스페인어, 포르투갈어, 힌디어, 네덜란드어, 영어에 걸쳐 평균 **6.96% 오류율**을 기록합니다. 이는 15배 더 큰 모델인 Meta의 omniASR-LLM 7B보다 우수하며, Whisper large-v3보다도 앞섭니다. [영어](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) 트랙에서는 평균 **6.6% WER**를 기록합니다.

<figure class="image text-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/size-vs-wer-en.png" alt="Average error rate against model size for open-weight models on the Open ASR Leaderboard multilingual track, including English">
  <img class="hidden dark:block" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/size-vs-wer-en-dark.png" alt="Average error rate against model size for open-weight models on the Open ASR Leaderboard multilingual track, including English">
  <figcaption>The best score for its size</figcaption>
</figure>

### 속도

Rumble-2-Flash는 속도도 빠릅니다. 처리량은 배치가 얼마나 채워져 있는지에 따라 달라집니다. 배치가 계속 가득 차 있는 HF의 Open ASR Leaderboard 대규모 Datasets에서는 **7,000 RTFx 초과**에 도달합니다. 이는 약 0.5초 만에 1시간 분량의 오디오를 처리하는 속도입니다.

### 실제 전화 통화

깨끗한 오디오는 쉽습니다. 전화 통화는 압축되고 잡음이 많으며, 서로 억양이 매우 다른 화자가 참여하는 경우도 많습니다. [AppTek의 콜센터 벤치마크](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues)(헬프라인 통화 302시간, 영어 억양 14종, AppTek 자체 스크립트로 평가)에서 Rumble-2-Flash는 **7.86% WER**를 기록해 7.78%인 NVIDIA의 Nemotron 3.5 Transcribe (30B)와 대등한 성능을 보입니다. 이는 같은 모델 계열의 Qwen3-ASR 0.6B(8.7%)보다 훨씬 우수하며, 비교 대상 중 더 나은 성능을 보이는 유일한 모델인 Qwen3-ASR 1.7B(7.4%)에 근접한 결과입니다.

<figure class="image text-center">
  <img class="block dark:hidden" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/apptek.png" alt="WER on the AppTek call-centre benchmark">
  <img class="hidden dark:block" src="https://huggingface.co/datasets/Reson8/public-media/resolve/main/blog/rumble-2-flash/apptek-dark.png" alt="WER on the AppTek call-centre benchmark">
  <figcaption>Real phone calls, 14 English accents</figcaption>
</figure>

## 사용해 보기 {#section-4}

가중치는 Hugging Face Hub의 [Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash)에서 확인할 수 있습니다. 또한 저희 [API](https://www.reson8.dev)를 통해서도 사용할 수 있습니다.

여러분이 이 모델로 무엇을 만들지 정말 궁금합니다. 무엇을 만들었는지, 어떤 점이 마음에 드는지, 다음 버전에서 무엇이 바뀌기를 원하는지 [Discussions 탭](https://huggingface.co/Reson8/Rumble-2-Flash/discussions)에서 알려 주세요.
