---
layout: post
title: "Phonon TTS 자세히 살펴보기: 벤치마크를 뛰어넘는 고품질 온디바이스 음성 출시"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/phonon-closer-look/thumbnail.png
image: assets/images/blog/posts/2026-11-05-phonon-tts/thumbnail.png
authors:
  - user: olesiagradium
slug: "phonon-tts"
source_url: "https://huggingface.co/blog/phonon-tts"
source_published_date: "2026-11-05"
source_published_at: "2026-11-05T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [A Closer Look at Phonon TTS. Launching High-Quality On-Device Speech That Tops the Benchmarks.](https://huggingface.co/blog/phonon-tts)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/phonon-tts -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Phonon TTS 자세히 살펴보기: 벤치마크를 뛰어넘는 고품질 온디바이스 음성 출시

<video src="https://huggingface.co/datasets/olesiagradium/video/resolve/main/Benchmark%20First%20Place%20Highlight%202026-11-02%2010.46.mp4" controls autoplay muted loop playsinline width="100%"></video>

2주 전, 자체 디바이스에서 실행되는 소형 text-to-speech 모델 Phonon을 공개했습니다. 서버도 네트워크도 필요하지 않으며, 입력한 내용은 여러분의 곁에 그대로 남습니다.

지금까지의 반응에 매우 만족하고 있습니다. 그래서 이번 출시의 이야기를 전해드리려 합니다. Phonon을 만든 이유, Phonon이 할 수 있는 일, 함께 공개된 데모, 그리고 Hugging Face의 Open TTS Leaderboard에서 어떤 성과를 거뒀는지 살펴보겠습니다.

## 음성은 디바이스에 있어야 하는 이유 {#section-1}

대부분의 음성 AI는 클라우드에서 실행됩니다. 모든 문장이 서버를 오가는 왕복 과정을 거치죠. 반면 Phonon은 여러분의 디바이스에서 직접 실행됩니다. 텍스트는 디바이스에 남고, 음성은 네트워크 없이 작동합니다. 가중치도 공개했습니다. Phonon 31m은 Apache 2.0이므로 상업용 제품에서도 무료로 사용할 수 있습니다. 다른 모델들은 CC BY-NC이므로 비상업적 프로젝트에서 무료로 사용할 수 있습니다. 이 모델들로 상업용 제품을 만든다면 요금은 생성하는 음성의 양이 아니라 사용자 수를 기준으로 책정됩니다.

작은 디바이스에서 좋은 음성을 실행하는 일은 언제나 어려웠습니다. 모델을 작게 만들면 음질이 좋지 않았고, 음질을 높이면 모델이 디바이스에 들어가지 않았습니다. 우리는 두 문제를 모두 해결했습니다.

4월에 Phonon을 발표한 뒤, 다음 6개월 동안 파트너 및 초기 사용자들과 함께 모델을 발전시켰습니다. 이들의 피드백은 이번 출시의 방향을 잡는 데 큰 도움이 되었습니다. 모델은 더 작아졌고, 더 많은 언어를 추가했으며, voice cloning과 voice design을 통한 사용자 지정 음성도 지원하게 되었습니다.

Phonon은 [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) 계보를 바탕으로 하며, 이에 따라 오디오를 생성하는 데 flow matching을 사용합니다. 이는 다른 TTS 시스템에서 사용되는 모델링 기법과 매우 다릅니다. 덕분에 이렇게 작은 모델로도 이 수준의 품질을 구현할 수 있었지만, 모델을 학습하는 과정에서는 새로운 과제가 생겼습니다. 학습 초기에는 수치적으로 불안정했으며, 5개 언어 모두에서 품질 좋은 결과를 만들기 위해 여러 가지 기법이 필요했습니다.

## 출시한 내용 {#section-2}

Phonon은 약 65MB와 75MB의 두 가지 크기로 제공됩니다. 두 모델을 합치면 영어, 프랑스어, 독일어, 스페인어, 포르투갈어를 지원합니다. 100개가 넘는 기본 제공 음성이 있으며, [Gradium Studio](https://studio.gradium.ai/)에서 직접 음성을 만들 수도 있습니다. 또한 Phonon은 대부분의 다른 TTS 시스템이 자주 어려움을 겪는 전화번호, 날짜, 코드와 같은 중요한 요소도 처리합니다.

휴대폰, 노트북, 싱글보드 컴퓨터, 브라우저 탭에서 실행되므로 [지금 바로 들어볼 수 있습니다](https://huggingface.co/spaces/gradium/phonon). Phonon 31m은 영어를 지원하며 Apache 2.0이므로 상업용 제품을 포함해 자유롭게 기반으로 사용할 수 있습니다. Phonon 40m은 비상업적 라이선스에 따라 5개 언어를 모두 지원합니다.

## 데모와 첫 프로젝트 {#section-3}

[출시](https://gradium.ai/blog/release-phonon-open-weight)와 함께 친구들과 우리 팀이 만든 데모도 공개되었습니다. Wi-Fi를 끈 Apple Watch에서는 Phonon이 커피를 내리는 과정을 단계별로 안내합니다. 신호가 없는 iPhone에서는 운동 반복 횟수를 세며 계속할 수 있도록 격려합니다. Autonomous의 Lamp라는 로봇은 분류 모자를 쓰고 캐릭터에 맞춰 이야기합니다. 브라우저에서는 오프라인 공룡 러너 게임이 Rex가 진행하는 라디오 쇼로 변합니다. 플레이어의 자체 머신에서 음성이 재생되는 SF 게임과, 페이지마다 소리 내어 읽어주는 Proust도 있습니다.

Hugging Face 팀이 Transformers 통합을 구축했기 때문에 개발자들은 첫날부터 Transformers에서 Phonon을 사용할 수 있었습니다.

개발자들도 Phonon을 기반으로 만들기 시작했습니다. [ExecuTorch 변환](https://huggingface.co/executorch-community/Phonon-31M-EN-ExecuTorch-WebGPU)을 사용하면 브라우저 방문자의 GPU에서 Phonon을 실행할 수 있고, [Micdrop](https://micdrop.dev/docs/ai-integration/provided-integrations/phonon.md)은 Node.js 앱에 Phonon을 제공하며, [smallest-s2s](https://huggingface.co/spaces/Steveeeeeeen/smallest-s2s)와 같은 Spaces도 Phonon을 기반으로 구축됩니다.

Phonon은 출시 후 첫 2주 동안 200,000회 넘게 다운로드되었으며, 여전히 Hugging Face에서 트렌딩 중입니다.

## Hugging Face Open TTS Leaderboard에서 Phonon의 성과 {#section-4}

[Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard)는 Hugging Face가 만든 매우 새로운 프로젝트이며, 우리는 이를 매우 좋아합니다. [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-to-speech)에 9,000개가 넘는 TTS 모델이 있는 가운데, 이 리더보드는 개발자들이 한곳에서 오픈 모델을 비교하고 나란히 들어볼 수 있도록 해줍니다. 23개 언어에 걸쳐 모델은 **명료성**(*출력의 단어 오류율*), **속도**(*실시간 계수의 역수 및 첫 오디오까지 걸리는 시간*), **자연스러움**(*영어의 UTMOS 및 한국어의 KOMOS*)을 기준으로 평가됩니다. 고정 음성과 voice cloning을 평가하며, 후자의 경우에는 추가로 화자 유사성 지표도 사용합니다. 리더보드에는 Listen 탭도 있어 TTS 출력을 직접 편리하게 비교하고 가장 마음에 드는 결과에 투표할 수 있습니다.

명료성에서 Phonon은 [독일어](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard?languages=de)와 [스페인어](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard?languages=es)에서 1위를 차지했으며, 평가한 모든 언어에서 상위 3위 안에 들었습니다. Phonon보다 최대 100배 큰 모델들과 나란히 기록한 결과입니다.

속도에서는 Phonon의 진가가 더욱 드러납니다. 첫 오디오까지 걸리는 시간 기준으로 모델들이 [CPU 스트리밍 벤치마크](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard?streaming-cpu)에서 가장 높은 성능을 기록하기 때문입니다.

![Bar chart of the top 10 models by 95th-percentile time to first audio. Phonon 31m en is first at 28 ms and Phonon 40m en is second at 38 ms, ahead of Pocket TTS at 165 ms. Lower is better.](https://huggingface.co/datasets/olesiagradium/video/resolve/main/phonon-ttfa-chart.png)

## 추론 {#section-5}

Phonon을 위해 특별히 새로운 추론 프레임워크 [ptts](https://github.com/gradium-ai/xn-ptts)를 구축했습니다. ptts의 목표는 매우 다양한 디바이스에서 최고의 성능을 내는 것이며, 이를 위해 10개가 넘는 백엔드를 지원합니다. CPU(Intel, Arm, WebAssembly), GPU(CUDA, Vulkan, Metal, WebGPU), NPU(Apple Core ML, Snapdragon®)를 지원합니다.

Phonon은 참조 PyTorch 구현으로 🤗 Transformers v5.21부터도 사용할 수 있습니다. `ptts`가 디바이스에서 단일 스트림을 실행하는 반면, Transformers는 GPU 처리량을 목표로 합니다. 배치 생성과 전체 그래프 `torch.compile`를 지원하므로, 대규모 음성 데이터셋 합성 등에 사용할 수 있습니다. 네이티브 PyTorch로 구현되어 있어 Phonon을 자체 스택에 통합하거나 모델 자체를 연구하는 등, 그 위에 무엇이든 구축할 수 있는 기반이 됩니다.

Pocket TTS 계보에 속하므로 먼저 Pocket TTS를 Transformers에 추가했습니다. Phonon은 그 위에 구축된 [모듈식](https://huggingface.co/docs/transformers/modular_transformers) 모델로, 다른 부분만 설명하는 [약 200줄](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gradium_phonon_tts/modular_gradium_phonon_tts.py)의 코드로 구성됩니다.

Phonon은 오디오 잠재 표현을 자기회귀 방식으로 생성합니다. 각 단계에서 잠재 표현을 출력하고, 이를 다음 입력으로 다시 전달하며, 그에 대해 학습 손실을 계산합니다. 잠재 표현은 언어 모델에서 텍스트 토큰이 하는 역할처럼 모델의 입력이자 출력이며 타깃입니다. 따라서 오디오를 잠재 표현으로 매핑하는 코덱은 토크나이저 역할을 합니다. Transformers에서는 이를 다음과 같이 [오디오 토크나이저](https://huggingface.co/docs/transformers/model_doc/pocket_tts_audio_tokenizer)로 제공하며, processor가 텍스트 토크나이저와 나란히 관리합니다:

```py
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration

model_id = "gradium/phonon-31m-en"
processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
model = GradiumPhononTTSForConditionalGeneration.from_pretrained(model_id, device_map="auto")

print(f"{model.num_parameters():,}")  # 30,788,354
print(f"{processor.audio_tokenizer.num_parameters():,}")  # 10,305,313
```


음성은 녹음 파일이 아니라 화자 잠재 표현의 텐서입니다. Phonon은 오디오 토크나이저로 오디오를 직접 인코딩하지 않습니다. 이 토크나이저는 디코더 전용으로 제공되며, 음성은 미리 계산되어 제공됩니다: [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices).

```py
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

path = hf_hub_download("gradium/phonon-voices", "voices/LMDIYE0Oa377Qh6Z.safetensors")
voice = load_file(path)["speaker_wavs"][0].T  # (num_frames, 512)
```


프롬프트 레이아웃(음성 위치, 텍스트, 오디오를 여는 토큰)은 Transformers의 다른 멀티모달 모델과 마찬가지로 체크포인트에 chat template으로 함께 제공됩니다. 음성이 들어갈 위치에 빈 `audio` 항목을 넣어 표준 대화를 작성하면, 체크포인트가 달라져도 코드는 그대로 유지됩니다. 그러면 processor가 음성 플레이스홀더를 프레임마다 하나의 위치로 확장하고, 텍스트를 정규화 및 토큰화한 뒤 입력을 배치로 묶습니다:

```py
conversation = [
    {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Hello from Phonon."}]}
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    add_generation_prompt=True,
    processor_kwargs={"audio_prompt_latents": [voice]},
).to(model.device)
# prompt: <|audio_prompt|> × num_frames, "Hello from Phonon.", <|audio_bos|>
```


`generate`는 잠재 표현을 반환하며, processor가 이를 디코딩해 오디오로 변환합니다:

```py
outputs = model.generate(**inputs)  # audio_latents: (batch_size, num_frames, 32)
audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)
processor.save_audio(audio, "phonon.wav")
```


더 자세한 내용은 [문서](https://huggingface.co/docs/transformers/model_doc/gradium_phonon_tts) 🤗를 참고하세요!

## 다음 단계 {#section-6}

Phonon은 어떤 음성으로든 말할 수 있습니다. 화자의 동의를 받아 짧은 녹음에서 음성을 클론하거나, [Gradium Studio](https://studio.gradium.ai/)에서 일반 텍스트로 새로운 음성을 설명할 수 있습니다. 그런 다음 음성 파일을 다운로드해 Phonon에 불러오면 디바이스에서 완전히 실행됩니다.

또한 설계 파트너들과 함께 사용자 지정 Phonon 모델을 학습하고 있습니다. 메모리 예산이 빠듯하다면 모델을 더 작게 만들 수 있습니다. 한국어, 아랍어 또는 일본어가 필요하다면 해당 언어로 학습할 수도 있습니다. 제품에 고유한 어휘가 있다면 그 어휘를 기반으로 fine-tuning할 수도 있습니다. [연락해 주세요](https://gradium.ai/contact?subject=Sales). 필요한 사항을 알려주시면 됩니다.

Hugging Face 팀과 Phonon을 첫 2주 동안 사용해 주신 모든 분께 감사드립니다.

Phonon은 우리가 처음으로 공개한 오픈 릴리스입니다. [직접 사용해 보고](https://huggingface.co/spaces/gradium/phonon) 무엇을 만들었는지 알려주세요.
