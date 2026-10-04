---
layout: post
title: "Reachy Mini를 로컬에서 완전히 실행하기"
author: KREW
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/local-reachy-mini-conversation/thumbnail.png
image: assets/images/blog/posts/2026-05-27-local-reachy-mini-conversation/thumbnail.png
authors:
  - user: A-Mahla
  - user: andito
slug: "local-reachy-mini-conversation"
source_url: "https://huggingface.co/blog/local-reachy-mini-conversation"
source_published_date: "2026-05-27"
source_published_at: "2026-05-27T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Reachy Mini goes fully local](https://huggingface.co/blog/local-reachy-mini-conversation)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/local-reachy-mini-conversation -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Reachy Mini를 로컬에서 완전히 실행하기

Reachy Mini를 만든 후에는 [conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app)를 설치하고 대화를 시작합니다. 지금까지는 오디오를 서버로 전송해야 했지만, 이제는 다릅니다. 오늘은 전체 스택을 로컬에서 실행하는 방법을 안내합니다.

이 스택은 [`speech-to-speech`](https://github.com/huggingface/speech-to-speech)로 구동됩니다. 이는 VAD → STT → LLM → TTS 파이프라인을 계층적으로 연결한 것으로, Realtime API-호환 `/v1/realtime` WebSocket을 노출합니다. 백엔드를 시작하면 UI에서 로봇이 그 백엔드를 가리키도록 설정합니다.

캐스캐이드는 오늘날 오픈 소스 생태계에서 가장 유연한 옵션이며, 올바른 구성 요소를 사용하면 가장 빠릅니다. 우리는 우리가 가장 좋아하는 구성 요소를 추천하겠지만, 캐스캐이드는 그것들을 교체할 수 있다는 점이 핵심입니다. 매주 새로운 모델이 출시됩니다.

> **TL;DR**
> - Reachy Mini용 로컬 음성 백엔드 배포합니다.
> - 우리는 `speech-to-speech` 라이브러리, 캐스캐이드 방식입니다.
> - 권장: **llama.cpp**와 **Gemma 4**, **Silero VAD**, **Parakeet-TDT 0.6B v3 STT**, **Qwen3-TTS**.

---

## 빠른 시작 {#section-1}

이 블로그는 Reachy Mini와의 대화를 완전히 로컬에서 실행하는 방법을 안내합니다. 클라우드도, API 키도, 데이터가 기기에서 벗어나지 않습니다. 아래는 이를 실시간으로 보여주는 영상입니다:

<video controls width="360" style="display: block; max-width: 100%; height: auto; margin: 0 auto;">
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/local_reachy_mini_conversation/Reachy mini local.mp4" type="video/mp4">
</video>

### 로컬에서 LLM 서빙

LLM을 서빙하려면 Hugging Face의 `llama.cpp`를 사용할 예정입니다. 설치가 필요하면 가장 간단한 방법은 `brew install llama.cpp` 또는 `winget install llama.cpp`이며, 추가 도움은 [check the docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md)를 참조하십시오. 먼저 다음을 실행합니다:
```bash
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF -np 2 -c 65536 -fa on --swa-full
```

완료! 처음 실행 시 모델을 다운로드하지만, 이후 실행은 빠릅니다.

<details>
<summary>What do those flags do?</summary>

- `-hf ggml-org/gemma-4-E4B-it-GGUF` — 모델을 `-hf ggml-org/gemma-4-E4B-it-GGUF`에서 직접 가져옵니다. 첫 실행은 다운로드하고, 이후 실행은 캐시를 사용합니다.
- `-np 2` — 두 개의 병렬 슬롯. 서버가 두 번째 요청을 처리하게 해 주며, 첫 번째 요청이 차단되지 않습니다.
- `-c 65536` — 64k 컨텍스트 윈도우, 슬롯 간 공유. 긴 대화에 충분한 여유 공간이 있습니다.
- `-fa on` — 플래시 어텐션. 더 빠르고 메모리 사용량이 적으며, 현대 하드웨어에서 사실상 비용이 없습니다.
- `--swa-full` — 전체 슬라이딩 윈도우 어텐션 캐시를 유지해 재계산 없이 더 빠른 프롬프트 처리를 가능하게 합니다. Gemma에서 메모리 사용을 약간 늘려도 속도를 높입니다.

</details>

### 음성-음성 설정

우선 간단히 라이브러리를 설치합니다.

```bash
uv pip install speech-to-speech
```


그런 다음 LLM을 다른 터미널에서 서빙하는 동안, 간단히 실행합니다:

```bash
speech-to-speech --responses_api_base_url "http://127.0.0.1:8080" --responses_api_api_key "" --mode local
```


터미널을 통해 모델과 대화를 시작할 수 있습니다! 처음 실행 시 Parakeet-TDT 0.6B v3와 Qwen3TTS를 다운로드해야 하지만, 이후 실행은 빠릅니다.

다음은 로컬 대화 모드를 보여주는 영상입니다:

<video controls width="100%">
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/local_reachy_mini_conversation/s2s-llamacpp.mp4" type="video/mp4">
</video>

이제 ``--mode local``에서 시도한 후에는 그 옵션 없이 명령을 다시 실행해 로봇에 speech-to-speech를 제공할 수 있습니다.

### Reachy Mini를 음성-음성으로 연결하기

llama.cpp와 speech-to-speech가 실행되면 로봇을 데스크탑 앱으로 시작하고 대화 앱을 실행할 수 있습니다. 대화 앱의 UI에서 HF 백엔드의 'edit connection'(연결 편집)을 클릭해 로컬 모드를 선택해야 합니다. 아래는 이를 수행하는 방법을 보여주는 영상입니다:

<video controls width="100%">
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/local_reachy_mini_conversation/setting_up_conv_app.mp4" type="video/mp4">
</video>

이제 끝났습니다. 로봇과 대화를 시작할 수 있습니다. 파이프라인의 각 단계는 트레이드오프가 있습니다: 더 빠른 TTS 모델은 품질이 낮고, 더 느린 STT 모델은 품질이 더 높습니다. 우리는 다국어를 위해 최적화했으며, 단일 언어에 맞추려면 최적화를 원할 수 있습니다. 남은 블로그 글에서 커스터마이즈 방법을 다룹니다.

## 더 깊이 들어가기 {#section-2}

### 왜 자체 Speech-to-Speech 서버를 운영하나요?

호스팅된 실시간 백엔드는 편리하지만, 자체 엔진을 실행하면 세 가지를 가능하게 합니다:

- **개인정보 보호.** 오디오가 네트워크를 벗어나지 않으며, 전체 파이프라인이 사용자가 제어하는 하드웨어에서 실행됩니다.
- **API 비용 없음.** 분당 요금이나 토큰당 요금이 없습니다.
- **파이프라인에 대한 완전한 제어.** VAD, STT, LLM, TTS를 자유롭게 교체합니다. Hub 🤗에 더 나은 것이 새로 도입될 때마다 교체 가능합니다.

`speech-to-speech` 저장소는 이를 한 번에 한 CLI로 제공합니다. `/v1/realtime`에서 Reachy Mini가 이미 알고 있는 동일한 프로토콜로 작동하는 WebSocket 서버를 시작합니다.

### 우리의 고정 관점 기본값: VAD, STT, TTS

캐스캐이드된 음성 파이프라인은 네 단계로 구성됩니다: VAD, STT, LLM, TTS. 이 중 세 단계에 대해 우리는 확실한 기본값을 선택해 LLM에 집중할 수 있도록 합니다:

| 단계 | 선택 | 이유 |
|-------|--------|-----|
| VAD | **Silero VAD v5** | 작고 정확하며 CPU에서 실행됩니다. 오픈 소스 음성 에이전트 세계의 사실상 기본값입니다. |
| STT | **Parakeet-TDT 0.6B v3** | 스트리밍에 친화적이고 매우 빠르며 영어 품질이 훌륭합니다. |
| TTS | **Qwen3-TTS** | 표현력이 풍부하고 지연이 낮으며 다국어를 지원하고 맞춤 음성을 지원합니다. |

이 선택들에 대해 우리는 주관적 입장을 가지고 있습니다. 선호하는 경우 자유롭게 바꿔도 좋습니다.

### LLM 선택하기

LLM은 지연 시간과 시스템 전반 성능에 가장 큰 영향을 주는 계층입니다. 두 가지 옵션을 지원합니다: **로컬에서 모델 실행**(llama.cpp, MLX, Transformers, vLLM) 또는 **Responses API를 통해 서버 사용**(OpenAI, Gemini, HF Inference Endpoints, llama.cpp, vLLM 등).

#### The Responses API: 뇌와 음성 루프의 분리

시스템의 주요 병목은 LLM 추론 지연입니다. 이를 해결하기 위해, Responses API 프로토콜을 통해 외부 추론 엔진을 노출합니다.

따라서 `speech-to-speech` 엔진은 LLM이 Responses API 프로토콜을 말하는 한 별도 프로세스에서 실행되는 두 번째 모드를 지원합니다. 한 터미널에서 모델 서버를 시작하고, 다른 터미널에서 음성 루프를 시작하며, 두 프로세스는 HTTP로 통신합니다.

##### 옵션 1: llama.cpp를 한 터미널에서, speech-to-speech를 다른 터미널에서

**터미널 1: llama.cpp 서버:**

```bash
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF -np 2 -c 65536 -fa on --swa-full
```


**터미널 2: speech-to-speech 클라이언트:**

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "ggml-org/gemma-4-E4B-it-GGUF" \
  --responses_api_base_url "http://127.0.0.1:8080/v1"
```


##### 옵션 2: vLLM을 한 터미널에서, speech-to-speech를 다른 터미널에서

> **vLLM ≥ 0.21.0이 필요합니다.** Speech-to-speech 백엔드에서 사용하는 도구 호출 스트리밍을 포함한 Responses API 프로토콜에 대한 완전한 지원은 vLLM 0.21.0에서 도입되었습니다. 이전 버전은 부트는 되지만 도구 호출 시 문제가 발생합니다.

이 파이프라인을 위해 vLLM에서 모델을 서빙할 때는 사실상 세 가지 플래그가 필요합니다:

- `--enable-auto-tool-choice` — 모델의 원시 출력을 구조화된 도구 호출로 변환하는 파서를 선택합니다. (예: `qwen3_coder`는 Qwen3 지시형 모델용, `llama3_json`은 Llama 3용, `hermes`은 Hermes 스타일 모델용, …).
- `--default-chat-template-kwargs '{"enable_thinking":false}'` : `<think>` 추론 채널을 비활성화합니다. 더 어려운 에이전트 작업의 경우 이를 `true`로 바꿔 모델이 추론하게 할 수 있지만, 자연스러운 대화를 원한다면 이를 끈 상태로 두는 것을 강력히 권합니다: 생각하는 토큰 하나하나가 로봇이 말을 시작하기까지의 지연으로 들립니다.

**터미널 1: vLLM 추론 서버 (`Qwen/Qwen3-4B-Instruct-2507`):**

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8000 \
  --host 127.0.0.1 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}'
```


> `--speculative-config` 라인은 Multi-Token Prediction (MTP)을 가능하게 합니다. 이는 선택적이지만 엔드-투-엔드 지연에 큰 영향을 미칩니다. 모델이 이를 지원할 때는 활성화해 두십시오.

**터미널 2: speech-to-speech 클라이언트:**

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --responses_api_base_url "http://127.0.0.1:8000/v1"
```


##### 옵션 3: Hugging Face Inference Endpoints

같은 프로토콜이지만 모델은 Hugging Face의 관리형 GPU에서 실행됩니다. 어떤 채팅 모델이든 Inference Endpoint로 배포한 뒤, 음성 루프를 엔드포인트 URL로 가리키세요:

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --responses_api_base_url "https://<your-endpoint>.endpoints.huggingface.cloud/v1" \
  --responses_api_api_key "$HF_TOKEN"
```


##### 옵션 4: Hugging Face Inference Providers

직접 엔드포인트를 관리하고 싶지 않다면 [Inference Provider](https://huggingface.co/docs/inference-providers)를 사용하세요. Hugging Face는 요청을 제3자 백엔드(예: Together, Fireworks, Replicate)로 단일 URL로 라우팅합니다:

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "Qwen/Qwen3.6-35B-A3B:deepinfra" \
  --responses_api_base_url "https://router.huggingface.co/v1" \
  --responses_api_api_key "$HF_TOKEN"
```


##### 옵션 5: OpenAI(또는 OpenAI 호환 공급자)

원Infra 없이 최전방 모델을 테스트하고 싶다면 같은 플래그를 OpenAI에 지정하세요:

```bash
speech-to-speech \
  --mode realtime \
  --stt parakeet-tdt \
  --tts qwen3 \
  --llm_backend responses-api \
  --model_name "gpt-5.4" \
  --responses_api_api_key "$OPENAI_API_KEY"
```


The `--responses_api_*` 플래그는 프로토콜을 구현하는 모든 공급자(OpenRouter, Together, Fireworks, …)에서도 동일하게 작동합니다. 기본 URL과 API 키를 바꾸고 나머지 파이프라인은 동일하게 유지합니다.

---

#### LLM을 인-프로세스로 실행하기

##### 옵션 1: MLX(Apple Silicon)에서 로컬 LLM 실행

맥에서 실행하는 경우 MLX는 합리적인 지연으로 실제 모델을 실행하는 가장 낮은 마찰의 방법입니다. **Qwen3-4B-Instruct-2507**를 추천합니다. 이 모델은 M 시리즈 칩에서 즉시 반응을 느낄 만큼 작고, 대화도 충분히 버틸 수 있을 정도로 강력합니다.

```bash
speech-to-speech \
  --llm_backend mlx-lm \
  --model_name "mlx-community/Qwen3-4B-Instruct-2507-bf16"
```


서버는 기본적으로 `ws://127.0.0.1:8765/v1/realtime`에서 수신합니다. 실행 상태를 유지하고 로컬 백엔드에 대화 앱을 연결하면 로봇과 대화를 시작할 수 있습니다.

##### 옵션 2: Transformers에서 로컬 LLM 실행 (CUDA / CPU / MPS)

동일한 아이디어이지만 일반적인 `transformers`를 사용합니다. CUDA 박스에서 실행 중이거나 Linux를 사용할 경우, 또는 MLX용 가중치를 다시 변환하지 않고 자유롭게 모델을 바꾸고 싶을 때 이 옵션을 사용하세요.

```bash
speech-to-speech \
  --llm_backend transformers \
  --model_name "Qwen/Qwen3-4B-Instruct-2507"
```


> **팁.** `Qwen3-4B-Instruct-2507`은 단일 컨슈머 GPU에서 속도/품질 균형을 잘 제공하기 때문에 LLM에 또 다른 좋은 선택지입니다. 백엔드가 지원하는 어떤 HF 모델이든 `--model_name`으로 가리킬 수 있습니다 — 예를 들어 더 큰 Gemma, Qwen, 또는 Mistral 같은 모델도 가능합니다.

### 노트북에서 엔진을 실행하고 로봇에서 앱을 실행하기

노트북에서 음성 엔진을 실행하고 Reachy Mini Wireless에서 대화 앱을 실행하는 경우, 바뀌는 것은 URL뿐입니다. 엔진이 LAN 주소에 바인딩되도록 하고, UI에서 IP를 선택할 때 로봇의 IP를 노트북의 IP로 사용하세요.

IP를 모르는 경우, 찾는 방법은 아래와 같습니다:

<details>
<summary>macOS</summary>

```bash
ipconfig getifaddr en0    # wifi
ipconfig getifaddr en1    # ethernet (sometimes en0, varies)
```


</details>

<details>
<summary>Linux</summary>

```bash
hostname -I
```


</details>

<details>
<summary>Windows</summary>

```powershell
ipconfig
```


활성 어댑터에서 IPv4 주소를 찾으십시오.

</details>

당신은 `192.168.x.x` 또는 `10.x.x.x` 중 하나를 원합니다. `169.254.x.x`가 보이면 네트워크에 실제로 연결되어 있지 않은 것입니다.

---

## 마무리 {#section-3}

이제 완전히 로컬 음성 루프가 있습니다:

- 로봇이 **Silero**로 듣고,
- **Parakeet-TDT 0.6B v3**로 전사하고,
- 선택한 LLM으로 사고한 뒤, 로컬 MLX, 로컬 Transformers, vLLM 또는 옆의 llama.cpp 서버이든, 혹은 호스팅된 Responses API 엔드포인트이든 간에,
- **Qwen3-TTS**로 응답합니다.

[`huggingface/speech-to-speech`](https://github.com/huggingface/speech-to-speech)와 [`pollen-robotics/reachy_mini_conversation_app`](https://github.com/pollen-robotics/reachy_mini_conversation_app)에 별표를 표시해 주시고, 로봇에 어떤 오픈 소스 cascade를 실행하게 되었는지 토론에 남겨 주세요.

<!-- publication refresh -->
