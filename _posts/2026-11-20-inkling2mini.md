---
layout: post
title: "Welcome Inkling 2 Mini, 로컬 환경을 위한 최첨단 성능"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/inkling2mini/thumbnail.png
image: assets/images/blog/posts/2026-11-20-inkling2mini/thumbnail.png
authors:
  - user: merve
  - user: pcuenq
  - user: ariG23498
  - user: guarin
slug: "inkling2mini"
source_url: "https://huggingface.co/blog/inkling2mini"
source_published_date: "2026-11-20"
source_published_at: "2026-11-20T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Welcome Inkling 2 Mini, Cutting-edge Performance for your Local Setup](https://huggingface.co/blog/inkling2mini)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/inkling2mini -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Welcome Inkling 2 Mini, 로컬 환경을 위한 최첨단 성능

TLDR; Inkling 2 Mini는 새로운 루프형 transformer 모델입니다. 토큰당 \~26B *physical* 총 파라미터와 3B 활성 파라미터를 사용하는 멀티모달 Mixture-of-Experts입니다. 용량을 늘리기 위해 일부 레이어를 루프에서 세 번 반복 방문하는 흥미로운 기법을 사용하며, 이를 통해 더 많은 파라미터를 가진 것처럼 작동합니다. 1M 토큰 컨텍스트 크기를 지원하고 이미지와 텍스트 입력을 모두 받습니다. Thinking Machines Lab은 NVFP4 및 MXFP8 expert weight를 사용하는 mixed-precision 형식으로 모델을 제공합니다. 적은 메모리 비용으로 추가 속도 향상을 제공하는 DSpark drafter도 포함되어 있습니다.

Inkling 2 Mini는 Inkling 2 Small보다 훨씬, 훨씬 작으며 로컬 AI 환경을 위해 설계되었습니다. 우리는 이 모델이 신선하고 훌륭한 새로운 로컬 모델이라고 생각합니다. vibe 테스트에서는 Qwen 3.8 27B와 적어도 비슷한 수준이었습니다. 추가 검증이 필요하지만, 이번 릴리스를 매우 기대하고 있습니다\!

Transformers, Llama App (llama.cpp), MLX를 지원합니다. SGLang은 데이터센터 하드웨어에서 프로덕션 배포를 위한 day-zero 추론 지원도 제공합니다.

이 글에서는 Inkling Mini를 차별화하는 요소와 몇 가지 vibe 테스트, 그리고 자신의 환경에서 사용하는 방법을 살펴봅니다.

## Inkling 2 Mini의 새로운 점은 무엇인가요? {#section-1}

Inkling 2 Mini는 24개의 decoder layer를 가진 mixture-of-experts 모델이지만, 가운데 12개 레이어는 루프에서 세 번 실행됩니다. 따라서 이 모델은 24개의 *physical* 레이어만으로 48개(6+ 3\*12 \+ 6)의 *logical* 레이어를 가진 것처럼 작동합니다.

Inkling 2 Small과 마찬가지로 Inkling 2 Mini는 토큰당 48회의 layer visit을 실행합니다. hidden size는 4,608이 아니라 2,304입니다. 각 layer visit에서 두 변형 모두 토큰을 8개의 expert로 라우팅하지만, Mini는 레이어당 128개의 expert 중에서 선택하고 Small은 384개 중에서 선택합니다\.  
Decoder는 layers 0–5를 한 번, layers 6–17을 세 번, layers 18–23을 한 번 실행합니다. layers 6–17을 통과하는 세 번의 패스는 모두 동일한 weight를 재사용합니다.

Nanbeige와 같은 아키텍처는 이전에도 루프를 탐구했습니다. 하지만 Inkling 2 Mini는 흥미로운 접근 방식을 탐구합니다. 루프형 코어(layers 6–17)를 통과하는 모든 패스는 처음 6개 레이어의 출력인 동일한 입력에서 시작합니다. 패스 간 정보는 cross-pass `messages`를 통해 이동합니다. 각 looped layer는 자체적으로 학습된 query를 사용해 이전 패스의 출력에서 메시지를 읽습니다(query, key, value 및 MLP 입력 각각에 하나의 메시지). 그런 다음 작은 gating controller가 레이어의 현재 입력과 메시지를 모두 살펴봅니다. 그리고 현재 입력을 얼마나 잊고 메시지를 얼마나 추가할지 결정합니다. 따라서 각 패스는 이전 패스가 계산한 결과를 정제합니다.

![Inkling 2 Mini architecture](https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/architecture.png)

> * 각 6-layer block에는 5개의 sliding-window layer와 1개의 full-attention layer가 있습니다. 로컬 윈도우는 최대 1,024개의 토큰을 담습니다.  
> * Attention은 24개의 query head와 8개의 KV head를 사용합니다. 학습된 relative position logit, 학습된 sink, output gate를 사용합니다. Full-attention layer는 dynamic key 및 value convolution을 사용합니다.  
> * Router는 각 토큰에 대해 128개의 SwiGLU expert 중 8개를 선택합니다. 선택된 score를 정규화하고 weight를 8배로 조정합니다\.  
> * passes 2와 3에서 각 core layer는 이전 패스의 두 residual block을 대상으로 attention을 수행합니다. 4개의 rank-64 gate가 query, key, value 및 MoE 입력에 별도의 메시지를 혼합합니다.

>

텍스트 decoder의 width는 2,304이고 각 expert의 intermediate width는 1,152입니다. core weight를 재사용하면 checkpoint에 저장되는 서로 다른 레이어 수가 줄어듭니다. 모델은 여전히 48회의 visit을 모두 실행하며, 각 visit은 자체 KV 및 short-convolution state를 유지합니다.

플래그십 Inkling 2 Mini 모델과 함께 연구용 artifact도 공개되었습니다. 동일한 아키텍처를 공유하지만, 이 변형은 루프를 가변적인 횟수만큼 통과하도록 학습되었습니다 🤯. 3회 대신 1회 또는 2회로 설정하면 추론이 훨씬 빨라지지만, 생성 품질은 약간 낮아질 수 있습니다. 이 변형을 테스트한 결과 작동했지만, 심층적으로 평가하지는 않았습니다.

## 벤치마크 결과 {#section-2}

Inkling 2 Mini는 Artificial Analysis Intelligence Index v5(AAv5)에서 동급 최고 수준이며, 다양한 범주의 여러 벤치마크에서도 최고 수준입니다. Gemma 4 31B, Muse Glimmer, Qwen3.8-27B, Nemotron 3.5 Super VL과 비교했을 때 Inkling 2 Mini는 48개 중 32개 벤치마크에서 앞섭니다\. 여기에는 coding 및 autoresearch, knowledge work를 위한 agentic search 및 tool use, vision interpretation 및 creation, reasoning 및 knowledge, safety 및 alignment가 포함됩니다. 전체 벤치마크와 점수 목록은 [Inkling 2 Mini 모델 카드](https://huggingface.co/thinkingmachines/Inkling-2-Mini)에서 확인할 수 있습니다.

Inkling 2 Mini는 뛰어난 로컬 코딩 동반자입니다. FrontierCode v2.0에서 앞서고, Terminal-Bench 4.0에서는 Qwen3.8-27B와 동률이며, AutomationBench-AA v1.0.6에서는 Qwen에 뒤처집니다. 아래에서 coding 및 knowledge work와 관련된 몇 가지 주요 결과를 확인할 수 있습니다. AA Briefcase는 ELO rating이며, 높을수록 좋다는 점에 유의하세요.

![Inkling 2 Mini selected benchmarks](https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/selected-benchmarks.png)

## 추론 지원 {#section-3}

이 섹션에서는 transformers, llama.cpp, MLX 및 SGLang에서 Inkling 2 Mini를 사용하는 방법을 확인할 수 있습니다.

### Transformers

Inkling 2 Mini(고정 루프)와 `TML-d24-varloop-research`(가변 루프) 연구용 변형을 모두 지원합니다.

#### Inkling 2 Mini에서 고정 루프 사용

`AutoModelForMultimodalLM` 클래스를 사용할 수 있습니다.

```py
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling-2-Mini"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
)

messages = [{"role": "user", "content": [{"type": "text", "text": "Why is the sky blue?"}]}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```


#### 가변 루프

연구용 checkpoint에서 루프 횟수를 설정하려면 다음과 같이 `text_config`의 `loop_num_passes`을 수정하면 됩니다. 아래 예제에서는 두 개의 루프를 사용합니다.

```py
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/TML-d24-varloop-research"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    text_config={"loop_num_passes": 2},
)

messages = [{"role": "user", "content": [{"type": "text", "text": "Why is the sky blue?"}]}]

inputs = processor.apply_chat_template(
    messages, reasoning_effort="medium", tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", 
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))

```


#### 이미지와 텍스트

이미지 프롬프트를 사용하려면 두 예제 중 하나의 messages를 아래 목록으로 바꾸고 평소처럼 추론하면 됩니다.

```py
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.jpg"},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]
```


#### 추론 수준

요청의 effort를 설정하려면 `processor.apply_chat_template(...)`에 `reasoning_effort="medium"`을 전달합니다. chat template은 숫자 값을 system message로 프롬프트에 추가합니다. 기본값은 `high`(0.9)입니다.

```py
inputs = processor.apply_chat_template(
    messages, reasoning_effort="medium", tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", 
).to(model.device)
```


사용 가능한 수준은 `none`(0.0), `minimal`(0.1), `low`(0.2), `medium`(0.7), `high`(0.9; 기본값), `xhigh`(0.99), `max`(0.99)입니다. `reasoning_effort=0.35`과 같이 0.0부터 0.99 사이의 숫자를 전달할 수도 있습니다.

#### Tool calling

모델이 몇 가지 도구를 호출하도록 해 보겠습니다.

이 예제에서는 모델에 특정 위치의 현재 기온을 묻습니다. 프롬프트와 함께 이를 수행할 도구(`get_current_temperature`)도 제공합니다. 모델이 tool call을 요청하면, 대신 도구를 호출한 다음 그 출력을 다음 instruction set에 추가합니다. 모델은 도구의 출력을 사용해 최종 결과를 제공합니다.

tool calling에 대해 더 알고 싶다면 [공식 문서](https://huggingface.co/docs/transformers/en/chat_extras)를 확인하는 것을 권장합니다.

```py
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling-2-Mini"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")


def get_current_temperature(location: str, unit: str):
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
    """
    return 22.0  # A real function would call a weather API here


def ask(messages):
    inputs = processor.apply_chat_template(
        messages, tools=[get_current_temperature], reasoning_effort="low", tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024)[0][inputs["input_ids"].shape[-1] :]
    parsed = processor.parse_response(output, prefix=inputs["input_ids"][0])
    return parsed


messages = [{"role": "user", "content": "Hey, what's the temperature in Paris right now?"}]

reply = ask(messages)
call = reply["tool_calls"][0]["function"]
print("Model calls:", call["name"], call["arguments"])  # the model fills in the arguments itself

temperature = get_current_temperature(**call["arguments"])
messages += [reply, {"role": "tool", "name": call["name"], "content": str(temperature)}]
print(ask(messages)["content"])

# Model calls: get_current_temperature {'location': 'Paris, France', 'unit': 'celsius'}
# Right now in Paris it’s about **22°C** (roughly **71.6°F**).
```


### Llama App 및 Llama.cpp

[Llama App](https://llama.app/)을 사용하면 macOS(메뉴 막대에서) 또는 Windows(시스템 트레이에서)에서 로컬 언어 모델을 쉽게 실행하고 관리할 수 있습니다. llama.cpp를 기반으로 하며 동일한 팀이 개발했습니다. 또한 coding agent, editor 및 script와 쉽게 통합할 수 있도록 로컬 OpenAI 호환 API를 제공합니다.

Inkling 2 Mini 릴리스에서는 두 모델 모두에 대해 GGUF 4-bit(`IQ4_NL)` 및 8-bit(`Q8_0`) quant를 제공합니다.

Linux를 사용하거나 CLI에서 모델을 제공하려는 경우, 고정 루프 모델에 다음 명령을 실행할 수 있습니다.

```sh
llama serve -hf ggml-org/Inkling-2-Mini-GGUF:IQ4_NL
```


가변 루프 모델을 실행하려면 다음을 실행하면 됩니다.

```sh
llama serve -hf ggml-org/TML-d24-varloop-research-GGUF:IQ4_NL
```


8-bit weight를 로드하려면 suffix를 `:Q8_0`으로 변경합니다.

여기서는 4-bit 모델에 식물을 licensed therapist와 매칭하는 SaaS인 plant therapy의 웹사이트를 만들도록 요청합니다. 모델은 상당히 자율적으로 작동하며, 웹사이트의 시각 요소를 위한 자체 SVG를 만들었고 매칭 로직까지 구축했습니다.

<video controls muted loop playsinline preload="metadata" aria-label="Plant Therapy (Inkling 2 Mini)" src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/plant-therapy.mp4"></video>

### SGLang

Inkling 2 Mini는 SGLang 내부에서 Day 0 지원을 제공합니다. 최신 버전의 SGLang을 설치하고 다음을 실행하면 서버를 시작할 수 있습니다. 해결해야 할 한 가지 사전 조건은 `SM103`에서 실행하는 것입니다.

```sh
export MODEL="thinkingmachines/Inkling-2-Mini"

python -m sglang.launch_server \
  --model-path $MODEL \
  --served-model-name inkling \
  --attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16 \
  --tp 1 --page-size 128 --grammar-backend xgrammar \
  --host 0.0.0.0 --port 30000
```


서버가 시작되면 다음과 같이 API를 요청할 수 있습니다.

```sh
curl -s localhost:30000/health_generate && echo OK
curl -s localhost:30000/v1/models | python -m json.tool

curl -s localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "inkling",
        "messages": [{"role": "user", "content": "What is 17 times 24?"}],
        "max_tokens": 512,
        "reasoning_effort": "low"
      }' | python -m json.tool
```


`--enable-multimodal`을 설정하여 멀티모달 기능을 활성화할 수도 있습니다.

### Speculative decoding

SGLang은 3-layer DSpark draft를 사용해 target model이 검증할 토큰을 제안할 수 있습니다. draft는 decoder visits 5, 23, 41의 입력과 index 48의 최종 normalized state를 읽습니다. draft weight가 있으면 서버 명령에 `--speculative-algorithm DSPARK --context-length 131072`을 추가합니다.

speculative decoding을 사용한 경우와 사용하지 않은 경우에 모델을 벤치마크하고, 단일 NVIDIA B300 GPU에서 생성 속도를 측정했습니다. 모든 요청은 1,024-token prompt를 사용하고 정확히 1,024개의 토큰을 생성했으며, concurrency는 1, 4, 16 및 64였습니다.

<details>
<summary>Show SGLang benchmarking script</summary>

```sh
MODEL=thinkingmachines/Inkling-2-Mini

for TAG in base mtp; do
  EXTRA=""; [ $TAG = mtp ] && EXTRA="--speculative-algorithm DSPARK"

  # Start the server (with or without MTP)
  python -m sglang.launch_server --model-path $MODEL \
    --attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16 \
    --tp 1 --page-size 128 --context-length 131072 --max-running-requests 64 \
    --port 30000 $EXTRA > $TAG.log 2>&1 &
  until curl -sf localhost:30000/health_generate > /dev/null; do sleep 10; done

  # Benchmark at increasing concurrency
  for C in 1 4 16 64; do
    python -m sglang.bench_serving --backend sglang --port 30000 --model $MODEL \
      --dataset-name random --apply-chat-template \
      --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
      --num-prompts $(( C*4 < 32 ? 32 : C*4 )) --max-concurrency $C \
      --output-file results_$TAG.jsonl
  done

  kill %1; wait
done
```


</details>

![SGLang throughput and time per token with and without MTP](https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/mtp-benchmark.png)

### MLX

멀티모달 이미지 및 텍스트 입력을 포함한 MLX 지원을 mlx-lm 프로젝트에 upstream했으며, mlx-community organization을 통해 quant도 공유했습니다.

MLX 추론은 원본 checkpoint에서도 작동합니다. 이 경우 native nvfp4 \+ mxfp8 quantization scheme을 재변환 없이 사용하므로, 저자가 의도한 원래 품질을 얻을 수 있습니다.

다음은 CLI 호출 예제입니다.

```sh
mlx_lm generate --model thinkingmachines/Inkling-2-Mini --image image.jpg --prompt "Describe this image.<|end_message|><|message_user|><|content_image|><|unused_200054|>" -m 1000
```


### 데모와 Vibe 테스트

Inkling 2 Mini를 여러 재미있는 작업에 사용해 보았습니다. 이 섹션에서 확인해 보세요.

#### 3D shader flythrough

동일한 프롬프트로 procedural mountain과 반사되는 물 위를 지나는 두 개의 sunset flythrough가 생성되었으며, 각각 단일 WebGL fragment shader로 렌더링되었습니다.

**Sunset Ridge (Inkling 2 Mini)**

<video controls muted loop playsinline preload="metadata" aria-label="Sunset Ridge (Inkling 2 Mini)" src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/3d-worlds/sunset-ridge.mp4"></video>

**Over the Still Sea (Qwen 3.8 27B)**

<video controls muted loop playsinline preload="metadata" aria-label="Over the Still Sea (Qwen 3.8 27B)" src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/3d-worlds/over-the-still-sea.mp4"></video>

<details>
<summary>Show shared flythrough prompt</summary>
<blockquote><p>Build the most impressive original real-time scene you can: a slow, cinematic camera flight over procedural mountains, through volumetric clouds and low over water with reflections, all lit by a sunset. Deliver it as a single self-contained index.html in the current directory that renders everything in one full-screen WebGL fragment shader, purely from code (no textures, images, fonts, libraries, external files or network requests of any kind). It must run smoothly in desktop Chrome at 1920x1080, aiming for 60 fps on a desktop GPU, with an automatic resolution-scale fallback that lowers the internal render resolution when the frame rate drops, and the flight must keep going indefinitely without glitches. To check your work, run <code>node shot.cjs index.html shot.png 6</code> (the third argument is how many seconds to wait; a comma list such as <code>3,12,25</code> saves one screenshot per time, and an optional fourth argument sets the size, default 1280x720): it renders the page in headless Chrome, saves the screenshot(s), and prints the WebGL renderer, a rough frame rate and any console errors; then look at the images. The headless renderer on this machine may be software-only, so its frame rate can be far below what a real desktop GPU achieves and is not the target. Decide for yourself when the result is finished, then reply with a short description of what you built.</p></blockquote>
</details>

#### 별이 빛나는 밤, 생명을 얻다

<div style="display: flex; gap: 12px; align-items: flex-start;">
  <div style="flex: 1 1 0; min-width: 0;">
    <img src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/starry-night/reference.png" alt="Van Gogh’s original Starry Night painting, with a swirling blue sky above a village" width="400" style="width: 100%; height: auto;" loading="lazy">
    <p>Vincent van Gogh, <em>The Starry Night</em> (1889)</p>
  </div>
  <div style="flex: 1 1 0; min-width: 0;">
    <video src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/starry-night/starry-night.mp4" aria-label="Inkling 2 Mini’s animated recreation of The Starry Night" width="400" style="width: 100%; height: auto;" controls muted loop playsinline preload="metadata"></video>
    <p>Inkling 2 Mini recreation</p>
  </div>
</div>

Pi coding harness에서 Inkling 2 Mini에 Van Gogh의 그림 참고 이미지와 screenshot helper를 제공했습니다. Inkling은 단일 WebGL2 fragment shader를 포함한 독립 실행형 HTML 페이지를 작성했습니다. 이 shader는 참고 이미지나 다른 asset을 로드하지 않고 코드만으로 장면을 그립니다. 시간을 사용해 소용돌이치는 하늘, 별빛의 광채, 흔들리는 사이프러스 나무, 마을 창문의 깜박임을 애니메이션으로 표현합니다.

Inkling은 24라운드의 screenshot 확인을 거치며 결과물을 렌더링하고 검토했습니다. Apple M4 Max에서 최종 애니메이션을 1080p 및 60 fps로 프레임 단위 기록했습니다.

<details>
<summary>Show prompt</summary>
<blockquote><p>In this folder, starry-night.jpg is Vincent van Gogh's The Starry Night (1889, public domain). Recreate it as a living painting: a single self-contained index.html (all code inline, no external files, libraries or network requests) that fills the browser window and runs smoothly in desktop Chrome at 1920x1080. Everything must be generated procedurally in code, for example with WebGL shaders; the page must not load, embed or sample starry-night.jpg or any other image. Stay faithful to the painting's composition and palette (the great spiral and swirling currents of the sky, the haloed stars, the glowing crescent moon, the dark flame-like cypress on the left, the rolling hills, and the village with its church spire), rendered in Van Gogh's thick, directional brushstrokes, and bring it to life: the sky's swirls flow, the stars and moon pulse with light, the cypress sways, and the village windows glow and flicker. Aim for the most impressive result you can make. A helper is provided: run <code>node shot.js --help</code> to see how to screenshot index.html at several moments; look at the screenshots next to the reference and keep improving until you are satisfied, then reply with a short summary of what you made.</p></blockquote>
</details>

#### Napkin에서 App으로

<div style="display: flex; gap: 12px; align-items: flex-start;">
  <div style="flex: 2.6 1 0; min-width: 0;">
    <a href="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/napkin-to-app/napkin-sketch.png"><img src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/napkin-to-app/napkin-sketch.png" alt="Hand-drawn Trip Split wireframe with Expenses, Add expense, and Balances screens" width="540" style="width: 100%; height: auto;" loading="lazy"></a>
    <p>Input napkin sketch</p>
  </div>
  <div style="flex: 1 1 0; min-width: 0;">
    <a href="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/napkin-to-app/trip-split-app.png"><img src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/napkin-to-app/trip-split-app.png" alt="Inkling’s Trip Split app showing three trip expenses and a $405 total" width="210" style="width: 100%; height: auto;" loading="lazy"></a>
    <p>Generated Trip Split app</p>
  </div>
</div>

Pi coding harness에서 Inkling 2 Mini에 손으로 그린 Trip Split wireframe 사진과 한 문단의 프롬프트를 제공했습니다. Inkling은 스케치의 label, sample expense 및 handwritten note를 사용해 단일 HTML 파일 안에 vanilla CSS와 JavaScript로 expense list, add-expense form 및 balances view를 구축했습니다. 앱은 expense가 변경되면 total과 balance를 업데이트하고, 여행 비용을 정산할 payment를 제안하며, 브라우저의 localStorage에 데이터를 저장해 새로고침 후에도 유지합니다.

Inkling을 세 번 독립적으로 실행하고 Playwright를 사용해 각 build를 클릭하며, 시각적 판단이 필요한 부분에서는 screenshot을 확인했습니다. 여기 표시된 첫 번째 시도는 13개의 검사를 모두 통과했습니다. 스케치의 \$90짜리 저녁 식사를 추가하자 total이 \$405에서 \$495로 증가했고 각 사람이 지불해야 할 금액이 다시 계산되었습니다.

<details>
<summary>Show prompt</summary>
<blockquote><p>The photo sketch.jpg in this folder is a product designer's hand-drawn wireframe of a small app. Build that app for real, as the most impressive, original, production-quality version you can make: a single self-contained index.html in this folder using vanilla HTML, CSS and JavaScript only (no frameworks, CDNs, external files or network requests; localStorage is fine), which must run smoothly in desktop Chrome at 1920x1080. Treat everything on the paper as the spec: every screen, label, button and number, the example data, and each red note describe behaviour that must actually work, while the visual design, typography and motion are yours to make beautiful. Look at the sketch carefully with your vision, and use your vision again to check your own work: <code>node shot.cjs --help</code> explains a provided helper that screenshots index.html in headless Chrome at any window size after clicks or other JavaScript steps (Playwright is also installed if you prefer your own scripts), and you can look at any screenshot with the read tool. Click through every feature, compare each screen against the sketch, and keep improving until you are fully satisfied; then reply with a short summary of what you built.</p></blockquote>
</details>

#### 4-bit 모델

4-bit GGUF quantized Inkling 2 Mini(IQ4\_NL)에 식물을 licensed therapist와 매칭하는 SaaS인 plant therapy의 웹사이트를 만들도록 요청합니다.

모델은 상당히 자율적으로 작동합니다. 짧은 프롬프트 하나만으로 웹사이트의 시각 요소를 위한 자체 SVG를 만들고 매칭 로직까지 구축했습니다: "Can you make a website for my saas called plant therapy where I match house plants with licensed therapists so plants don’t die, return single HTML”.

<video controls muted loop playsinline preload="metadata" aria-label="Plant Therapy (Inkling 2 Mini)" src="https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/plant-therapy-website.mp4"></video>

#### “스스로를 브라우저 agent로 만들어라"

Inkling 2 Mini를 Pi coding agent와 결합해 Helium, Chromium 및 Hugging Face Sandboxes를 사용하여 스스로를 browser agent로 구축하도록 했습니다. trace와 최종 파일(browser agent로 사용할 수 있음)은 [이 bucket에서](https://huggingface.co/buckets/huggingface-projects/inkling-2-mini-release-assets/resolve/browser-agent-demo) 제공합니다.

#### 마무리

로컬 환경을 위한 성능 좋은 중형 모델인 Inkling 2 Mini를 출시하기 위해 Thinking Machines Lab과 협력하게 되어 매우 기쁩니다. 이 모델을 사용해 보고, 의견을 댓글로 알려주세요\!
