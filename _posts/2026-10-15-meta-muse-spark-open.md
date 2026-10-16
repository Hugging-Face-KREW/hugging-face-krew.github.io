---
layout: post
title: "Muse Spark Open 소개: Meta의 오픈 웨이트 프런티어 멀티모달 Mixture-of-Experts 모델"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/meta-muse-spark-open/thumbnail.png
image: assets/images/blog/posts/2026-10-15-meta-muse-spark-open/thumbnail.png
authors:
  - user: pcuenq
  - user: merve
  - user: burtenshaw
  - user: guarin
  - user: qgallouedec
  - user: alvarobartt
  - user: Serveurperso
  - user: Jundot
slug: "meta-muse-spark-open"
source_url: "https://huggingface.co/blog/meta-muse-spark-open"
source_published_date: "2026-10-15"
source_published_at: "2026-10-15T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Welcome Muse Spark Open: open-weight frontier multimodal mixture-of-experts model from Meta](https://huggingface.co/blog/meta-muse-spark-open)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/meta-muse-spark-open -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Muse Spark Open 소개: Meta의 오픈 웨이트 프런티어 멀티모달 Mixture-of-Experts 모델

Muse Spark Open은 Meta의 오픈 웨이트 모델입니다. 이 mixture-of-experts 모델은 약 8160억 개의 파라미터와 토큰당 약 420억 개의 활성 파라미터를 사용하며, 최대 100만 토큰의 context window를 지원하면서 텍스트, 이미지, 오디오, 비디오를 하나의 모델로 처리합니다. 이 모델은 텍스트를 생성하므로 개발자는 다양한 유형의 입력에 대해 질문할 수 있는 공통 인터페이스를 사용할 수 있습니다.

Muse Glimmer에 이어 Muse Spark Open은 훨씬 더 큰 규모의 mixture-of-experts 아키텍처를 도입했으며, 현재까지 가장 뛰어난 미국 오픈 웨이트 모델입니다.

Meta는 에이전틱 코딩을 위해 Muse Spark Open을 학습했습니다. 여기에는 코드베이스 읽기, 코드 작성 및 디버깅, 도구 호출이 포함됩니다. 또한 이미지, 오디오, 비디오를 입력으로 받아들이므로, 코드 작업 중에도 동일한 모델이 스크린샷을 읽고, 음성 메모를 듣고, 화면 녹화를 볼 수 있습니다. 이 글에서는 모델 실행 방법, fine-tuning 방법, 그리고 이 모델로 구축한 내용을 소개합니다. 모든 모델은 [here](https://huggingface.co/collections/meta-models/muse-spark-open)에서 확인할 수 있습니다.

## 전체 기능 및 아키텍처 {#section-1}

Meta는 Muse Spark Open을 멀티 파일 리팩터링과 장시간의 디버깅 세션 등 실제 개발자 워크플로에 맞춰 학습한 코딩 최적화 모델로 설명합니다. Muse Spark Open은 웹에서 이미지를 포함한 작업을 수행하는 등, 장시간 작업에서 계획하고, 행동하고, 관찰하고, 스스로 수정하는 에이전틱 모델입니다.

| 기능 | Muse Spark Open |
| :---- | :---- |
| 모델 크기 | 총 ~816B 파라미터; 토큰당 활성 ~42B |
| Expert 라우팅 | MoE 레이어당 256개 expert; 토큰당 8개 선택 |
| 입력 | 텍스트, 이미지, 오디오, 비디오 |
| 출력 | 텍스트 |
| Context window | 최대 1M 토큰 |
| Checkpoint 형식 | BF16 및 NVFP4 |

Checkpoint repository는 Muse-Spark-Open-816B-A42B 및 Muse-Spark-Open-816B-A42B-NVFP4입니다.

![model diagram](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/model.png)

Muse Spark Open은 Muse 모델 시리즈의 풀 사이즈 모델입니다. Meta가 8월에 출시한 Muse Glimmer는 같은 시리즈의 소형 dense 모델로, 로컬 실행에 적합합니다. 두 모델은 동일한 tokenizer와 vocabulary를 사용합니다. Chat template은 새로운 modality를 위한 추가 사항과 함께 Muse Glimmer template을 따릅니다. Muse Spark Open에서는 입력 embedding과 출력 layer가 분리된 행렬이며, 각각 약 1.66B개의 파라미터를 가집니다.

### Mixture of Experts

언어 모델은 62개 layer와 8192의 hidden dimension을 사용합니다. 각 layer에는 128개의 query head와 16개의 key-value head(grouped-query attention)가 있습니다. 각 head의 dimension은 64입니다. Attention block에는 output gate와 output normalization layer가 있습니다. 또한 모델은 query와 key에도 normalization(QK-norm)을 적용합니다. 각 block은 단순한 합이 아니라 학습된 residual gate를 통해 입력과 출력을 결합합니다.

Muse Glimmer는 각 layer에서 하나의 dense feed-forward network를 사용합니다. Muse Spark Open은 이 dense network를 Mixture-of-Experts(MoE) block으로 대체합니다. 각 MoE block에는 256개의 expert가 있으며, router가 각 토큰에 대해 8개의 expert를 선택합니다. 각 expert는 8192에서 4096 dimension으로 투영한 뒤 다시 8192로 투영하고, 출력에 자체 normalization을 적용하는 feed-forward network입니다. Expert가 전체 모델 파라미터의 ~97%를 차지합니다.

토큰당 8개의 expert만 실행되므로 모델은 각 토큰을 약 42B개의 파라미터로 계산합니다. 256개의 expert는 모두 메모리에 유지되므로 배포에는 816B 파라미터 모델에 해당하는 메모리가 필요하지만, 실행 속도는 42B 모델에 해당할 정도로 훨씬 빠릅니다. BF16 checkpoint는 safetensors 형식으로 약 1.5 TB입니다. NVFP4 checkpoint는 정확도 손실을 무시할 수 있는 수준으로 파라미터를 4비트 정밀도로 저장하며, NVIDIA Blackwell GPU를 위해 특별히 제작되었습니다. transformers에서는 BF16 checkpoint가 tensor parallelism(TP=32)을 사용해 8개의 H200 GPU가 있는 4개 node에서 실행되고, NVFP4는 단일 B200 node에서 실행됩니다. Muse Spark Open을 배포하려면 아래의 `llama.cpp`, vLLM 및 SGLang recipe를 확인하세요.

### Perception Encoder

Perception encoder는 Muse Glimmer와 마찬가지로 이미지와 비디오 프레임을 언어 모델용 토큰으로 변환합니다. Encoder는 1.8B개의 파라미터, 50개 layer, 1536의 hidden dimension, 16개의 attention head를 사용합니다. 각 이미지를 14×14 patch로 나누고 4096 dimension의 feature를 생성합니다. 위치 정보는 학습된 absolute embedding과 2D RoPE에서 얻습니다. 각 layer의 feed-forward network는 LayerNorm과 GELU를 사용합니다. 네 번째 layer마다 그리고 마지막 layer에서는 full attention을 사용하고, 나머지 layer에서는 sparse attention을 사용합니다.

![image pipeline diagram](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/image_pipeline.png)

Muse Glimmer 30B는 동일한 아키텍처의 encoder를 사용합니다. 이 설계는 Meta의 [Perception Encoder](https://huggingface.co/papers/2504.13181) 작업을 따릅니다. 비디오 입력은 이미지와 동일한 encoder를 통과합니다.

### Audio Encoder

![audio diagram](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/audio_encoder.png)

Muse Spark Open은 audio encoder를 통해 오디오를 입력으로 받습니다. Audio tokenizer가 오디오를 토큰으로 변환하고, chat template은 `<|audio|>` 토큰으로 오디오 위치를 표시합니다. Encoder는 80ms마다 하나의 오디오 토큰을 출력하며, 이는 초당 12.5토큰에 해당합니다. 예를 들어 10초 길이의 오디오가 있다면 입력 오디오 토큰 수는 125개입니다.
Encoder 자체는 convolutional pyramid이며, 각 stage는 서로 다른 patch size의 patch embedder와 stride가 1이고 1d kernel size가 7인 2개의 convolutional block으로 구성됩니다. 이 stage는 [16, 6, 5, 4]의 patch size에 대해 반복됩니다. 마지막 stage의 끝에서는 temporal frame이 RMSNorm → linear projection → RMSNorm으로 구성된 최종 block에 입력됩니다.

### Long Context

Inference에서 context window는 1M 토큰입니다. Configuration은 `max_position_embeddings`을 1,048,576으로 설정합니다. 여기에는 텍스트와 동일한 window 안의 이미지, 비디오, 오디오 토큰이 포함됩니다.

Grouped-query attention은 이 크기의 모델에서 KV cache를 작게 유지합니다. 각 토큰은 cache에 2 × 16 heads × 64 dimensions × 62 layers = 126,976개의 값을 추가합니다. BF16에서는 토큰당 약 250 KB이며, 1M 토큰 전체 context에서는 약 266 GB입니다.

## Inference Support {#section-2}

### Transformers

Muse Spark Open 구현은 멀티모달 입력에서 텍스트를 생성하기 위한 `MuseSparkForConditionalGeneration`과 대화를 준비하기 위한 `MuseSparkProcessor`을 제공합니다. Processor는 tokenizer, image 및 video processor, audio feature extractor, chat template을 결합합니다.

병렬 처리 설정을 구성하려면 `DistributedConfig`을 사용합니다. `tp_size`은 dense layer가 shard되는 device 수이고, `ep_size`는 expert가 분산되는 device 수입니다.

다음과 같이 모델과 processor를 초기화할 수 있습니다.

```py
import os

from transformers import AutoModelForMultimodalLM, AutoTokenizer, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

tokenizer = AutoTokenizer.from_pretrained("meta-models/Muse-Spark-Open-816B-A42B
")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Spark-Open-816B-A42B",
    distributed_config=distributed_config,
)

messages = [
    {"role": "user", "content": "Explain in one sentence why the sky is usually blue."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    reasoning_effort="medium",  # default: xhigh, options: low/medium/high/xhigh
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)

input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=2000)

```


#### Text-only Inference

대화를 준비한 다음 메시지를 `processor.apply_chat_template`에 전달하고 `add_generation_prompt=True`, `tokenize=True`, `return_dict=True`, `return_tensors="pt"`를 활성화합니다. Processor 출력을 `model.generate`에 전달하고 입력 prefix를 제외한 새로 생성된 토큰을 decode합니다. Processor에는 reasoning effort를 조절하는 레버가 있으며, symbolic level(low/medium/high/xhigh)을 모델이 직접 허용하는 정수(16, 64, 256, 512)에 매핑합니다. Muse Spark Open에서는 reasoning을 끌 수 없으므로, 더 쉬운 작업을 해결할 때 적은 토큰을 사용하려면 모델 reasoning을 “low”로 유지하세요. System prompt에 `Reasoning strength: 1`과 같은 줄을 포함해 숫자로 effort를 설정할 수도 있습니다.

```py
messages = [
    {"role": "system", "content": "You should only answer with a number."},
    {"role": "user", "content": "What is 17 * 23?"},
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort="high", # options:low/medium/high/xhigh
).to(model.device).to(model.dtype)

output = model.generate(**inputs, max_new_tokens=2000)
generated_tokens = output[0][inputs["input_ids"].shape[1] :]
print(processor.decode(generated_tokens, skip_special_tokens=False))


```


#### Images and Text

Muse Spark Open processor는 각 이미지 marker를 처리된 이미지에 필요한 patch token 수만큼 확장하고, 그 앞뒤를 이미지 boundary token으로 감쌉니다. 이러한 확장을 통해 텍스트 위치가 모델에 전달되는 시각적 feature와 정렬됩니다. Processor는 하나의 prompt 안에 여러 이미지가 있는 경우도 처리합니다.

```py
image_url = (
    "https://huggingface.co/datasets/merve/vl-test-suite/"
    "resolve/main/pills.jpg"
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {
                "type": "text",
                "text": "Do any of the components in this supplement interact?",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="medium",
    return_dict=True,
    return_tensors="pt",
).to(model.device).to(model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=2000)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

processor.parse_response(response)

```


#### Video Inference

비디오 prompt에는 시각적 콘텐츠뿐 아니라 frame timing도 포함됩니다. Processor는 오디오가 포함된 비디오를 처리하며, 이는 audio encoder 덕분에 Muse Glimmer에서 개선된 부분입니다. Processor는 비디오 frame과 오디오를 downsample해 frame을 서로 정렬합니다. 기본적으로 비디오는 2 FPS로 sampling되며 최대 96 frame으로 제한됩니다. 비디오 processor의 fps를 설정해 이를 변경할 수 있습니다:

```
processor.video_processor.fps = 4.0
```


일반적으로 비디오는 시간 순서에 따른 frame으로 구성되며, downsample을 많이 할수록 특정 이벤트 사이의 frame을 더 많이 잃게 됩니다. 따라서 더 정확한 생성을 원한다면 fps를 높게 유지해야 합니다. 그러나 더 많은 메모리가 필요하고 prefill latency가 증가한다는 trade-off가 있습니다. Frame은 timestamp(예: “Time: 3.5s”)를 나타내는 텍스트 입력과 interleave됩니다. 오디오가 포함된 비디오를 전달하려면 `load_audio_from_video=True`을 `apply_chat_template`에 전달합니다.

`return_metadata=True`으로 입력을 준비할 때 비디오의 metadata를 유지해야 processor가 올바른 framerate를 지정할 수 있습니다.

```py
import torch

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Describe what happens in this video. "
                    "Use timestamps and include relevant speech or sounds."
                ),
            },
            {
                "type": "video",
                "url": "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/ramen.mp4"
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    load_audio_from_video=True,
    return_metadata=True,
).to(model.device).to(model.dtype)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

print(response)
```


### Audio Inference

Muse Spark Open audio tower는 raw waveform을 처리합니다. Processor는 24 kHz의 mono audio를 기대하며, waveform을 1,920 samples의 배수로 padding합니다. 이는 80ms마다 하나의 audio embedding을 생성하는 것에 해당합니다. 다른 sample rate의 오디오를 processor에 직접 제공하려면 먼저 resampling해야 합니다. transformers의 `load_audio` util을 사용하면 됩니다. `load_audio`은 torchcodec, librosa 또는 torchaudio backend를 사용하므로 inference를 실행하기 전에 이 중 하나가 로컬에 설치되어 있는지 확인하세요.

메시지와 함께 `apply_chat_template`만 간단히 사용하려는 경우에는 이 모든 과정이 내부적으로 처리됩니다.

```py
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

model_id = "meta-models/Muse-Spark-Open-816B-A42B"
processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForMultimodalLM.from_pretrained(model_id).eval()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "https://huggingface.co/buckets/huggingface/audio-samples/resolve/bbaudio_question_39.mp3",
            },
            {
                "type": "text",
                "text": "Answer the question.",
            },
        ],
    }
]

# Downloads the audio, decodes it as mono, resamples it to 24 kHz, pads it, creates input_values_mask, and expands the audio placeholders.
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    load_audio_backend="auto",
).to(model.device).to(model.dtype)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

# Decode only newly generated tokens
prompt_length = inputs["input_ids"].shape[1]
generated_text = processor.batch_decode(
    output_ids[:, prompt_length:],
    skip_special_tokens=True,
)

print(generated_text[0])
```


Processor는 audio boundary marker 사이에 일치하는 수의 audio token을 삽입합니다. 또한 실제 sample을 식별하는 mask를 반환하므로 모델이 padding된 audio embedding을 제외할 수 있습니다.

#### Tool Calling

Muse Spark Open은 text-only 및 멀티모달 tool calling을 수행할 수 있습니다. 다음과 같이 모델이 도구를 호출하도록 할 수 있습니다.

```py

from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "meta-models/Muse-Spark-Open-816B-A42B"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "weather.get",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/merve/vl-test-suite/resolve/main/SF.png"},
            {"type": "text", "text": "I'm going to the city in this picture. What clothes should I wear?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tools=tools,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
    reasoning_strength="high"
).to(model.device).to(model.dtype)

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs)
response = processor.parse_response(
      completion,
      prefix=inputs["input_ids"][0],
      tools=tools,
  )
print(response)

```


Muse Spark Open은 channel 기반 chat template(Harmony, gpt-oss에서 처음 도입)을 사용합니다. 내부 동작을 알 필요는 없지만, 출력 품질이 낮을 때를 대비해 모델이 라우팅하는 channel을 확인하면 유용합니다. 아래 입력 메시지에서는:

```
<|begin_of_text|>
  <|start|>system<|message|>
  You are a helpful AI assistant.
  Knowledge cutoff: 2026-01-04.
  Current date: ...
  ...
  <|eot|>
  <|start|>user<|message|>...
  <|eot|>
  <|start|>assistant
```


`<|start|>assistant`은 의도적으로 열어 둔 상태이며, `to=`을 사용해 recipient를 선택하는 것은 모델입니다. 모델에는 서로 다른 recipient가 있으며, 예를 들어 `self`는 모델에 reasoning을 요청합니다.

```
to=self<|message|>the model reasons here<|eom|> # reasoning
<|start|>assistant to=user<|message|>final answer<|eot|> # final answer
```


`self`은 reasoning에 사용되고, `commentary`은 최종 사용자에게 표시되는 tool call 전후의 comment이며, `user`는 최종 응답입니다.  
Tool-call routing은 다음과 같이 parse됩니다(`tools`을 `apply_chat_template`에 전달했다고 가정):

```
<|start|>assistant to=weather.get<|message|>
  <atem:function_calls>
  <atem:invoke name="weather.get">
  <atem:parameter name="city">San Francisco</atem:parameter>
  </atem:invoke>
  </atem:function_calls>
  <|eot|>
```


### Llama.cpp

<video controls width="720"> <source src="https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/webui.mp4" type="video/mp4" /> </video> <p><small>Llama Server에서 실행되는 Muse Spark 2-bit</small></p>

Muse Spark Open은 llama.cpp에 통합되어 있으며 prebuilt binary로 제공됩니다. 다음 명령으로 설치할 수 있습니다.

```
curl -LsSf https://llama.app/install.sh | sh

```


다음과 같이 model ID에 quant suffix를 붙여 server를 시작할 수 있습니다.

```

llama serve -hf ggml-org/Muse-Spark-Open-816B-A42B-GGUF:Q2_K

```


Quant를 포함하는 repository는 [here](https://huggingface.co/ggml-orgf/Muse-Spark-Open-816B-A42B-GGUF)입니다. Server가 시작되면 localhost:9931로 이동해 내장 WebUI에서 채팅할 수 있습니다. Web UI에서 inference를 위한 오디오와 이미지를 추가하고 reasoning lever를 조정할 수 있습니다.   
Muse Spark Open에서는 reasoning을 끌 수 없으므로, 더 쉬운 작업을 해결할 때 적은 토큰을 사용하려면 모델 reasoning을 “low”로 유지하세요.   
Mac M3 Ultra(512GB)에서 quant를 테스트했습니다. Imatrix-calibrated 2-bit quant는 18 tokens/s로 실행되며, KV cache를 위한 여유 공간도 충분했습니다(모델이 사용한 메모리는 304 GB).

다음과 같이 server에 query를 보낼 수도 있습니다. 멀티모달 기능, tool call 등을 확인하려면 [the API reference](https://llama.app/docs/serve)을 참고하세요.

```
## Text-only inference
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a limerick about python exceptions"}
        ]
    }'

## Image-text inference
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer <your-key>" \
  -d '{"model":"muse-spark-open","messages":[{"role":"user","content":[
        {"type":"text","text":"What is in this image?"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
      ]}],"temperature":0}'

## Audio-text inference
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer <your-key>" \
  -d '{"model":"muse-spark-open","messages":[{"role":"user","content":[
        {"type":"text","text":"Transcribe this audio."},
        {"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,..."}}
      ]}],"temperature":0}'
```


llama.cpp team은 다양한 quantized version을 제공합니다. Apple Silicon chip에서 Q2_K, Q4_K_S 및 Q_8을 테스트했습니다. quant id를 suffix로 추가해 각 quant를 실행할 수 있습니다(`llama serve -hf ggml-org/Muse-Spark-Open-816B-A42B:Q2_K`).

### SGLang

SGLang용 Muse Spark Open 통합은 텍스트, 이미지, 오디오, 비디오 입력을 지원합니다. NVFP4 reference configuration은 4개의 GB300 GPU에서 실행됩니다.

SGLang 통합 환경에서 Hugging Face Hub로부터 checkpoint를 다운로드하고 server를 시작합니다.

```
export MODEL_ID=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
export CKPT="$(hf download "$MODEL_ID")"

SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 \
uv run sglang_meta serve \
  --model-path "$CKPT" \
  --served-model-name "$MODEL_ID" \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_trtllm \
  --tp-size 4 \
  --ep-size 4 \
  --port 30000
```


Checkpoint에는 chat template이 포함되어 있으며, 이를 명시적으로 전달합니다. Launcher는 Muse reasoning 및 tool-call parser를 구성합니다.

OpenAI-compatible API를 통해 request를 보냅니다.

```
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "messages": [{
      "role": "user",
      "content": "Explain how a mixture-of-experts model works."
    }],
    "max_tokens": 2048
  }'
```


응답은 `reasoning_content`의 reasoning trace와 `content`의 최종 답변을 분리합니다.

### vLLM

vLLM용 Muse Spark Open reference integration은 Blackwell series GPU가 필요한 NVFP4 checkpoint를 사용해 텍스트 및 이미지 입력을 지원합니다.

Reference plugin과 dependency를 설치한 후 Hugging Face Hub에서 checkpoint를 다운로드하고 server를 시작합니다.

```
export MODEL_ID=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
export CKPT="$(hf download "$MODEL_ID")"

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_PLUGINS=musespark1x_models,ptq_nvfp4 \
vllm serve "$CKPT" \
  --served-model-name "$MODEL_ID" \
  --tokenizer-mode musespark1x \
  --config-format auto \
  --load-format internal_safetensor \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser musespark1x \
  --tool-call-parser musespark1x \
  --enable-auto-tool-choice \
  --port 8000
```


이 configuration은 128K-token serving context를 설정합니다. Plugin은 model implementation, checkpoint loader, reasoning 및 tool-call parser를 제공합니다.

동일한 Hub model ID를 사용해 request를 보냅니다.

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "messages": [{
      "role": "user",
      "content": "Explain how a mixture-of-experts model works."
    }],
    "max_tokens": 2048
  }'
```


## MLX {#section-3}

Muse Spark Open은 현재 [oMLX](https://github.com/jundot/omlx) 및 [mlx-lm](https://github.com/ml-explore/mlx-lm)에서 사용할 수 있습니다. oMLX에 최적화된 oQ quant는 [3-bit](https://huggingface.co/mlx-community/Muse-Spark-Open-816B-A42B-oQ3e) 및 [4-bit](https://huggingface.co/mlx-community/Muse-Spark-Open-816B-A42B-oQ4e)에서 제공되며 즉시 사용할 수 있습니다.

M3 Ultra(512 GB)에서 oQ3e quant를 테스트했습니다. 모델은 약 344 GB를 사용해 KV cache를 위한 공간을 남기며, 1024-token prompt와 128개의 생성 토큰을 사용했을 때 prefill은 약 225 tok/s, generation은 약 21 tok/s로 실행됩니다.

oMLX interface를 통해 모델을 다운로드하면 내장 chat UI에서 채팅할 수 있습니다. Chat UI에서 이미지를 추가하고 reasoning effort를 low부터 xhigh까지 즉시 조정할 수 있습니다. 모델의 reasoning은 최종 답변과 분리된 접을 수 있는 thinking block으로 표시됩니다.

Reasoning을 끌 수 없으며 기본값(Auto)은 xhigh를 사용합니다. 더 쉬운 작업에서 적은 토큰을 사용하려면 “low”로 설정하세요.
![omlx](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/omlx_muse_spark_open_reasoning_screenshot.png)

언어 모델은 공식 mlx-lm repository의 [this PR](https://github.com/ml-explore/mlx-lm/pull/1975)에서도 사용할 수 있습니다. 다음과 같이 동일한 고품질 oQ3e quant에서 실행할 수 있습니다.

```
mlx_lm.generate --model mlx-community/Muse-Spark-Open-816B-A42B-oQ3e \
  --prompt "Explain convolutional networks like I am 5" \
  --max-tokens 5000
```


이 모델은 크기가 크며, 위에서 설명한 것처럼 3-bit quantization조차 256 GB 이상의 RAM을 필요로 합니다. 여러 컴퓨터에 걸쳐 distributed inference를 실행할 수도 있습니다. 자세한 내용은 documentation을 참고하세요.

## Inference Endpoints로 Muse Spark Open 배포 {#section-4}

NVFP4 variant를 위한 사전 검증된 SGLang recipe를 사용해 Inference Endpoints에 Muse Spark Open을 배포할 수 있습니다. 이 variant는 131072 context length에서 8 × NVIDIA RTX Pro 6000 98 GB 또는 1048576 context length에서 8 x NVIDIA H200 141GB로 실행됩니다.

배포하려면 Hugging Face CLI로 다음 한 명령을 실행합니다.

```sh
hf endpoints catalog deploy --repo meta-models/Muse-Spark-Open-816B-A42B-NVFP4 --name muse-spark
```


또는 [endpoints.huggingface.co/new/meta-models/Muse-Spark-Open-816B-A42B-NVFP4](https://endpoints.huggingface.co/new/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)을 열고 Create Endpoint를 선택합니다.

Endpoint는 `/v1/chat/completions`의 Chat Completions API와 `/v1/responses`의 Responses API를 포함한 OpenAI-compatible API를 제공합니다.

```sh
curl "https://<your-endpoint>.endpoints.huggingface.cloud/v1/chat/completions" \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/merve/vl-test-suite/resolve/main/pills.jpg"}},
        {"type": "text", "text": "Do any of the components in this supplement interact?"}
      ]
    }],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```


모든 OpenAI-compatible client는 동일한 URL과 API key로 사용하는 Hugging Face token을 통해 작동합니다.

## TRL을 사용한 Fine-tuning {#section-5}

TRL `SFTTrainer`을 사용해 816B checkpoint를 fine-tuning할 수 있습니다. 이 절에서는 8개의 H100 GPU가 있는 8개 node에서 TRL로 fine-tuning한 방법을 보여줍니다. Recipe는 attention layer에 LoRA adapter를 학습합니다. Expert는 고정된 상태로 유지됩니다. 전체 script는 [TRL examples](https://github.com/huggingface/trl/tree/main/examples/sft_muse_spark_open)에 있습니다.

먼저 모델을 64개의 GPU device에 분할해야 합니다. 모델 크기가 크기 때문에 여러 축에 걸쳐 parallelize해야 합니다. FSDP로 모델을 shard하고 `DistributedConfig`을 사용해 expert를 GPU 간에 분할합니다. 이 설정은 모델을 로드할 때 어떻게 분할할지 `transformers`에 알려줍니다.

```py
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(fsdp_size=64, ep_size=8)
```


위 snippet에서 `fsdp_size=64`은 FSDP2를 사용해 64개 GPU 전체에 모델을 shard하고, `ep_size=8`은 각 layer의 256개 expert를 8개 group으로 나눕니다. Transformers는 각 토큰을 해당 expert를 보유한 GPU로 보내므로, 각 GPU는 batch의 자체 slice를 학습합니다.

이제 training run을 위한 `SFTConfig`을 구성할 수 있습니다. 여기서는 open source dataset에서 모델의 chat template을 적용하는 간단한 작업을 수행했지만, 자체 use case에도 적용할 수 있습니다.

```py
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers.distributed import DistributedConfig
from trl import SFTConfig, SFTTrainer

train_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train[:500000]")

training_args = SFTConfig(
    output_dir="muse-spark-lora-sft",
    model_init_kwargs={"dtype": torch.bfloat16, "distributed_config": DistributedConfig(fsdp_size=64, ep_size=8)},
    per_device_train_batch_size=1,
    max_steps=500,
    max_length=2048,
    packing=True,
    save_only_model=True,
    logging_steps=1,
    report_to="trackio",
)

trainer = SFTTrainer(
    model="meta-models/Muse-Spark-Open-816B-A42B",
    args=training_args,
    train_dataset=train_dataset,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
)
trainer.train()
```


여기서 핵심 부분은 padding 없이 example을 2048-token sequence로 결합하는 `packing=True`입니다. Block-diagonal mask는 example을 서로 분리합니다. `save_only_model=True`은 adapter만 저장하고 optimizer state는 저장하지 않습니다. Trainer는 load time에 shard된 모델을 재개할 수 없기 때문입니다. `LoraConfig`은 네 개의 attention projection을 대상으로 하며 각 layer의 256개 expert는 고정된 상태로 유지됩니다.

그런 다음 각 node에서 다음 명령을 실행할 수 있습니다.

```sh
torchrun --nnodes 8 --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint "$MASTER_IP:29700" sft_muse_spark_open.py
```


500 step을 학습하는 데 64개의 H100에서 약 2시간이 걸렸으며, 속도는 초당 약 8,000토큰이었습니다. Mean token accuracy는 72%에서 84%로 상승했고 loss는 1.26에서 0.57로 감소해 양호한 fine-tuning 실행임을 보여주었습니다.

![Loss](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/training_loss.png)

![Acc](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/token_acc.png)

## Demos {#section-6}

Muse Spark Open은 코딩 및 에이전틱 모델입니다. 이를 직접 확인하기 위해 모델의 이미지 및 오디오 입력을 사용하는 멀티모달 및 에이전틱 작업을 조합해 시도했습니다. 아래의 모든 항목은 직접 실행하거나 fork할 수 있습니다.

### 3D Asset Generator

llama.cpp에서 serving되는 2-bit quantized Muse Spark open을 Pi coding agent와 결합해 [Three.js](http://Three.js)에서 3D asset을 생성하고 반복적으로 개선했습니다. Prompt는 [here](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/pi-3d-prompt.txt)에서 확인할 수 있습니다. 모델이 quantized되어 있으므로 조금 더 많은 steering이 필요합니다. Mac Ultra에서 llama.cpp를 사용했을 때 18 TPS를 얻었습니다.

![Pi demo](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/pi-demo.png)

먼저 asset을 설명하면 Pi가 모델을 사용해 해당 asset의 Three.js code를 생성하고, browser에서 결과를 렌더링한 뒤 여러 각도에서 screenshot을 캡처합니다. 그런 다음 이 이미지를 llama server의 동일한 모델로 보내 시각적 feedback을 받습니다. Pi는 해당 feedback을 사용해 code를 수정하고 이 과정을 반복합니다.

### OpenEnv 및 Hugging Face Sandbox를 사용한 Browser 상호작용

Spark를 computer use task에서 테스트한 결과 성능에 깊은 인상을 받아, computer use agent의 작동 방식을 알려주는 lean demo를 구축했습니다. 필요에 따라 이를 추가로 customize할 수도 있습니다. 교육용 example이면서 몇 줄의 코드로 실행할 수 있는 예제이기도 합니다. [here](https://huggingface.co/docs/openenv/main/en/environments/helium_browser)에서 확인할 수 있습니다.

<video controls width="720"> <source src="https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/browser_agent.mp4" type="video/mp4" /> </video> <p><small>OpenEnv 및 HF Sandboxes를 사용하는 Muse Spark NVFP4 Browser Agent</small></p>

[Hugging Face Sandboxes](https://huggingface.co/docs/huggingface_hub/en/guides/sandbox)은 간편한 code execution을 위해 Hugging Face Jobs 위에 구축된 sandbox입니다. Demo는 세 가지 component로 구성됩니다.

1. Inference Endpoints로 serving되는 Muse Spark Open,  
2. helium 및 selenium을 사용해 web driver action(click, type, scroll 등)을 감싸고 browser environment의 reset, step, observation, termination interface를 정의하는 OpenEnv,  
3. Chromium web browser

여기서 remote sandbox에는 아래 그림과 같이 environment와 browser가 포함됩니다.

![Diagram](https://huggingface.co/datasets/merve/meta-muse-spark-open-assets/resolve/main/browser_server.png)

예제에 설명된 몇 가지 단계만으로 쉽게 시도할 수 있습니다. Muse Spark Open으로 Inference Endpoint를 시작합니다(NVFP4도 잘 작동합니다!). Endpoint URL을 environmental variable로 추가하고, OpenEnv server를 Spaces에 publish한 뒤 해당 주소도 제공합니다. 그러면 agent loop를 실행할 수 있습니다.
Muse Spark Open은 action을 수행하기 위해 이전 action, system prompt, URL, browser screenshot만 보고 그 외에는 아무것도 보지 않습니다(DOM도, HTML code도, 그 무엇도 없음). 따라서 진정한 멀티모달 browser demo입니다.

### Browser 호출하기

마지막으로 Spark의 에이전틱 멀티모달 기능을 한계까지 시험하고자 모델의 오디오, 이미지, 에이전틱 특성을 browser agent로 결합하는 [this space](https://huggingface.co/spaces/burtenshaw/muse-spark-call-your-browser)을 구성했습니다. 필요한 물건을 일반적으로 설명하면 모델이 통합된 browser와 상호작용해 Amazon에서 선택지를 찾습니다.

<video controls width="720"> <source src="https://huggingface.co/buckets/huggingface/muse-spark-blog-assets/resolve/browser.mp4" type="video/mp4" /> </video>

Muse Spark Open은 음성 입력을 받아 설명에 부합하는 제품 유형을 reasoning하고, Amazon에서 검색한 뒤 기준에 따라 가장 적절한 제품을 선택합니다.

## 마무리 {#section-7}

Muse Spark Open은 Muse 시리즈의 풀 사이즈 오픈 웨이트 모델입니다. [BF16](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B) 또는 [NVFP4](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) checkpoint로 시작하거나, 훌륭한 `llama.cpp quants` 중 하나를 사용해 보세요. Muse Spark Open으로 무언가를 만들었다면 공유하고 [repository’s discussions.](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B/discussions)에서 알려주세요.
