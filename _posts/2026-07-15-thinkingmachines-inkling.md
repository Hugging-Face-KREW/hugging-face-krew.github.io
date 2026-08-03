---
layout: post
title: "Thinking Machines의 Inkling에 오신 것을 환영합니다"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-07-15-thinkingmachines-inkling/thumbnail.png
authors:
  - user: burtenshaw
  - user: merve
  - user: pcuenq
  - user: ariG23498
slug: "thinkingmachines-inkling"
source_url: "https://huggingface.co/blog/thinkingmachines-inkling"
source_published_date: "2026-07-15"
source_published_at: "2026-07-15T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Welcome Inkling by Thinking Machines](https://huggingface.co/blog/thinkingmachines-inkling)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/thinkingmachines-inkling -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Thinking Machines의 Inkling에 오신 것을 환영합니다

[Inkling](https://huggingface.co/thinkingmachines/Inkling)은 대형(1조 매개변수!)의 오픈 모델로 이미지, 텍스트, 오디오 입력을 네이티브로 수용합니다.

요약: Thinking Machines의 Inkling은 허깅페이스에 공개되었습니다. Inkling은 이미지, 오디오, 텍스트를 모두 이해하는 대형 멀티모달 LLM이며, 에이전트형 기능을 가지며, 1M 컨텍스트를 지원합니다. BF16 전체 체크포인트와 잘 보정된 NVFP4 변형으로 제공되며, 추론 속도를 높이기 위한 예측적 MTP 계층을 포함합니다. 트랜스포머, SGLang, 및 llama.cpp에서 데이-제로(day-0) 지원이 있습니다.

## Inkling이 특별한 이유는 무엇인가? {#section-1}

Inkling은 대략 1조 매개변수에 달하는 대형 오픈 모델로, 이미지, 텍스트, 오디오 입력을 네이티브로 받을 수 있으며, 45조 토큰의 텍스트, 이미지, 오디오 및 비디오로 학습되었습니다. 오디오, 이미지, 텍스트 등의 모달리티 간 추론에 집중하며, 파인튜닝을 통한 도메인 적응을 목표로 합니다. 이 모델을 바탕으로 데모를 몇 가지 만들고 아키텍처를 탐구해 왔으며, 멀티모달 추론 애플리케이션의 새로운 물결을 이끌어내기에 좋다고 생각합니다.

## 전반적인 기능과 아키텍처 {#section-2}

Inkling은 전체 975B 매개변수 중 활성 매개변수 41B를 가진 디코더 전용 멀티모달 Mixture-of-Experts 모델이다. 여러 가지가 얽혀 있어 각 부분을 하나씩 살펴보자:

- 디코더 전용(Decoder-only): 이는 아키텍처가 대부분의 최첨단 LLM들처럼 인과적 자동회귀 생성을 지원한다는 것을 의미합니다.
- 멀티모달(Multimodal): 모델은 텍스트, 오디오, 이미지를 입력으로 받을 수 있습니다.
- Mixture of Experts (MoE): 각 층 내부의 피드포워드 네트워크가 희소하게 작동하여 추론이 더 빨라지며, 언제나 41B 매개변수만 활성화됩니다. 모델은 256명의 전문가를 갖고 있습니다. 나중에 자세히 보겠습니다.

다음은 아키텍처의 간략한 개요입니다.

**Relative attention:** 트랜스포머 모델에 위치 정보를 주입하는 일반적인 방법인 RoPE 대신 Inkling은 위치 정보를 인코딩하기 위해 상대 주의를 사용합니다. 각 어텐션 레이어는 위치를 어텐션 로짓에서 직접 학습합니다. key-query-values 외에, 토큰별, 헤드별 상대 피처 R을 생성하는 네 번째 투영이 있습니다. 이 투영 텐서는 키 벡터와 쿼리 벡터 사이의 거리 정보(거리 정보)를 보정한 다음 어텐션 모듈로 전파됩니다.

![Inkling relative attention architecture](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/relative_attention.png)

**Hybrid attention:** 디코더 레이어는 전역 어텐션(전체 컨텍스트 길이에 한꺼번에 주의)과 슬라이딩 윈도우 어텐션(고정 컨텍스트 윈도우를 슬라이딩 방식으로 주의)을 번갈아가며 구성됩니다. 아키텍처는 5:1의 슬라이딩 윈도우 대 글로벌 어텐션 레이어 패턴을 가지고 있습니다. 이 하이브리드 어텐션 방식은 계산 효율성을 제공합니다. 마지막 레이어는 글로벌 어텐션을 사용하여 특징이 풍부한 표현을 구축하는 데 도움이 됩니다.

**Short convolution:** 모델은 은닉 상태 위에 독특한 짧은 1D 합성곱, 또는 `SConv`를 사용합니다. SConv는 현재 토큰과 이전 `W-1`개의 은닉 상태를 읽고, `W`를 슬라이딩 윈도우 크기로 둡니다. 여기서의 직관은 SConv가 로컬 어텐션을 돕는 한편, 어텐션 및 MoE 모듈을 로컬 표현으로부터 해방하는 것입니다.

![Inkling short convolution architecture](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/sconv.png)

**MoE with shared experts sink:** Inkling에서 라우터는 경로화된 전문가와 공유 전문가 모두의 점수를 매깁니다. 상위-k 선택은 6명의 전문가에 대해 수행되며, 항상 활성화되는 2명의 공유 전문가가 추가로 있습니다.

**Vision understanding:** 이 모델은 여러 선형 계층으로 구성된 간단한 계층적 MLP 패처를 포함합니다. 각 계층은 픽셀을 점진적으로 융합하며, 마지막 계층에서 패치당 하나의 임베딩을 생성합니다.

**Audio understanding:** 아키텍처는 이산화된 멜 스펙트로그램을 사용합니다. 각 100 ms 길이의 오디오 청크를 멜 스케일로 변환한 뒤, 정확한 멜 스펙트로그램 빈으로 분류합니다.

멀티모달 타워는 비교적 간단한 모듈로, 각 모달리티마다 별도의 인코더를 사용하는 다른 모델들처럼 복잡하지 않습니다. 각 이미지 패치는 이미지 임베딩 타워를 통과하고, 오디오 청크는 오디오 임베딩 타워를 통과하여 두 미디어 임베딩을 얻습니다. 이미지 입력은 비디오 처리를 위한 추가 시간 차원도 포함합니다. 이 기능은 다운스트림 파인튜닝에 유용할 것으로 예상되지만, 즉시 비디오 성능은 평가하지 않았습니다. 타워는 패치 격자를 접고, 인접한 토큰의 작은 로컬 블록을 채널 차원으로 쌓아 hMLP를 통과합니다. 오디오 파형은 멜 스케일로 변환되며, 디스크리트 멜 빈으로 분류됩니다. 이 멜 빈 값들은 오디오 임베딩 타워에 임베딩되고, 이 임베딩들을 합산하여 최종 오디오 입력을 구성합니다.

## 추론 지원 {#section-3}

Inkling은 데이-제로(day-0) 트랜스포머 지원을 제공하며, SGLang과 vLLM 같은 주요 추론 엔진에서도 지원됩니다.

이 모델은 거대합니다. bf16 체크포인트는 VRAM 2 TB가 필요하고, nvfp4 버전은 VRAM 600 GB가 필요합니다. Inference Providers와 같은 서버리스 추론 라우터를 통해 모델을 시도해 볼 수 있으며, llama.cpp와 함께 ggml 양자화를 사용해 로컬 배포를 할 수 있습니다.

### 트랜스포머

다이렉트하게 `transformers`로 추론하는 가장 쉬운 방법은 `any-to-any` 파이프라인을 사용하는 것입니다. Hopper 이후의 GPU에서 16비트 `"thinkingmachines/Inkling"`를 사용하거나, Blackwell Nvidia GPU에서 양자화된 NVFP4 체크포인트 `"thinkingmachines/Inkling-NVFP4"`를 사용할 수 있습니다. 최신 버전의 트랜스포머(오늘 출시된 5.14.0)를 사용하는지 확인하십시오 (`pip install -U transformers`).

```python
from transformers import pipeline

model_id = "thinkingmachines/Inkling"
# model_id = "thinkingmachines/Inkling-NVFP4"

pipe = pipeline("any-to-any", model=model_id)
```


파이프라인을 초기화한 후 프롬프트를 다음과 같이 전달할 수 있습니다.

```python
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
                "text": "Do components in this supplement interact with each other?",
            },
        ],
    },
]
output = pipe(
    messages,
    max_new_tokens=2000,
    return_full_text=False,
    reasoning_effort="medium",
)
output[0]["generated_text"]
```


한 단계 아래에서 Auto 클래스도 사용할 수 있습니다. 추론에는 모델용 `AutoModelForMultimodalLM` 클래스와 프로세서용 `AutoProcessor` 클래스를 사용할 수 있습니다. 서로 다른 추론 작업에는 토크나이저가 `reasoning_effort` 인수를 받습니다. 추론 노력의 기존 옵션은 `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`입니다.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

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
    reasoning_effort="high",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=2000)
generated_tokens = output[0][inputs["input_ids"].shape[1] :]
print(processor.decode(generated_tokens, skip_special_tokens=False))
```


멀티모달 추론의 경우에도 동일한 클래스를 사용할 수 있습니다. 모델 카드에 각 모달리티에 대한 예제 스니펫을 제공합니다.

<details>
<summary>Text with image inference</summary>

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

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
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=2000)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

processor.parse_response(response)
```


</details>

Inkling은 또한 오디오 입력을 받습니다. 아래는 같은 `AutoModelForMultimodalLM` 클래스를 사용하는 예제 추론 스니펫입니다.

<details>
<summary>Text with audio inference</summary>

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

audio_url = (
    "https://huggingface.co/datasets/merve/vl-test-suite/"
    "resolve/main/example_audio.mp3"
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech to text."},
            {
                "type": "audio",
                "audio": audio_url,
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

processor.parse_response(response)
```


</details>

여러 노드의 클러스터에서 더 현실적인 병렬 배치를 원하신다면 아래의 [Slurm](#slurm-scripts) 섹션을 참조하십시오.

### SGLang

SGLang은 Inkling의 출시 시점에 가장 빠른 배포 프레임워크 중 하나로, 커스텀 모델 구현을 포함합니다. 아래의 런치 명령은 모델을 8개의 GPU에 걸쳐 샤딩하고 포트 30000에서 OpenAI 호환 API를 제공합니다.

```shell
pip install sglang

python3 -m sglang.launch_server \
 --model-path thinkingmachine/Inkling \
 --tp-size 8 \
 --served-model-name inkling \
 --host 0.0.0.0 \
 --port 30000
```


`--tp-size`를 GPU 수에 맞추십시오. KV 캐시를 위한 여유 공간이 더 필요하면 `--mem-fraction-static`을 추가하십시오(예: `0.85`).

### vLLM

vLLM은 생산 서비스에 강력합니다. 하나의 `vllm serve` 명령으로 허브에서 가중치를 다운로드하고, 텐서 병렬화로 GPU에 모델을 샤딩하며, 포트 8000에서 OpenAI 호환 서버를 시작합니다.

```shell
pip install vllm

vllm serve thinkingmachine/Inkling \
  --tensor-parallel-size 8 \
  --served-model-name inkling
```


실제로는 여러 노드와 SLURM 같은 분산 도구가 필요합니다(아래 참조). 핵심 매개변수는 노드의 GPU 수에 따라 `--tensor-parallel-size`를 설정하고, KV-캐시 메모리 한계에 도달하면 컨텍스트 윈도우를 제한하기 위해 `--max-model-len`를 사용합니다.

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```


### 원격 추론: Hugging Face 추론 제공자와 함께

이 모델은 허깅페이스를 통해 여러 추론 제공자를 사용하여 추론할 수 있습니다. [here](https://huggingface.co/thinkingmachines/inkling?inference_provider=fastest&language=python&client=openai&inference_api=true)를 사용하는 모든 코드 스니펫을 확인할 수 있습니다. 아래는 OpenAI 클라이언트와 함께 사용하는 방법입니다.

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="thinkingmachines/Inkling:auto",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?",
        },
    ],
)

print(completion.choices[0].message)
```


설정에서 선호하는 공급자에 접미사 `“:auto”`를 사용합니다. 또한 `“cheapest”` 또는 `“:fastest”`를 사용할 수도 있습니다. 이번 릴리스에서는 모든 사용자를 위해 2시간의 추론 비용을 커버합니다.

참고: Inference Providers에서의 오디오 지원은 작업 중이며 곧 추가될 예정입니다.

### 로컬 추론 with llama.cpp와 Unsloth

제한된 하드웨어에서 양자화된 버전의 모델을 실행하려면 `llama.cpp`를 사용할 수 있습니다. Unsloth는 모델을 1비트 정밀도로 양자화하여 원본 모델 대비 VRAM 사용량을 95% 감소시켰습니다.

```shell
llama serve -hf unsloth/inkling-GGUF:UD-IQ1_S
```


여기에서 [`http://localhost:8000`](http://localhost:8000)`/v1`에서 실행되는 OpenAI 호환 서버를 시작하고, 선호하는 도구나 클라이언트를 통해 연결합니다. 그곳으로 이동하면 모델과 대화를 시작하고, 좋아하는 MCP로 설정하며, 이미지나 파일을 편리하게 전달하는 등 다양한 작업을 수행할 수 있습니다.

Llama cpp는 도구, MCP 및 에이전트형 워크로드를 지원하는 내장 UI도 함께 제공합니다. llma 앱에서 1비트 정밀도에서 Inkling이 실행되는 것을 확인해 보십시오:

<video controls width="100%" autoplay loop muted>
  <source src="https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/thinky.mp4" type="video/mp4">
</video>

Inkling GGUF도 Unsloth Studio에서 동적 1비트 GGUF로 실행 가능하며, 탑 1% 정확도 대비 약 74.2%를 유지하면서 크기는 86% 작습니다.

![Inkling running in Unsloth Studio](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/unsloth.png)

## 사용 사례 {#section-4}

### Pi로 에이전트형 코딩

Pi는 다양한 언어 모델과 함께 사용할 수 있는 최소한의 코딩 에이전트 허브입니다. Pi를 추론 엔진 서버 엔드포인트(예: llama.cpp)로 사용하거나, 설치 후 `~/.pi/agent/models.json`에 이를 추가하여 허깅페이스의 추론 제공자와 함께 사용할 수 있습니다.

```json
{
  "providers": {
    "inference-providers": {
      "baseUrl": "https://router.huggingface.co/v1",
      "api": "openai-completions",
      "apiKey": "hf_...",
      "models": [
        {
          "id": "thinkingmachines/Inkling"
        }
      ]
    }
  }
}

```


그런 다음 프로젝트 디렉토리에서 `pi`를 호출하여 Pi를 시작하면 됩니다! 이 데모에서는 모델에 어려운 수학적 추론 문제를 부여하고, Pi의 도구를 이용해 해결합니다.

![Visual reasoning gif demo](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/visual-reasoning.gif)

Inkling은 광범위한 멀티모달 추론과 낮은 토큰 소모에 집중하므로 문서 처리나 오디오 작업으로 먼저 사용해 보십시오.

### 다중 토큰 예측 드래프터(MTP)

MTP는 한 번에 여러 토큰을 예측하는 추가 계층을 모델에 더합니다. 추론 중에는 이 추가 계층이 예측적 디코딩의 ‘드래프터(drafters)’ 역할을 하여 성능을 해치지 않으면서 생성 속도를 높입니다. MTP를 사용하면 드래터를 서비스하는 데 따른 VRAM의 작은 비용에도 불구하고 동일한 생성 결과를 얻을 수 있습니다. Thinking Machines는 이번 릴리스에서 MTP 드래터도 제공합니다.

```python
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("thinkingmachines/Inkling")
model = AutoModelForMultimodalLM.from_pretrained(
    "thinkingmachines/Inkling",
    dtype=torch.bfloat16,
    device_map="auto",
)

# Preprocess the inputs.
...
generated = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False,
    use_mtp=True,
)
print(processor.decode(generated[0], skip_special_tokens=True))
```


### 멀티모달 비전

전문가 수준의 소스와 대학 입시 문제를 포함한 소규모 추론 문제 세트를 준비했습니다. 스크린샷에 워터마크를 찍어 모델에 도전했고, 모델은 이 문제들을 모두 높은 수준의 한계에서 해결했고, 최고 수준 및 중간 추론 시도에서 하나를 실패했습니다. 따라서 모델의 해답을 확인하고, 해결에 소요된 토큰 수를 제공하는 모델 답안을 링크합니다. 이 분위기 평가에서는 시스템 프롬트를 제공하지 않으며, 이러한 추론 문제는 보통 좋은 시스템 프롬프트와 함께 실행하는 것이 바람직합니다. 분위기 평가 이미지와 결과는 [here](https://huggingface.co/buckets/merve/inkling)에 있습니다.

| 범주 | 질문 | 추론 노력 중간의 토큰 수 | 추론 노력 높음의 토큰 수 | 추론 effort 최대의 토큰 수 |
| :---- | :---- | :---- | :---- | :---- |
| 자유 형식 약물 상호작용 | 여기서 어떤 구성 요소가 상호작용합니까? | 1,893 ✅ |  2,367 ✅ | 3,688 ✅ |
| 물리 문제 (MMMU-Pro) | 이미지의 질문에 답하십시오. | 1,357 ✅ | 3,323 ✅ | 3,314 ✅ |
| 다국어 물리 문제 | 이미지에 주어진 터키어 질문에 답하십시오. | 1,435 ✅ | 2,129 ✅ |  3,162 ✅ |
| 변호사 시험 | 이미지의 질문에 답하십시오. | 1,117 ✅ | 2,137 ✅ | 1,676 ✅ |
| 인포그래픽 질의 응답(주관식) | 제시된 정보를 바탕으로 북극의 여름 기온 상승 기간이 관측된 기간 대비 대략 몇 배 더 긴가요? | 1,378 ❌ | 3,859 ✅ | 6000 토큰(토큰 예산 초과) |

**분위기 평가에 대한 추가 메모:**

- 인포그래픽의 질문에 직접 답하기보다는 모델이 먼저 이미지의 텍스트를 텍스트로 변환해 근거를 마련합니다.
- 추론에서 토큰을 절약하려면 프롬프트가 매우 중요한데, 예를 들어 약의 뒷면 이미지에 '여기서 어떤 구성 요소가 상호작용합니까?'와 같은 모호한 질문을 제시하면 모델은 먼저 '상호작용'의 의미를 파악해야 합니다.
- 객관식 문제의 정답은 모델의 자체 추론 구조를 크게 돕지만, 개방형 질문의 경우 MCQ에 비해 성능이 떨어지는 경향이 있어 많은 모델에서 공통적으로 보이는 이슈입니다. 일반적으로 생각의 흐름은 OCR → 특성화 → 각 옵션 평가 → 답변의 순서입니다.
- 0.7 수준의 추론 노력(중간)이 좋은 절충점을 제공하는 것으로 보입니다.

### 멀티모달 오디오

우리는 BigBenchAudio의 일부 오디오 추론 예제와 [GlobeAudio](https://huggingface.co/datasets/iNLP-Lab/GlobeAudio)의 몇 가지 다국어 오디오 예제(러시아어 및 중국어 다지선다형 문제의 마지막 단어를 묻는 문제)에 대해 분위기 평가를 수행했습니다. 우리가 테스트한 [BigBenchAudio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) 예제는 논리적 진술과 질문으로 구성되며, 형식적 오류를 묻는 문제(오디오의 맥락에서 논증이 논리적으로 도출될 수 있는지 여부)나 객체 개수 세기(오디오에 나타난 여러 개의 특정 객체의 총 개수를 묻는 문제)일 수 있습니다. 이 벤치마크는 원래 음성-대-음성 추론을 위해 만들어졌지만, 이 모델의 오디오 추론 능력을 확인하고자 합니다. GlobeAudio의 경우 질문이 비교적 간단하여 추론 노력 0.1로 진행했습니다. GlobeAudio의 각 언어의 첫 번째 예제를 실행했습니다. 모든 질문과 노력에서 테스트를 통과했으며, 최소 노력에서 두 번째 형식적 오류 예제만 실패했고, 따라서 각 문제의 토큰 사용량만 추론 노력에 비례하여 제공합니다. 분위기 평가 결과 및 오디오 파일은 [here](https://huggingface.co/buckets/merve/inkling)에 있습니다.

| GlobeAudio | 질문 | 최소 추론 노력 시 완료 토큰 수 | 중간 추론 노력 시 완료 토큰 수 |
| :---- | :---- | :---- | :---- |
| Russian (마지막 단어를 묻는 문제) | 오디오 녹음의 마지막 단어는 무엇입니까? 1. Россия 2. Свидетелем 3. Москва 4. Событий 정답 하나의 정확한 텍스트로 답하십시오. | 130 | 179 |
| Russian (발화자의 직업을 묻는 문제) | 화자가 어떤 직업을 가지고 있을 가능성이 있습니까? 1. Репортершей 2. Блоггершей 3. Учительницей истории 4. Ведущей развлекательного шоу 정답 하나의 정확한 텍스트로 답하십시오. | 105 | 136 |
| Chinese (발화 속도 묻기) | 播报员의 말하는 속도에 변화가 있습니까? 1. 갑자기 빨라짐 2. 갑자기 느려짐 3. 변하지 않음 4. 시시각각 변함  정답 하나의 정확한 텍스트로 답하십시오. | 111 | 289 |

| Big Bench Audio  | 최저 추론 노력 토큰 | 중간 추론 노력 토큰 | 최고 추론 노력 토큰 |
| :---- | :---- | :---- | :---- |
| Formal Fallacy (10) | 285 | 335 | 444 |
| Formal Fallacy (39) | 275 (실패) | 555 | 778 |
| Object Counting (680) | 150 | 233 | 161 |

**분위기에 대한 추가 메모:**

- 비전과 유사하게, 모델은 먼저 음성을 글로 전사한 뒤 질문에 답합니다.
- 추론에서 토큰 절약은 프롬프트가 큰 차이를 만듭니다. 예를 들어 러시아어 테스트에서 오디오 속의 다른 답들이 나타나더라도 모델은 정답을 선택했습니다.
- 다지선다형 문제의 해답은 모델의 자체 추론 구조를 도와주지만, 개방형 질문의 경우 MCQA에 비해 성능이 떨어지는 경향이 있습니다. 일반적으로 생각의 흐름은 문자 인식(OCR) → 특성화(특징 파악) → 각 옵션 평가 → 답변입니다.
- 0.7 추론 노력(중간)이 좋은 절충점으로 보입니다.

### 포스트 트레이닝(Post-training)

Inkling을 포스트 트레이닝에 사용하고 싶다면 Thinking Machines가 포스트 트레이닝 오픈 가중치 모델 관리 도구인 `tinker`를 구축했습니다. 그들의 쿡북에는 파인튜닝, 디스틸레이션(distillation), 강화학습에 대한 예제가 포함되어 있습니다.

Inkling을 tinker와 OpenEnv라는 에이전트형 RL 환경 도구로 포스트 트레이닝했습니다. 환경 없이도 환경을 예측하도록 모델을 학습시키는 ECHO 알고리즘을 사용했고, 환경에서 생성된 토큰에 대해 다음 토큰 교차 엔트로피 손실을 적용하는 한편, 에이전트의 행동에 대한 일반적인 정책 학습을 함께 수행했습니다. 이는 별도의 모델, 교사, 또는 추가 롤아웃 없이 정책에 암시적 세계 모델을 가르칩니다. [example](https://github.com/huggingface/OpenEnv/blob/main/examples/echo_world_model/backends/tinker_echo_demo.py)를 확인해 보십시오.

![Inkling post-training metrics](https://huggingface.co/buckets/huggingface/inkling-blog-assets/resolve/trackio.png)

<details>
<summary>RL Example with Tinker and OpenEnv</summary>

```
git clone https://github.com/huggingface/OpenEnv.git
cd OpenEnv

# Add TINKER_API_KEY=... to .env, then run:
uv run --env-file .env \
  examples/echo_world_model/backends/tinker_echo_demo.py

```


</details>

**Transformer 강화학습(Transformers Reinforcement Learning)을 다루는 경우, 지식 증류 설정에서 Inkling을 교사 모델로 사용하는 것을 권장합니다. 예를 들어 Inkling의 문서 이해 능력을 활용하여 더 작은(on-device) 모델의 성능을 향상시킬 수 있습니다. [this example](https://github.com/huggingface/trl/blob/main/examples/scripts/gold.py)에서는 트랜스포머 강화학습 라이브러리와 GOLD 알고리즘을 사용하여 지식을 증류합니다. GOLD는 서로 다른 토크나이저 간에 토큰 로짓을 매칭하므로 허브의 어떤 모델로도 쉽게 증류할 수 있습니다.

## Slurm 스크립트 {#section-5}

클러스터에 Inkling을 배포하기 위해 트랜스포머 API로 서비스를 제공하는 SLURM 스크립트와 다양한 모달리티로 엔드포인트를 쿼리하는 방법을 제공합니다. 명령을 업데이트하여 이 스크립트를 vLLM이나 SGlang에 맞게 조정할 수 있습니다. 이 스크립트는 [here](https://huggingface.co/buckets/merve/inkling)에 있습니다.

- [Submit inference job](https://huggingface.co/buckets/merve/inkling/tree/slurm/submit_inkling_generate.sbatch)
- [Python generation script](https://huggingface.co/buckets/merve/inkling/tree/slurm/generate_inkling.py)

## 벤치마크 결과 {#section-6}

|     |     | Inkling | Nemotron 3 Ultra | Kimi K2.5 | Kimi K2.6 | GLM 5.2 | DeepSeek V4 Pro | Gemini 3.1 Pro (고급) | Claude Fable 5 (최대) | GPT 5.6 Sol (초고) |
|-----|-----|---------|------------------|-----------|-----------|---------|-----------------|-----------------------|----------------------|---------------------|
| **추론** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | HLE(텍스트 전용) | 29.7%   | 26.6%            | 29.4%     | 35.9%     | 40.1%   | 35.9%           | 44.7%                 | 53.3%                | 47.2%               |
|     | HLE(도구 포함) | 46.0%   | 37.4%            | 50.2%     | 54.0%     | 54.7%   | 48.2%           | 51.4%                 | 64.5%                | 55.0%               |
|     | AIME 2026 | 97.1%   | 94.2%            | 95.8%     | 96.4%     | 99.2%   | 96.7%           | 98.3%                 | –                    | 99.9%               |
|     | GPQA Diamond | 87.2%   | 86.7%            | 87.9%     | 91.1%     | 89.5%   | 88.8%           | 94.1%                 | 92.6%                | 94.1%               |
| **에이전트형(코딩)** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | SWEBench Verified | 77.6%   | 70.7%            | 76.8%     | 80.2%     | –       | 80.6%           | 80.6%                 | 95.0%                | –                   |
|     | SWEBench Pro (Public) | 54.3%   | 46.4%            | 50.7%     | 58.6%     | 62.1%   | 55.4%           | 54.2%                 | 80.0%                | 64.6%               |
|     | Terminal Bench 2.1 (Best Harness) | 63.8    | 56.4             | 51.3         | 71.3      | 82.7    | 64              | 73.8                  | 84.6                 | 89.5                |
|     | GDPVal-AA v2 | 1233    | 1164             | 1009         | 1190      | 1514    | 1307            | 962                   | 1760                 | 1748                |
| **에이전트형(일반)** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | MCP Atlas | 74.1%   | 44.7%                | 64.0%     | 68.1%     | 77.8%   | 73.2%           | 78.2%                 | 83.3%                | 81.8%               |
|     | Tau 3 Banking | 23.7%   | 13.8%            | 13.2%     | 20.6%     | 26.8%   | 25.8%           | 16.5%                 | 26.8%                | 33.0%               |
| **사실성** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | BrowseComp (w/ Ctx) | 77.1%   | –                | 74.9%     | 83.2%     | –       | 83.4%           | 85.9%                 | 88.0%                | 89.4%               |
|     | SimpleQA Verified | 43.9%   | 32.4%            | 36.9%     | 38.7%     | 38.1%   | 57.0%           | 77.3%                 | 68.3%                | 71.6%               |
|     | AA Omniscience | 1.0%    | -1.0%            | -8.0%     | 6.0%      | 4.0%    | -10.0%          | 33.0%                 | 40.0%                | 22.0%               |
| **챗(Chat)** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | IFBench | 79.8%   | 81.4%            | 70.2%     | 76.0%     | 73.3%   | 76.5%           | 77.1%                 | 63.5%                | 72.7%               |
|     | Global-MMLU-Lite | 88.7%   | 85.6%            | 84.0%     | 88.4%     | 89.2%   | 89.3%           | 92.7%                 | 93.3%                | 91.8%               |
| **비전** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | MMMU Pro (Standard 10) | 73.3%   | –                | 75.0%     | 79.0%     | –       | –               | 82.0%                 | 84.2%                | 83.0%               |
|     | Charxiv RQ | 78.1%   | –                | 77.5%     | 80.4%     | –       | –               | 80.2%                 | 86.5%                | 84.7%               |
|     | Charxiv RQ (with python) | 82.0%   | –                | 78.7%     | 86.7%     | –       | –               | 89.9%                 | 89.4%                | 87.8%               |
| **오디오** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | Audio MC | 56.6%   | –                | –         | –         | –       | –               | 66.8%                 | –                    | –                   |
|     | MMAU | 77.2%   | –                | –         | –         | –       | –               | 82.5%                 | –                    | –                   |
|     | VoiceBench | 91.4%   | –                | –         | –         | –       | –               | 94.3%                 | –                    | –                   |
| **안전(Safety)** |     |         |                  |           |           |         |                 |                       |                      |                     |
|     | FORTRESS (Adversarial) | 78.0%   | 77.6%            | 54.1%     | 65.6%     | 71.3%   | 36.0%           | 65.2%                 | 96.0%                | 82.4%               |
|     | FORTRESS (Benign) | 95.9%   | 90.5%            | 98.3%     | 97.2%     | 90.0%   | 98.5%           | 98.0%                 | 55.1%                | 98.1%               |
|     | StrongREJECT | 98.6%   | 98.7%            | 99.5%     | 99.8%     | 98.5%   | 98.6%           | 98.0%                 | 98.7%                | 98.5%               |
