---
layout: post
title: "Welcome Beam: 소유하고, 커스터마이즈하고, 확장할 수 있는 501B 오픈 모델"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/reflection-beam/thumbnail.jpg
image: assets/images/blog/posts/2026-10-16-reflection-beam/thumbnail.jpg
authors:
  - user: burtenshaw
  - user: merve
  - user: pcuenq
slug: "reflection-beam"
source_url: "https://huggingface.co/blog/reflection-beam"
source_published_date: "2026-10-16"
source_published_at: "2026-10-16T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Welcome Beam: a 501B open model you can own, customize, and scale](https://huggingface.co/blog/reflection-beam)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/reflection-beam -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Welcome Beam: 소유하고, 커스터마이즈하고, 확장할 수 있는 501B 오픈 모델

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/beam-banner.jpg" alt="Hugging Face mascot with colorful concentric waves on a blue background." width="624" style="max-width: 100%; height: auto; background-color: white;">

Reflection의 Beam이 Hugging Face에 공개되었습니다! 새로운 미국 연구소 Reflection의 첫 번째 모델이며, 오픈 모델입니다! 501B 파라미터(토큰당 23B 활성)를 사용하는 mixture-of-experts(MoEs) 모델로, 텍스트를 입력으로 지원하고 **1M 토큰의 컨텍스트 윈도우**를 제공하며 Apache 2.0 라이선스로 공개되었습니다.

Beam은 *코딩* 및 *에이전트* 사용 사례에 초점을 맞춥니다. 자체 하드웨어, 인프라 또는 클라우드에서 프런티어 규모의 모델을 실행하고, 자체 사용 사례에 맞게 커스터마이즈하려는 팀을 위한 모델입니다. Beam에는 모델에서 최고의 성능을 끌어내는 전용 오픈 소스 Rust 기반 하네스가 함께 제공됩니다.

Beam에는 네이티브 transformers 구현과 함께 vLLM 및 SGLang 같은 다운스트림 라이브러리 지원도 제공됩니다. Reflection 팀과 협력해 이 모델을 필요에 맞게 구축할 수 있도록 추론, SFT(지도 파인튜닝), RL(강화학습) 레시피를 준비했습니다. transformers, SGLang, vLLM에서 day-0 지원을 제공하는 것뿐만 아니라, 이 글에서는 모델 실행 방법, 사후 학습 방법, 그리고 이 모델로 구축한 내용을 소개합니다.

**TL;DR**

> * 토큰당 23B가 활성화되는 501B 파라미터 MoEs LLM. 52개 레이어, MoE 레이어당 256개 전문가, 토큰당 8개 라우팅.
> * **1M 토큰의 컨텍스트 윈도우**를 지원하는 텍스트 입력 및 텍스트 출력.
> * 체크포인트 형식: [**BF16**](https://huggingface.co/reflection/Beam-1.0-501B-A23B)**, [FP8](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8),** 및 [**NVFP4**](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4)
> * 턴 간 컨텍스트에 유지할 수 있는 추론(`preserve_thinking`), 전용 문법을 사용하는 도구 호출. 모델은 "low", "medium"(기본값), "high", "xhigh", "max"의 다섯 가지 추론 값을 지원합니다.
> * 토크나이저에 포함된 응답 템플릿.
> * DFlash2를 사용하는 speculative decoding 드래프트 모델.
> * Day-0 라이브러리: transformers, SGLang, vLLM. Miles를 사용한 전체 파라미터 RL.
> * [Rust-based custom harness.](https://developers.reflection.ai/mirror-overview)와 함께 제공됩니다.
> * Apache 2.0.

## Beam으로 구축하기 {#section-1}

Reflection은 Beam을 사용하는 최고의 프로젝트에 $10,000 상당의 Hub 크레딧을 제공합니다. 여러분이 할 일은 무언가를 만들고 Hub에서 공유하는 것뿐입니다.

* Spaces에서 Beam 모델을 사용하는 데모를 만들 수 있습니다.
* 선호하는 하드웨어에 맞게 양자화할 수 있습니다.
* 선호하는 문제를 위한 합성 데이터셋을 생성할 수 있습니다.
* Beam을 사용하거나 Beam을 위한 RL 환경을 구축할 수 있습니다.
* Beam을 벤치마크하고 결과를 공유할 수 있습니다.
* 우리를 깜짝 놀라게 할 수도 있습니다!

Beam으로 무언가를 구축했다면 Hub에서 공유하고 $10,000를 받을 기회를 얻어 보세요. Space, 모델 또는 데이터셋 저장소를 구축해 [model’s repo page](https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5)에서 공유하기만 하면 됩니다. 모든 링크를 수집하고 최고의 프로젝트를 선정한 다음 Hub 계정으로 크레딧을 보내 드립니다.

## Beam이란? {#section-2}

Beam은 긴 컨텍스트(1M 컨텍스트 길이) 추론과 에이전트 작업을 위해 구축된 텍스트 전용 모델입니다. 희소 MoE 모델로, 각 토큰은 네트워크의 작은 일부만으로 계산되므로 대형 모델의 용량을 유지하면서 훨씬 작은 모델에 가까운 속도로 실행됩니다.

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/beam-architecture.png" alt="Beam architecture: three dense layers followed by 49 mixture-of-experts layers, with hybrid attention and weighted residual connections." width="624" style="max-width: 100%; height: auto; background-color: white;">

### Attention

디코더는 hidden size 5,120의 52개 레이어로 구성됩니다. 대부분의 레이어는 이전 512개 토큰에 대해 sliding-window attention을 사용하며, 6번째 레이어마다 그리고 마지막 레이어에서는 전체 컨텍스트에 어텐션을 적용합니다. 따라서 sliding-window 레이어는 43개, global 레이어는 9개입니다.

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/beam-attention.png" alt="Attention block with grouped-query attention, RoPE or scalable-softmax query scaling, and a learned sigmoid gate." width="624" style="max-width: 100%; height: auto; background-color: white;">

*어텐션 블록 내부(두 레이어 유형 모두)*

두 종류의 레이어는 위치 정보를 서로 다르게 처리합니다. sliding-window 레이어는 짧은 윈도우 내에서 상대적 위치가 가장 중요하므로 RoPE를 사용합니다. **Global 레이어는 위치 인코딩을 사용하지 않습니다**. 대신 각 쿼리에 보이는 토큰 수에 따라 느리게 증가하는 스케일러를 곱합니다(`(1 + (pos + 1) / 2048) ** 0.1`). 이는 temperature처럼 작동합니다. 컨텍스트가 길어져도 softmax가 평탄화되지 않고 정밀도를 유지합니다. 구현에서는 이를 scalable-softmax query scaling이라고 부릅니다.

어텐션은 48개의 쿼리 헤드와 8개의 key-value 헤드를 사용하는 grouped-query attention을 적용하며, 쿼리와 키에는 RMSNorm을 사용합니다. 레이어 입력에서 계산되는 학습된 sigmoid 게이트가 출력 프로젝션 전에 모든 헤드의 모든 채널을 스케일링합니다.

### Mixture of Experts

처음 3개 레이어는 dense SwiGLU MLP를 사용합니다. 나머지 49개 레이어는 이를 256개 전문가로 구성된 Mixture-of-Experts 블록으로 대체합니다. 각 전문가는 intermediate size가 2,560인 SwiGLU MLP이며, 약 39M개의 파라미터를 가집니다. 전문가가 모델 파라미터의 98.4%를 차지합니다.

라우터는 DeepSeek-V3의 group-limited 방식을 따릅니다. sigmoid로 256개 전문가 모두에 점수를 매기고, 이를 32개씩 8개 그룹으로 나눈 다음, 각 그룹의 상위 두 점수의 합으로 순위를 매겨 가장 좋은 4개 그룹을 유지하고, 그 안에서 상위 8개 전문가를 선택합니다. Beam은 DeepSeek-V3와 동일한 그룹 크기를 사용하지만 shared expert는 사용하지 않습니다. 전문가 출력은 정규화된 라우터 가중치로 결합하고 2.5를 곱해 스케일링합니다.

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/beam-expert-routing.png" alt="Beam router scores 256 experts and selects groups before routing each token to eight experts." width="624" style="max-width: 100%; height: auto; background-color: white;">

토큰당 8개의 전문가만 실행되므로 모델은 각 토큰을 약 23B개의 파라미터로 계산합니다. 256개 전문가 모두 메모리에 유지되므로 배포에는 501B 모델에 해당하는 메모리가 필요하지만, 실행 속도는 23B 모델과 비슷합니다. BF16 체크포인트는 safetensors 형식으로 약 1 TB입니다. NVFP4 체크포인트는 NVIDIA Blackwell GPU를 위해 대부분의 파라미터를 4비트 정밀도로 저장하며, 정확도를 위해 일부 파라미터는 BF16으로 유지합니다. Hub에는 Hopper 및 Ampere 시리즈 칩에 적합한 FP8 체크포인트도 있습니다.

Beam을 직접 배포하려면 아래의 vLLM 또는 SGLang 레시피를 확인하세요.

### Residual Stream

Residual connection이 가장 독특한 부분입니다. 표준 transformer는 각 서브레이어의 출력을 가중치 1로 스트림에 더하므로, 깊이가 깊어질수록 스트림이 커집니다.

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/standard-residual-stream.png" alt="Standard transformer residual stream adds each sublayer output with weight one." width="624" style="max-width: 100%; height: auto; background-color: white;">

Beam은 입력 임베딩을 1단계로 계산합니다. 단계 `k`에서 서브레이어 출력에 가중치 `1/sqrt(k)`을 적용해 더하고, 기존 스트림은 가중치 `sqrt((k-1)/k)`로 유지합니다. 두 가중치의 제곱 합은 1이므로 첫 번째 레이어부터 마지막 레이어까지 스트림은 일정한 스케일을 유지합니다. 총 104번의 덧셈이 끝나면 최종 hidden state에서 모든 서브레이어 출력과 입력 임베딩이 동일한 가중치를 갖습니다.

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/beam-residual-stream.png" alt="Beam residual stream uses depth-dependent weights to keep the accumulated hidden state at a constant scale." width="624" style="max-width: 100%; height: auto; background-color: white;">

모든 서브레이어는 전후로 정규화되며, 최종 logits는 FP32로 계산되는 ±30 범위의 tanh softcap을 통과합니다. 구현에서는 router, SwiGLU product, norms, RoPE, query scaling, softcap을 FP32로 계산하도록 위치를 고정합니다.

### Long Context

Hybrid attention은 긴 컨텍스트에서 효과적입니다. 각 레이어는 토큰당 2 x 8 KV heads x 128 dims = 2,048개의 값을 캐시합니다. 52개 레이어 모두가 모든 토큰을 유지한다면 캐시는 토큰당 208 KiB씩 증가합니다. 전체 시퀀스를 유지하는 것은 9개의 global 레이어뿐입니다. 43개의 sliding-window 레이어는 각각 고정된 512토큰 윈도우를 유지하므로 더 이상 커지지 않습니다. 따라서 캐시는 토큰당 36 KiB씩 증가하며, 단순 계산한 증가율의 약 6분의 1입니다.

| 컨텍스트 | 52개 레이어가 모두 global인 경우 | Beam |
| :---- | :---- | :---- |
| 32K 토큰 | 6.5 GiB | 1.2 GiB |
| 128K 토큰 | 26 GiB | 4.6 GiB |
| 전체 컨텍스트(1M) | 208 GiB | **36.08 GiB** |

이 수치는 transformers dynamic cache를 사용하는 BF16 기준 시퀀스당 수치입니다.

## Benchmark Results {#section-3}

다음은 Reflection 팀이 제공한 다양한 정밀도 수준에서의 벤치마크 결과입니다.

|  | *Beam (BF16)* | *Beam (FP8)* | *Beam (NVFP4)* |
| :---- | :---: | :---: | :---: |
| **Agentic Coding / Terminal**  |  |  |  |
| DeepSWE v1.1 | 43.3% | 43.0% | 42.7% |
| SWE Bench Pro v2-Hard | 76.5% | 78.4% | 78.6% |
| SWE Bench Pro v1 | 66.1% | 65.9% | 65.6% |
| Terminal Bench v2.1 | 80.1% | 78.9% | 78.2% |
| SWE Atlas Codebase QnA | 31.8% | 32.2% | 31.4% |
| SWEBench Multilingual | 78.3% | 78.1%  | 77.5% |
| SWEBench Verified | 81.6% | 80.2% | 80.7% |
| **Reasoning** |  |  |  |
| AIME 2026 | 97.7% | 98.0% | 97.5% |
| HLE (no tools) | 36.0% | 36.0% | 34.8% |
| HLE (with tools) | 51.1% | 51.1% | 51.1% |
| SciCode | 49.2% | 48.3% | 46.9% |
| CriPT AA | 16.3% | 17.4% | 15.4% |
| GPQA Diamond | 90.6% | 90.3% | 89.3% |
| LiveCodeBench | 88.3% | 88.1% | 87.4% |
| **Tool Calling / Search** |  |  |  |
| AutomationBench \- public split only | 39.3 | 38.2 | 39.8% |
| MCP Atlas | 80.8% | 81.5% | 80.5% |
| Tau3 Banking | 36.1% | 36.8% | 35.4% |
| BrowseComp w/ context management | 76.8% | 76.7% | 75.6% |
| DeepSearchQA w/ context management | 80.5% | 79.6% | 81.1% |
| **General Capabilities**  |  |  |  |
| AA-LCR | 78.2% | 79.4% | 78.7% |
| LongBench v2 | 67.5% | 67.6% | 67.2% |
| IFBench | 79.6% | 80.7% | 79.2% |
| AA Omniscience (index) \- public split only | 10.8 | 8.8 | 9.8 |

## Hardware Requirements {#section-4}

모델을 제공하는 가장 실용적인 방법은 bf16 체크포인트의 품질 대부분을 유지하는 NVFP4 체크포인트를 사용하는 것입니다. NVFP4는 일부 가중치를 4비트로 저장하므로 BF16에서 1 TB였던 501B 파라미터를 \~280 GB로 줄입니다. Blackwell GPU는 NVFP4를 네이티브로 실행하지만 H200처럼 Blackwell보다 오래된 GPU에서는 실행할 수 없습니다. FP8 체크포인트는 Blackwell보다 오래된 GPU에 적합합니다.

B200 하나에는 180 GB의 HBM3e가 있으므로, 두 개(360 GB)면 Beam 가중치를 저장할 수 있지만 활성값과 KV 캐시를 위한 여유 공간은 거의 남지 않습니다. 장시간 컨텍스트와 동시 요청을 위한 충분한 공간을 확보하려면 B200 4개(720 GB)를 계획하세요. 8xB200 전체 노드에서 제공하거나, 각각 288 GB인 B300 2개 또는 8xRTX Pro 6000에서도 제공할 수 있습니다. Blackwell 하드웨어가 없다면 Inference Providers(서버리스)를 사용해 Reflection을 테스트하세요.

## Inference Support {#section-5}

Beam은 transformers, SGLang, vLLM에서 day-0 지원을 제공합니다.

### Transformers

최신 버전의 transformers를 설치하세요.

```sh
pip install -U "transformers>=5.20.1"
```


Beam을 실행하는 가장 쉬운 방법은 text-generation pipeline을 사용하는 것입니다.

```py
from transformers import pipeline

pipe = pipeline(
    "text-generation", model="reflection/Beam-1.0-501B-A23B", device_map="auto"
)

messages = [
    {"role": "user", "content": "Write a limerick about Python exceptions."}
]
output = pipe(messages, max_new_tokens=2000)
output[0]["generated_text"][-1]

```


pipeline은 전체 채팅을 반환합니다. 마지막 메시지는 응답 템플릿에 의해 이미 파싱된 Beam의 응답입니다.

한 단계 더 낮은 수준에서는 `AutoModelForCausalLM` 및 `AutoTokenizer`을 사용합니다.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "reflection/Beam-1.0-501B-A23B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype="auto", device_map="auto"
)

```


#### Chat

```py
messages = [
    {"role": "system", "content": "You should only answer with a number."},
    {"role": "user", "content": "What is 17 * 23?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort="medium" # by default
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=2000)

tokenizer.parse_response(
    outputs[0][input_len:], prefix=inputs["input_ids"][0]
)
```


Beam의 토크나이저에는 응답 템플릿이 포함되어 있습니다. `parse_response`은 이를 읽고 원시 출력을 메시지 dict로 변환합니다. 이 dict에는 추론, 답변, 도구 호출이 각각 별도 필드로 들어갑니다. 따라서 추론, 답변, 도구 호출을 분리하기 위한 일반적인 regex가 필요하지 않습니다. 또한 규칙이 토크나이저와 함께 제공되므로, 파싱된 구조가 Beam이 학습하도록 설계된 출력과 일치합니다. 채팅 템플릿은 assistant turn의 일부를 미리 작성할 수 있으며, parser는 이를 확인해야 합니다.

#### Tool calling

<details>
<summary>Complete tool-calling example</summary>

```py
def get_weather(city: str, unit: str = "celsius"):
    """
    Get the current weather for a city.

    Args:
        city: The city to look up.
        unit: Temperature unit, "celsius" or "fahrenheit".
    """
    ...

tools = [get_weather]
messages = [
    {"role": "user", "content": "I land in Paris tonight. Do I need a coat?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=1000)

message = tokenizer.parse_response(
    outputs[0][input_len:], prefix=inputs["input_ids"][0], tools=tools
)
message.get("tool_calls")

```


</details>

`tools`도 `parse_response`에 전달하세요. Beam은 자체 XML 스타일 문법으로 도구 호출 인자를 인라인으로 작성하므로, 단순한 `7`는 숫자일 수도 있고 문자열일 수도 있습니다. 도구 스키마가 있으면 parser가 각 인자를 선언된 타입으로 변환합니다.

#### Multi-GPU and multi-node

Beam의 config는 expert-parallel plan을 정의합니다. Expert parallelism은 각 전문가를 하나의 디바이스에 배치합니다. Attention, dense 레이어, 임베딩, LM head는 모든 디바이스에 복제됩니다. 이는 8B 파라미터, 즉 BF16에서 16 GB입니다. `DistributedConfig`을 사용해 모델을 로드하세요.

```py
import os

from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "reflection/Beam-1.0-501B-A23B",
    distributed_config=DistributedConfig(
        tp_size=int(os.environ["WORLD_SIZE"]), ep_size=int(os.environ["WORLD_SIZE"]), tp_plan={}
    ),
)

```


16개의 H200 GPU에서 `torchrun`을 사용해 스크립트를 실행하세요. 그러면 각 GPU가 78 GB의 가중치를 보유합니다. GPU가 8개라면 각각 139 GB를 보유하게 되어 141 GB H200에는 여유 공간이 남지 않습니다. 디바이스 수는 전문가 수인 256을 나눌 수 있어야 합니다.

### vLLM

Beam은 vLLM에서 day-0 지원을 제공합니다. 아래 명령은 16개 GPU(8x H200 노드 2개)에 걸쳐 tensor parallelism을 사용해 BF16 체크포인트를 제공하고 OpenAI 호환 API를 노출합니다.

```sh
vllm serve reflection/Beam-1.0-501B-A23B --tensor-parallel-size 16 --served-model-name beam
```


더 빠른 생성을 위해 DFlash2 speculative decoding을 추가하세요. 작은 드래프트 모델이 최대 15개의 다음 토큰을 제안하고 Beam이 이를 한 번의 forward pass에서 모두 확인하므로, 각 단계에서 여러 토큰을 수락할 수 있습니다. 비용은 드래프트 모델을 위한 약간의 추가 메모리입니다.

```
vllm serve reflection/Beam-1.0-501B-A23B \
  --tensor-parallel-size 16 \
  --speculative-config '{ "method": "dflash", "model": "reflection/dflash2-drafter-preview", "num_speculative_tokens": 15 }'
```


두 명령 모두 OpenAI 호환 서버를 시작하므로 모든 OpenAI 클라이언트를 `http://localhost:8000/v1`에 연결할 수 있습니다. 추가 serving 구성은 [proposed Beam recipe for vLLM](https://github.com/vllm-project/recipes/pull/1132)을 참조하세요.

### SGLang

Beam은 SGLang에서도 day-0 지원을 제공합니다. 아래 두 명령에서 모두 활성화하는 DFlash2 speculative decoding과 함께 제공하는 것을 권장합니다. Blackwell에서는 NVFP4 체크포인트가 B300 4개에 들어갑니다.

```sh
python -m sglang.launch_server \
--model-path reflection/Beam-1.0-501B-A23B-NVFP4 \
--served-model-name beam \
--tp 4 \
--attention-backend trtllm_mha \
--page-size 64 \
--context-length 1048576 \
--mem-fraction-static 0.85 \
--enable-unified-memory \
--speculative-algorithm DFLASH \
--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2 \
--speculative-dflash-block-size 8 \
--tool-call-parser beam --reasoning-parser beam \
--host 0.0.0.0 --port 30000
```


Hopper GPU에서는 대신 8개의 H200에서 FP8 체크포인트를 사용하세요. 변경되는 플래그는 세 개뿐입니다. 체크포인트, tensor-parallel size(4 대신 8), attention backend(trtllm_mha 대신 fa3)입니다. trtllm_mha에는 Blackwell이 필요합니다.

```
python -m sglang.launch_server \
--model-path reflection/Beam-1.0-501B-A23B-FP8 \
--served-model-name beam \
--tp 8 \
--attention-backend fa3 \
--page-size 64 \
--context-length 1048576 \
--mem-fraction-static 0.85 \
--enable-unified-memory \
--speculative-algorithm DFLASH \
--speculative-draft-model-path reflection/Beam-1.0-501B-A23B-DFlash2 \
--speculative-dflash-block-size 8 \
--tool-call-parser beam --reasoning-parser beam \
--host 0.0.0.0 --port 30000

```


### Inference Providers

Beam은 Baseten, Fireworks, Together가 제공하는 Hugging Face Inference Providers에서 사용할 수 있으며, 곧 더 많은 제공자가 추가될 예정입니다.

```py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="reflection/Beam-1.0-501B-A23B:auto",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(completion.choices[0].message)

```


### Inference Endpoints

[FP8](https://endpoints.huggingface.co/new/reflection/Beam-1.0-501B-A23B-FP8) 또는 [NVFP4](https://endpoints.huggingface.co/new/reflection/Beam-1.0-501B-A23B-NVFP4)의 카탈로그에서 Inference Endpoints에 원클릭으로 배포하거나, `hf` cli로 다음과 같이 설치할 수 있도록 Beam 레시피를 준비했습니다.

```sh
hf endpoints catalog deploy --repo reflection/Beam-1.0-501B-A23B --name beam
```


## What we tested {#section-6}

Reflection 플랫폼을 통해 다섯 가지 작업에서 Reflection Beam-501B-A23B의 vibe evals를 공유합니다. 각 작업은 낮은 추론 노력과 높은 추론 노력으로 한 번씩 실행했습니다. 이는 벤치마크가 아닌 vibe evals입니다. Beam 사용을 계획하고 있다면 특정 사용 사례에서 어떻게 동작하는지, 그리고 중요한 점으로 어디에서 강점과 약점을 보이는지 확인해 볼 수 있는 좋은 방법입니다.

아래에는 실행한 프롬프트와 Hub의 출력 trace를 요약했습니다. 전체 입력, 출력, API 사용량, 작업 fixture, grader, 대화 trace는 [this public Hugging Face bucket](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12)에 있습니다.

| 범주 | 프롬프트 설명 | 낮은 추론(토큰) | 높은 추론(토큰) |
| :---- | :---- | :---- | :---- |
| 다단계 도구 호출 | ORD-742를 처리합니다. 재고, 예약, 추가 요금, 도착 날짜를 비교하고 자격을 충족하는 가장 저렴한 창고를 예약합니다. | ✅ [544](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/multi_step_tools-low.jsonl) | ✅ [811](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/multi_step_tools-high.jsonl) |
| 코드베이스에 대한 긴 컨텍스트 검색 | 세 개의 완전한 소스 파일에서 Click 명령 구성과 두 가지 호출 결과를 도출하고, 구현 심볼 여섯 개를 인용합니다. | 🔴 [1,613](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/long_context-low.jsonl) \* | ✅ [5,007](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/long_context-high.jsonl) |
| 샌드박스의 터미널 작업 | CSV 원장 CLI를 검사하고 수정합니다. 모든 버그를 고치고, 테스트를 보존하며, 테스트와 CLI 검사를 실행합니다. | ✅ [582](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/terminal-low.jsonl) | ✅ [1,548](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/terminal-high.jsonl) |
| 수학 추론(AIME 스타일) | {1, …, 16}의 8원소 부분집합 중 합이 17로 나누어떨어지는 것은 몇 개인가요? 근거를 제시하세요. | 🟠 [829](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/math-low.jsonl) † | ✅ [2,809](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/math-high.jsonl) |
| 구조화된 출력 | 순서가 뒤섞인 수정 로그를 엄격한 JSON으로 변환합니다. 패치, 취소, null, 날짜, 통화, 합계를 병합합니다. | ✅ [1,040](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/structured_output-low.jsonl) | ✅ [2,088](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12/tree/traces/structured_output-high.jsonl) |

*\* 낮은 추론의 긴 컨텍스트 응답은 모든 코드 동작과 여섯 개의 소스 인용을 정확히 맞혔습니다. 하지만 응답이 parser 형식과 일치하지 않았으므로, 이는 부분적으로 프롬프트의 모호성 때문입니다.*

† 낮은 추론은 정답을 반환했지만 요청된 근거를 생략했습니다. 높은 추론은 독립적으로 검토된 올바른 unity roots 증명을 제시했습니다.

우리의 말만 믿지 마세요. [bucket](https://huggingface.co/buckets/burtenshaw/beam-vibe-evals-2026-10-12)을 자세히 살펴보고, 여러분의 사용 사례나 프로젝트에 맞게 다시 작성해 Beam을 한계까지 밀어붙여 보세요.

## Post-training Beam {#section-7}

Beam을 자체 사용 사례에 맞게 파인튜닝할 수 있습니다. 여기에서는 지도 파인튜닝과 강화학습을 시작할 수 있도록 TRL과 Miles를 사용하는 몇 가지 예제를 공유합니다.

### SFT with TRL

자체 데이터에 Beam을 적용하는 가장 빠른 방법은 TRL의 SFTTrainer를 사용한 지도 파인튜닝입니다. 메모리 사용량을 관리할 수 있도록 501B 파라미터 전체를 업데이트하는 대신 PEFT로 LoRA adapter를 학습합니다.
`SFTTrainer`을 사용해 Beam을 간단한 목표에 맞게 조정했습니다. 사용자가 요청한 언어로 추론을 유지하는 것입니다. LoRA를 사용해 [Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking)에서 파인튜닝했으며, 이 방식은 base model을 고정한 채 소수의 adapter 가중치만 학습합니다.

이 설정은 로드 시 모델 샤딩, expert parallelism, FlashAttention 4, sequence packing을 결합합니다. LoRA SFT는 8개의 B300 GPU가 장착된 단일 노드에 들어가며 초당 약 19,000토큰에 도달합니다. 이에 해당하는 H100 설정은 4개 노드를 사용합니다.

Multilingual-Thinking을 한 epoch 학습한 후 Beam은 16개의 held-out 프롬프트 중 16개 모두에서 요청된 언어로 추론했습니다. 반면 base model은 16개 중 3개였습니다. 이 소규모 평가는 추론 자체의 정확도가 아니라 언어 준수 여부를 측정합니다.

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/trl-sft-training.png" alt="Training loss and mean token accuracy over a 500-step LoRA supervised fine-tuning run on one eight-GPU B300 node." width="624" style="max-width: 100%; height: auto; background-color: white;">

*8-GPU B300 노드 하나에서 open-r1/Mixture-of-Thought의 추론 하위 집합을 사용해 별도로 500-step 실행을 진행하는 동안의 학습 loss(왼쪽)와 평균 토큰 정확도(오른쪽)입니다. 이 학습 지표는 공식 [TRL example PR](https://github.com/huggingface/trl/pull/7629)에서 가져왔습니다.*

### GRPO with Miles

Reflection과 RadixArk는 BF16에서 Miles로 GRPO를 사용해 모델을 학습하는 레시피를 구축했습니다. 전체 모델을 업데이트하거나, base 가중치를 고정한 채 LoRA로 adapter를 학습할 수 있습니다. Megatron이 학습하고, SGLang이 rollout을 생성하며, Miles는 각 업데이트 후 새 가중치를 rollout 엔진에 게시합니다. 이 레시피를 검토했습니다.

| 워크로드 | 참조 구성 |
| :---- | :---- |
| 추론 / 평가, BF16 | 엔진당 16x H200 |
| 전체 파라미터 GRPO, BF16(Miles) | 96x H200: 학습 64개, rollout 32개. 또는 colocated 64x H200. |
| LoRA GRPO, BF16(Miles) | 32x H200: 학습 16개, rollout 16개. |
| 전체 파라미터 GRPO, BF16, 128토큰 컨텍스트(Miles) | 48x H200: 학습 32개, rollout 16개. Reflection과 RadixArk가 실제 체크포인트에서 처음 검증한 설정입니다. |

이 레시피는 DAPO-Math-17K에서 GRPO를 실행하고 AIME25에서 평가합니다. 이미지를 빌드하고 데이터를 다운로드한 다음, 8x H200 노드 12개로 구성된 Ray 클러스터에 작업을 제출하세요.

```sh
docker build -f docker/Dockerfile.beam -t miles-beam .

python scripts/run_beam.py prepare \
  --model-id reflection/Beam-1.0-501B-A23B \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets

export MASTER_ADDR=10.0.0.1
export RAY_ADDRESS=http://${MASTER_ADDR}:8265

python scripts/run_beam.py execute \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets \
  --output-dir /root/shared_data/beam-run \
  --num-rollout 100
```


rollout에서 일어나는 과정:

**샘플링:** 32개의 프롬프트를 각각 8개의 응답과 함께 모델에 전달합니다. 두 개의 SGLang 엔진이 각각 16개 GPU에서 rollout을 제공합니다.

**점수:** boxed answer를 label과 비교해 확인하며, 길이 제한에 가까운 응답에는 소프트 패널티를 적용합니다.

**업데이트:** 256개 샘플 전체에 대해 optimizer step을 한 번 수행합니다. truncated importance sampling을 사용하는 group-relative advantage이며, KL 또는 entropy penalty는 없습니다.

**병렬 학습:** tensor parallelism 8, expert parallelism 8, data parallelism 8을 사용하며, sequence parallelism과 full recompute를 적용합니다.

**Optimizer:** 일정한 1e-6의 Adam을 사용합니다. 그래디언트는 BF16, master weights는 FP32, moments는 BF16입니다. Optimizer state는 노드 로컬 NVMe를 통해 스트리밍되므로 각 trainer 노드에 2 TB의 scratch 공간이 필요합니다.

**평가:** rollout 5회마다 AIME25를 평가하며, 문제당 8개의 샘플을 사용합니다. 체크포인트는 20회마다 저장합니다.

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/miles-grpo-aime2025.png" alt="Full-parameter GRPO evaluation on AIME 2025: accuracy increases from 46.25% to 55.8% with a 4,000-token output limit." width="624" style="max-width: 100%; height: auto; background-color: white;">

현재 rollout을 학습하는 동안 다음 rollout이 생성되므로 응답은 한 업데이트 뒤처집니다. 마지막 체크포인트에서 계속하려면 `--resume`을 추가하세요. `execute-colocate`은 8개 노드에서 동일한 레시피를 실행하며, 모든 GPU가 차례로 학습과 서빙을 수행합니다.

### Train with LoRA

8× H200 노드 4개에서 LoRA와 함께 GRPO를 실행할 수도 있습니다. 학습에는 GPU 16개, rollout에는 GPU 16개를 사용합니다. 이 레시피는 base 가중치를 고정한 채 Beam의 attention, dense MLP, expert 레이어에 rank-16 adapter를 학습합니다. SGLang과 동기화되는 것은 adapter 가중치뿐이며, optimizer state는 GPU 메모리에 유지되므로 NVMe optimizer offloading이 필요하지 않습니다.

```sh
python scripts/run_beam_lora.py execute \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets \
  --output-dir /root/shared_data/beam-run \
  --num-rollout 100
```


<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/miles-lora-grpo-aime2025.png" alt="LoRA GRPO evaluation on AIME 2025: accuracy rises from 42.9% to 64.2% while mean output length falls from 3,188 to 2,197 tokens." width="624" style="max-width: 100%; height: auto; background-color: white;">

## Build with Beam: Demos {#section-8}

Beam으로 무언가를 구축했다면 Hub에서 공유하고 $10,000를 받을 기회를 얻어 보세요. Space, 모델 또는 데이터셋 저장소를 구축해 [model’s repo page](https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5)에서 공유하기만 하면 됩니다. 영감을 얻을 수 있도록 저희가 구축한 몇 가지를 소개합니다.

### Hey Beam, 네이티브 로컬 MacOS 애플리케이션을 만들어 줘!

Hugging Face에서는 하드웨어에서 네이티브로 실행되는 오픈 및 로컬 모델을 좋아하지만, Mac에서 네이티브 앱을 만드는 일은 어려울 수 있습니다! Beam이 도움을 줄 수 있는지 살펴보겠습니다.

Beam에 음성을 텍스트로 변환하고 편리하게 커서 위치에 입력하는 네이티브 MacOS 애플리케이션을 만들어 달라고 요청했습니다. Mac Metal을 최대한 활용하기 위해 MLX를 사용하고, 뛰어난 음성-텍스트 변환을 위해 Parakeet Model을 사용하고자 합니다.

```
Build BeamSpeak, a native macOS menu bar dictation app with a compact recording pill and no main window or account.

Use ⌘⇧Space to start and stop microphone recording, showing elapsed time, microphone level, and transcription status. Run speech recognition entirely locally using a Hugging Face model, preferably Parakeet v3 through MLX, and keep the model loaded between recordings. Paste the transcript into the original app if it remains focused and Accessibility permission is granted; otherwise copy it to the clipboard, preserving newer clipboard changes and never pressing Enter. Handle permissions, cancellation, silence, errors, a two-minute recording limit, and temporary audio cleanup. Deliver source code, a runnable BeamSpeak.app, setup instructions, and verification of offline transcription, the physical shortcut, microphone capture, and text delivery.
```


Pi coding harness와 함께 Beam을 사용해 MLX로 Parakeet을 실행하는 초기 네이티브 macOS 메뉴 바 앱을 구축했습니다. 저장된 실행 기록에는 후속 프롬프트가 포함되어 있으며, 최종 소스에는 이후 Codex 수정도 포함되어 있습니다. 위 프롬프트는 구축 요구 사항을 축약한 버전입니다.

<figure class="image text-center">
  <video controls width="100%">
    <source src="https://huggingface.co/buckets/burtenshaw/beamspeak-demo/resolve/beamspeak-demo.mp4" type="video/mp4">
  </video>
</figure>

Beam이 pi에서 이를 구축한 agent trace는 [this trace](https://huggingface.co/buckets/burtenshaw/beamspeak-demo/tree/trace/beam-build.sanitized.jsonl)에서 확인할 수 있습니다. 데모를 사용해 보려면 [source code](https://huggingface.co/buckets/burtenshaw/beamspeak-demo/resolve/beamspeak-source.zip)을 다운로드하거나 [setup instructions](https://huggingface.co/buckets/burtenshaw/beamspeak-demo/resolve/RUNNING.md)을 읽어 보세요.

### Hey Beam, 여러 에이전트에서 어떻게 확장해?

프로그래밍 작업을 여러 에이전트로 나누면 Beam이 더 빠르게 진행할 수 있는지 확인하고자 했습니다. 오픈 소스 Pi coding agent와 Beam을 결합하고 세 가지 설정을 비교했습니다. 1. 혼자 작업하는 에이전트 하나, 2. 공유 Git 저장소를 통해 협력하는 지속형 peer 5개, 3. 비동기 subagent에 작업을 위임하는 lead agent입니다. 하네스는 Hugging Face jobs에서 실행했으며, 추론은 Reflection 플랫폼 API를 사용했습니다.

작업에는 ProgramBench를 사용했습니다. 에이전트는 소스 코드나 평가자의 테스트를 보지 않고 컴파일된 실행 파일과 문서만으로 프로그램을 재현해야 합니다. 각 설정에는 8M API 토큰의 동일한 상한과 작업당 2시간의 solver 제한이 적용되었으며, 이는 해당 설정의 에이전트 전체가 공유했습니다. 구현 snapshot을 저장하고 [ProgramBench-Tests dataset](https://huggingface.co/datasets/programbench/ProgramBench-Tests)의 고정된 hidden-test suite를 기준으로 평가했습니다.

저희 [score-versus-latency plot](https://burtenshaw-beam-pi-programbench.hf.space/study)은 통과한 테스트의 평균 비율을 보여 줍니다. 시간 축은 경과한 wall time이 아니라 모델 토큰에 대한 고정된 기준 속도와 측정된 도구 시간을 사용합니다. 처음에는 다섯 가지 작업을 실행했습니다. 현재 그래프에는 jq, xsv, FIGlet, hexyl의 사후 subset이 표시되며, 저장된 산출물의 채점이 너무 느려 ripgrep은 세 설정 모두에서 제외했습니다. 게시된 결과에는 원래의 15개 episode가 모두 포함됩니다.

<figure>
<iframe
	src="https://burtenshaw-beam-pi-programbench.hf.space/study"
	title="Beam ProgramBench study: test pass rate versus estimated latency"
	loading="lazy"
	frameborder="0"
	width="100%"
	height="780"
></iframe>
</figure>

[Trackio dashboard](https://burtenshaw-beam-pi-programbench.hf.space/?project=beam-pi-programbench-20261009)에서 실행 기록을 검사하고, [published result data](https://huggingface.co/spaces/burtenshaw/beam-pi-programbench/tree/main/results/beam-pi-programbench-20261009)을 탐색하며, [study harness](https://huggingface.co/spaces/burtenshaw/beam-pi-programbench/tree/main/source)을 재사용할 수 있습니다.

### Hey Beam, 내 코드베이스를 위한 decision model을 파인튜닝해 줘!

Decision model은 많은 사용 사례에서 매우 빠르고 실용적이기 때문에 현재 큰 인기를 얻고 있습니다. Beam의 할당량이 소진된 후 Pi에서 Beam으로 시작해 개발자 incident triage를 위한 맞춤형 decision model을 학습하고 평가했습니다.

코딩 에이전트가 incident-triage policy를 작고 빠른 decision model로 바꿀 수 있는지 확인하고자 했습니다. 작업은 개발자 incident를 읽고 담당 팀, 심각도, 즉시 누군가에게 알림을 보내야 하는지를 선택하는 것이었습니다. Unsloth를 사용해 Qwen3.5-0.8B를 파인튜닝하여 이러한 선택지를 직접 점수화했습니다.

먼저 워크플로는 training, calibration, testing을 위해 incident cluster별로 나눈 2.3k개의 incident가 포함된 합성 데이터셋을 구축했습니다. Science Slurm 클러스터의 H100 80GB 하나에서 4비트 backbone을 사용해 LoRA adapter와 decision head를 3개 epoch 동안 학습했습니다.

기준선으로는 예시 없이 전체 triage policy와 ticket을 사용해 적응하지 않은 모델에 프롬프트를 제공하고, 세 가지 결정을 JSON으로 생성하도록 요청했습니다. 동일한 320개의 합성 테스트 ticket에서 파인튜닝된 decision model은 세 가지 답을 모두 맞힌 비율이 76.25%였으며, prompted zero-shot은 4.69%였습니다. 평균 ticket 지연 시간은 1,218.06 ms에서 102.58 ms로 줄어 11.9배 빨라졌습니다.

<img src="https://huggingface.co/buckets/huggingface/Beam-blog-assets/resolve/images/decision-model-results.png" alt="Incident triage decision model: accuracy rises from 4.69% to 76.25% and mean latency falls from 1,218.06 ms to 102.58 ms." width="624" style="max-width: 100%; height: auto; background-color: white;">

이 실험을 직접 시도해 보려면 데이터, 코드, 체크포인트, 결과가 포함된 [artifact bucket](https://huggingface.co/buckets/burtenshaw/qwen35-incident-triage-reproduction)을 확인하세요.

## Wrapping Up {#section-9}

Beam은 자체 하드웨어에서 실행하고 변경할 수 있는 501B 오픈 모델입니다. [**BF16**](https://huggingface.co/reflection/Beam-1.0-501B-A23B) 체크포인트로 시작하거나 [**FP8**](https://huggingface.co/reflection/Beam-1.0-501B-A23B-FP8)**,** [**NVFP4**](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4) 중 하나를 사용해 보세요. Beam으로 무언가를 구축했다면 [discussions tab](https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions)에서 공유해 주세요.
