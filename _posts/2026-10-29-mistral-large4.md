---
layout: post
title: "Hugging Face의 Le Chonk 소개: Mistral Large 4"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/ml4/thumbnail.png
image: assets/images/blog/posts/2026-10-29-mistral-large4/thumbnail.png
authors:
  - user: merve
  - user: alvarobartt
  - user: ror
  - user: cyrilvallez
  - user: qgallouedec
  - user: ehcalabres
  - user: juanjucm
  - user: ngxson
  - user: pcuenq
  - user: burtenshaw
  - user: ariG23498
slug: "mistral-large4"
source_url: "https://huggingface.co/blog/mistral-large4"
source_published_date: "2026-10-29"
source_published_at: "2026-10-29T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Welcome Le Chonk: Mistral Large 4 on Hugging Face](https://huggingface.co/blog/mistral-large4)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/mistral-large4 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Hugging Face의 Le Chonk 소개: Mistral Large 4

Mistral Large 4(Le Chonk라고도 함)는 Mistral의 멀티모달 다국어 Mixture-of-Experts(MoE) 모델입니다. 텍스트와 이미지를 입력으로 받아 텍스트를 생성합니다. 총 **약 1T개, 활성 파라미터 49B개**를 가지며, 컨텍스트 윈도우는 **1M 토큰**입니다.

이 모델은 두 가지 공식 변형으로 출시되었습니다:

1. [FP8](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)
2. [NVFP4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B-NVFP4)

[Unsloth](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF)과 llama.cpp 팀이 만든 [GGUFs](https://huggingface.co/ggml-org)도 있습니다. Mistral Ecosystem Open-Weight License(모델을 서비스로 제공하는 경우에 대한 제한을 설정한 MIT 파생 라이선스)에 따라 배포됩니다.

Mistral Large 4는 특히 사이버 보안, 코딩 에이전트, 멀티모달리티에 강합니다. 이 글에서는 모델의 작동 방식, 기능, 아키텍처 등을 살펴봅니다!

## 목차 {#section-1}

* [Mistral Large 4의 새로운 점](#section-2)
* [아키텍처](#section-3)
* [벤치마크 결과](#section-4)
* [추론 지원](#section-5)
* [TRL을 사용한 미세 조정](#section-6)
* [Inference Endpoints](#section-7)
* [Dell Enterprise Hub](#section-8)
* [직접 사용해 보기](#section-9)

## Mistral Large 4의 새로운 점 {#section-2}

[Mistral Large 4](https://huggingface.co/collections/mistralai/mistral-large-4)는 대규모 expert 풀과 sparse attention, [Pixtral](https://huggingface.co/docs/transformers/en/model_doc/pixtral) vision encoder를 결합합니다. 언어 모델은 각 토큰에 사용할 소수의 expert 그룹을 선택하고, attention indexer는 주의를 기울일 이전 토큰 위치를 선택합니다. 이는 서로 다른 두 가지 sparsity 형태로, 하나는 feed-forward network를 선택하고 다른 하나는 컨텍스트 위치를 선택합니다.

모든 체크포인트는 [Mistral Large 4 collection](https://huggingface.co/collections/mistralai/mistral-large-4)에서 확인할 수 있습니다.

### 사이버 보안 기능

Mistral Large 4는 오픈 소스 소프트웨어의 실제 취약점을 재현한 다음 소프트웨어를 패치하도록 모델에 요청하는 Artificial Analysis Cyber Index 테스트에서 82%를 기록했습니다. Mistral에 따르면 일부 폐쇄형 frontier 모델은 작업 시도를 거부하기 때문에 동일한 테스트에서 거의 0점에 가까운 점수를 기록합니다. 이는 방어자에게 매우 중요합니다. 취약점을 수정하려면 일반적으로 취약점을 재현해야 하기 때문입니다. Mistral의 내부 테스트에서 Mistral Large 4는 모델이 명시적으로 학습하지 않은 악성 코드 분석, 취약점 우선순위 지정, 탐지 규칙 작성에도 유용했습니다.

Mistral은 주로 방어적 보안 작업을 위해 이 모델을 포지셔닝합니다. 취약점 연구와 사고 대응은 모델 입력에서 공격처럼 보이는 경우가 많기 때문에 폐쇄형 소스 필터가 이를 차단할 수 있습니다. API에 의존하는 팀은 사고가 발생한 중간에 액세스 권한을 잃을 수도 있습니다. 중요한 점은 오픈 웨이트 모델을 사용하면 사용 정책과 배포를 모두 사용자가 제어할 수 있다는 것입니다. 다만 이 모델도 유해한 요청은 거부합니다. JailbreakBench, StrongREJECT, AgentHarm의 사이버 프롬프트에서 평균 거부율은 다른 어떤 오픈 웨이트 모델보다 높습니다.

커뮤니티 빌더에게 Mistral Large 4와 같은 오픈 웨이트 모델은 다양한 다운스트림 사용 사례를 가능하게 합니다. 자체 코드베이스, 크래시 로그, 악성 코드 샘플을 중심으로 사이버 보안 시스템을 구축할 수 있습니다. 예를 들어 [TRL you could fine-tune](#fine-tuning-with-trl) Mistral Large 4 trace가 코드베이스 패치를 수행하도록 할 수 있습니다. 실제로 재현 후 패치 작업은 검증 가능한 보상을 사용하므로, 더 작은 보안 모델을 대상으로 추가 RL을 수행하는 데 사용할 수 있는 RL 환경에 적합합니다. 또한 오픈 웨이트를 사용하면 제공자의 설명에 의존하지 않고 연구자가 모델의 거부 동작을 직접 측정할 수 있습니다.

요컨대 ML4와 같은 사이버 보안 기능을 오픈 상태로 제공하는 것은 커뮤니티에 큰 힘이 됩니다.

> [!NOTE]
> 모델의 사이버 보안 기능에 대해 Pierre Stock과 인터뷰했습니다. [you can watch it here](https://www.youtube.com/embed/FczbSauVvjw)

## 아키텍처 {#section-3}

![ML4](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml4/ml4.png)

Mistral Large 4의 텍스트 디코더는 72개 레이어, 6,144의 hidden dimension, 64개의 attention head로 구성됩니다. attention 블록은 Multi-head Latent Attention(MLA)과 DeepSeek Sparse Attention(DSA) indexer를 결합합니다(이는 GLM 5.3과 같은 다른 모델에서도 사용되는 방향입니다). MLA는 압축된 KV 표현을 유지하고, DSA는 각 query에 대해 최대 2,048개의 key 위치를 선택합니다.

![Attention](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml4/attention.png)  
각 decoder block에는 3:1 RoPE 대 NoPE 패턴이 적용됩니다. 즉, decoder layer 3개는 RoPE를 사용하고 네 번째 레이어마다 NoPE를 사용합니다. 또한 모델은 레이어 간에 KV state를 공유합니다.  
첫 번째 decoder layer는 dense feed-forward network를 사용합니다. 나머지 71개 레이어는 MoE block을 사용하며, 이 블록에서는 softmax router가 각 토큰에 대해 128개 expert 중 4개를 선택합니다. 선택된 expert와 함께 하나의 shared expert가 항상 실행되므로 토큰당 활성 expert는 총 5개입니다.

RoPE와 관련해 Mistral은 주파수마다 하나의 학습 가능한 multiplier를 사용하는 YaRN-scaled RoPE인 *LeRope*를 사용합니다.

![Experts](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml4/experts.png)

### Vision encoder

Mistral Large 4는 1.6B Pixtral encoder를 사용합니다. Pixtral은 30개 레이어, 2,048의 hidden dimension, 16개의 attention head로 구성됩니다. 14 × 14 patch와 spatial merge size가 2인 patch-merge projector를 사용합니다. 아래에서 encoder pipeline을 확인할 수 있습니다.

![Image Encoder](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml4/img_encoder.png)

## 벤치마크 결과 {#section-4}

Mistral Large 4는 코딩, 사이버 보안, 멀티모달 이해, 에이전트 워크플로 전반에서 강력한 벤치마크 성능을 보여 오픈 웨이트 생태계에 경쟁력 있는 모델로 자리매김했습니다. 2026년 10월 29일 기준으로 이 모델은 Artificial Analysis Intelligence index에서 다섯 번째로 우수한 오픈 웨이트 모델입니다.

코딩에서 ML4는 **DeepSWE v1.1 61.7%**, **SWE-Atlas-QnA 59.4%**, **Terminal-Bench 4 28.3%**를 달성했습니다. 종합 **Coding Agent Index 점수 49.8%**로 DeepSeek V4 Pro 0813과 Qwen3.8 Max를 앞섭니다.

ML4의 코딩 기능은 지식 업무로도 확장되어, 비즈니스 워크플로와 장기 전문 작업을 평가하는 **AutomationBench 59.9%** 및 **AA-Briefcase 1,393 Elo**를 기록했습니다.

사이버 보안은 Mistral Large 4의 두드러진 기능 중 하나로, 실제 취약점 평가와 보안 대회 작업 전반에서 강력한 결과를 보였습니다. 세 가지 벤치마크가 모델의 사이버 보안 성능을 잘 보여줍니다:

* **Artificial Analysis Cyber Index:** 소프트웨어 취약점을 찾고 수정하는 모델의 능력을 독립적으로 평가하는 이 테스트에서 ML4는 **전 세계 상위 5개 모델**에 속합니다.  
* **취약점 재현 및 패치:** **82%**를 달성했으며, 이는 해당 index 내 이 테스트에서 보고된 최고 점수입니다. 이 작업은 오픈 소스 소프트웨어의 실제 취약점을 재현한 다음 패치해야 하므로 조사와 해결 능력을 모두 평가합니다.  
* **Cybench:** **보안 대회에서 엄선한 40개 과제**로 구성된 테스트에서 ML4는 **93%**를 해결해 실전 보안 문제에서 강력한 성능을 보여줍니다.

Mistral Large 4는 Lakera의 B3 벤치마크에서 **93.3%의 공격 저항성**도 보여 프롬프트 인젝션 공격에 대한 견고성을 뒷받침하는 근거를 추가했습니다. 벤치마크 외에도 내부 테스트에서 악성 코드 분석, 취약점 우선순위 지정, 탐지 규칙 생성에서 유망한 결과가 나타났습니다.

## 추론 지원 {#section-5}

Mistral Large 4는 transformers, vLLM, llama.cpp에 대한 day-0 지원과 함께 제공됩니다. GGUF 양자화 모델은 GGUF와 [llama.cpp team](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF)에서 사용할 수 있습니다. Inference Endpoints를 사용한 원격 배포 옵션과 Inference Providers를 통한 서버리스 추론도 제공합니다.

### Transformers

Transformers 구현은 `AutoModelForMultimodalLM`을 제공하며, `AutoProcessor`이 대화와 이미지 입력을 준비합니다. Mistral Large 4는 추론 수준으로 `low`, `medium`, `high`를 사용할 수 있습니다. 기본값은 `high`이며, 추론은 비활성화할 수 없다는 점에 유의해야 합니다. 따라서 `reasoning_effort`: `none`을 사용해도 여전히 `high`을 사용합니다.

모델을 실행하려면 transformers 패키지를 업그레이드합니다.

```bash
pip install --upgrade transformers
```


```python
import os
from transformers import AutoProcessor, AutoModelForMultimodalLM, DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

model_id = "mistralai/Mistral-Large-4-1T-A52B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    distributed_config=distributed_config,
)

# only text
text_messages = [
    {
        "role": "user",
        "content": "Who are you?"
    }
]

# text + image
multimodal_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/cats-image/resolve/main/cats_image.jpeg",
            },
            {
                "type": "text",
                "text": "How many cats are in the image?",
            },
        ],
    },
]

for messages in [text_messages, multimodal_messages]:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        reasoning_effort="medium",
    ).to(model.device).to(model.dtype)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=100)
    if int(os.environ["RANK"]) == 0:
        print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```


위 스크립트를 실행하려면 python 파일 `script.py`에 저장한 다음 다음 명령을 실행합니다.

```bash
torchrun --nproc-per-node 8 script.py
```


### vLLM

vLLM은 고품질 NVFP4 Mistral Large 4 체크포인트를 지원합니다. 이 체크포인트는 메모리 사용량이 더 적으며 NVIDIA Blackwell GPU가 필요합니다. 또한 vLLM에서 추가 속도 향상을 제공하는 DSpark drafter가 모델에 포함되어 있습니다. 다음과 같이 서버를 실행합니다:

```sh
vllm serve mistralai/Mistral-Large-4-1T-A52B-NVFP4 \
  --config-format mistral \
  --load-format mistral \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --enable-expert-parallel \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enforce-eager \
  --mm_processor_cache_gb 0.0
```


서버가 실행되면 다음 명령을 실행할 수 있습니다.

```sh
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":512}'
```


### Llama.cpp

Llama.cpp와 [Llama App](https://llama.app/)는 이미 Mistral Large 4를 지원합니다.

다음 명령으로 Llama App을 설치할 수 있습니다.

```sh
curl -LsSf https://llama.app/install.sh | sh
```


그런 다음 다음과 같이 서버를 시작합니다. 이때 모델 ID 뒤에 quant suffix를 지정합니다.

```sh
llama serve -hf ggml-org/Mistral-Large-4-1T-A52B-GGUF:Q2_K
```


[Here’s the repository](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF)은 FP8 체크포인트에서 파생된 quants를 사용하며, [this is the NVFP4 one](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-NVFP4-GGUF)은 공식 NVFP4 repo에서 변환되었습니다.

서버가 시작되면 localhost:9931로 이동해 내장 WebUI에서 채팅할 수 있습니다. 웹 UI에서 추론에 사용할 이미지를 추가하고 reasoning lever를 조정할 수 있습니다.  
다음과 같이 서버를 쿼리할 수도 있습니다. 멀티모달리티, tool call 등의 기능은 [the API reference](https://llama.app/docs/serve)을 확인해 보세요.

```sh
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a limerick about python exceptions"}
        ]
    }'
```


다음과 같이 image-text 추론을 실행할 수 있습니다.

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <your-key>" \
    -d '{
        "model": "muse-spark-open",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,..."
                        }
                    }
                ]
            }
        ],
        "temperature": 0
    }'
```


llama.cpp 팀은 추가 변환을 위해 다음 quants를 제공합니다: `IQ2_XXS`, `Q2_K`, `Q4_K_M`, `NVFP4`, `Q8_0`, `BF16`. 2-bit quants는 통합 RAM 512GB를 갖춘 단일 Apple Silicon Ultra 컴퓨터에서 실행됩니다. 나머지는 분산 추론이 필요합니다. 여러 NVIDIA 또는 AMD 카드에서도 실행할 수 있습니다.

단일 M3 Ultra에서 `Q2_K`을 실행했을 때 prefill 처리량은 초당 약 90토큰, 생성 속도는 15 tok/s였습니다. 4 x RTX6000 Pro에서는 다음과 같은 결과를 확인했습니다:

* NVFP4: 32K context size, prefill 130-590 t/s, gen speed ~10 t/s  
* IQ2_XXS: 800K context size. 짧은 컨텍스트 크기에서는 prefill이 500 tok/s이고 생성 속도는 30 tok/s입니다. 그러나 긴 컨텍스트를 사용하면 성능이 129 tok/s(prefill) 및 3 tok/s(생성)로 떨어집니다.

## TRL을 사용한 미세 조정 {#section-6}

TRL을 사용한 Mistral Large 4 미세 조정도 day 0부터 지원합니다. `transformers`, `trl`, `peft`를 최신 버전으로 업그레이드하기만 하면 됩니다.

```sh
pip install --upgrade transformers trl peft
```


다음 예제에서는 모델이 다국어 사고를 수행하도록 학습합니다. 미세 조정된 모델은 system prompt가 요청하는 언어로 추론할 수 있습니다\!

이 스크립트는 8xB300s 노드에서 FP8 변형을 실행합니다. Slurm launcher를 포함한 전체 예제는 [TRL's examples](https://github.com/huggingface/trl/tree/main/examples/sft_mistral_large_4)에 있습니다.

```py
# torchrun --nproc-per-node 8 sft.py
import os

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, DistributedConfig
from trl import SFTConfig, SFTTrainer

world_size = int(os.environ["WORLD_SIZE"])
model_id = "mistralai/Mistral-Large-4-1T-A52B"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(fsdp_size=world_size, ep_size=world_size),
    dtype="auto",
)

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
# the assistant turns carry reasoning, so render them in reasoning mode
dataset = dataset.map(lambda x: {"chat_template_kwargs": {"reasoning_effort": "high"}})

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        packing=True,
        max_length=4096,
        assistant_only_loss=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
    ),
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=r".*language_model.*\.(q_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|gate_proj|up_proj|down_proj)",
    ),
)
trainer.train()
```


## Inference Endpoints {#section-7}

Mistral Large 4는 Inference Endpoints에서도 사용할 수 있으며, 8 x NVIDIA H200 GPU에 바로 배포할 수 있는 recipe가 제공됩니다. 이 recipe는 tensor 및 expert parallelism을 사용해 8개 GPU 전체에 FP8 weight를 분할하고, 모델에 내장된 DSpark draft head를 사용해 speculative decoding으로 생성 속도를 높입니다. 최대 131K 토큰의 컨텍스트를 제공하고 한 번에 최대 16개의 요청을 처리하며, 다음과 같은 결과를 확인했습니다:

* 단일 요청: 1K 토큰 프롬프트에서 첫 토큰까지 약 0.6초가 걸렸고, 생성 속도는 약 65 tok/s였습니다. DSpark의 draft가 승인되는 채팅 프롬프트에서는 120–138 tok/s에 도달했습니다.  
* 긴 프롬프트: 16K 토큰 프롬프트 4개를 한 번에 처리할 때 총 약 4,300 tok/s의 속도로 prefill되었고, 첫 토큰까지 8–9초가 걸렸습니다.  
* 부하 상태: 1,000개의 ShareGPT 대화 전체에서 총 생성 속도는 약 480 tok/s였습니다.

`hf` CLI를 사용해 다음과 같이 배포할 수 있습니다:

```sh
hf endpoints catalog deploy --repo mistralai/Mistral-Large-4-1T-A52B --name mistral-large-4
```


endpoint가 실행되면 endpoint URL과 Hugging Face token을 사용해 OpenAI 호환 API를 쿼리할 수 있습니다. `reasoning_effort` field는 모델이 답변하기 전에 얼마나 오래 추론할지 제어하며, `low`, `medium`, `high`(기본값)으로 설정할 수 있습니다.

```sh
curl "$(hf endpoints describe mistral-large-4 --json | jq -r .status.url)/v1/chat/completions" \
    -H "Authorization: Bearer $(hf auth token 2>/dev/null)" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistralai/Mistral-Large-4-1T-A52B",
        "messages": [
            {"role": "user", "content": "Write a limerick about python exceptions"}
        ],
        "reasoning_effort": "medium"
    }'
```


답변은 `choices[0].message.content`으로 반환되고 추론 과정은 `choices[0].message.reasoning`에 반환됩니다. endpoint는 표준 Chat Completions field를 통한 tool call, structured output, 이미지 입력, streaming도 지원합니다.

## Dell Enterprise Hub {#section-8}

Dell AI Server(NVIDIA H200, B200, GB200 또는 GB300)를 가장 쉽게 시작하는 방법은 `dell-ai` CLI를 설치하는 것입니다:

```sh
pip install dell-ai && dell-ai login
```


그런 다음 이를 사용해 예를 들어 NVIDIA B300 노드에서 Mistral Large 4에 최적화된 recipe를 실행합니다.

```sh
dell-ai models deploy --model-id mistralai/Mistral-Large-4-1T-A52B-NVFP4 \
    --engine docker \
    --platform-id xe9780-nvidia-b300 \
    --gpus 4
```


이번 릴리스에 대한 자세한 내용은 [Dell Enterprise Hub (DEH)](https://dell.huggingface.co/updates/mistral-large-4)에서 확인할 수 있습니다.

## 직접 사용해 보기 {#section-9}

[Mistral Large 4 collection](https://huggingface.co/collections/mistralai/mistral-large-4)에서 weight를 다운로드하고, [GGUFs](https://huggingface.co/ggml-org/Mistral-Large-4-1T-A52B-GGUF)을 사용해 로컬에서 실행하거나 Inference Endpoints로 배포해 보세요. 여러분이 Le Chonk로 무엇을 만들어낼지 기대됩니다. Hub에서 미세 조정 모델, quants 및 기타 artifact를 공유해 주세요!
