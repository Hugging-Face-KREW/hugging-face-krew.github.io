---
layout: post
title: "Nunchaku 4비트 확산 추론을 Diffusers에 도입하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/nunchaku-diffusers/thumbnail.png
image: assets/images/blog/posts/2026-07-23-nunchaku-diffusers/thumbnail.png
authors:
  - user: rootonchair
slug: "nunchaku-diffusers"
source_url: "https://huggingface.co/blog/nunchaku-diffusers"
source_published_date: "2026-07-23"
source_published_at: "2026-07-23T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Bringing Nunchaku 4-bit Diffusion Inference to Diffusers](https://huggingface.co/blog/nunchaku-diffusers)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nunchaku-diffusers -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Nunchaku 4비트 확산 추론을 Diffusers에 도입하기

대형 확산 트랜스포머는 멋진 이미지(또는 비디오, 오디오 조각, 이제 텍스트까지)도 만들 수 있지만, BF16 정밀도로 최신 텍스트-투-이미지 모델을 로드하는 데에는 종종 20-30 GB의 VRAM이 필요합니다. 이로 인해 이들 모델은 대부분의 일반 사용자 GPU에서 손에 닿지 않는 영역이 됩니다. 양자화는 이 문제에 대한 강력한 해결책이며, Diffusers는 이미 bitsandbytes, GGUF, torchao, Quanto 등 여러 양자화 백엔드를 통합하고 있는데, 이는 [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization)에서 다룬 바 있습니다.

대부분의 이러한 백엔드는 _weight-only_ 입니다. 이는 가중치를 저정밀도로 저장하고 계산 시 다시 고정밀도로 디퀀타이즈해 사용하는 것을 의미합니다. 이렇게 하면 메모리 사용량이 크게 감소하지만, 일반적으로 추론 속도를 높이지 못하고, 심지어 약간의 지연 오버헤드를 추가할 수 있습니다.

[SVDQuant](https://arxiv.org/abs/2411.05007), 대중적인 [Nunchaku](https://github.com/nunchaku-tech/nunchaku) 추론 엔진의 양자화 방법은 다른 접근 방식을 취합니다. 주 트랜스포머 계층을 4비트 가중치와 활성화(W4A4)로 실행하여 메모리를 줄이면서도 디노이징 루프를 가속합니다. 아래에 자세한 내용이 다루어져 있지만, 지금까지 이러한 체크포인트를 사용하려면 별도의 추론 라이브러리가 필요했습니다.

현재 Diffusers에서는 `from_pretrained()`를 호출하는 것만으로 Nunchaku 체크포인트를 로드하는 것이 가능하며, 로컬 CUDA 컴파일이 필요 없도록 [`kernels`](https://github.com/huggingface/kernels) 패키지가 제공됩니다. 또한 동반 도구 [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor)를 사용하면 새 아키텍처를 직접 양자화하고 일반 Diffusers 저장소로 게시할 수 있습니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nunchaku-diffusers/contact_sheet_top3_metrics_bold.png" alt="Nunchaku Lite image quality and performance comparison">
</figure>

## 목차 {#section-1}

* [Nunchaku Lite 시작하기](#section-2)
* [배경: SVDQuant와 Nunchaku](#section-3)
* [Nunchaku Lite 소개](#section-4)
* [Diffusers에서의 네이티브 로딩](#section-5)
* [더 빠른 속도와 더 낮은 메모리 사용량 얻기](#section-6)
* [벤치마크](#section-7)
* [직접 모델 양자화하기](#section-8)
* [바로 사용 가능한 체크포인트](#section-9)
* [결론](#section-10)
* [감사의 말씀](#section-11)

## Nunchaku Lite 시작하기 {#section-2}

먼저 요구사항을 설치합니다. Diffusers의 최신 버전과 Hugging Face `kernels` 패키지가 필요합니다:

```bash
pip install -U diffusers transformers accelerate kernels bitsandbytes
```


그런 다음 Diffusers 모델처럼 사전 양자화된 파이프라인을 로드합니다:

```python
import torch
from diffusers import ErnieImagePipeline

pipe = ErnieImagePipeline.from_pretrained(
    "lite-infer/ERNIE-Image-Turbo-nunchaku-lite-nvfp4_r32-bnb4-text-encoder",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt="A cinematic portrait of a red fox in a misty forest at sunrise, "
           "detailed fur, volumetric light",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=1.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("output.png")
```


<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nunchaku-diffusers/fox_bf16_vs_nunchaku_no_metrics.png" alt="BF16 and Nunchaku Lite outputs for a red fox prompt">
</figure>

커스텀 파이프라인 클래스나 별도의 추론 엔진이 필요하지 않으며, 로컬에서 컴파일할 필요도 없습니다. NVFP4 커널은 사용 시 처음으로 [Nunchaku Lite kernels page](https://huggingface.co/kernels/rootonchair/nunchaku-lite-kernels)를 통해 허브에서 다운로드됩니다. 이 체크포인트는 Nunchaku NVFP4 트랜스포머와 bitsandbytes NF4 텍스트 인코더를 쌍지으며, RTX 5090에서 약 1.7초 만에 1024x1024 이미지를 생성하고 피크 메모리 사용량은 약 12GB로, BF16 파이프라인의 약 24GB와 비교됩니다. Nunchaku Lite 체크포인트 포맷에 대한 자세한 내용은 [official Diffusers documentation](https://huggingface.co/docs/diffusers/main/en/quantization/nunchaku)에서 확인할 수 있습니다.

> [!참고]
> NVFP4 체크포인트는 NVIDIA Blackwell GPU( RTX 50 시리즈, RTX PRO 6000, B200 )가 필요합니다. 초기 세대의 경우 INT4 변형을 사용하십시오. 자세한 내용은 아래 [hardware support](#hardware-support) 표를 참조하십시오.

## 배경: SVDQuant와 Nunchaku {#section-3}

**SVDQuant**은 **Nunchaku**의 양자화 방식이며, 그것의 참조 CUDA 추론 엔진입니다. 표준 4비트 양자화는 가중치와 활성화에 큰 이상값(outliers)이 포함되어 있어 확산 트랜스포머에 대해 어렵습니다. SVDQuant는 활성화의 이상값을 가중치로 옮겨 각 가중치 행렬의 가장 어렵운 부분을 16비트 저랭크 분기로 표현하고, 남은 잔여 부분을 4비트로 양자화합니다. Nunchaku는 4비트 경로와 저랭크 분기에 대한 융합 커널로 이를 빠르게 만듭니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nunchaku-diffusers/svdquant_kernel_fusion.png" alt="Nunchaku kernel fusion: the low-rank down projection is fused with input quantization, and the low-rank up projection is fused with the 4-bit matmul">
  <figcaption>Nunchaku fuses the low-rank down projection with the quantization kernel and the low-rank up projection with the 4-bit compute kernel, eliminating the memory access overhead of the 16-bit branch. Figure from the <a href="https://arxiv.org/abs/2411.05007">SVDQuant paper</a>.</figcaption>
</figure>

## Nunchaku Lite 소개 {#section-4}

**원래의 [Nunchaku engine](https://github.com/nunchaku-ai/nunchaku)는** [model-specific fused execution paths](#quantizing-models-with-structural-rewrites)에서 비롯된 속도의 대부분을 얻습니다. 예를 들어 QKV 프로젝션과 GELU/MLP 커널의 융합 등이 그러합니다. 이러한 최적화는 각 아키텍처의 모듈 구성 및 체크포인트 형식에 묶여 있어, 새로운 모델 패밀리를 지원하려면 보통 아키텍처 특화 통합 작업이 필요합니다.

**Nunchaku Lite**는 Diffusers의 새로운 통합 경로입니다. 이를 통해 Diffusers는 커스텀 파이프라인이나 별도의 추론 엔진 없이 Nunchaku 스타일의 체크포인트를 로드할 수 있습니다. 내부적으로, Nunchaku Lite는 재고 Diffusers 모델의 관련 `nn.Linear` 모듈을 로드되기 전에 런타임 SVDQ/AWQ 선형 계층으로 패치합니다. CUDA 커널은 `kernels` 패키지를 통해 Hub에서 제공합니다. 두 가지 커널 계열이 사용됩니다:

*   **`svdq_w4a4`**: SVDQuant 저랭크 보정이 적용된 4비트 가중치와 활성화. 이 계층은 트랜스포머의 어텐션과 MLP 프로젝션에 사용되며, 계산의 거의 전부가 이 계층에 소요되며, INT4 및 NVFP4 변형으로 제공됩니다.
*   **`awq_w4a16`**: 4비트 가중치와 16비트 활성화를 사용하며, 적응 정규화 및 변조 프로젝션에 사용되는 예로 FLUX `adanorm_single` / `adanorm_zero` 또는 Qwen-Image 변조 계층. 이 계층은 메모리 바운드이며 정밀도에 민감하므로 AWQ가 메모리와 공간을 절약하면서도 정밀도를 유지하는 데 잘 맞습니다.

그 결과, 아키텍처별 융합 커널과 모듈이 없으면 Nunchaku Lite가 원래의 Nunchaku 엔진의 속도 향상을 따라잡지 못하는 단점이 있습니다. 다만 기본 구현은 여전히 약 **30%의 속도 향상**과 함께 같은 수준의 **VRAM 감소**를 제공합니다.

## Diffusers에서의 네이티브 로딩 {#section-5}

bitsandbytes나 torchao를 Diffusers에서 사용해 보셨다면 메커니즘이 친숙하게 느껴질 것입니다. Nunchaku Lite 모델 저장소는 일반 Diffusers 저장소입니다. 유일한 특별한 부분은 트랜스포머의 `config.json` 내부에 있는 `quantization_config` 블록입니다:

```json
"quantization_config": {
    "quant_method": "nunchaku_lite",
    "compute_dtype": "bfloat16",
    "svdq_w4a4": {
        "precision": "nvfp4",
        "group_size": 16,
        "rank": 32,
        "targets": [
            "layers.0.self_attention.to_q",
            "layers.0.self_attention.to_k",
            "..."
        ]
    },
    "awq_w4a16": {
        "precision": "int4",
        "group_size": 64,
        "targets": [
            "adaLN_modulation.1",
            "..."
        ]
    }
}
```


이 구성은 Diffusers에 어떤 모듈이 양자화되었는지, 어떤 스킴을 사용하는지, 그리고 어떤 Nunchaku Lite 런타임 계층을 인스턴스화할지(`SVDQW4A4Linear` 또는 `AWQW4A16Linear`)를 알려줍니다.

양자화된 모델은 밀집형 모델의 정확한 모듈 구조를 그대로 유지하기 때문에, 아래로 이어지는 모든 구성 요소(스케줄러, LoRA 로딩 훅, 오프로딩, `torch.compile` 등)가 일반 Diffusers 모델처럼 보게 됩니다.

### 하드웨어 지원

Nunchaku Lite는 GPU 세대와 체크포인트 정밀도에 따라 서로 다른 커널 변형을 사용합니다:

| 스킴 | 정밀도 | 지원되는 GPU |
|---|---|---|
| `svdq_w4a4` | `nvfp4` | Blackwell (RTX 50 시리즈, RTX PRO 6000, B200) |
| `svdq_w4a4` | `int4` | Turing / Ampere / Ada (RTX 30 및 40 시리즈, A100, L40S) |
| `awq_w4a16` | `int4` | Turing / Ampere / Ada (RTX 30 및 40 시리즈, A100, L40S) |

> [!경고]
> Volta 및 Hopper GPU는 현재 4비트 커널에서 지원되지 않습니다. 양자화 도구는 로드 시 GPU의 CUDA 기능을 검증하고, 잘못된 출력 대신 명확한 오류를 발생시킵니다.

## 더 빠른 속도와 더 낮은 메모리 사용량 얻기 {#section-6}

Nunchaku Lite는 Diffusers의 다른 메모리 및 속도 최적화와 결합하여 사용할 수 있습니다.

**`torch.compile`.** 트랜스포머를 컴파일하면 엔드투엔드 속도향상이 1.35배에서 1.8배로 향상됩니다:

```python
pipe.transformer.compile(fullgraph=True)

# or compile_repeated_blocks() for faster compilation

pipe.transformer.compile_repeated_blocks(fullgraph=True)
```


**양자화된 텍스트 인코더.** 트랜스포머만이 메모리 소모가 큰 유일한 구성 요소가 아닙니다. T5나 Qwen3 같은 텍스트 인코더도 자체적으로 수 기가바이트를 차지할 수 있습니다. bitsandbytes NF4로 텍스트 인코더를 추가로 양자화하면 벤치마크에서 피크 VRAM이 약 22% 감소합니다.

**오프로딩.** Diffusers의 오프로딩 도구들인 `enable_model_cpu_offload()` 및 `enable_sequential_cpu_offload()`은 파이프라인을 더 작은 GPU에 맞추고자 할 때 보통대로 작동합니다.

## 벤치마크 {#section-7}

아래 모든 수치는 [rootonchair/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder](https://huggingface.co/rootonchair/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder)를 사용하여 1024x1024 해상도에서 NVIDIA RTX PRO 6000(Blackwell)으로 측정되었습니다.

### 엔드투엔드 지연 및 메모리

| 구성 | 전체 파이프라인 | 디노이즈 루프 | 피크 VRAM | 속도향상 |
|---|---|---|---|---|
| BF16 baseline | 3.00 s | 2.86 s | 31.1 GB | 1.0x |
| Nunchaku Lite NVFP4 | 2.27 s | 2.13 s | 20.6 GB | 1.35x |
| Nunchaku Lite NVFP4 + `torch.compile` | 1.68 s | 1.53 s | 20.6 GB | 1.8x |
| Nunchaku Lite NVFP4 + NF4 text encoder | 2.29 s | 2.13 s | 16.0 GB | 1.35x |

위에서 보듯이, Nunchaku는 피크 VRAM을 최대 50%까지 감소시키면서도 지연(latency)을 대략 30% 개선합니다. 남은 오버헤드는 주로 추가 커널 실행에서 기인하는데, 이 부분은 `torch.compile`으로 완화할 수 있어 전체 파이프라인을 1.68초로 낮추고 BF16 기준선 대비 약 1.8배 빠릅니다.

### 이미지 품질

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/nunchaku-diffusers/quality_grid.png" alt="Quality comparison grid">
  <figcaption>BF16 vs 4-bit outputs with identical seeds and settings.</figcaption>
</figure>

## 직접 모델 양자화하기 {#section-8}

Diffusers에서 Nunchaku Lite 지원은 아키텍처에 독립적이며, [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor) 도구킷은 Diffusers 모델에 대한 엔드 투 엔드 SVDQuant 워크플로를 제공합니다: 보정(calibrate), 양자화(quantize), 패키징(package), 게시(publish).

아래에, FLUX.2 Klein 4B를 예로 들어 양자화하는 과정을 살펴봅니다. 주요 단계는 모델 검사, 트랜스포머 보정 및 양자화, 결과를 Diffusers 파이프라인으로 패키징한 뒤, 이를 확인하고 허브에 업로드하는 것입니다. [full tutorial](https://github.com/rootonchair/diffuse-compressor/blob/main/docs/quantize_new_hf_model.md)은 모든 플래그를 자세히 다룹니다.

### 1. 양자화될 내용 확인

일반 스캐너가 모델을 순회하며 타깃을 결정합니다: 반복된 트랜스포머 블록 스택 내부의 호환 가능한 선형 계층은 SVDQ W4A4 타깃이 되고, 인식된 변조 선형은 AWQ W4A16 타깃이 되며, 그 밖의 모든 것은 밀집(Dense) 상태로 남습니다.

```bash
python examples/text_to_image/quantize_hf.py black-forest-labs/FLUX.2-klein-4B \
  --precision int4 --rank 32 --inspect-config
```


양자화하기 전에 이 보고서를 항상 읽으십시오. FLUX.2 Klein 4B의 예상 결과는 100개의 SVDQ 타깃, 3개의 AWQ 타깃, 그리고 6개의 밀집 외부 선형으로, 패턴 누락이나 중복된 이름이 없어야 합니다.

### 2. 양자화 실행

다음 명령은 트랜스포머에 대해 SVDQuant를 실행하고 양자화된 체크포인트를 `outputs/checkpoints/svdq-int4_r32-flux-2-klein-4b.safetensors`로 작성합니다:

```bash
python examples/text_to_image/quantize_hf.py black-forest-labs/FLUX.2-klein-4B \
  --precision int4 \
  --output outputs/checkpoints/svdq-int4_r32-flux-2-klein-4b.safetensors
```


`--precision int4`를 `nvfp4`로 바꿔 Blackwell-네이티브 가중치를 구축합니다.

### 3. Diffusers 파이프라인 패키징

변환기는 양자화된 트랜스포머를 기본 파이프라인의 다른 구성 요소와 결합하고, 간결한 `nunchaku_lite` 구성을 `transformer/config.json`에 기록하며, 필요에 따라 텍스트 인코더를 NF4로 변환할 수 있습니다:

```bash
python examples/convert_nunchaku_lite_diffusers.py \
  --checkpoint outputs/checkpoints/svdq-int4_r32-flux-2-klein-4b.safetensors \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --bnb4-text-encoder text_encoder \
  --compute-dtype bfloat16 \
  --output-dir outputs/diffusers/FLUX.2-klein-4B-nunchaku-lite-int4-bnb4-text-encoder
```


### 4. 로드, 확인 및 허브로 푸시

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "outputs/diffusers/FLUX.2-klein-4B-nunchaku-lite-int4-bnb4-text-encoder",
    device_map="cuda",
)
image = pipe(
    "A glass robot in a greenhouse, cinematic lighting",
    num_inference_steps=4, guidance_scale=1.0,
    generator=torch.Generator("cuda").manual_seed(12345),
).images[0]
```


출력이 올바르게 보이면 `pipe.push_to_hub("your-name/your-model-nunchaku-lite-int4")`를 실행합니다. 다른 사용자는 위와 동일한 `from_pretrained()` 패턴으로 로드할 수 있습니다.

### 구조적 재작성으로 모델 양자화

일반 경로는 아키텍처가 구조적 재작성 없이 양자화될 수 있다고 가정합니다. 추가 속도 향상을 위해 원래의 Nunchaku 엔진은 Diffusers 계층의 묶음을 융합 모듈로 재구성합니다. 일반 경로는 Q, K, V 투영을 하나의 모듈로 결합하거나 융합된 투영을 여러 모듈에 걸쳐 분할하는 등의 변경을 스스로 추론할 수 없습니다.

FLUX.1-dev의 QKV 투영은 구체적인 예입니다. [Diffusers defines three separate modules](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py#L313-L329):

```python
self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
```


다음은 [Nunchaku FLUX module combines those layers](https://github.com/nunchaku-ai/nunchaku/blob/main/nunchaku/models/transformers/transformer_flux_v2.py#L63-L79)를 하나의 양자화된 `to_qkv` 모듈로 묶은 예시입니다:

```python
to_qkv = fuse_linears([other.to_q, other.to_k, other.to_v])
self.to_qkv = SVDQW4A4Linear.from_linear(to_qkv, **kwargs)
```


이 그룹화된 모듈은 Nunchaku의 융합 연산자가 QKV 투영, Q/K 정규화, 로터리 임베딩을 함께 소모하기 때문에 필요합니다. 비교적으로, [default Diffusers path](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py#L45-L116)은 이를 각각 따로 실행합니다:

```python
query = attn.to_q(hidden_states)
key = attn.to_k(hidden_states)
value = attn.to_v(hidden_states)

query = query.unflatten(-1, (attn.heads, -1))
key = key.unflatten(-1, (attn.heads, -1))
value = value.unflatten(-1, (attn.heads, -1))

query = attn.norm_q(query)
key = attn.norm_k(key)

if image_rotary_emb is not None:
    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
```


[Nunchaku path](https://github.com/nunchaku-ai/nunchaku/blob/main/nunchaku/models/attention_processors/flux.py#L69-L93)은 그룹화된 투영, 정규화 모듈 및 로터리 임베딩을 하나의 융합 연산자에 제공합니다:

```python
qkv = fused_qkv_norm_rottary(
    hidden_states, attn.to_qkv, attn.norm_q, attn.norm_k, image_rotary_emb
)
```


이것은 일반 경로가 추론할 수 없는 구조적 재작성입니다. Diffusers는 `to_q`, `to_k`, 및 `to_v` 매개변수 접두사를 가진 세 개의 대상 모듈을 가지며, Nunchaku는 `to_qkv` 아래 하나의 그룹화 모듈을 가집니다. 모델별 대상 구성(target config) 또는 어댑터는 Q, K, V 매개변수를 출력 차원에 따라 순서대로 연결(concatenate)하고 `to_qkv`로 로드되어야 한다고 명시해야 합니다.

이러한 구조적 재작성은 양자화 중 모델별 대상 구성(target config)에 의해 설명되며, 체크포인트가 로드될 때 소형 런타임 어댑터에 의해 처리됩니다. [FLUX.2 Klein 4B quantization script](https://github.com/rootonchair/diffuse-compressor/blob/main/examples/text_to_image/quantize_flux2_klein_4b.py)는 구조적으로 재작성된 체크포인트를 생성하기 위한 구체적인 대상 구성 예를 제공하고, [rootonchair/nunchaku-lite](https://github.com/rootonchair/nunchaku-lite)은 묶음 QKV 텐서를 로드하고 융합된 투영을 분할하는 등 다른 융합 작업을 로드하는 데 필요한 런타임 어댑터를 제공합니다. 전체 워크플로우는 [Adding A New Model](https://github.com/rootonchair/diffuse-compressor/blob/main/docs/adding_new_model.md) 가이드를 확인하면 됩니다.

## 바로 사용 가능한 체크포인트 {#section-9}

즉시 시작하려면 아래 저장소를 확인하십시오:

- [rootonchair/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder](https://huggingface.co/rootonchair/ERNIE-Image-Turbo-nunchaku-lite-int4-bnb4-text-encoder): bitsandbytes NF4 텍스트 인코더를 갖춘 INT4 ERNIE-Image-Turbo
- [rootonchair/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder](https://huggingface.co/rootonchair/ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder): bitsandbytes NF4 텍스트 인코더를 갖춘 NVFP4 ERNIE-Image-Turbo
- [OzzyGT/Krea_2_Turbo_nunchaku_lite_nvfp4](https://huggingface.co/OzzyGT/Krea_2_Turbo_nunchaku_lite_nvfp4): NVFP4 Krea 2 Turbo 체크포인트
- [lite-infer](https://huggingface.co/lite-infer): 더 많은 Nunchaku Lite 체크포인트 및 컬렉션

## 결론 {#section-10}

Nunchaku의 SVDQuant 커널은 소비자 하드웨어에서 확산 트랜스포머를 효율적으로 실행하는 가장 효과적인 방법 중 하나이며, 이제 Diffusers에서 네이티브로 지원됩니다. 사전 양자화된 체크포인트는 `from_pretrained()`로 로드되며, diffuse-compressor 도구 모음은 엔진 지원을 기다리지 않고도 새로운 아키텍처를 양자화할 수 있게 해줍니다. 가중치와 활성화를 모두 양자화하는 W4A4 경로는 메모리 사용을 줄이는 동시에 디노이징 대기 시간을 개선하고 BF16 원본에 가까운 이미지 품질을 유지합니다.

새 모델을 양자화하고 게시하셨다면 소식을 듣고 싶습니다. Hub에 공유해 주세요! 이 기능에 대해 궁금한 점이 있다면 언제든지 [Discord](https://discord.gg/G7tWnz98XR)에 참여해 주세요.

자세한 내용을 보려면 아래 리소스를 확인하세요:

- [Diffusers Nunchaku documentation](https://huggingface.co/docs/diffusers/quantization/nunchaku)
- [The integration PR (huggingface/diffusers#14100)](https://github.com/huggingface/diffusers/pull/14100)
- [SVDQuant paper](https://arxiv.org/abs/2411.05007) 및 [Nunchaku engine](https://github.com/nunchaku-tech/nunchaku)와 함께
- [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor)
- 이전 게시물: [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization) 및 [Memory-efficient Diffusion Transformers with Quanto and Diffusers](https://huggingface.co/blog/quanto-diffusers)

## 감사의 말씀 {#section-11}

Diffusers 유지보수자들에게 통합 전반에 걸친 리뷰와 지도를 해 주신 데에 감사드립니다. 또한 원래 SVDQuant 작업에 대해 MIT HAN Lab / Nunchaku 팀에도 감사드립니다. 블로그 포스트에 대한 피드백을 제공해 준 Marc Sun에게도 감사드립니다. `nunchaku-lite`를 시도해 보고 피드백을 제공해 준 Álvaro Somoza에게도 감사합니다.

`rootonchair`도 이 작업을 지원하고 이 개발의 많은 부분이 이 환경에서 이루어지도록 해 준 SilverAI에 감사드립니다.
