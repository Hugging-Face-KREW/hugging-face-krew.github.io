---
layout: post
title: "3D 생성 모델을 위한 결정판 미세 조정 가이드"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/3d-lora-finetuning/thumbnail.gif
image: assets/images/blog/posts/2026-11-12-3d-lora-finetuning/thumbnail.gif
authors:
  - user: suvadityamuk
  - user: merve
  - user: pcuenq
slug: "3d-lora-finetuning"
source_url: "https://huggingface.co/blog/3d-lora-finetuning"
source_published_date: "2026-11-12"
source_published_at: "2026-11-12T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [The Definitive Fine-Tuning Guide for 3D Generation Models](https://huggingface.co/blog/3d-lora-finetuning)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/3d-lora-finetuning -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 3D 생성 모델을 위한 결정판 미세 조정 가이드

![Hero GIF](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/sole_orbit_base_vs_lora.gif)

> 한눈에 보기: 이 블로그에서는 Hugging Face Jobs와 로컬 환경 모두에서 실행할 수 있는, 널리 사용되는 3D 모델 TRELLIS.2를 여러분의 3D 에셋에 맞게 미세 조정할 수 있는 스크립트와 컨테이너를 제공합니다. 미세 조정에 관심이 있다면 몇 가지 인자만으로 [저희 스크립트를 사용할 수 있으며](#walkthrough)(데이터셋을 Hugging Face Hub에 업데이트했다고 가정), 직접 수정해 볼 수도 있습니다 [이 저장소에서](https://github.com/suvadityamuk/trellis2-finetuning). 더 자세히 알아보고 싶다면 [스크립트의 작동 방식](#inside-the-training-pipeline)을 살펴봅니다. 시작해 보겠습니다!

## 소개 {#section-1}

3D 생성은 로보틱스, 시뮬레이션, 미디어, 엔터테인먼트 등 다양한 분야에서 활용됩니다. 이러한 분야에서는 생성된 객체가 엄격한 규칙을 따르도록 하고 모델과 데이터 표현을 표준화하기가 어렵습니다. [GLB](https://en.wikipedia.org/wiki/GlTF)와 [PLY](https://en.wikipedia.org/wiki/PLY_\(file_format\)) 등 파일 형식도 다양합니다. 또한 [신경 방사 필드(NeRF)](https://www.matthewtancik.com/nerf)와 [3D Gaussian splatting(3DGS)](https://en.wikipedia.org/wiki/Gaussian_splatting) 등 [다양한 표현 방식](https://huggingface.co/spaces/suvadityamuk/3d-representations-guide)이 있습니다. 각각 장단점이 다릅니다.

[TRELLIS](https://huggingface.co/papers/2412.01506)는 unified latent representation이라는 공통 내부 표현을 도입했습니다. 디코더는 이 표현을 필요한 출력 형식의 3D 모델로 변환합니다. 따라서 표현 형식마다 서로 다른 모델을 사용할 필요가 없습니다.

이후 같은 팀은 3D 구조 표현을 개선한 [TRELLIS.2](https://huggingface.co/papers/2512.14692)를 출시했습니다. 하지만 두 모델 모두 특정 작업에 쉽게 적용할 방법을 제공하지는 않습니다.

### 개선하려는 점

TRELLIS.2와 같은 범용 모델은 특정 애플리케이션에서 중요한 세부 사항에 오류를 낼 수 있습니다. 예를 들어 사용자는 가구 모델에 대해 정확한 형상을 필요로 할 수 있습니다. 해당 작업에 맞게 미세 조정하지 않으면 사전 학습 모델이 충분히 정확하지 않은 형상을 생성할 수 있습니다. 실험에는 188개의 신발 객체가 포함된 GSO의 부분집합을 사용합니다. 초기 테스트에서 사전 학습 모델은 대체로 올바르게 보이는 신발을 생성했지만 밑창은 정확하지 않았습니다. LoRA가 밑창을 더 정확하게 만들 수 있는지 테스트합니다.

![Sole Orbit Base](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/sole_orbit_base.gif)

목표는 [Google Scanned Objects (GSO)](https://huggingface.co/datasets/hugging-3d/google-scanned-objects) 데이터셋을 사용해 특정 객체 카테고리에서 TRELLIS.2의 성능을 개선하는 것입니다. 어떤 파이프라인 구성 요소를 미세 조정하고 학습 중 어떤 구성 요소를 변경하지 않을지 설명합니다. 파이프라인은 구조, 세부 형상, 재질을 분리합니다. 따라서 작업에 중요한 구성 요소만 미세 조정합니다.

![Meme](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/meme.jpg)

> *무엇을 미세 조정할지 선택하는 것만큼 무엇을 동결할지 선택하는 것도 중요합니다.*

### 무엇을 사용할까요?

#### LoRA

low-rank adaptation의 약자인 LoRA는 원래 가중치를 동결한 채 작은 어댑터 행렬을 학습해 모델을 미세 조정합니다. 이를 통해 학습 가능한 파라미터 수와 해당 파라미터의 gradient 및 optimizer state에 필요한 메모리를 줄일 수 있습니다. 이러한 어댑터를 추가하고 학습하기 위해 Hugging Face의 PEFT 라이브러리를 사용합니다. 자세한 내용은 [LoRA 가이드](https://huggingface.co/docs/peft/package_reference/lora)를 참고하세요.

#### Hugging Face Jobs

Hugging Face Jobs는 클라우드 하드웨어에서 컨테이너화된 워크로드를 실행합니다. 전처리와 학습에 이를 사용합니다. 대신 자신의 GPU를 사용하려면 아래의 로컬 Docker 명령을 참고하세요. 시작하기 전에 다음을 준비했는지 확인하세요.

- [여기에서 얻을 수 있는](https://huggingface.co/settings/tokens/new) HF_TOKEN  
- DINOv3는 gated model이므로 Hugging Face 계정에 [DINOv3에 대한 액세스 권한이 있어야 합니다](https://huggingface.co/collections/facebook/dinov3)  
- 자체 데이터셋을 사용하는 경우 시작하기 전에 해당 메시를 Hub에 업로드해야 합니다.

## TRELLIS.2의 작동 방식 {#section-2}

학습에 앞서 TRELLIS.2가 3D 객체를 생성하는 방식과 LoRA 어댑터를 어디에 추가할지 살펴보겠습니다.

TRELLIS.2에는 이미지 배경 제거, 이미지 인코딩, 거친 구조 예측, 세부 형상 생성, 재질 생성이라는 다섯 가지 주요 단계가 있습니다. 그런 다음 파이프라인은 결과를 텍스처가 적용된 메시로 내보냅니다.

![TRELLIS.2 Pipeline](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/pipeline_lr.png)

### 입력 이미지 인코딩

먼저 BiRefNet이 입력 이미지에서 배경을 제거합니다. 그런 다음 DINOv3-L이 이미지를 feature token으로 인코딩합니다. 생성 모델은 이 토큰을 사용해 3D 객체를 유도합니다.

![Pixels to feature tokens](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/image_to_tokens.png)

### 희소 구조 생성(Sparse Structure Flow)

sparse-structure flow DiT는 객체의 거친 형상을 예측합니다. 이는 16³ × 8 크기의 작은 latent를 생성하는 diffusion transformer(DiT)입니다. 그런 다음 variational autoencoder(VAE) 디코더가 이 latent를 64 × 64 × 64 occupancy grid로 변환합니다. 그리드의 각 voxel은 해당 위치가 객체의 일부인지 나타냅니다. voxel은 3차원의 픽셀입니다. 출력이 geometry 단계를 유도하므로 이 모델에 LoRA 어댑터를 추가합니다.  
![Stage 1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/stage1_sparse_structure.png)

### O-Voxel 및 Structured Latents

O-Voxel은 TRELLIS.2의 3D 형식입니다. 이는 희소 voxel grid입니다. 객체 표면에 있는 voxel만 유지합니다. 각 voxel은 메시 정점과 같은 로컬 geometry 및 재질 속성을 유지합니다. 형식이 작기 때문에 CPU는 몇 초 만에 메시를 O-Voxel로 변환할 수 있습니다. GPU는 몇 초 만에 O-Voxel을 다시 메시로 변환할 수 있습니다.

Structured sparse latent는 TRELLIS에서 유래합니다. 3D 데이터를 sparse tensor에 저장하므로 계산이 더 빨라집니다. 모델은 그리드 voxel의 약 10%만 사용하므로 빈 voxel에 연산을 소비하지 않습니다. 따라서 shape flow DiT와 texture flow DiT는 sparse transformer입니다. sparse-structure flow DiT는 dense transformer입니다.  
![O-Voxel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/ovoxel_sparse.png)

### Geometry Generation(Shape Flow)

shape flow DiT는 occupancy grid와 DINOv3 feature token을 사용해 latent를 만듭니다. 또 다른 VAE 디코더가 이 latent를 512 × 512 × 512 해상도의 O-Voxel geometry로 변환합니다.

이 단계에서는 객체의 형상을 제어하므로 DiT에 LoRA 어댑터를 추가합니다.

![Stage 2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/stage2_shape.png)

### 재질 생성 + 메시 생성

마지막 단계에서 texture flow DiT는 shape latent와 DINOv3 feature를 입력으로 사용합니다. geometry 표면의 재질과 텍스처를 정의합니다. 그런 다음 VAE 디코더가 latent를 각 voxel의 base color, metallic, roughness, alpha 값으로 변환합니다. 이 단계에는 LoRA 어댑터를 추가하지 않습니다. 데이터셋에는 diffuse color map이 있지만 이 단계의 학습에 필요한 별도의 material map은 없습니다. 또한 문제의 대상은 신발의 형상입니다. 기본 모델은 이미 색상과 텍스처를 잘 생성합니다.

마지막으로 파이프라인은 geometry와 material attribute를 텍스처가 적용된 메시로 결합하고 GLB 파일로 내보냅니다.

![Stage 3](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/stage3_texture.png)

## 계획 {#section-3}

### 미세 조정할 구성 요소

sparse-structure flow DiT와 shape flow DiT에만 LoRA 어댑터를 추가합니다. 이 DiT들이 객체의 형상과 geometry를 제어합니다. 현재 기능을 유지하기 위해 texture flow DiT는 동결합니다.

데이터셋에는 baked lighting 효과가 포함된 diffuse color map이 있습니다. texture flow DiT를 학습하려면 별도의 material map이 필요합니다. 데이터셋의 color map은 표면 색상과 조명을 섞고 있습니다. 따라서 이 학습의 올바른 target이 아닙니다.

이러한 map으로 texture flow DiT를 학습하면 조명을 표면 속성으로 표시하는 법을 학습할 수 있습니다. 또한 이전에 학습한 내용을 잊을 수 있는데, 이를 catastrophic forgetting이라고 합니다.

### LoRA를 사용하는 이유

부분집합에는 GSO의 1,030개 객체 중 신발 188개가 포함되어 있습니다. 이처럼 작은 데이터셋에는 전체 미세 조정보다 LoRA가 더 적합합니다. 전체 미세 조정은 비용이 훨씬 많이 듭니다. 또한 모델이 이미 잘 수행하는 것(학습된 prior)을 덮어쓸 수 있습니다. 현재 구성에서는 대상 DiT 두 개에 대해 총 약 930만 개의 어댑터 파라미터를 학습합니다. sparse-structure flow DiT의 attention 및 feed-forward(MLP) 레이어에는 rank 8 어댑터를 사용합니다. shape flow DiT의 attention 레이어에는 rank 16 어댑터를 사용합니다. rank는 어댑터 행렬의 크기를 결정합니다. rank가 높을수록 어댑터가 더 많은 것을 학습할 수 있습니다.

### 실행할 작업

3D 모델 설정에는 많은 종속성과 설치 단계가 필요할 수 있습니다. 필요한 소프트웨어가 포함된 Docker 이미지 두 개를 제공합니다. 이 이미지는 [Jobs](https://hf.co/jobs) 또는 자신의 컴퓨터에서 실행할 수 있습니다. 하나는 전처리용이고 다른 하나는 미세 조정용입니다.

LoRA 어댑터를 학습하고 Hub에 업로드하려면 다음을 수행합니다.

1. 컨테이너가 데이터에 액세스하도록 합니다.  
2. 작업에 필요한 파라미터를 설정합니다.  
3. 전처리 컨테이너를 실행합니다.  
4. 미세 조정 컨테이너를 실행합니다.

그런 다음 미세 조정 스크립트와 설정, 그리고 각 설정이 학습에 미치는 영향을 설명합니다.

### 실험 개요

실험은 다음 세 부분으로 진행합니다.

1. **데이터를 준비합니다.** GLB 또는 OBJ 메시를 TRELLIS.2가 학습에 사용하는 latent 표현으로 변환하고 입력 이미지를 렌더링합니다.  
2. **어댑터를 학습합니다.** 준비된 데이터셋과 LoRA 구성으로 각 DiT마다 하나의 job을 실행합니다.  
3. **코드를 설명합니다.** 컨테이너가 데이터를 로드하고, 어댑터를 추가하고, 모델을 학습하는 방식을 살펴봅니다.

전처리된 데이터셋을 저장해 모든 학습 실행에서 재사용할 수 있도록 합니다.

## 따라 하기 {#section-4}

### Hugging Face Jobs에서 실행

#### 전처리

전처리 컨테이너는 Blender를 사용해 각 객체의 16개 뷰를 렌더링합니다. LoRA 학습은 이 뷰를 conditioning image로 사용합니다. Blender는 xvfb-run 유틸리티를 통해 디스플레이 없이 실행됩니다. 컨테이너는 각 객체를 [전처리 파이프라인](https://github.com/microsoft/TRELLIS.2/blob/main/data_toolkit/README.md)으로도 처리합니다. 중간 출력 예시는 [여기에서 footwear 데이터셋을](https://huggingface.co/datasets/suvadityamuk/gso-footwear-trellis2-latents) 확인할 수 있습니다. 188개 객체의 전체 실행에는 a100-large job에서 약 1시간이 걸렸습니다. 전처리 파이프라인의 간소화된 버전은 [이 companion repository](https://github.com/suvadityamuk/trellis2-finetuning/blob/master/preprocess.py)에서도 제공됩니다.

```sh
export HF_TOKEN=hf_xxx  

# Replace your-username with your Hugging Face username

hf jobs run -d --flavor a100-large --timeout 6h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-prep:latest \
  trellis2-prep \
    --meshes your-username/my-meshes \
    --push-to your-username/my-meshes-trellis2 \
    --private --holdout 0.05 \
    --views 16 --workers 8
```


Hugging Face Jobs에서는 다음 인자를 전달합니다.

| flag | 의미 |
| ----- | ----- |
| `-d` | Detach: job ID와 URL을 출력한 다음 반환합니다. job을 추적하려면 `hf jobs logs -f <id>`을 실행합니다. |
| `--flavor a100-large` | 하드웨어입니다. a100-large(80 GB)를 사용합니다. 모든 옵션을 보려면 hf jobs hardware를 실행합니다. |
| `--timeout 6h` | 이 시간이 지나면 job이 중지됩니다. 두 명령 모두 재개할 수 있습니다. 전처리는 완료된 객체를 건너뜁니다. 학습은 --push-to의 최신 checkpoint부터 계속됩니다. |
| `--secrets HF_TOKEN` | 토큰(HF_TOKEN 또는 hf auth login에서 가져옴)을 암호화된 환경 변수로 job에 전달합니다. 토큰은 이미지에 기록되지 않습니다. |
| `cr.hf.co/...:latest` | 실행할 이미지입니다. 그 뒤의 모든 텍스트는 컨테이너에서 실행되는 명령입니다. |

이 컨테이너는 Hugging Face Jobs에서 실행하든 Docker로 실행하든 몇 가지 인자를 더 받으며, [여기에서 문서화된 내용을 확인할 수 있습니다](https://github.com/suvadityamuk/trellis2-finetuning#1-preprocess-trellis2-prep). 전처리 파이프라인의 단계는 뒤의 [이 섹션](#preprocessing-into-latents)에 설명되어 있습니다.

#### LoRA 미세 조정

학습 컨테이너는 준비된 데이터셋과 구성 preset 또는 파일을 로드합니다. Trackio로 학습 metric을 기록합니다. 실행을 모니터링하려면 Trackio dashboard를 사용하세요.

LoRA 학습에는 두 가지 모드가 있으며 이를 나타내기 위해 ‘ss’와 ‘shape’를 사용합니다. ‘ss’ 모드는 Sparse-Structure Flow DiT만 미세 조정하는 반면, ‘shape’ 모드는 Shape Flow DiT를 미세 조정합니다. 내부적으로는 각 단계에서 사용할 서로 다른 config.json 파일을 참조할 뿐입니다. 프로세스에 대한 자세한 내용은 뒤의 [이 섹션](#training)에서 확인할 수 있습니다.

```sh
export HF_TOKEN=hf_xxx  # Replace your-username with your Hugging Face username

hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config ss \
    --latents your-username/my-meshes-trellis2 \
    --push-to your-username/my-trellis2-lora \
    --run-name ss_lora \
    --push-path ss_lora \
    --trackio-space your-username/trellis2-trackio

hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config shape \
    --latents your-username/my-meshes-trellis2 \
    --push-to your-username/my-trellis2-lora \
    --run-name shape_lora \
    --push-path shape_lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500
```


`--push-to`을 설정하면 컨테이너가 어댑터 checkpoint를 Hub에 업로드합니다. 컨테이너는 [간단한 문서로도 설명된](https://github.com/suvadityamuk/trellis2-finetuning#3-train-trellis2-train) 파라미터를 받습니다. 간소화된 스크립트는 [같은 저장소](https://github.com/suvadityamuk/trellis2-finetuning/blob/master/finetune.py)에서도 제공됩니다.

[이미 처리된 데이터셋](https://huggingface.co/datasets/suvadityamuk/gso-footwear-trellis2-latents)으로 직접 실험하고 미세 조정해 보고 싶다면 다음 명령 세트를 사용해도 됩니다.

```sh
export HF_TOKEN=hf_xxx  

# Replace your-username with your Hugging Face username

# For the Sparse-Structure Flow DiT
hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train \
    --config ss \
    --latents suvadityamuk/gso-footwear-trellis2-latents \
    --push-to your-username/trellis2-footwear-lora \
    --run-name ss_lora \
    --push-path ss_lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500

# For the Shape Flow DiT
hf jobs run -d --flavor a100-large --timeout 12h --secrets HF_TOKEN \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train \
    --config shape \
    --latents suvadityamuk/gso-footwear-trellis2-latents \
    --push-to your-username/trellis2-footwear-lora \
    --run-name shape_lora \
    --push-path shape_lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500
```


### Docker로 로컬에서 실행

##### 전처리

명령과 flag는 Hugging Face Jobs에서와 같습니다. Docker 옵션만 다릅니다.

```sh
# For preprocessing into latents
docker run --gpus all --rm --shm-size 16g -e HF_TOKEN=hf_xxx \
  -v hf-cache:/root/.cache/huggingface -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-prep:latest \
  trellis2-prep --meshes your-username/my-meshes --push-to your-username/my-meshes-trellis2
```


##### LoRA 미세 조정

```sh
# For Sparse-structure Flow DiT
docker run --gpus all --rm --shm-size 16g \
  -v hf-cache:/root/.cache/huggingface -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config ss --latents your-username/my-meshes-trellis2 \
    --push-to your-username/my-trellis2-lora --trackio-space your-username/trellis2-trackio

# For Shape Flow DiT
docker run --gpus all --rm --shm-size 16g \
  -v hf-cache:/root/.cache/huggingface -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train --config shape --latents your-username/my-meshes-trellis2 \
    --push-to your-username/my-trellis2-lora --trackio-space your-username/trellis2-trackio
```


[이미 전처리된 데이터셋](https://huggingface.co/datasets/suvadityamuk/gso-footwear-trellis2-latents)을 원하는 로컬 컴퓨터에서 사용하려면 다음 명령을 실행합니다.

```sh
export HF_TOKEN=hf_xxx  # Replace your-username with your Hugging Face username

# For Sparse-structure Flow DiT
docker run --gpus all --rm --shm-size 16g -e HF_TOKEN \
  -v hf-cache:/root/.cache/huggingface \
  -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train \
    --config ss \
    --latents suvadityamuk/gso-footwear-trellis2-latents \
    --push-to your-username/trellis2-footwear-lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500

# For Shape Flow DiT
docker run --gpus all --rm --shm-size 16g -e HF_TOKEN \
  -v hf-cache:/root/.cache/huggingface \
  -v "$PWD/work":/workspace/work \
  cr.hf.co/suvadityamuk/trellis2-finetuning-train:latest \
  trellis2-train \
    --config shape \
    --latents suvadityamuk/gso-footwear-trellis2-latents \
    --push-to your-username/trellis2-footwear-lora \
    --trackio-space your-username/trellis2-trackio \
    --set max_steps=3000 i_save=500
```


학습된 LoRA 어댑터는 이 저장소에 있습니다: [suvadityamuk/trellis-2-finetune-gso](https://huggingface.co/suvadityamuk/trellis-2-finetune-gso)

## 학습 파이프라인 내부 {#section-5}

#### latent로 전처리

![Preprocessing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/preprocessing.png)
TRELLIS.2의 동결된 encoder를 사용해 입력 메시를 학습 latent로 변환합니다. 이 latent를 저장하므로 이후 실행에서는 메시를 다시 인코딩할 필요가 없습니다.

또한 Blender에서 각 객체의 16개 뷰를 렌더링합니다. 학습 중에는 각 sample마다 하나의 random view를 conditioning image로 선택합니다. 188개 객체 중 9개는 평가용으로 남겨 두고 나머지 179개로 학습합니다. 이렇게 제외한 객체는 학습 중 모델이 보지 못한 객체에서도 어댑터가 작동하는지 보여줍니다.

![Conditioning](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/conditioning_views.png)

> *한 객체에 대해 Blender가 렌더링한 뷰 중 일부*

첫 번째 snippet은 데이터를 설정합니다. `FLOW`은 학습할 flow DiT(`ss` 또는 `shape`)를 선택하고, `STEPS`은 optimizer step 수를 설정합니다. batch size 4와 gradient accumulation step 4를 사용하면 effective batch size는 16입니다. 그런 다음 snippet은 선택한 flow DiT의 LoRA config 파일을 로드하고, TRELLIS.2 형식으로 데이터셋을 만들고, 제외한 객체를 제거하고, 미세 조정을 위한 data loader를 만듭니다.

```py
cfg = json.load(
    open(
        {
            "ss": "/workspace/TRELLIS.2/configs/lora/ss_flow_lora_gso.json",
            "shape": "/workspace/TRELLIS.2/configs/lora/slat_flow_img2shape_lora_gso.json",
        }[FLOW]
    )
)

key, latents = {
    "ss": ("ss_latent", "ss_latents/ss_enc_conv3d_16l8_fp16_64"),
    "shape": ("shape_latent", "shape_latents/shape_enc_next_dc_f16c32_fp16_512"),
}[FLOW]


roots = json.dumps(
    {
        "gso": {
            key: f"{DATA}/{latents}",
            "render_cond": f"{DATA}/renders_cond",
            "meta": DATA,
        }
    }
)

ds = getattr(datasets, cfg["dataset"])(roots, **cfg["dataset_args"])

holdout = set(open(f"{DATA}/holdout.txt").read().split())
ds.instances = [(root, sha) for root, sha in ds.instances if sha not in holdout]

batches = cycle(
    DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        collate_fn=getattr(ds, "collate_fn", None),
    )
)
```


#### LoRA 어댑터 준비

![Preparing LoRA Adapters](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/lora_adapters.png)
`peft` 라이브러리를 사용해 선택한 레이어에 LoRA 어댑터를 삽입합니다. TRELLIS.2에는 custom forward pass가 있으므로 표준 `PeftModel()` wrapper는 작동하지 않습니다. 대신 `inject_adapter_in_model()`을 호출합니다. 이 함수는 어댑터를 모델에 직접 추가하고 custom interface를 유지합니다.

수치 안정성을 위해 학습 가능한 어댑터 가중치는 32-bit floating point(`fp32`)로 유지하고 forward pass에는 bfloat16(`bf16`) autocasting을 사용합니다. Autocasting을 사용하면 PyTorch가 일부 연산에 더 낮은 정밀도를 사용할 수 있어 메모리 사용량이 줄어듭니다. Autocasting은 동결된 backbone 가중치의 저장 방식은 변경하지 않습니다.

```py

model = models.from_pretrained(cfg["ckpt"])
for block in model.blocks:
    block.use_checkpoint = True

inject_adapter_in_model(
    LoraConfig(
        r=cfg["rank"], lora_alpha=cfg["alpha"], target_modules=cfg["lora_targets"]
    ),
    model,
)

params = []
for name, p in model.named_parameters():
    p.requires_grad_("lora_" in name)
    if p.requires_grad:
        p.data = p.data.float()  # fp32 master weights for LoRA
        params.append(p)

model.cuda().train()


```


#### 학습

![Training Loop](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/training_loop.png)
여기서 objective function은 널리 사용되는 이미지 생성 모델 [Stable Diffusion 3](https://huggingface.co/collections/stabilityai/stable-diffusion-3), [Flux 2](https://huggingface.co/collections/black-forest-labs/flux2) 등에서 볼 수 있는 전통적인 rectified flow matching이라는 점에 유의해야 합니다. TRELLIS.2도 pretraining에서 같은 objective를 사용했으므로 미세 조정에 적합합니다. 이론은 flow matching을 다루는 [이 블로그](https://huggingface.co/spaces/ariG23498/flow-matching)를 참고하세요.

sparse-structure 및 shape DiT에서 LoRA 파라미터만 업데이트합니다. 원래 모델 가중치는 모두 동결된 상태로 유지됩니다.

>> 아래 snippet은 한 단계의 핵심 학습 loop를 보여줍니다. 컨테이너 코드보다 단순하며 일부 설정 단계는 포함하지 않습니다. 전체 workflow에는 컨테이너 명령을 사용하세요.

optimizer를 초기화하고 입력 이미지에서 feature를 추출하기 위해 DINOv3를 로드합니다.

```py
opt = torch.optim.AdamW(
    params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95)
)

dino = DinoV3FeatureExtractor(
    cfg["image_cond_model"], image_size=cfg["image_size"]
).cuda()

```


그런 다음 학습 loop 안에서 다음 단계를 수행합니다.

* 각 sample에 대해 timestep *t*를 sampling합니다(TRELLIS.2 config에 따라 sparse-structure flow DiT에는 logitNormal, shape flow DiT에는 uniform 사용).  
* 데이터 shape에 맞는 random noise를 생성하고 broadcasting을 위해 *t*를 reshape합니다.  
* *t*에 따라 clean data와 noise를 혼합합니다(*t = 0*은 noise가 최소인 data이고 *t = 1*은 pure noise입니다).  
* pretraining 조건에 맞추기 위해 *t × 1000*에서 velocity를 예측합니다.  
* 예측 velocity와 target velocity(*x_t*의 first derivative) 사이의 mean squared error로 loss를 계산합니다.

```py
def flow_loss(x0, cond):
    s = cfg["t_schedule"]

    t = (
        torch.rand(x0.shape[0])
        if s["name"] == "uniform"
        else torch.sigmoid(
            torch.randn(x0.shape[0]) * s["args"]["std"] + s["args"]["mean"]
        )
    )
    t = t.cuda()

    noise = (
        x0.replace(torch.randn_like(x0.feats))
        if cfg["sparse"]
        else torch.randn_like(x0)
    )

    tv = t.view(-1, *[1] * (len(x0.shape) - 1))

    x_t = (1 - tv) * x0 + (cfg["sigma_min"] + (1 - cfg["sigma_min"]) * tv) * noise

    pred, target = model(x_t, t * 1000, cond), (1 - cfg["sigma_min"]) * noise - x0

    return (
        F.mse_loss(pred.feats, target.feats)
        if cfg["sparse"]
        else F.mse_loss(pred, target)
    )

```


학습 loop는 loss를 계산하고 어댑터 가중치를 업데이트합니다. Gradient accumulation은 각 optimizer step 전에 4개의 작은 batch에서 나온 gradient를 더합니다. Gradient checkpointing은 backpropagation 중 일부 중간 값을 다시 계산해 GPU 메모리를 절약합니다. 외부 loop는 지정된 optimizer step 수만큼 실행되고, 내부 loop는 정해진 batch 수에 걸쳐 gradient를 누적합니다.

각 batch에서 DINOv3가 conditioning image를 인코딩합니다. [classifier-free guidance](https://huggingface.co/papers/2207.12598)(CFG) dropout을 위해 loop는 sample의 random fraction(p_uncond)에 대해 conditioning을 0으로 설정합니다. 그런 다음 bf16 autocasting으로 flow-matching loss를 계산합니다. 내부 loop가 끝나면 매우 큰 update를 방지하기 위해 gradient를 clip하고 optimizer step을 수행합니다. 10 step마다 loss를 기록하고 500 step마다 어댑터 checkpoint를 저장합니다.

```py

for step in range(1, STEPS + 1):
    for _ in range(ACCUM):
        batch = next(batches)
        x0 = recursive_to_device(batch["x_0"], "cuda")

        with torch.no_grad():
            cond = dino(batch["cond"])
            cond[torch.rand(len(cond), device="cuda") < cfg["p_uncond"]] = 0

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_loss(x0, cond) / ACCUM

            loss.backward()

    torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])

    opt.step()
    opt.zero_grad()
```


### 결과

두 개의 어댑터를 학습했습니다. 하나는 sparse-structure DiT용이고 다른 하나는 shape DiT용입니다. 컨테이너 이미지의 LoRA config 파일에는 전체 학습 설정이 포함되어 있습니다. sparse-structure flow DiT에는 [`ss_flow_lora_gso.json`](https://github.com/suvadityamuk/trellis2-finetuning/blob/master/container/trellis2/configs/lora/slat_flow_img2shape_lora_gso.json), shape flow DiT에는 [`slat_flow_img2shape_lora_gso.json`](https://github.com/suvadityamuk/trellis2-finetuning/blob/master/container/trellis2/configs/lora/slat_flow_img2shape_lora_gso.json)을 사용합니다. 각 실행은 Hugging Face Jobs에서 A100 80 GB 하나를 사용했습니다. sparse-structure 실행에는 7.2시간, shape 실행에는 3.6시간이 걸렸습니다.

sparse-structure flow DiT의 training loss는 시작할 때 빠르게 감소한 뒤 더 느리게 감소합니다. shape flow DiT의 loss는 대략 같은 수준을 유지합니다. 이러한 curve는 training objective가 어떻게 변하는지 보여주지만 생성된 geometry의 품질을 직접 측정하지는 않습니다.  
![Loss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/training_loss.png)

Flow-matching loss는 sampling된 객체, noise, timestep에 따라 달라집니다. 미세 조정이 geometry를 개선하는지 확인하려면 제외한 객체에 대한 output도 비교해야 합니다.

아래 그림은 제외한 9개 객체 중 5개에 대한 output을 비교합니다. 입력 이미지, base model의 생성 output, rotation adjustment 후의 base output, footwear adapter를 사용한 output, reference scan을 보여줍니다. 이 예시에서 미세 조정 모델의 밑창은 scan에 더 가깝습니다.

![Heldout Meshes](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/heldout_meshes.png)

또한 test set에서 geometry를 평가하기 위해 세 가지 metric을 사용합니다.

| Metric | 의미 |
| ----- | ----- |
| Chamfer distance | 한 표면의 점에서 다른 표면의 가장 가까운 점까지의 거리로, 양방향으로 측정합니다. 낮을수록 좋습니다. |
| F-score @ 0.02 | precision(생성된 표면 중 reference에 가까운 부분의 비율)과 recall(생성된 표면이 포함하는 reference 표면의 비율)의 조합입니다. 여기서 matching threshold는 scan bounding-box diagonal의 2%입니다. 높을수록 좋습니다. |
| Normal consistency | 표면 방향이 얼마나 일치하는지 나타냅니다. Surface normal은 표면에 수직인 벡터입니다. 높을수록 좋습니다. |

### ![Heldout Metrics](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/heldout_metrics.png)

9개 객체를 비교한 chart에서 base model의 평균 F-score는 0.276인 반면 footwear adapter를 사용한 모델은 0.803이었습니다. Chamfer distance는 0.174에서 0.033으로 감소했고 normal consistency는 0.530에서 0.804로 증가했습니다. 두 어댑터를 모두 사용했을 때 F-score는 제외한 9개 객체 모두에서 더 좋았습니다.   
![Object Metrics](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/3d-lora-finetuning/per_object_fscore.png)

모든 실행의 통계는 이 [Trackio space](https://suvadityamuk-trellis2-geoflow-lora-trackio.hf.space/media?project=trellis2-geoflow-lora&runs=blog-results)에서 확인할 수 있습니다.

## 향후 방향 {#section-6}

드디어 마무리되었습니다! geometry를 제어하는 두 모델 구성 요소에 LoRA 어댑터를 추가해 신발 생성을 위해 TRELLIS.2를 미세 조정했습니다. 데이터를 한 번 준비하고, 별도의 job에서 어댑터를 학습했으며, 제외한 객체에 대한 output을 평가했습니다.

핵심 교훈은 문제와 보유한 데이터에 부합하는 training target을 선택해야 한다는 것입니다. 이번 실험에서는 structure 및 shape 단계를 미세 조정하고 texture 단계는 동결했습니다.

다음으로 [Pixal3D](https://github.com/TencentARC/Pixal3D)와 [UniLat3D](https://github.com/UniLat3D) 같은 다른 3D 생성 모델에서도 같은 선택이 도움이 되는지 살펴볼 수 있습니다.

[HunyuanWorld 2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)와 [NVIDIA Lyra 2.0](https://nv-tlabs.github.io/Project-Lyra/) 같은 Gaussian-splat 및 world-generation 모델도 다음 단계가 될 수 있습니다. 이러한 모델은 서로 다른 표현 방식과 학습 파이프라인을 필요로 할 수 있습니다.

끝까지 함께해 주셔서 감사합니다. 다음 글에서 만나요! 👋🤗
