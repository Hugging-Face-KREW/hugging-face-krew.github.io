---
layout: post
title: "미세 조정 NVIDIA Cosmos Predict 2.5 with LoRA/DoRA for Robot Video Generation"
author: dailybot
categories: [Translation, HuggingFace]
slug: "cosmos-fine-tuning-for-robot-video-generation"
source_url: "https://huggingface.co/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation"
source_published_date: "2026-05-18"
source_published_at: "2026-05-18T16:00:21+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

> Source: https://huggingface.co/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation

* TOC
{:toc}
<!--toc-->

_이 글은 Hugging Face 블로그의 [Fine-Tuning NVIDIA Cosmos Predict 2.5 with LoRA/DoRA for Robot Video Generation](https://huggingface.co/blog/nvidia/cosmos-fine-tuning-for-robot-video-generation)를 한국어로 번역한 글입니다._

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 미세 조정 NVIDIA Cosmos Predict 2.5 with LoRA/DoRA for Robot Video Generation

## 동기

NVIDIA [Cosmos Predict 2.5](https://arxiv.org/abs/2511.00062)는 텍스트, 이미지, 또는 비디오 클립을 조건으로 물리적으로 타당한 비디오를 생성할 수 있는 대규모 [월드 모델](https://www.nvidia.com/en-us/glossary/world-models/?ncid=so-nvsh-876275)입니다. 로봇 조작이나 특정 카메라 시점과 같은 특정 도메인에 맞추려면 여전히 표적화된 미세 조정이 필요합니다.

로봇 정책(로봇 제어 정책)을 훈련하려면 시범 데이터가 필요하지만, 실제 로봇 궤적을 수집하는 것은 느리고 비용이 많이 듭니다. 미세 조정된 비디오 월드 모델로 합성 궤적을 생성하는 것은 확장 가능한 대안입니다. 다만 2B 매개변수 모델의 전체 미세 조정은 비용이 많이 들고 일반 지식의 재앙적 망각(catatastrophic forgetting)의 위험이 있습니다. [LoRA](https://arxiv.org/abs/2106.09685)와 [DoRA](https://arxiv.org/abs/2402.09353)는 동결된 기본 모델에 작은 학습 가능한 어댑터 모듈을 주입하여 메모리 요구를 줄이고 어댑터 파일 크기를 작고 휴대 가능하게 유지합니다. 이를 통해 단일 GPU에서의 미세 조정을 실용적으로 수행하고 추론 시 서로 다른 도메인에 대해 어댑터를 유연하게 교체할 수 있습니다.

이 가이드는 `diffusers`와 `accelerate` 라이브러리를 사용해 단일 및 다중 GPU 학습을 지원하는 LoRA와 DoRA를 이용한 Cosmos Predict 2.5의 매개변수 효율적 미세 조정을 안내합니다. 그런 다음 미세 조정된 모델을 사용해 downstream [로봇 학습](https://www.nvidia.com/en-us/use-cases/robot-learning/) 작업에 사용할 합성 로봇 궤적을 생성하는 방법을 보여줍니다.

## 요구사항

- Python 3.10+

- PyTorch 2.5+ with CUDA

- `diffusers` (자동으로 `transformers`와 `peft`를 가져옴), `accelerate`

- 선택사항: 학습 모니터링을 위해 `wandb` 설치

- 최소한 하나의 80 GB GPU가 단일 GPU 학습에 필요; 빠른 반복을 위해 8× H100 권장

머신에 의존성 설치:

```
pip install -U "diffusers[torch]" transformers accelerate peft wandb 
```

## 데이터 준비

diffusers를 설치한 후 [examples/cosmos](https://github.com/terarachang/diffusers/tree/cosmos_predict_2.5_lora_clean/examples/cosmos)로 이동해 예제 코드를 살펴봅니다. 우리는 [GR00T Dreams post-training recipe](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/end2end/gr00t-dreams/post-training.html)에서 사용하는 동일한 데이터셋을 사용합니다:

- [Training Dataset](https://huggingface.co/datasets/nvidia/GR1-100): 픽 앤 플레이(Task) 설명이 포함된 92개의 로봇 조작 비디오

- [Test Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Eval): (프롬프트, 이미지) 쌍 50개. 모델은 입력 텍스트 프롬프트와 초기 프레임 이미지를 바탕으로 비디오를 생성해야 합니다.

학습 및 테스트 데이터셋 다운로드 및 전처리는 [download_and_preprocess_datasets.sh](https://github.com/terarachang/diffusers/blob/cosmos_predict_2.5_lora_clean/examples/cosmos/download_and_preprocess_datasets.sh)를 사용합니다:

```
bash download_and_preprocess_datasets.sh
```

생성된 학습 데이터셋 폴더 구조는 다음과 같습니다:

```
gr1_dataset/train
├── metas/
│   └── *.txt
├── videos/
│   └── *.mp4
└── metadata.csv
```

평가 데이터셋은 (프롬프트, 이미지) 쌍의 `.txt`와 `.png` 파일이 평탄한 디렉터리로 구성됩니다:

```
gr1_dataset/test
├── filename1.txt
├── filename1.png
├── filename2.txt
├── filename2.png
└── ...
```

## Training

본 섹션에서는 [train_cosmos_predict25_lora.py](https://github.com/terarachang/diffusers/blob/cosmos_predict_2.5_lora_clean/examples/cosmos/train_cosmos_predict25_lora.py)에 구현된 내용을 따라갑니다.

### VideoDataset

`VideoDataset`은 각 샘플을 `args.train_data_dir`에서 `(caption, video)` 쌍으로 로드합니다(우리 예제의 경우 `gr1_dataset/train`). `args.num_frames`보다 긴 비디오의 경우 매 에포크마다 임의의 연속 윈도우를 샘플링해 시간적 증강을 가능하게 합니다. 내부적으로 `diffusers.video_processor`의 `VideoProcessor`가 원시 프레임을 리사이즈하고 정규화하여 (channels, frames, height, width) 형태의 텐서로 변환합니다.

```
train_dataset = VideoDataset(
    dataset_dir=args.train_data_dir,
    num_frames=args.num_frames,
    video_size=[args.height, args.width],
)
```

### 어댑터 초기화

Cosmos Predict 2.5는 세 개의 서브모듈로 구성됩니다:

- 비디오를 래티엔트(latents)로 인코딩하는 VAE

- 텍스트 프롬프트를 프롬프트 임베딩으로 인코딩하는 텍스트 인코더

- 래티엔트 공간에서의 확산을 위한 DiT

학습 중에는 모든 VAE, 텍스트 인코더, DiT 가중치가 고정됩니다. LoRA 어댑터는 DiT의 주의(attention) 프로젝션(`to_q`, `to_k`, `to_v`, `to_out.0`) 및 피드포워드 계층(`ff.net.0.proj`, `ff.net.2`)에 주입됩니다. 학습 가능한 LoRA 매개변수는 BF16 혼합 정밀도 하에서 수치 안정성을 위해 float32로 업캐스팅됩니다.

```
from
 diffusers 
import
 Cosmos2_5_PredictBasePipeline

from
 peft 
import
 LoraConfig

pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
    
"nvidia/Cosmos-Predict2.5-2B"
,
    revision=
"diffusers/base/post-trained"
,
    torch_dtype=torch.bfloat16,
)


# freeze all base weights

dit = pipe.transformer
vae = pipe.vae
text_encoder = pipe.text_encoder

dit.requires_grad_(
False
)
vae.requires_grad_(
False
)
text_encoder.requires_grad_(
False
)

lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=[
'to_q'
, 
'to_k'
, 
'to_v'
, 
'to_out.0'
, 
'ff.net.0.proj'
, 
'ff.net.2'
],
    use_dora=args.use_dora,  
# set True to switch to DoRA

)
dit.add_adapter(lora_config)
cast_training_params(dit, dtype=torch.float32)  
# LoRA params in fp32
```

`use_dora=True`를 전달하면 DoRA로 전환되며, 각 가중치를 저랭크 업데이트를 적용하기 전으로부터 크기(magnitude)와 방향(direction)으로 분해합니다. 학습 루프에 다른 변경은 필요하지 않습니다.

### 손실 함수

Cosmos Predict 2.5는 rectified flow를 사용합니다: 모델은 노이즈 샘플을 원래의 "깨끗한" 데이터로 선형적으로 운반하는 속도(velocity)를 예측하도록 학습됩니다. 구체적으로, 타임스텝 t에서 노이즈 보정(interpolation) `xt = σt·noise + (1−σt)·clean`이 샘플링된 노이즈 레벨에서 구성되고, 모델은 평균 제곱 오차(MSE 손실)를 통해 목표 속도 `noise − clean`를 예측하는 것을 학습합니다. 비디오의 처음 두 프레임은 조건으로 사용되므로 해당 잠재(latent)에는 노이즈가 추가되지 않습니다.

학습 손실은 Cosmos Predict 2.5에서 사용하는 rectified flow 공식에 따릅니다:

```
# Sample timestep with logit-normal distribution

sigma_t = sample_train_sigma_t(bsz, distribution=
'logitnormal'
, device=device)


# Rectified flow interpolates between clean latent and noise

xt = noise * sigma_t + clean_latent * (
1
 - sigma_t)


# Conditional generation: DiT conditions on the first two frames of the video, the timestep, and the prompt embeds


# `cond_indicator` and `cond_mask` have values = 1 for the first two frames and 0 for other frames

xt = clean_latent * cond_mask + xt * (
1
 - cond_mask)
in_timestep = cond_indicator * 
0.0001
 + (
1
 - cond_indicator) * sigma_t


# Forward

pred_velocity = dit(
                    hidden_states=xt,
                    condition_mask=cond_mask,
                    timestep=in_timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=
False
,
                )[
0
]


# MSE loss is computed only on the non-conditioned frames

target_velocity = noise - clean_latent
pred_velocity = target_velocity * cond_mask + pred_velocity * (
1
 - cond_mask)
loss = F.mse_loss(pred_velocity.
float
(), target_velocity.
float
())
```

### 옵티마이저와 스케줄러

최적화 기법으로 `torch.optim.AdamW`를 사용하고, 스케줄러로는 diffusers.optimization의 `get_linear_schedule_with_warmup`을 사용합니다. 스케줄러는 학습률을 스케줄러 워밍업 스텝 수(`scheduler_warm_up_steps`) 동안 선형으로 증가시키고, 최대점에서 `scheduler_f_max × learning_rate`에 도달한 뒤 남은 `num_training_steps` 동안 선형적으로 감소시킵니다.

```
lora_params = [p 
for
 p 
in
 dit.parameters() 
if
 p.requires_grad]

optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.weight_decay)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.scheduler_warm_up_steps,
    num_training_steps=args.num_training_steps,
    f_min=args.scheduler_f_min,
    f_max=args.scheduler_f_max,
)
```

### 체크포인트 저장

LoRA 가중치는 각 에폭마다 diffusers 포맷으로 저장됩니다: `args.checkpointing_epochs` 에폭마다:

```
if
 (epoch+
1
) % args.checkpointing_epochs == 
0
:
    
if
 accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, 
f"checkpoint-
{epoch}
"
)
        accelerator.save_state(save_path)
```

`accelerator.save_state()`는 `pytorch_lora_weights.safetensors` 파일을 저장하여 inference 시 파이프라인에 전달할 어댑터 파일로 사용합니다.

### Training Command

시작점으로 제공된 셸 스크립트를 사용합니다:

```
export MODEL_NAME="nvidia/Cosmos-Predict2.5-2B"
export DATA_DIR="gr1_dataset/train"
export OUT_DIR=YOUR_OUTPUT_DIR
lora_rank=32

accelerate launch --mixed_precision="bf16" train_cosmos_predict25_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --revision diffusers/base/post-trained \
  --train_data_dir=$DATA_DIR \
  --train_batch_size=1 \
  --num_train_epochs=500 \
  --checkpointing_epochs=100 \
  --seed=0 \
  --output_dir=$OUT_DIR \
  --report_to=wandb \
  --height 432 --width 768 \
  --allow_tf32 --gradient_checkpointing \
  --lora_rank $lora_rank --lora_alpha $lora_rank
```

`lora_rank`는 저랭크 분해의 랭크를 제어합니다. 랭크가 높을수록 학습 가능한 매개변수가 많아지고 표현 능력이 커지지만 메모리 사용량과 어댑터 파일 크기가 증가합니다. 시작점으로 rank=32를 사용하면 약 50M 학습 가능 매개변수를 얻습니다.

`lora_alpha`는 LoRA 업데이트에 적용되는 스케일링 계수입니다: 가중치 변화량은 원래 고정 가중치에 더해지기 전에 `lora_alpha / lora_rank`로 스케일링됩니다. 여기서 `lora_alpha = lora_rank`로 설정하면 이 스케일이 1.0이 되어 LoRA 업데이트가 추가 완강도 없이 전체 강도로 적용됩니다.

LoRA 대신 DoRA를 사용하려면 커맨드에 `--use_dora`를 추가합니다.

다중 GPU 학습의 경우 `accelerate`가 자동으로 분산을 처리합니다. 경험적으로 100 에폭의 학습으로도 이 작업에서 상당한 성능 향상을 얻을 수 있으며, 이는 단일 H100에서 약 17시간, 8개의 H100 GPU에서는 약 2.5시간이 소요됩니다.

## LoRA로 학습한 모델로 추론 실행

학습이 완료되면 [eval_cosmos_predict25_lora.py](https://github.com/terarachang/diffusers/blob/cosmos_predict_2.5_lora_clean/examples/cosmos/eval_cosmos_predict25_lora.py)을 사용해 평가 데이터셋에서 비디오를 생성합니다. 이 스크립트는 `gr1_dataset/test`에서 쌍(pair) `.png`와 `.txt` 파일을 읽고 각 파일에 대한 비디오를 생성한 뒤 `--output_dir`에 `.mp4` 파일로 저장합니다.

### ImageDataset

`ImageDataset`은 `.txt` 파일을 프롬프트 문자열로 읽고 `diffusers.utils`의 `load_image`를 사용해 `.png`를 `PIL.Image.Image`로 로드합니다:

```
def
 
__getitem__
(
self, idx
):
    img_path, txt_path, stem = self.samples[idx]
    image = load_image(img_path)
    
with
 
open
(txt_path) 
as
 f:
        prompt = f.read().strip()
    
return
 {
"image"
: image, 
"prompt"
: prompt, 
"stem"
: stem}
```

### 파이프라인과 LoRA/DoRA 가중치 로딩

```
from
 diffusers 
import
 Cosmos2_5_PredictBasePipeline

pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
    
"nvidia/Cosmos-Predict2.5-2B"
,
    revision=
"diffusers/base/post-trained"
,
    device_map=
"cuda"
,
    torch_dtype=torch.bfloat16,
)

pipe.load_lora_weights(
"/path/to/lora/checkpoint"
)
pipe.fuse_lora(lora_scale=
1.0
)
```

`fuse_lora`는 어댑터 가중치를 기본 모델에 합쳐 LoRA/DoRA 분해로 인한 추론 오버헤드를 제거합니다.

### 초기 잠재 노이즈 생성

재현성을 보장하기 위해, `arch_invariant_rand` 함수는 [NumPy](https://www.nvidia.com/en-us/glossary/numpy/)를 통해 초기 잠재 노이즈를 생성하여 GPU 아키텍처에 독립적이도록 만듭니다. 재현성이 중요하지 않다면 파이프라인에 입력 노이즈를 제공할 필요가 없습니다.

```
# generation starts from random noise with the same shape as the latent

latent_shape = pipe.get_latent_shape_cthw(args.height, args.width, args.num_output_frames)
noises = arch_invariant_rand(
    (args.batch_size, *latent_shape), dtype=torch.float32, device=args.device, seed=args.seed
)

frames = pipe(
    image=image,           
# PIL Image: the conditioning first frame

    prompt=prompt,
    num_frames=args.num_output_frames,
    num_inference_steps=args.num_steps,
    height=args.height,
    width=args.width,
    latents=noises,        
# optional

).frames[
0
]

export_to_video(frames, 
"output.mp4"
, fps=
16
)
```

### 추론 명령

```
export LORA_DIR=YOUR_ADAPTER_DIR
export DATA_DIR="gr1_dataset/test"
export OUT_DIR=YOUR_EVAL_OUTPUT_DIR

python eval_cosmos_predict25_lora.py \
  --data_dir $DATA_DIR \
  --output_dir $OUT_DIR \
  --lora_dir $LORA_DIR \
  --height 432 --width 768 \
  --num_output_frames 93 \
  --num_steps 36 \
  --seed 0
```

LoRA 없이 기본 모델만 평가하려면 `--lora_dir`을 생략합니다.

## 평가 지표

### Sampson Error

Sampson Error는 대응되는 키포인트가 대응하는 에피폴라 선으로부터의 거리를 측정하는 기하학적 오류 지표입니다. 생성된 비디오의 맥락에서 낮은 Sampson 에러는 프레임 간(또는 카메라 뷰 간)의 운동이 기하학적으로 일관됨을 의미합니다. 값이 높아지면 지터, 환영 동작, 다중 뷰 간의 불일치를 나타낼 수 있습니다.

우리는 [Cosmos Predict 평가 가이드](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/evaluation/evaluation_predict.html)를 따르며 생성된 비디오의 기하학적 품질을 두 가지 지표로 평가합니다:

- 시간적 Sampson Error: 한 카메라 뷰 내 연속 프레임 간의 시간적 안정성 측정

- 교차 뷰 Sampson Error: 서로 다른 카메라 뷰의 동시 프레임 간 기하학적 정합성 측정

### LLM-판정자

우리는 [Cosmos Reason2](https://huggingface.co/nvidia/Cosmos-Reason2-2B)를 LLM 평가자로 사용하여 각 예제를 1에서 5까지 점수화합니다. 두 가지 루브릭을 설계했습니다:

- 물리적 타당성(비디오 물리 규칙 평가, [video_physics.yaml](https://github.com/terarachang/diffusers/blob/cosmos_predict_2.5_lora_clean/examples/cosmos/llm_judge_prompts/video_physics.yaml)): 프롬프트를 보지 않고도 비디오가 물리적 상식을 준수하는지 판단합니다.

- 지시 수행 여부([video_IF.yaml](https://github.com/terarachang/diffusers/blob/cosmos_predict_2.5_lora_clean/examples/cosmos/llm_judge_prompts/video_IF.yaml)): 프롬프트와 비디오를 입력으로 받아 주어진 작업이 올바르게 수행되었는지 평가합니다.

video_physics.yaml

```
system_prompt:
 
"You are a helpful assistant."


user_prompt:
 
|


  You are a helpful video analyzer. Evaluate whether the video follows physical commonsense.



  
Evaluation Criteria:

  
1
.
 
**Object
 
Behavior:**
 
Do
 
objects
 
behave
 
according
 
to
 
their
 
expected
 
physical
 
properties
 
(e.g.,
 
rigid
 
objects
 
do
 
not
 
deform
 
unnaturally,
 
fluids
 
flow
 
naturally)?

  
2
.
 
**Motion
 
and
 
Forces:**
 
Are
 
motions
 
and
 
forces
 
depicted
 
in
 
the
 
video
 
consistent
 
with
 
real-world
 
physics
 
(e.g.,
 
gravity,
 
inertia,
 
conservation
 
of
 
momentum)?

  
3
.
 
**Interactions:**
 
Do
 
objects
 
interact
 
with
 
each
 
other
 
and
 
their
 
environment
 
in
 
a
 
plausible
 
manner
 
(e.g.,
 
no
 
unnatural
 
penetration,
 
appropriate
 
reactions
 
on
 
impact)?

  
4
.
 
**Consistency
 
Over
 
Time:**
 
Does
 
the
 
video
 
maintain
 
consistency
 
across
 
frames
 
without
 
abrupt,
 
unexplainable
 
changes
 
in
 
object
 
behavior
 
or
 
motion?


  
Instructions for Scoring:

  
-
 
**1:**
 
No
 
adherence
 
to
 
physical
 
commonsense.
 
The
 
video
 
contains
 
numerous
 
violations
 
of
 
fundamental
 
physical
 
laws.

  
-
 
**2:**
 
Poor
 
adherence.
 
Some
 
elements
 
follow
 
physics,
 
but
 
major
 
violations
 
are
 
present.

  
-
 
**3:**
 
Moderate
 
adherence.
 
The
 
video
 
follows
 
physics
 
for
 
the
 
most
 
part
 
but
 
contains
 
noticeable
 
inconsistencies.

  
-
 
**4:**
 
Good
 
adherence.
 
Most
 
elements
 
in
 
the
 
video
 
follow
 
physical
 
laws,
 
with
 
only
 
minor
 
issues.

  
-
 
**5:**
 
Perfect
 
adherence.
 
The
 
video
 
demonstrates
 
a
 
strong
 
understanding
 
of
 
physical
 
commonsense
 
with
 
no
 
violations.


  
Does
 
this
 
video
 
adhere
 
to
 
the
 
physical
 
laws?
```

video_IF.yaml

```
system_prompt:
 
"You are a helpful assistant."


user_prompt:
 
|


  You are a helpful video analyzer. Evaluate whether the video follows the given instruction.



  
Instruction:
 {
instruction
}

  
Evaluation Criteria:

  
1
.
 
**Task
 
Completion:**
 
Does
 
the
 
video
 
show
 
the
 
task
 
described
 
in
 
the
 
instruction
 
being
 
completed?

  
2
.
 
**Action
 
Accuracy:**
 
Are
 
the
 
actions
 
performed
 
in
 
the
 
video
 
consistent
 
with
 
what
 
the
 
instruction
 
specifies?

  
3
.
 
**Object
 
Interaction:**
 
Does
 
the
 
robot
 
or
 
agent
 
interact
 
with
 
the
 
correct
 
objects
 
as
 
described
 
in
 
the
 
instruction?

  
4
.
 
**Goal
 
Achievement:**
 
Is
 
the
 
final
 
state
 
of
 
the
 
video
 
consistent
 
with
 
the
 
expected
 
outcome
 
of
 
the
 
instruction?

  
5
.
 
**Correct
 
Hand
 
Usage:**
 
Does
 
the
 
video
 
show
 
the
 
correct
 
hand
 
performing
 
the
 
action?


  
Instructions for Scoring:

  
-
 
**1:**
 
No
 
adherence
 
to
 
the
 
instruction.
 
The
 
video
 
shows
 
actions
 
completely
 
unrelated
 
to
 
the
 
instruction.

  
-
 
**2:**
 
Poor
 
adherence.
 
Some
 
elements
 
match
 
the
 
instruction,
 
but
 
major
 
deviations
 
are
 
present.

  
-
 
**3:**
 
Moderate
 
adherence.
 
The
 
video
 
follows
 
the
 
instruction
 
for
 
the
 
most
 
part
 
but
 
contains
 
noticeable
 
deviations.

  
-
 
**4:**
 
Good
 
adherence.
 
Most
 
elements
 
in
 
the
 
video
 
match
 
the
 
instruction,
 
with
 
only
 
minor
 
issues.

  
-
 
**5:**
 
Perfect
 
adherence.
 
The
 
video
 
fully
 
follows
 
the
 
instruction
 
with
 
no
 
deviations.


  
Does
 
this
 
video
 
follow
 
the
 
instruction?
```

## Results

### 질적 분석

루트 모델(미세 조정 이전), LoRA, DoRA로 생성된 비디오를 평가 데이터셋의 처음 두 예제에서 비교합니다.

프롬프트: 왼손으로 어두운 초록색 오이를 원형 회색 매트 위에서 들어 올려 벤치 색상 베이지 그릇 위로 올리기.

프롬프트: 오른손으로 주황색 주스 카톤을 분홍색 접시의 중앙에서 초록색 그릇의 중앙으로 옮기기.

미세 조정 이전에는 기본 모델이 여러 면에서 어려움을 겪었습니다: 로봇 손이 분포 밖의 손으로 보이고, 이후 프레임에서 사람 손이 환각되며, 프롬프트에 명시된 손을 일관되게 사용하지 못하고, 생성된 비디오에 눈에 띄는 지터가 나타났습니다. LoRA와 DoRA로의 미세 조정은 이 세 가지 문제를 모두 해결합니다.

### 정량 분석

우리는 서로 다른 설정으로 네 가지 어댑터를 미세 조정합니다: 랭크 8과 32의 LoRA와 DoRA. 각 테스트 예제마다 서로 다른 시드로 5개의 비디오를 생성하고, 평가 지표 섹션의 세 가지 지표를 사용해 시드 간 평균 점수를 보고합니다.

결론: 100 에폭 학습(8× H100에서 약 2.5시간)은 이 세 가지 지표를 대폭 향상시키기에 충분합니다. LoRA와 DoRA는 유사한 성능으로 수렴하며, DoRA의 추가적인 크기-방향 분해가 아주 낮은 랭크에서 도움이 될 수 있지만 여기서는 필요하지 않음을 확인합니다.

더 큰 랭크(32 대 8)는 지시 수행 능력(instruction following)을 높이지만 기하학적 일관성이나 물리적 타당성을 개선하지는 못합니다. 이는 기하학적 및 물리적 사전 지식이 대개 세계 모델의 고정 가중치에 의해 포착되기 때문이라고 가설합니다. LoRA 어댑터는 도메인 내 로봇 외관과 작업 구조로 분포를 이동시키면 충분히 달성 가능하고, 랭크 8에서도 가능하다고 봅니다.

DoRA vs LoRA를 언제 사용할까: 메모리가 매우 타이트하거나 어댑터 파일 크기가 중요한 경우 LoRA r=8로 시작하세요. 랭크가 낮은 상태에서 LoRA의 학습 불안정성을 관찰한다면 DoRA r=32가 합리적인 대안이며, 크기-방향 분해가 학습 안정화를 도울 수 있습니다.

- Cosmos Cookbook의 단계별 워크플로우, 기술 레시피, 구체적 예제에 대한 자세한 내용은 [Cosmos Cookbook](https://nvda.ws/4qevli8)을 방문하세요.

- 새로운 오픈 코스모스 모델과 데이터셋은 [Hugging Face](https://huggingface.co/nvidia/collections?search=cosmos)와 [GitHub](https://github.com/nvidia-cosmos)에서 확인하거나 [build.nvidia.com](https://nvda.ws/3Yg0Dcx)에서 모델을 시도해 보세요.

- 커뮤니티에 참여하고 우리의 [Cosmos Discord 채널](https://discord.gg/u23rXTHSC9)에 합류하세요.

- 이미 Cosmos를 사용 중이신가요? [기여 방법](https://nvda.ws/4aQcBkk)을 알아보세요.
