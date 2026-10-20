---
layout: post
title: "TorchNeuron: AWS Trainium에서 HF 모델을 그 어느 때보다 쉽게 실행하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/hf-torchneuron/thumbnail.png
image: assets/images/blog/posts/2026-10-19-hf-torchneuron/thumbnail.png
authors:
  - user: regisss
  - user: pagezyhf
  - user: michaelbenayoun
  - user: Jingya
  - user: tengomucho
  - user: dacorvo
  - user: jeffboudier
slug: "hf-torchneuron"
source_url: "https://huggingface.co/blog/hf-torchneuron"
source_published_date: "2026-10-19"
source_published_at: "2026-10-19T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [TorchNeuron: running HF models on AWS Trainium is easier than ever](https://huggingface.co/blog/hf-torchneuron)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/hf-torchneuron -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# TorchNeuron: AWS Trainium에서 HF 모델을 그 어느 때보다 쉽게 실행하기

TorchNeuron은 AWS Trainium에서 PyTorch를 위한 새로운 네이티브 백엔드를 제공합니다. Transformers, Diffusers 및 Kernels가 이를 네이티브로 지원하므로, 그 어느 때보다 쉽게 Trainium에서 모델을 실행하고 Hugging Face Hub에서 공유할 수 있습니다.

## TorchNeuron {#section-1}

[TorchNeuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html)은(는) eager mode, `torch.compile` 및 FSDP와 TP 같은 네이티브 분산 API를 지원하는 Trainium용 새로운 백엔드를 제공하는 PyTorch 플러그인입니다.
TorchNeuron에 대해 자세히 알아보려면 Pytorch [blog](https://pytorch.org/blog/introducing-torchneuron-native-pytorch-backend-for-aws-trainium/)을(를) 읽어보세요.

2026년 10월 20일부터 TorchNeuron을 공개 베타로 사용할 수 있습니다. 연구자와 개발자는 `pip install torch-neuronx`을(를) 사용해 이를 설치할 수 있습니다.

## HF 라이브러리의 새로운 백엔드와 네이티브 인터페이스 {#section-2}

지금까지 AWS Trainium 및 Inferentia에서의 Hugging Face 경험은 — 각각 비용 효율적인 학습과 대규모 추론을 위해 설계된 AI 칩입니다 — [optimum-neuron](https://github.com/huggingface/optimum-neuron)을(를) 통해 제공되었습니다. 이 라이브러리는 PyTorch XLA를 기반으로 하는 torch-neuronx 패키지에 의존합니다. 당시에는 자연스러운 선택이었지만, 이는 컴파일러처럼 생각해야 한다는 의미였습니다. 즉, eager 실행 대신 lazy tensor를 사용하고, 정적 shape만 지원하며, 그래프가 끊길 때마다 다시 컴파일해야 했습니다. Optimum Neuron은 이러한 복잡성을 숨기기 위해 존재했으며, 이를 위해 사용자 지정 코드(modeling class, trainer 등)를 유지 관리했습니다.

TorchNeuron은 이를 바꿉니다. PyTorch에 직접 통합되어 `neuron`을(를) 네이티브 디바이스로 노출하며, eager 실행, PyTorch 자체의 분산 primitive(DTensor, FSDP2) 및 `torch.compile`을(를) 제공합니다.

Hugging Face 라이브러리에서 Trainium은 이제 다른 모든 디바이스와 같은 디바이스가 됩니다. Transformers, Diffusers, Accelerate, PEFT 및 TRL을 다른 accelerator에서 사용하는 것과 동일한 class와 script로 Trainium에서 직접 실행할 수 있습니다. 이제부터 Neuron 사용자는 Optimum Neuron을 거치지 않고 이러한 라이브러리를 직접 사용하는 것을 권장합니다.

### Transformers

#### 추론

이제 다양한 모델에서 텍스트 생성을 Transformers에서 네이티브로 지원합니다. TorchNeuron을 사용하면 가장 많이 사용되는 상위 100개 텍스트 생성 모델 아키텍처 중 89%가 기본 제공으로 지원됩니다. Optimum Neuron v0.4.5에서는 46%였습니다. 코드에 `device_map=neuron` 인자만 전달하면 모델이 Neuron 디바이스로 이동합니다.

다음은 tensor parallelism을 사용한 텍스트 생성 예제이며, 이를 통해 Neuron 인스턴스를 최대한 활용할 수 있습니다. Neuron에서는 잦은 재컴파일을 방지하기 위해 generation에 static cache를 사용하는 것을 강력히 권장하므로, 이 예제에서도 static cache를 사용합니다.

```py
import os

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401
from torch.distributed import DeviceMesh
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.distributed import DistributedConfig
from transformers.generation.configuration_utils import CompileConfig

MODEL_ID = "Qwen/Qwen3.8-27B"
MAX_SEQ_LEN = 1024
PREFILL_CHUNK_SIZE = 512

dist.init_process_group(backend="neuron")
world_size = dist.get_world_size()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=world_size),
    device_mesh=DeviceMesh("neuron", list(range(world_size))),
).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "The future of AI is"}],
    add_generation_prompt=True,
    padding=True,
    tokenizer_kwargs={"pad_to_multiple_of": PREFILL_CHUNK_SIZE},
    return_tensors="pt",
)

generation_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=False,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    cache_implementation="static",
    max_cache_len=MAX_SEQ_LEN,
    prefill_chunk_size=PREFILL_CHUNK_SIZE,
    compile_config=CompileConfig(backend="neuron", fullgraph=False, dynamic=False),
)

with torch.no_grad():
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=generation_config,
    )

if dist.get_rank() == 0:
    prompt_len = inputs["input_ids"].shape[1]
    print(tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True))
```


#### 학습

Neuron이 이제 네이티브 PyTorch 디바이스가 되었으므로 Trainium에서의 학습에는 다른 accelerator와 동일한 Hugging Face 라이브러리를 사용합니다. 즉, Transformers, Accelerate, PEFT 및 TRL을 사용합니다. 더 이상 전용 trainer나 Neuron 전용 model code fork가 필요하지 않습니다. 표준 TRL fine-tuning script는 몇 가지 Neuron 전용 설정만으로 Trainium 인스턴스에서 실행됩니다.

다음은 TRL의 `SFTTrainer`을(를) 기반으로 구축한 supervised fine-tuning script입니다. 이전에 GPU에서 TRL로 모델을 fine-tuning한 적이 있다면 익숙하게 느껴질 것입니다.

```py
import os

import torch
from datasets import load_dataset
from transformers.distributed import DistributedConfig

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser


def main(script_args, training_args, model_args):
    # The Neuron backend can't compile Dynamo's automatic dynamic shapes yet (https://github.com/huggingface/trl/issues/7660)
    torch._dynamo.config.automatic_dynamic_shapes = False

    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": training_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
        "device_map": None,
    }
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # The model is sharded on the TP x FSDP grid as its weights are loaded
        training_args.model_init_kwargs["distributed_config"] = DistributedConfig(
            tp_size=int(os.environ.get("TP_SIZE", "1")),
            fsdp_size=int(os.environ.get("FSDP_SIZE", "1")),
        )

    # Prepare dataset
    train_dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    # UltraChat also has a string `prompt` column, which makes TRL treat it as a prompt-completion dataset.
    train_dataset = train_dataset.select_columns(["messages"])

    # Training
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.end()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
```


병렬 처리는 Transformers에서 네이티브로 처리됩니다. tensor-parallel (TP) 및 FSDP 크기가 포함된 `DistributedConfig`을(를) `model_init_kwargs`에 넣으면, `SFTTrainer`이(가) 이를 `from_pretrained`에 전달하고 모델은 weight를 로드하면서 shard됩니다. Trainer는 모델이 이미 분산되어 있음을 인식하므로 다시 래핑하지 않습니다. `--fsdp` 플래그나 Accelerate FSDP config가 필요하지 않습니다.

이 script에는 Trainium에 종속된 내용이 없습니다. 동일한 code가 어떤 hardware에서도 실행됩니다. 다만 Trainium에서 실행할 때 알아둘 만한 몇 가지 사항이 있습니다.

* **정적 shape.** 이 script는 `torch.compile`에서 automatic dynamic shapes를 비활성화하고 wrapped packing을 사용해 최대 2048개의 token으로 이루어진 sequence를 생성합니다. TRL ≥ 1.16에서는 Neuron의 batch가 128 token의 배수로 padding되므로 input shape 수가 제한됩니다. 컴파일된 graph는 동일한 shape를 사용하는 이후 step에서 재사용됩니다.
* **`torch.compile` 활성화.** `--torch_compile true`을(를) 전달하는 것만으로 충분합니다. Trainium 인스턴스에서는 `torch.compile`이(가) 디바이스를 감지하고 Neuron backend를 자동으로 선택하므로 추가로 구성할 항목이 없습니다.
* **Async weight loading 비활성화.** Transformers에 내장된 asynchronous weight loading은 Neuron에서 지원되지 않으므로, 현재는 `HF_DEACTIVATE_ASYNC_LOAD=1`을(를) 사용해 비활성화합니다.

아래 command는 `torchrun`을(를) 사용해 학습을 시작하며, NeuronCore당 하나의 process를 시작합니다. `TP_SIZE` 및 `FSDP_SIZE`는 parallelism layout을 설정하고, 두 값의 곱이 process 수입니다. 여기서는 4 x 8 = 32로, `--nproc_per_node`와 일치합니다. 32개의 process는 4 x 8 grid를 구성합니다. Tensor parallelism은 TP group의 4개 core에 각 layer를 분할하고, FSDP는 8개 TP group에 parameter, gradient 및 optimizer state를 shard합니다.

```sh
TP_SIZE=4 FSDP_SIZE=8 HF_DEACTIVATE_ASYNC_LOAD=1 uv run torchrun --nproc_per_node=32 sft_neuron_example.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --dataset_train_split train_sft \
    --output_dir Qwen3-8B-sft-tp4-fsdp8 \
    --dtype bfloat16 \
    --max_length 2048 \
    --packing true \
    --packing_strategy wrapped \
    --learning_rate 2e-5 \
    --warmup_steps 0.03 \
    --num_train_epochs 1 \
    --logging_steps 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_only_model true \
    --torch_compile true
```


8개의 Trainium2 chip(32 NeuronCore)에서 실행했으며, 이는 `trn2.48xlarge` instance의 절반입니다.

학습은 끝까지 실행되며, `trainer.save_model`은(는) 표준 Transformers checkpoint를 작성합니다. checkpoint의 어떤 부분도 Neuron에 종속되지 않으므로, 그대로 Hugging Face Hub에 push한 다음 다른 hardware에서 로드할 수 있습니다.

### Diffusers

#### 추론

Diffusers library의 많은 pipeline이 eager mode와 compiled mode에서 TorchNeuron과 호환됩니다. FLUX.1-dev, FLUX.2, Qwen-Image, Qwen-Image-2.1 및 H3도 tensor parallelism과 호환되므로 Neuron device를 최대한 활용할 수 있습니다. 다음은 TP를 사용하는 방법을 보여주는 예제입니다.

```py
import torch, torch.distributed as dist, torch_neuronx # TP
from torch.distributed.device_mesh import DeviceMesh # TP
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, TensorParallelConfig

# Distributed devices initialization
dist.init_process_group(backend="neuron") # TP
dev = torch.neuron.current_device()
mesh = DeviceMesh("neuron", list(range(8))) # TP=8

# Pipeline initialization
model = "Qwen/Qwen-Image-2.1"
transformer = QwenImage21Transformer2DModel.from_pretrained(
    model,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    parallel_config=TensorParallelConfig(mesh=mesh), # TP
)
pipe = QwenImage21Pipeline.from_pretrained(
    model,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.vae.to(dev)
pipe.transformer.compile_repeated_blocks(backend="neuron", fullgraph=True, dynamic=False)

# Image generation
emb, mask, _ = pipe.encode_prompt(prompt="A capybara wearing a wizard hat, oil painting", device="cpu")  # text encoder stays on CPU
image = pipe(
    prompt_embeds=emb.to(dev),
    prompt_embeds_mask=mask.to(dev) if mask is not None else None,
    height=1024,
    width=1024,
    num_inference_steps=40,
    generator=torch.Generator("cpu").manual_seed(0), # same noise on every rank
).images[0]
if dist.get_rank() == 0:
    image.save("out.png")
```


다음과 같이 이 code를 실행할 수 있습니다: `NEURON_LOGICAL_NC_CONFIG=2 torchrun --nproc_per_node=8 run.py`.

앞으로 점점 더 많은 pipleine이 TP와 호환될 예정입니다. 원하는 pipeline이 호환되지 않는 경우에도 [this guide](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#writing-a-tensor-parallelism-plan)을(를) 따라 지원을 추가할 수 있습니다.

### Kernels

점점 더 많은 Neuron kernel이 모델을 가속하기 위해 Hub에서 제공될 예정이므로 [kernels from the Neuron team](https://huggingface.co/aws-neuron/kernels)을(를) 꼭 확인해 보세요.
Neuron kernel은 [Neuron Kernel Interface](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html) (NKI)을(를) 사용해 작성할 수 있으며, 이를 통해 operation을 최적화할 수 있는 더 낮은 수준의 제어를 제공합니다. 이를 통해 사용자는 자체 custom operation을 실행하고 Hub에서 공유할 수 있습니다.

예를 들어 [this kernel](https://huggingface.co/kernels/aws-neuron/vit-encoder-neuron-kernels)을(를) 사용하면 vision transformer를 가속하여 Trainium에서 최대 x2.4의 속도 향상을 얻을 수 있습니다!

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hf-torchneuron/kernel.png)

## 감사의 말 {#section-3}

TorchNeuron을 개발하고 HF 통합을 최대한 개선하기 위해 저희의 피드백에 신속하게 대응해 준 AWS 팀에 특별히 감사드립니다.
