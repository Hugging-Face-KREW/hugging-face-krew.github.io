---
layout: post
title: "Hugging Face ❤️ TorchTPU: TPU를 네이티브로"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/hf-torchtpu/thumbnail.png
image: assets/images/blog/posts/2026-10-20-hf-torchtpu/thumbnail.png
authors:
  - user: tengomucho
  - user: Jingya
  - user: pagezyhf
  - user: jeffboudier
  - user: regisss
slug: "hf-torchtpu"
source_url: "https://huggingface.co/blog/hf-torchtpu"
source_published_date: "2026-10-20"
source_published_at: "2026-10-20T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Hugging Face ❤️ TorchTPU: TPUs, Natively](https://huggingface.co/blog/hf-torchtpu)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/hf-torchtpu -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Hugging Face ❤️ TorchTPU: TPU를 네이티브로

<img src="https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/banner.jpg" alt="The Hugging Face mascot and a smiling PyTorch logo cheering over a glowing chip, on a manga-style background." width="624" style="max-width: 100%; height: auto;">

PyTorch를 TPU에서 실행해 보신 적이 있다면, 아마 코드를 `torch_xla`로 포팅했거나 JAX를 배워야 했을 것입니다. 이제 네이티브 PyTorch 백엔드인 [TorchTPU](https://github.com/google-pytorch/torch_tpu)를 사용하면 TPU는 그저 또 하나의 디바이스가 됩니다. 이를 통해 Hugging Face 생태계에서 TPU를 네이티브로 지원할 수 있게 되었습니다. Transformers, Diffusers, Kernels가 TPU에서 네이티브로 실행됩니다. Hugging Face Hub에서 모델을 고르고, `"tpu"`로 보내면 끝입니다.

## 잠깐, TPU가 뭐죠? 🤔 {#section-1}

TPU는 [Tensor Processing Unit](https://cloud.google.com/tpu)를 의미합니다. TPU는 Google의 자체 AI 가속기입니다. 신경망이 대부분의 시간을 사용하는 행렬 연산이라는 한 가지 작업을 위해 만들어진 칩입니다. Google은 약 10년 동안 TPU를 설계해 왔으며, Gemini를 비롯한 자체 모델의 학습과 서빙에 사용하고 있습니다.

TPU는 [Google Cloud](https://docs.cloud.google.com/tpu/docs/intro-to-tpu)에서 사용할 수 있으며, 단일 칩부터 수천 개 칩으로 구성된 팟까지 지원합니다. 학습이나 대규모 서빙을 위해 많은 연산 능력이 필요할 때 특히 강점을 발휘합니다.

항상 걸림돌이었던 것은 소프트웨어였습니다. TPU 스택은 TensorFlow와 JAX를 중심으로 발전해 왔고, PyTorch 사용자는 진입하기 위해 더 많은 수고를 해야 했습니다.

## 예전 방식 {#section-2}

지금까지 TPU에서 PyTorch를 실행하려면 PyTorch/XLA(`torch_xla`)를 거쳐야 했습니다. 작동하긴 했지만 나름의 특성이 있었습니다. 텐서는 지연 실행 방식이었고(여러분이 아니라 텐서가요), 그래서 코드가 생각한 시점에 항상 실행되지는 않았습니다. 작은 변경만으로도 그래프가 다시 컴파일될 수 있었고, 디버깅의 동작을 추적하기도 어려웠습니다.

이를 쉽게 만들기 위해 Optimum TPU를 개발했습니다. 이를 통해 Hugging Face 모델에서 TPU 추론과 학습을 지원했습니다. 이제 TorchTPU가 네이티브 지원을 제공하므로, Transformers와 Diffusers로 Hugging Face 모델을 실행하는 데 Optimum TPU는 더 이상 필요하지 않습니다.

## TorchTPU의 등장 🚀 {#section-3}

[TorchTPU](https://github.com/google-pytorch/torch_tpu)는 PyTorch에 TPU를 새로운 백엔드로 추가합니다. 일반적인 PyTorch 코드를 작성하면 됩니다. TPU에 맞게 조정된 Pallas 커널, 분산 학습 지원, 더 빠른 속도가 필요할 때 사용할 수 있는 `torch.compile` 통합 기능을 제공합니다. TPU VM에 설치하는 데는 명령 하나면 충분합니다.

```shell
pip install torch-tpu --extra-index-url https://download.pytorch.org/whl/cpu
```


추가 인덱스를 사용하면 `torch`의 CPU 빌드를 가져옵니다. 이를 지정하지 않으면 pip가 훨씬 큰 CUDA 빌드를 가져오는데, TPU에서는 필요하지 않습니다.

## 이제 TPU에서 사용할 수 있는 여러분이 가장 좋아하는 라이브러리 {#section-4}

### Transformers

Transformers v5는 모델 정의의 단일 진실 공급원이 되어 어디서나 실행될 수 있도록 한다는 한 가지 아이디어를 중심으로 만들어졌습니다. 이제 "어디서나"에는 TPU도 포함됩니다. TPU에서 모델을 로드하려면 `device_map="tpu"`를 전달하면 됩니다. 그게 전부입니다.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3.5-9B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="tpu", # will load model directly on the TPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("The future of AI is", return_tensors="pt").to("tpu")
out = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```


모델은 기본적으로 eager 모드에서 실행됩니다. 더 나은 성능을 위해 `torch.compile`를 사용할 수도 있고, 분산 설정에서 실행할 수도 있습니다.

### 이미지? 동영상? Diffusers가 TPU에서 그려냅니다 🎨

이제 TPU는 Diffusers에서 일급 디바이스입니다. 프로토타이핑이나 디버깅에는 eager 모드에서 `pipe.to("tpu")`만 있으면 됩니다. 배포 시에는 컴파일된 모드에서 `compile` 호출을 하나 추가하면 됩니다. 파이프라인과 코드는 그대로 유지되며, 별도로 유지 관리해야 할 TPU 포크도 없습니다.

```py
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", dtype=torch.bfloat16
)
pipe.to("tpu")  # eager mode: that's the whole change
# optional: compiled mode
pipe.transformer.compile(backend="tpu", fullgraph=True, dynamic=False)

image = pipe("a corgi astronaut planting a flag on a pizza moon").images[0]
```


첫 번째 호출에서 컴파일이 이루어지고, 이후 같은 이미지 크기로 호출할 때마다 그래프를 재사용합니다.

한 칩에 담기에는 모델이 너무 큰가요? `TensorParallelConfig`를 `from_pretrained`에 전달하면 로드되는 동안 트랜스포머가 칩 전체에 샤딩되므로, 전체 모델이 한 곳에 들어갈 필요가 없습니다. 인기 있는 많은 파이프라인이 기본적으로 이를 지원하며, 그렇지 않은 파이프라인도 `tp_plan`만 추가하면 됩니다. 전체 레시피는 [Diffusers TPU guide](https://huggingface.co/docs/diffusers/main/optimization/tpu)에 있습니다.

다음은 Cloud TPU v6e-4에서 실행한 몇 가지 예시입니다.

<sub><i>TP=4 (각 모델은 네 개의 칩에 분할됨) · bf16 · batch 1</i></sub>

<table>
  <tr>
    <th><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3">MiniMax-H3</a> (video + audio)</th>
    <th colspan="2"><a href="https://huggingface.co/Qwen/Qwen-Image-2.1">Qwen-Image-2.1</a></th>
  </tr>
  <tr>
    <td><video controls src="https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/videos/minimax-h3-hf-trireme.mp4"></video></td>
    <td><img src="https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/qwen-image-21-capybara-1024.jpg" alt="A capybara wearing a wizard hat, reading a book by candlelight, oil painting"></td>
    <td><img src="https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/qwen-image-21-neon-2048.jpg" alt="Neon shop sign reading QWEN IMAGE 2.1"></td>
  </tr>
  <tr>
    <td><sub><i>"The fluffy plush Hugging Face mascot from the picture, a round yellow smiling face with rosy cheeks and two little open hands, stands at the prow of an ancient Greek trireme […] Sound of crashing waves, creaking timber, wind in the rigging, distant seagulls, and a low epic orchestral swell."</i> · 1344×768, 5.2 s, 49 steps · 188 s compiled, 262 s eager</sub></td>
    <td><sub><i>"A capybara wearing a wizard hat, reading a book by candlelight, oil painting"</i> · 1024×1024, 40 steps · 3.5 s compiled, 8.0 s eager</sub></td>
    <td><sub><i>"A neon shop sign that reads 'QWEN IMAGE 2.1', rainy night, reflections on wet pavement"</i> · 2048×2048, 40 steps</sub></td>
  </tr>
</table>

### Kernels

[Kernels Hub](https://huggingface.co/kernels?sort=trending&search=tpu)에는 바로 사용할 수 있는 TPU 커널이 포함되어 있습니다. Kernels 라이브러리로 하나를 로드해 모델의 속도를 높여 보세요.

TPU 커널은 [Pallas](https://docs.jax.dev/en/latest/pallas/tpu/)로 작성되므로 하드웨어를 낮은 수준에서 제어할 수 있습니다. [Tokamax](https://github.com/openxla/tokamax) 라이브러리의 커널을 가져와 사용할 수도 있습니다. 다음은 Tokamax의 ragged-dot 커널을 Mixture-of-Experts 모델에 연결한 예시입니다.

```py
from kernels import get_kernel
from transformers import AutoModelForCausalLM
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

repo_id = "tengomucho/ragged-dot-tpu"
kernel = get_kernel(repo_id, version=1, trust_remote_code=[repo_id])
ALL_EXPERTS_FUNCTIONS.register("tokamax_ragged_dot", kernel.experts_forward)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    experts_implementation="tokamax_ragged_dot",
)
```


배치가 클수록 성능 향상도 커집니다. batch size 64에서 디코딩은 Qwen3-30B-A3B에서 21.5배, Gemma 4 26B-A4B에서 12.8배 더 빠릅니다.

![Decode throughput with the Hub ragged-dot kernel versus the default grouped_mm, for Qwen3-30B-A3B and Gemma 4 26B-A4B](https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/kernel-ragged-dot-speedup.png)
*TPU v6e-8에서의 디코드 처리량, Hub 커널과 기본 구현 비교.*

## 여러분이 직접 테스트하지 않아도 되도록 테스트합니다 🧪 {#section-5}

라이브러리 테스트 스위트를 실행하며 Transformers의 TPU 지원을 지속적으로 개선하고 있습니다. TPU에서 nightly CI를 추가하고 결과를 (Transformers 대시보드)[https://huggingface.co/spaces/transformers-community/transformers-ci-dashboard]에서 확인할 수 있도록 작업 중입니다.

![Transformers CI dashboard with TPU test results next to other accelerators](https://huggingface.co/buckets/huggingface/hf-torchtpu-blog-assets/resolve/images/tpu-ci-dashboard.png)
*다른 가속기 결과와 나란히 표시된 TPU 결과가 포함된 Transformers CI 대시보드. 아직 작업 중입니다!*

아직 모든 테스트가 통과한 것은 아니며, 이를 자연스러운 과정으로 받아들이고 있습니다. TPU에서 실행되는 테스트 중 96.5%가 통과합니다. 발견하는 모든 실패는 작은 재현 스크립트와 TorchTPU 저장소의 이슈가 됩니다. 지금까지 34개의 이슈를 보고했고, 그중 19개는 이미 수정되었습니다. TPU에서 실행하면서 Transformers 자체의 버그도 몇 가지 발견했는데, `StaticCache`와 Qwen3 tensor-parallel 계획의 버그가 그 예입니다. 해당 문제는 upstream에서 수정했습니다.

TPU에서 문제가 발생한다면, 여러분보다 먼저 우리가 발견하고자 합니다.

## 시작하기 {#section-6}

이제 PyTorch 사용자에게 TPU는 더 이상 곁가지 작업이 아닙니다. TorchTPU를 설치하고, Hugging Face Hub에서 모델을 가져와 `"tpu"`로 보내세요.

- [TorchTPU on GitHub](https://github.com/google-pytorch/torch_tpu)
- [TPU kernels on the Hub](https://huggingface.co/kernels?sort=trending&search=tpu)
- [Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs)

여러분이 무엇을 만들어 낼지 정말 기대됩니다 🤗. 문제가 발생하면 이슈를 열어 알려 주세요.
