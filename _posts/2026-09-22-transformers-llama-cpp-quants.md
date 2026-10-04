---
layout: post
title: "이제 Transformers에서 llama.cpp 양자화 모델을 실행할 수 있습니다"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/transformers_llama_cpp_quants/thumbnail.png
image: assets/images/blog/posts/2026-09-22-transformers-llama-cpp-quants/thumbnail.png
authors:
  - user: marcsun13
  - user: ArthurZ
  - user: lysandre
slug: "transformers-llama-cpp-quants"
source_url: "https://huggingface.co/blog/transformers-llama-cpp-quants"
source_published_date: "2026-09-22"
source_published_at: "2026-09-22T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Transformers now runs llama.cpp quants](https://huggingface.co/blog/transformers-llama-cpp-quants)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/transformers-llama-cpp-quants -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 이제 Transformers에서 llama.cpp 양자화 모델을 실행할 수 있습니다

**GGUF 모델을 Transformers에서 효율적으로 실행할 수 있도록 지원을 추가하고 있습니다.** 따라서 노트북 메모리 크기에 맞게 조정된 체크포인트를 익숙한 Transformers API를 통해 사용할 수 있습니다. Hugging Face Hub에서 GGUF를 선택하고 `from_pretrained`으로 로드한 다음, 자신의 머신에서 생성을 시작해 보세요.

<video controls width="100%">
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformers-llama-cpp-quants/transformers-gguf.mp4" type="video/mp4">
</video>

노트북에서 AI 모델을 실행하는 일이 훨씬 쉬워졌으며, [llama.cpp](https://github.com/ggml-org/llama.cpp)가 그 과정에서 큰 역할을 해 왔습니다. [llama.cpp](https://github.com/ggml-org/llama.cpp)의 추론 엔진은 Ollama, LM Studio, Jan과 같은 로컬 AI 도구를 구동합니다. 또한 [MLX](https://github.com/ml-explore/mlx)와 같은 프로젝트와 함께 일상적인 사용에서 로컬 추론을 실용적인 선택지로 만드는 데 기여했습니다.

<p><em>A recent example of what local AI can feel like:</em></p>
<blockquote class="twitter-tweet" data-conversation="none"><p lang="en" dir="ltr">This is where we are right now. And i’m not gonna lie it feels pretty magical 🧙‍♀️<br><br>Qwen3.6 27B running inside of Pi coding agent via Llama.cpp on the MacBook Pro<br><br>For non-trivial tasks on the <a href="https://x.com/huggingface?ref_src=twsrc%5Etfw">@huggingface</a> codebases, this feels very, very close to hitting the latest Opus in Claude… <a href="https://t.co/lsIxLoUneU">pic.twitter.com/lsIxLoUneU</a></p>&mdash; Julien Chaumond (@julien_c) <a href="https://x.com/julien_c/status/2047647522173104145?ref_src=twsrc%5Etfw">April 24, 2026</a></blockquote>
<script async src="https://platform.x.com/widgets.js" charset="utf-8"></script>

**GGUF**는 llama.cpp 팀이 개발한 로컬 추론용으로 널리 사용되는 형식입니다. 이 팀은 [ggml-org on the Hub](https://huggingface.co/ggml-org)에서 양자화된 체크포인트도 공유합니다. [Unsloth](https://huggingface.co/unsloth), [LM Studio Community](https://huggingface.co/lmstudio-community), [bartowski](https://huggingface.co/bartowski)와 같은 게시자는 다양한 양자화 수준의 바로 사용할 수 있는 GGUF 체크포인트도 제공하므로, 사용자는 자신의 머신에 맞는 버전을 선택할 수 있습니다. GGUF 모델은 수백만 회 다운로드되었습니다.

이러한 모델을 Transformers로도 더 쉽게 로컬에서 실행할 수 있도록 하려 합니다. 모델을 실행하기 불편하다면 호환성만으로는 충분하지 않습니다. 성능을 llama.cpp에 가깝게 만들기 위해 [`kernels`](https://huggingface.co/docs/kernels/index) 라이브러리를 통해 기반이 되는 ggml 커널을 재사용하고, `generate`의 오버헤드를 줄이고 있습니다. 현재는 Apple Silicon에서의 로컬 추론에 초점을 맞추고 있으며, Qwen3.5 아키텍처부터 시작합니다.

## GGUF 파일 형식이란? {#section-1}

[GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)는 모델 가중치와 메타데이터를 토크나이저 정보 및 선택적 채팅 템플릿과 함께 하나의 파일에 패키징합니다. 다양한 양자화 수준을 지원하므로, 일부 정밀도를 줄이는 대신 메모리 사용량을 줄일 수 있습니다. `Q4_K_M`과 같은 변형은 텐서 정밀도를 혼합하여, 대부분의 가중치는 4비트로 유지하면서 민감한 텐서는 더 높은 정밀도로 유지합니다.

양자화에 따라 [Unsloth's Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main)의 파일 크기가 어떻게 달라지는지 살펴보겠습니다.

| GGUF 변형 | 파일 크기 | 절충점 |
|---|---:|---|
| `BF16` | 8.42 GB | 양자화하지 않은 기준 |
| `Q6_K` | 3.53 GB | 더 작은 변형보다 높은 정밀도 |
| `Q5_K_M` | 3.14 GB | 크기와 정밀도 사이의 절충안 |
| `Q4_K_M` | 2.74 GB | 로컬 추론을 위한 실용적인 시작점 |

`Q4_K_M`부터 시작한 다음, 메모리가 더 충분하다면 `Q5_K_M` 또는 `Q6_K`를 시도해 보는 것을 권장합니다. 더 공격적인 양자화는 더 큰 모델을 적합시키는 데 도움이 될 수 있지만, 품질의 절충점은 모델과 작업에 따라 달라집니다. 실제로 모델이 수행하게 하려는 작업을 기준으로 평가하세요. [Hub's GGUF documentation](https://huggingface.co/docs/hub/gguf#quantization-types)에는 사용 가능한 양자화 유형이 설명되어 있습니다.

## Transformers로 GGUF 로드하기 {#section-2}

시작하려면 다음이 필요합니다.

- **Apple Silicon Mac**.
- **공개된 [ggml-quantization kernel builds](https://huggingface.co/kernels/ggml-org/ggml-quantization)에서 지원하는 PyTorch 버전**. 일반적으로 최신 PyTorch 릴리스 두 개입니다.
- **최신 버전의 Transformers(현재는 다음 릴리스 전까지 main)와 호환되는 `kernels` 버전**.

```bash
pip install -U "git+https://github.com/huggingface/transformers.git" kernels
```


GGUF 모델을 로드하려면 Hub `model_id`와 파일명을 `gguf_file`로 전달하여 `from_pretrained`에 지정합니다.

추가 설정은 필요하지 않습니다. 가중치가 Metal에 패킹된 상태로 유지되면 Transformers가 호환되는 ggml/Metal 레이어 커널을 자동으로 로드하고 `ggml-org/ggml-attn`을 어텐션 구현으로 사용합니다. 해당 커널을 가져올 수 없으면 모델은 경고와 함께 `"sdpa"`로 대체되며, `attn_implementation="sdpa"`을 명시적으로 전달하여 언제든 `"sdpa"`을 강제로 사용할 수 있습니다. 더 많은 로딩 옵션은 [GGUF documentation](https://huggingface.co/docs/transformers/main/en/quantization/gguf)을 참조하세요.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/Qwen3.5-4B-GGUF"
filename = "Qwen3.5-4B-Q4_K_M.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    gguf_file=filename
)
```


이것이 GGUF와 관련된 유일한 단계입니다. 그 이후에는 모두 표준 Transformers API를 사용합니다.

```python
messages = [{"role": "user", "content": "Explain why the sky is blue in a few sentences."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=256)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


> [!WARNING]
> 호환되는 양자화 커널이 없으면 로더가 모델을 역양자화하는 방식으로 대체하며, 더 많은 메모리를 사용합니다.

## 원하는 인터페이스로 GGUF 제공하기 {#section-3}

동일한 체크포인트를 [`transformers serve`](https://huggingface.co/docs/transformers/main/en/serve-cli/serving)에서도 사용할 수 있으며, [`transformers serve`](https://huggingface.co/docs/transformers/main/en/serve-cli/serving)는 OpenAI 호환 API를 제공합니다.

```bash
pip install -U "transformers[serving] @ git+https://github.com/huggingface/transformers.git" kernels

transformers serve "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf"
```


모델 인자는 `<model_id>:<filename>.gguf`을 사용합니다. 콜론 앞에는 Hub 저장소(`unsloth/Qwen3.5-4B-GGUF`)가 오고, 뒤에는 로드할 파일(`Qwen3.5-4B-Q4_K_M.gguf`)이 옵니다. 이를 통해 여러 양자화 버전이 포함될 수 있는 저장소에서 특정 양자화를 선택합니다.

채팅 템플릿이 thinking을 지원하는 모델에서는 `--reasoning off`을 추가하여 이를 건너뛰거나 `--reasoning on`을 추가하여 활성화할 수 있습니다. 기본값인 `--reasoning auto`는 채팅 템플릿의 기본 설정을 따릅니다. 자세한 내용은 [reasoning options](https://huggingface.co/docs/transformers/main/en/serve-cli/serving#enable-reasoning-on-the-server)을 참조하세요.

[Jan](https://www.jan.ai/docs/desktop/remote-models/custom-endpoint) 또는 [Pi](https://pi.dev)과 같은 클라이언트를 연결하려면 다음 설정으로 사용자 지정 OpenAI 호환 provider를 추가하면 됩니다.

| 설정 | 값 |
|---|---|
| Base URL | `http://localhost:8000/v1` |
| Model ID | `unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-Q4_K_M.gguf` |

Transformers는 Mac에서 모델을 실행하고, 클라이언트는 대화 인터페이스를 제공합니다. 이 API를 지원하는 다른 클라이언트도 동일한 엔드포인트를 사용할 수 있습니다.

## llama.cpp와 비교한 벤치마크 {#section-4}

로컬 추론 성능의 기준은 llama.cpp입니다. 아래 비교는 소형 dense 모델, 대형 dense 모델, mixture-of-experts 모델 등 세 가지 GGUF 체크포인트에 초점을 맞춥니다.

llama.cpp 열의 수치는 [`llama-bench`](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) 도구(build `5f55650a7`, 릴리스 b10200, ggml 0.18.0의 Metal 백엔드)를 사용해 `llama-bench -m <file> -p 0 -n 128 -r 3`로 실행한 결과입니다. 이 도구는 `tg128`을 보고합니다. 즉, 프롬프트 처리를 제외하고 128개의 디코딩된 토큰에 대한 토큰 생성 속도를 세 번 반복하여 평균낸 값입니다. Transformers 열은 12토큰 프롬프트에서 동일한 128개 토큰을 생성하는 `generate`의 결과로, 워밍업된 세 번의 실행 중 가장 좋은 값이며 prefill을 포함합니다.

MacBook Pro M2 Max, 통합 메모리 32 GB, macOS 26.6, PyTorch 2.12.1, kernels 0.17.0 환경에서 전원을 연결한 상태로 측정했습니다.

<details>
  <summary>The benchmark script</summary>

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id, filename = "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"

model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
inputs = tokenizer("The capital of France is Paris. The capital of Germany is", return_tensors="pt")
inputs = inputs.to(model.device)


with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=8, min_new_tokens=8, do_sample=False)  # warm up
    torch.mps.synchronize()
    for _ in range(3):
        time.sleep(90)  # let the machine cool: back-to-back runs decay by 10% or more
        start = time.perf_counter()
        model.generate(**inputs, max_new_tokens=128, min_new_tokens=128, do_sample=False)
        torch.mps.synchronize()
        print(f"{128 / (time.perf_counter() - start):.1f} tok/s")
```


다른 열의 경우:

```bash
llama-bench -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M -p 0 -n 128 -r 3
```


</details>

![GGUF generation throughput compared with llama.cpp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformers-llama-cpp-quants/benchmark-comparison.svg)

세 체크포인트 모두에서 Transformers는 llama.cpp에 근접한 성능을 보입니다. 차트에는 위에서 설명한 동일한 측정값이 사용되었습니다. 다만 Transformers 측정에는 prefill이 포함되고 `llama-bench`은 디코드 전용 처리량을 보고하므로, 벤치마크 조건이 동일하다는 의미는 아닙니다.

## Transformers와 llama.cpp {#section-5}

[GGML and llama.cpp joined Hugging Face](https://huggingface.co/blog/ggml-joins-hf)에서 두 프로젝트의 상호 보완적인 역할을 설명했습니다. llama.cpp는 로컬 추론을 위한 기반을 제공하고, Transformers는 모델 정의를 위한 기반을 제공합니다. GGUF 지원은 이 두 가지를 더욱 가깝게 연결합니다.

**효율적인 로컬 추론이 우선순위라면 llama.cpp를 계속 권장합니다.** llama.cpp의 전용 런타임, 메모리 관리, 폭넓은 하드웨어 지원은 이러한 목표를 중심으로 구축되었습니다. 이번 통합은 개발자가 동일한 GGUF 체크포인트를 Transformers 내부에서 편리하게 사용할 수 있는 방법을 제공합니다.

- **Python과 PyTorch에서 GGUF를 실험합니다.** hook으로 중간 활성화를 검사하고, 모델의 forward pass를 수정하거나, 익숙한 PyTorch 도구를 사용해 사용자 지정 레이어의 프로토타입을 만들 수 있습니다.
- **GGUF 모델을 평가합니다.** 기존 Transformers 평가 워크플로를 사용해 양자화된 체크포인트의 품질을 측정할 수 있습니다.
- **GGUF 변환을 검증합니다.** 개발자 입장에서 원본 체크포인트와 GGUF 변환본을 Transformers에서 로드하면 양자화 오류를 고려하면서 가중치가 올바르게 변환되었는지 더 쉽게 확인할 수 있습니다.
- **새로운 디코딩 아이디어를 시도합니다.** `generate`과 함께 사용자 지정 logits processor와 stopping criteria를 사용하거나, Python으로 직접 생성 루프를 작성할 수 있습니다.
- **GGUF 체크포인트에서 파인튜닝합니다.** 가중치를 역양자화한 다음 표준 Transformers 학습 워크플로를 계속 진행할 수 있습니다.

마지막 경우에는 `GgufConfig(dequantize=True)`을 사용하세요.

```python
import torch
from transformers import AutoModelForCausalLM, GgufConfig

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3.5-4B-GGUF",
    gguf_file="Qwen3.5-4B-Q4_K_M.gguf",
    quantization_config=GgufConfig(dequantize=True),
    dtype=torch.bfloat16,
)
```


## GGUF를 넘어: 더 많은 모델을 위한 ggml 커널 {#section-6}

**더 큰 기회는 llama.cpp가 지원하지 않는 모델에 ggml의 성능을 제공하는 것입니다.**

Transformers는 이미 이러한 아키텍처의 PyTorch 구현을 제공합니다. PyTorch에서 ggml 커널과 양자화 방식을 사용할 수 있다면, 전체 모델을 먼저 llama.cpp에 구현하지 않고도 지원되는 연산을 가속하는 방향으로 나아갈 수 있습니다. 이는 새로운 아키텍처, 연구 모델, 그리고 전용 llama.cpp 구현을 제공받지 못할 수도 있는 사용자 지정 변형에 특히 유용합니다.

이러한 기회는 GGUF 형식 자체를 넘어섭니다. 커널은 텐서에서 작동하며, 모델 전체가 GGUF 파일에서 비롯되어야 할 필요는 없습니다. 동일한 구성 요소를 다른 Transformers 모델과 로딩 워크플로에 통합할 수 있습니다. 이는 다른 모달리티로 나아가는 길도 열어 줍니다. 컴퓨터 비전 모델, 오디오 모델, 멀티모달 모델은 llama.cpp에 전체 구현을 먼저 추가하지 않고도 호환되는 어텐션, 정규화, 행렬 곱셈 커널을 재사용할 수 있습니다. 각 아키텍처에는 여전히 통합과 검증이 필요하며, 여기의 초기 GGUF 예제는 텍스트 생성에 해당합니다.

## Python과 PyTorch를 사용한 빠른 로컬 추론 {#section-7}

모델과 생성 루프를 Python에 유지하면서 어느 정도까지 성능을 낼 수 있는지도 보여 드리고자 했습니다. **적절한 커널과 효율적인 생성 루프를 사용하면 Python과 PyTorch로도 강력한 로컬 추론 성능을 낼 수 있습니다.** 커널은 무거운 연산을 처리하고, 생성 루프는 불필요한 동기화를 피하여 GPU가 계속 작업하도록 합니다.

우리의 목표는 `torch.compile`을 요구하지 않으면서 eager execution을 빠르게 만드는 것이었습니다. 대화형 사용에서는 컴파일로 인한 일시 중지나 입력 형태가 바뀔 때의 재컴파일 없이 빠르게 시작하고 토큰이 꾸준히 생성되기를 원했습니다. 이 작업의 두 가지 핵심 요소는 커널과 `generate` 자체입니다.

### ggml의 Metal 커널 재사용

커널은 GPU에서 연산을 수행하는 작은 프로그램입니다. PyTorch는 범용 구현을 제공하지만, 특화된 커널은 작업량을 줄이거나 여러 연산을 결합하거나 저장된 형식 그대로 양자화된 가중치를 직접 읽을 수 있습니다.

`kernels` 라이브러리를 사용하면 ggml의 Metal 커널과 호환되는 빌드를 Hub에 배포하고 Transformers에서 호출할 수 있습니다. 이를 통해 별도의 추론 런타임으로 모델을 대체하지 않고 ggml의 작업을 PyTorch 모델에 가져올 수 있습니다.

| 커널 | 기능 |
|---|---|
| [`ggml-quantization`](https://huggingface.co/kernels/ggml-org/ggml-quantization) | 선택된 전문가를 포함하여 행렬 연산에 필요한 패킹된 양자화 가중치를 읽습니다. 각 디코드 연산 전에 전체 가중치 행렬을 확장하지 않습니다. |
| [`ggml-norm`](https://huggingface.co/kernels/ggml-org/ggml-norm) | Qwen3.5와 Qwen3.8에서 사용하는 zero-centered RMSNorm을 포함하여 정규화 연산을 융합합니다. |
| [`ggml-attn`](https://huggingface.co/kernels/ggml-org/ggml-attn) | 프롬프트 처리와 토큰 디코딩을 위해 ggml의 Metal flash attention을 제공합니다. |
| [`ggml-gated-delta-net`](https://huggingface.co/kernels/ggml-org/ggml-gated-delta-net) | Qwen3.5 및 Qwen3.8 하이브리드 아키텍처의 linear-attention 레이어에서 사용되는 gated delta network를 가속합니다. |
| [`topk`](https://huggingface.co/kernels/transformers-community/topk) | MoE 모델에서 각 토큰에 사용할 전문가를 선택하며, softmax와 top-k 라우팅을 결합합니다. 자체적으로 구현한 Metal입니다. |

처음 네 패키지는 ggml의 커널을 기반으로 하며, top-k 커널은 MoE 라우팅에서 별도의 병목을 해결합니다. 이들이 함께 작동하여 생성되는 각 토큰에 필요한 GPU 작업량을 줄입니다.

레이어 커널의 기여도를 보여 주기 위해 동일한 패킹된 GGUF 체크포인트를 커널 사용 여부에 따라 비교합니다. 양자화 커널은 두 구성에서 모두 활성화된 상태로 유지합니다. 이를 비활성화하면 가중치 표현 방식도 달라져 다른 절충점을 측정하게 되기 때문입니다.

![Throughput improvement from the layer kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformers-llama-cpp-quants/layer-kernel-benchmark.svg)

### CPU와 GPU의 협력 유지

GPU에 수행할 작업이 있을 때만 더 빠른 커널이 도움이 됩니다. 생성 중에는 CPU가 GPU 연산을 예약하고 다음 토큰을 생성하는 루프를 제어합니다. GPU에서 결과를 다시 읽으면 대기 중인 연산이 완료될 때까지 CPU가 기다려야 할 수 있습니다. 토큰마다 작은 대기가 반복되더라도 처리량이 눈에 띄게 감소할 수 있습니다.

`generate`에서는 두 가지 변경을 통해 이 문제를 해결하며, 그 결과 GGUF 파일을 실행할 때뿐 아니라 모든 Transformers 모델에서 개선이 이루어집니다.

- **[Drop an unnecessary attention mask early (#48814)](https://github.com/huggingface/transformers/pull/48814).** 지원되는 decoder-only 입력에 패딩이 없으면, 생성 시작 시 모든 값이 1인 padding mask를 제거할 수 있습니다. 이후의 어텐션 코드는 건너뛸 수 있는지 판단하기 위해 해당 마스크를 반복해서 검사할 필요가 없습니다. Causal attention은 계속 유지됩니다.
- **[Defer the stopping check (#47975)](https://github.com/huggingface/transformers/pull/47975).** 지원되는 경로에서 `generate`는 stopping 결정을 비동기적으로 복사하고 다음 단계에서 이를 사용합니다. GPU가 실행되는 동안 CPU는 계속 작업을 예약할 수 있습니다. 스트리밍 토큰도 동일한 방식을 사용하며, stopping 조건을 지난 추가 단계는 결과에서 제거됩니다.

이러한 변경은 모델을 둘러싼 생성 루프를 개선하므로 GGUF를 넘어 활용할 수 있습니다. 또한 커널 작업을 보완합니다. 커널은 연산 비용을 줄이고, 동기화 지점을 줄이면 CPU 스케줄링과 GPU 실행이 겹쳐 진행될 수 있습니다.

![Throughput improvement from the generation loop changes](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/transformers-llama-cpp-quants/generation-loop-benchmark.svg)

이 측정에서는 모든 레이어 커널을 활성화했으며, 막대는 생성 루프에 대한 변경 사항만 분리하여 보여 줍니다.

## 현재 제한 사항과 다음 단계 {#section-8}

초기 목표는 Apple Silicon에서 하나의 대화형 대화를 실행하는 것입니다. 다음과 같은 몇 가지 한계를 염두에 두어야 합니다.

- **현재 패킹된 추론 경로는 MPS 전용입니다.** 역양자화를 통한 GGUF 가져오기는 별도의 옵션으로 유지됩니다. 파일 형식을 지원한다고 해서 모든 장치에서 패킹된 커널을 사용할 수 있다는 의미는 아닙니다.
- **패딩과 배칭은 아직 개선이 필요합니다.** 패딩되지 않은 입력은 위에서 설명한 마스크 최적화의 이점을 얻습니다. 패딩된 배치는 동일한 지름길을 사용할 수 없으므로 성능이 낮아질 수 있습니다. 이 작업을 MPS의 `generate_batch`까지 확장하고자 합니다.
- **지원되는 아키텍처 범위가 제한적입니다.** 현재 패킹된 로더는 호환되는 Qwen3.8 체크포인트를 포함하여 Qwen3.5 dense 및 MoE 아키텍처를 지원합니다. 다른 아키텍처에 대한 지원을 추가하는 일은 비교적 간단하며, 지원 범위를 점진적으로 확대할 예정입니다.

Transformers에서 사용하고 싶은 GGUF 모델이 있다면 체크포인트와 사용 사례를 함께 [open an issue](https://github.com/huggingface/transformers/issues)해 주세요. 이를 통해 사람들이 로컬에서 실행하는 모델에 대한 지원의 우선순위를 정하는 데 도움이 됩니다.

## 감사의 말 {#section-9}

이번 작업을 시작하고 제 모든 PR을 검토해 주신 [Arthur Zucker](https://huggingface.co/ArthurZ)님, 그리고 `generate` PR을 맡아 주신 [Cyril Vallez](https://huggingface.co/cyrilvallez)님께 감사드립니다. 커널 통합을 도와주신 [Sayak Paul](https://huggingface.co/sayakpaul), [llama.cpp team](https://github.com/ggml-org/llama.cpp), Bertrand Chevalier께도 감사드립니다. 이 블로그 게시물을 검토해 주신 [Aritra Roy Gosthipaty](https://huggingface.co/ariG23498)님과 [Pedro Cuenca](https://huggingface.co/pcuenq)님, 프로젝트를 총괄해 주신 [Lysandre Debut](https://huggingface.co/lysandre)님께도 감사드립니다.
