---
layout: post
title: "파이토치의 프로파일링(3부): 어텐션이 전부다"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-07-10-torch-attention-profile/thumbnail.png
authors:
  - user: ariG23498
  - user: sergiopaniego
  - user: sayakpaul
  - user: ror
slug: "torch-attention-profile"
source_url: "https://huggingface.co/blog/torch-attention-profile"
source_published_date: "2026-07-10"
source_published_at: "2026-07-10T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Profiling in PyTorch (Part 3): Attention is all you profile](https://huggingface.co/blog/torch-attention-profile)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/torch-attention-profile -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 파이토치의 프로파일링(3부): 어텐션이 전부다

![Thumbnail of the blog post](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/profile-3-thumbnail.png)

<div style="
  border: 1px solid #1f5c48;
  border-radius: 6px;
  padding: 1.5rem 2rem;
  background: #17211f;
  color: #ddd;
  font-size: 1.1rem;
  line-height: 1.7;
">

  <p>
    This is the third post of <bold>Profiling in PyTorch</bold>, a series where we slowly build the skill of reading profiler traces and use it to drive optimization:
  </p>

  <ol>
    <li>
      <a href="https://huggingface.co/blog/torch-profiler" style="color: #10b981;">
        Profiling in PyTorch (Part 1): A Beginner's Guide to torch.profiler
      </a>
    </li>
    <li>
      <a href="https://huggingface.co/blog/torch-mlp-fusion" style="color: #10b981;">
        Profiling in PyTorch (Part 2): From nn.Linear to a Fused MLP
      </a>
    </li>
    <li>
      <a href="https://huggingface.co/blog/torch-attention-profile" style="color: #10b981;">
        Profiling in PyTorch (Part 3): Attention is all you profile
      </a>
      <em style="color: #aaa;">(current)</em>
    </li>
  </ol>

</div>

시리즈 "Profiling in PyTorch"는 프로파일러 트레이스와 표를 읽는 데 익숙해지게 만드는 것을 목표로 합니다. [Part 1](https://huggingface.co/blog/torch-profiler)에서는 덧셈과 곱셈 같은 기본 수학 연산을 프로파일링했습니다. 프로파일러 표가 핫스팟을 드러내는 방식과 프로파일러 트레이스가 알고리즘이 시간에 따라 실행되는 순서를 어떻게 보여주는지 보았습니다.

[Part 2](https://huggingface.co/blog/torch-mlp-fusion) 안에서 위의 덧셈과 곱셈을 torch 선형 계층으로 포장했고, 그 위에 여러 개의 선형 계층을 차례로 쌓아(다층 퍼셉트론) 그것을 프로파일링했습니다. 그 과정에서 융합된 커널과 수작업으로 튜닝된 커널도 함께 프로파일링했습니다.

트랜스포머 아키텍처의 관점에서, 프로파일링의 다음 논리적 단계는 또 다른 기본 알고리즘인 어텐션입니다. 어텐션은 이차 시간 복잡도로 악명 높지만, 이를 완화하고 빠르게 만드는 똑똑한 트릭이 많이 존재합니다. 여기서의 목표는 모든 트릭을 자세히 다루는 것이 아닙니다. 대신 각 트릭이 프로파일러 아래에서 어떻게 다르게 보이는지 보는 것입니다.

> [!NOTE]
> 이 블로그 포스트의 스크립트는 여기에서 실행됩니다: [`04_a_naive_attention.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/04_a_naive_attention.py), [`04_b_inplace_ops_attention.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/04_b_inplace_ops_attention.py), [`04_c_sdpa_attention.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/04_c_sdpa_attention.py), 그리고 [`04_d_kernels_attention.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/04_d_kernels_attention.py). 이전과 마찬가지로 읽으면서 코드를 따라가려면 별도의 탭에서 여는 것이 도움이 됩니다. 이 스크립트를 실행하기 위해 `NVIDIA A100-SXM4-80GB` GPU를 사용합니다. 허깅페이스 인프라에서 GPU를 설정하고 [Dev Mode with Spaces](https://huggingface.co/docs/hub/spaces-dev-mode)를 사용해 스크립트를 실험하는 것은 정말 쉽습니다. 또한 [Hugging Face Jobs pipeline](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)로도 스크립트를 실행할 수 있습니다.

## 나이브 어텐션 {#section-1}

어텐션은 쿼리(`q`), 키(`k`), 값(`v`)으로 작동합니다. 이들 간의 상호 작용은 간단한 일련의 단계로 작성할 수 있습니다:

1. 어텐션 스코어를 만듭니다 `scores`: `matmul(q, k.T)`
2. 스코어를 스케일합니다: `scores * scale`
3. 스코어에 인과 마스크를 적용합니다: `scores.masked_fill(mask, "-inf")`
4. 소프트맥스(softmax)로 스코어를 정규화하여 어텐션 가중치를 얻습니다 `attn`: `softmax(scores)`
5. 그 가중치로 값을 재가중합니다: `matmul(attn, v)`

그래서 어텐션은 실질적으로 원시 연산들의 모음입니다. 그 중 일부는 이미 알고 있는(matmul) 연산이고, 나머지는 쉽게 발견할 수 있습니다. PyTorch로 네이브 어텐션 모듈을 작성하고 이를 프로파일링해 봅시다.

```py
class NaiveCausalAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * self.scale
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out
```


트레이스를 열기 전, 보통의 연습대로 우리가 볼 수 있을 것을 추측해 봅시다. 이 모듈의 `forward`를 트레이스하면, 우리는 다음을 기대합니다:

- 매트멀 커널( `q . k.T` )
- 곱 커널(스케일링)
- 마스킹 연산
- 소프트맥스 커널
- 매트멀 커널( `atten . v` )

```bash
uv run 04_a_naive_attention.py
uvx trace-util -f traces/ -b <hf_uname>/traces
```


| ![CPU lane of the naive attention profiler trace, with the `attn_fwd` block expanded to show its matmul, mul, masked_fill and softmax operations](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cpu-profile-naive.png) |
| :--: |
| 그림 1: 네이브 어텐션의 프로파일 트레이스의 CPU 레인에 표시된 이산 연산들을 강조 |

그림 1은 프로파일의 CPU 레인( GPU 레인은 우리를 압도하지 않도록 접어 두었습니다)을 보여줍니다. 내부 `attn_fwd`(주석이 달린 순전파 호출)에서 우리가 추정한 정확한 연산들을 볼 수 있습니다. 매트멀은 이제 친숙한 친구이고, 새로 등장한 연산들은 쉽게 포착됩니다:

- `mul`: 스케일링
- `masked_fill`: 인과 마스킹
- `softmax`: 소프트맥스 커널

이제 GPU 레인을 펼쳐 실제로 어떤 커널이 실행되었는지 확인해 봅시다.

| ![Profiler trace of naive attention showing the CPU lane above the GPU lane, with each `attn_fwd` step mapping to a cluster of GPU kernels](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/gpu-profile-naive.png) |
| :--: |
| 그림 2: 네이브 어텐션의 프로파일 트레이스의 GPU 레인과 CPU 레인을 함께 보여주며, 하나의 프로파일러 스텝에 해당하는 커널들을 강조 |

그림 2는 GPU 레인 옆에 있는 CPU 레인을 보여줍니다. GPU 레인에서 하나의 `attn_fwd` 블록을 확대하여 커널을 하나씩 살펴봅시다.

| ![Zoomed-in GPU lane of naive attention showing the individual kernels for one step: two matmuls, a mul, a memory copy, a masking kernel and a softmax](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/each-kernels-naive.png) |
| :--: |
| 그림 3: 네이브 어텐션 구현의 프로파일 트레이스의 확대된 GPU 레인 |

그림 3은 한 프로파일러 스텝의 개별 커널을 읽어보게 해 줍니다:

1. matmul (쿼리와 키)
2. mul (스케일링)
3. 메모리 복사 🤔
4. 인과 마스킹
5. softmax (어텐션 가중치를 산출)
6. matmul (어텐션 가중치와 값)

다섯 가지가 예상됩니다. 메모리 복사는 이상한 하나인데, 이게 어디서 나온 걸까요? 힌트는 PyTorch에 in-place 연산이 있습니다는 점입니다. 텐서에 일반적인(out-of-place) 방식으로 연산하면, PyTorch는 종종 복사를 만들고 요청된 연산을 그 복사에 적용한 뒤 복사본을 반환합니다. 연산 순서를 보면 여기의 범인은 [`masked_fill`](https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.masked_fill.html)입니다.

이것을 in-place 연산으로 바꾼다면 어떨까요?

## 인플레이스 인과 마스크가 적용된 나이브 어텐션 {#section-2}

변경하는 것은 `masked_fill`에서 `masked_fill_`로의 교환뿐입니다(뒤의 밑줄은 PyTorch의 in-place 연산 표기법에 주의). 같은 스크립트를 실행합니다.

```diff
def forward(self, q, k, v, mask):
    # q, k, v: [batch, heads, seq, head_dim]
    scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq, seq]
    scores = torch.mul(scores, self.scale)
-    scores = scores.masked_fill(mask, float("-inf"))
+    scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)  # [batch, heads, seq, head_dim]
    return out
```


트레이스를 살펴보고 무언가 바뀌었는지 봅시다.

```bash
uv run 04_b_inplace_ops_attention.py
uvx trace-util -f traces/ -b <hf_uname>/traces
```


| Type | CPU 스트림 |
| :--: | :--: |
| 그림 4: 나이브 마스킹 | ![CPU lane of naive attention with out-of-place `masked_fill`, showing several dispatch ops for the masking step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cpu-profile-naive.png) |
| 그림 5: 인플레이스 마스킹 | ![CPU lane of naive attention with in-place `masked_fill_`, showing fewer dispatch ops for the masking step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cpu-profile-inplace.png) |

인플레이스 버전(Figure 5)은 마스킹 단계 안에 CPU 연산을 훨씬 적게 포장합니다. 이는 고무적인 신호다. GPU 레인을 펼쳐 무슨 일이 일어났는지 확인해 봅시다.

| Type | GPU 스트림 |
| :--: | :--: |
| 그림 6: 나이브 마스킹 | ![GPU kernels for naive attention including a separate Memcpy kernel before the masking](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/each-kernels-naive.png) |
| 그림 7: 인플레이스 마스킹 | ![GPU kernels for naive attention with in-place masking, with the Memcpy kernel gone](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/each-kernels-inpace.png) |

GPU 레인에서 `Memcpy` 커널은 완전히 사라졌다(그림 6, 7). 한 줄의 변경으로 순전파마다 커널 하나를 제거했습니다. 이것만으로는 큰 차이처럼 보이지 않을 수 있지만, 이건 단일 어텐션 연산에 불과합니다. 트랜스포머 기반의 대형 모델(LLMs, 확산 모델 등) 맥락에서 이는 레이어당 한 번 반복되며 레이어가 많으므로 절약 효과가 빠르게 누적됩니다(그리고 그것이 당신의 월급 인상에 기여합니다면, 우리와 최소 10%를 나누는 것이 공정합니다고 느낍니다).

> [!NOTE]
> Out-of-place는 PyTorch의 기본 설정입니다. 그래디언트를 계산하려면 autograd가 순전파에서 본 텐서 값을 기억해야 하며, 많은 역전파 공식이 그 값을 재사용하기 때문입니다. in-place 연산은 메모리의 그 값을 덮어써서 역전파가 잘못된 수치를 읽게 됩니다. `forward`를 `torch.no_grad` 아래에서 실행하기 때문에 우리에게는 in-place가 안전하며, 역전파가 없고 손상될 여지도 없다. 또한 in-place 연산은 시간 절약뿐 아니라(추가 복사 없이) 메모리 절약도 가능하므로 로짓처럼 큰 텐서에 특히 좋습니다.

## 스케일드 닷 프로덕트 어텐션 {#section-3}

우리는 막 원시 연산(primitives)으로 어텐션을 구성했고, 심지어 `Memcpy`까지 제거했습니다. 다행히 PyTorch 팀이 이 모든 것을 우리를 위해 처리했고, 파이프라인 전체를 단 한 줄의 함수로 패키지했습니다:

```py
from torch.nn import functional as F

F.scaled_dot_product_attention(q, k, v, is_causal=True)
```


이 한 줄이 우리의 직접 작성한 모듈을 대체하고, `is_causal=True`는 수동으로 마스크를 빌드하는 수고도 덜어줍니다. 이 한 호출이 얼마나 많은 것을 숨기는지 음미할 가치가 있습니다. 그리고 그것은 코드 줄 이상으로도 숨긴다. 스케일드 닷 프로덕트 어텐션(SDPA)은 단일 구현을 가지지 않습니다. 그 이면에서 여러 백엔드 중 하나로 _디스패치_되어 입력들(dtype, head dimension, mask, 하드웨어 등)을 지원하는 가장 빠른 백엔드를 선택합니다.

[official SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)가 이 선택 과정을 안내하고, 백엔드 목록 자체는 `torch.nn.attention.SDPBackend` 열거형에 나열되어 있습니다:

```python
from torch.nn.attention import SDPBackend

BACKENDS = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}
```


일반적으로 SDPA는 우리를 위해 선택하지만, `torch.nn.attention.sdpa_kernel` 컨텍스트 매니저로 특정 백엔드를 고정할 수 있습니다. 이것이 우리 스크립트에서 하는 일입니다. 이를 통해 각 백엔드를 독립적으로 프로파일하고, 트레이스에서 어떻게 다르게 나타나는지 읽어볼 수 있습니다. 하나씩 살펴봅시다.

### 수학 백엔드

```bash
uv run 04_c_sdpa_attention.py --backend math
uvx trace-util -f traces/ -b <hf_uname>/traces
```


아무 것도 열기 전에 추측해 봅시다. 이 모듈의 수작업 어텐션(matmul, mul, mask, softmax, matmul)을 한 줄로 대체했으니 트레이스가 _간단하고 빨라질 것입니다_. 커널 수가 적고, CPU 디스패치가 줄어들며, 어쩌면 융합 커널이 나올 수도 있습니다. 우선 프로파일러 표를 확인해봅시다.

| Metric | Where to look? | Naive in-place | SDPA math |
| :--: | :--: | :--: | :--: |
| `*_fwd` CUDA time avg | The "CUDA time avg" column for the `*_fwd` op | 1.955 ms | 7.239 ms |
| Self CUDA time total | At the bottom of the profiler table | 7.194 ms | 27.279 ms |

이것이 우리의 첫 번째 놀라움입니다. 한 줄이 `3.7x` 느리다.

| | Profiler Trace |
| :--: | :--: |
| 그림 8: 나이브 인플레이스 어텐션의 프로파일러 트레이스가 하나의 순전파에 대해 다섯 개의 GPU 커널 런치를 보여줌 | ![GPU lane of naive in-place attention with five kernel launches for one forward pass](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/inplace-kernel-launches.png) |
| 그림 9: SDPA 수학 백엔드의 프로파일러 트레이스가 단일 어텐션 순전파에 대해 20개의 GPU 커널 런치를 보여줌 | ![GPU lane of the SDPA math backend with twenty kernel launches for a single attention forward pass](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/math-kernel-launches.png) |

트레이스를 열어보면(그림 9) 경보가 울리는 이유를 보여줍니다. 수학 백엔드가 순전파당 `20`개의 GPU 커널을 실행하는 반면, 네이브 어텐션 구현(Figure 8)에서 실행된 `5`은 아닙니다. 이는 우리가 예상한 것의 정반대다. 왜 이런 일이 일어나는지 알아봅시다.

#### 텐서 코어가 남아 있습니다

다음과 같은 습관을 사용해 커널 이름을 읽는 방법을 배웠습니다. 이 습관을 이 자리에 적용해 봅시다:

| Run | matmul kernel |
| :--: | :--: |
| 그림 10: 네이브 어텐션 | ![Matmul kernel name for naive attention in Perfetto, carrying the s16816 bfloat16 Tensor-core GEMM signature](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cuda-core-kernels.png) |
| 그림 11: 수학 백엔드를 사용한 SDPA | ![Matmul kernel name for the SDPA math backend, carrying the sgemm FP32 CUDA-core signature](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/tensor-core-kernels.png) |

우리가 이 트레이스를 포착하는 데 사용한 A100은 [Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)와 함께 제공되며, 행렬 곱셈을 가속하기 위한 특화된 하드웨어로 일반 CUDA 코어보다 훨씬 빠른 것으로 알려져 있습니다. 이것이 여기서 왜 중요한지 보려면 GPU 내부에 무엇이 있는지 알아야 합니다. 스트리밍 멀티프로세서(SM)는 GPU의 계산 유닛이고, 각 SM은 두 종류의 산술 유닛(CUDA 코어와 텐서 코어)을 가진다. CUDA 코어는 범용적이고 한 번에 소수의 원소를 처리하며, 텐서 코어는 작은 행렬 타일을 한 명령으로 곱하고 누적합니다. 따라서 질문은 간단합니다. 각 백엔드가 실제로 빠른 경로를 사용하고 있는가?

커널 이름이 그것에 답합니다. 네이브 커널(Figure 10)의 `s16816`은 `bfloat16` 텐서 코어 매트멀의 시그니처이며( `16x8x16` 텐서 코어 명령), 따라서 네이브 버전은 빠른 경로에 있습니다. `sgemm`(Figure 11)는 일반 CUDA 코어에서 실행되는 단정도 싱글 프리시전(`FP32`) 매트멀입니다. 다시 말해, 수학 백엔드는 텐서 코어를 전혀 건드리지 않습니다: 속도와 수치 정확도 사이의 trade-off를 위해 텐서를 `FP32`로 업캐스트하고(입력이 `bf16`인 경우에도 데이터 이동을 두 배로 늘림) 느린 CUDA 코어로 되돌아간다.

#### 인과 마스크가 생성됩니다

네이브 버전에서는 인과 마스크를 한 번 만들고 재사용했습니다. 이 버전에서는 `is_causal=True`를 넘겨 주었고, 수학 백엔드가 매 호출마다 하나를 생성해 주었다. CPU 레인에서 그것이 어떻게 일어나는지 지켜볼 수 있습니다:

| ![CPU lane of the SDPA math backend showing the ops that rebuild the causal mask: aten::ones, aten::tril, aten::scalar_tensor, aten::fill_ and aten::where](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/mask-math.png) |
| :--: |
| 그림 12: 마스킹 연산에 대한 CPU 레인 표시 |

다음은 그림 12에서 보이는 모습입니다

```bash
aten::ones -> aten::tril            build a [seq, seq] lower-triangular matrix
aten::scalar_tensor -> aten::fill_  make the -inf fill value
aten::where                         turn it into an additive bias (0 or -inf)
```


GPU에서는 이것이 `triu_tril_kernel` 하나, 여러 개의 `where` 커널, 그리고 하나의 `add_`로 나타난다. 마스크를 생각하는 것을 중단하게 해 주는 편의 플래그는 작업 자체를 제거한 것이 아니라 한 층 아래로 옮겼으며, 순전파마다 마스크가 처음부터 새로 빌드됩니다.

#### 안전한 소프트맥스

직접 작성한 버전은 일반 `aten::softmax`이라고 불렀다. 수학 백엔드는 `aten::_safe_softmax`을 호출하고, 차이가 또 다른 커널들로 나타난다(그림 13):

| ![GPU lane of the SDPA math backend showing the extra kernels that aten::_safe_softmax launches compared to a plain softmax](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/safe-softmax-extra-kernels.png) |
| :--: |
| 그림 13: 일반 소프트맥스에 비해 추가 커널을 강조하는 Safe softmax |

전체가 마스킹된 행(모든 항목이 `-inf`인 행)은 일반 소프트맥스가 `exp(-inf)/sum(exp(-inf)) = 0/0 = NaN`를 계산하게 만들 것입니다. `_safe_softmax`는 바로 그것을 방지합니다. 우리의 네이브 커널은 그 경우를 전혀 처리하지 않았고, 그 귀퉁이 케이스에서 조용히 `NaN`를 만들어 냈을 것입니다.

#### 그래서 수학 백엔드는 무엇을 위한가?

종합하면, 수학 백엔드는 참조 구현입니다. 이는 원시 ATen 연산으로 어텐션을 dtype 안전하고 NaN 안전하게 분해하는 직관적 구현입니다. 본질적으로 우리가 손으로 작성한 네이브 어텐션과 같지만 더 조심스럽다. 그 신중함이 바로 그것을 매우 느리게 만듭니다.

그 임무는 빠르게 작동하는 것이 아니라 항상 작동하는 것입니다. 이것이 완벽한 베이스라인이 됩니다. 우리가 다음에 프로파일링하는 모든 백엔드(Flash, Efficient, cuDNN)는 `20` GPU 커널들을 본질적으로 하나의 융합 커널로 압축해서 bf16으로 유지하고 중간 행렬을 전혀 만들어 내지 않도록 하려 합니다.

### 효율적인 백엔드

```bash
uv run 04_c_sdpa_attention.py --backend efficient
uvx trace-util -f traces -b <hf_uname>/traces
```


| ![Profiler trace of the SDPA efficient backend showing a single fused fmha_cutlassF attention kernel per forward](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/efficient-backend.png) |
| :--: |
| 그림 14: sdpa의 효율적 백엔드에 대한 프로파일러 트레이스 |

수학 백엔드가 한 프로파일러 스텝에 20개의 커널을 실행했습니다면, 효율적 백엔드는 단 하나의 `fmha_cutlassF_bf16_aligned_64x64_rf_sm80`만 실행합니다(그림 14에서 볼 수 있습니다).

커널의 이름의 의미를 해석해 봅시다:

- `fmha` (퓨즈드 멀티헤드 어텐션): 어텐션의 모든 원시 연산이 하나의 연산으로 융합되어 있습니다.
- `cutlassF`: CUTLASS를 기반으로 한 텐서-코어 GEMM용 템플릿, forward에 대해 `F`.
- `bf16_aligned`: bf16으로 실행( FP32 업캐스트 없음, 수학과 다름).
- `64x64`: 타일 크기.
- `rf` (레지스터 파일): 작동 집합이 레지스터에 보관되어 칩에서 가장 빠른 메모리입니다.
- `sm80`: Ampere에 맞춰 컴파일(A100의 컴퓨트 수준 8.0).

메타(Meta)의 [xformers](https://github.com/facebookresearch/xformers) 라이브러리에서 나온 메모리 효율적인 어텐션 커널이며 PyTorch로 업스트림되었습니다. 사람들이 "xformers 백엔드"라고 말할 때 이 `fmha_cutlassF` 커널이 바로 그것입니다.

### Flash 백엔드

```bash
uv run 04_c_sdpa_attention.py --backend flash
uvx trace-util -f traces -b <hf_uname>/traces
```


| ![Profiler trace of the SDPA flash backend](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/flash-backend.png) |
| :--: |
| 그림 15: flash 백엔드 트레이스, forward당 하나의 융합 `pytorch_flash` 커널 |

`void pytorch_flash` 커널(Figure 15)은 [FlashAttention-2](https://arxiv.org/abs/2307.08691)(Tri Dao의 구현)이며 PyTorch에 벤더링되어 있습니다.

이제 트레이스를 더 읽기 전에, 지금까지 당신이 물어야 할 질문에 답하는 것이 가치 있습니다: _왜 "flash"라는 백엔드가 존재하고, 그것이 왜 이렇게 중요한가?_

#### 왜 flash 어텐션이 존재하는가?

잠시 수학 백엔드으로 돌아가 봅시다. 실제 문제는 20개의 커널 수가 아니라, 그 커널들이 서로에게 넘겨주는 데이터였습니다.

1단계는 전체 스코어 행렬 `attn = q . k.T`을 빌드하는데, 이는 head당 `[seq, seq]`입니다. 시퀀스 길이가 4096인 경우 단일 head에 대해 `4096 x 4096 ≈ 16 million`개의 숫자입니다. 그 행렬은 HBM(그래픽 카드의 주 메모리)로 기록되며 공간이 충분하면 기록합니다. 그런 다음 다시 읽어 시그널링을 위해 스케일하고, 마스크를 위해 다시 쓰고, 다시 소프트맥스를 위해 읽는 식으로 계속됩니다. 어텐션의 비용은 이 HBM으로의 왕복 트래픽에 의해 좌우되며, 매트멀 그 자체보다는 이 트래픽에서 더 많은 부분을 차지합니다.

FlashAttention은 정확히 이것을 겨냥합니다. 전체 `s` 행렬을 먼저 계산한 뒤 감소시키는 대신, **타일 단위로** `k`와 `v`를 따라가며 진행하고, 진행 방향에서 소프트맥스를 유지하는(“온라인 소프트맥스” 트릭) 방식으로 출력도 타일 하나씩 누적합니다. 전체 `[seq, seq]` 스코어 행렬은 **HBM에 한 번도 기록되지 않고**, 칩 안에만 존재합니다. 이것이 전체 어텐션 파이프라인을 bf16으로 유지되는 하나의 융합 커널로 단번에 축소시키는 유일한 아이디어입니다.

#### 왜 프로파일러에서 플래시가 "잘못 보이는"가?

| ![Perfetto footprint of the flash kernel reporting an estimated achieved occupancy of 13%](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/flash-occupancy.png) |
| :--: |
| 그림 16: 플래시 커널의 점유율 추정치가 13%로 보이는 모습 |

여기서 플래시가 프로파일러의 발자국을 읽는 사람들을 놀라게 합니다. 플래시는 가장 빠른 백엔드인데도 프로파일러가 매우 낮은 점유율로 보고합니다(그림 16에 표시). 왜 그것이 괜찮은지 보려면 세 가지 빠른 정의가 필요합니다.

GPU 커널은 본질적으로 다수의 작은 실행 유닛(스레드)에 의해 실행되는 일련의 명령입니다. 이 개별 실행 유닛들은 변수 로딩, 더하기, 저장 등을 처리합니다. 각 커널마다 많은 스레드를 실행하고, 이를 추적하기 위해 블록으로 묶는다.

블록은 스트리밍 멀티프로세서(SM)에 스케줄됩니다. 두고 온 블록은 한 SM에 완전히 남아 있으며, SM이 충분한 자원을 갖고 있으면 여러 블록을 한 번에 수용할 수 있습니다. 이러한 자원에는 레지스터, 공유 메모리, 최대 상주 스레드 수, 최대 상주 워프 수가 포함됩니다. 따라서 커널이 낮은 점유율(occupancy)을 가진다고 할 때, 이는 각 SM이 이론상 지원할 수 있는 것보다 더 적은 워프를 갖고 있음을 의미합니다.

> [!TIP]
> 스레드, 블록, 그리드 등에 대해 더 알고 싶다면 여기 [great resource](https://huggingface.co/blog/mi300kernels#a-quick-introduction-to-the-mi300x)가 있습니다.

트레이스에서 플래시 커널을 클릭하면 그 발자국이 이야기를 들려줍니다(그림 17).

| ![Resource footprint of the pytorch_flash kernel in Perfetto, showing a high per-thread register count and large shared memory usage per block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/flash-reg-count.png) |
| :--: |
| 그림 17: 커널 발자국, 블록당 레지스터와 공유 메모리가 많은 편 |

플래시는 블록당 많은 스레드와 상당량의 공유 메모리를 필요로 합니다. 예를 들어 블록이 128개의 스레드이고 각 스레드가 255개의 레지스터를 사용하면 그 블록은 `128 × 255 = 32,640` 레지스터가 필요합니다. 65,536 레지스터를 가진 Ampere SM에서 그런 두 블록만 한 번에 맞물린다. 각 128-스레드 블록은 `128 / 32 = 4` 워프를 가지므로 두 블록은 상주 워프가 겨우 8개다. 최대 64 상주 워프에 비하면 대략 13%의 점유율입니다. 플래시는 최적화가 부족해서가 아니라, 각 블록이 온칩 자원을 의도적으로 매우 “무겁게” 사용하기 때문입니다.

그리고 그것이 핵심입니다. 높은 점유율은 많은 워프를 실행 대기 중으로 유지해 지연을 숨기는 데 도움이 되지만, 작업 자체를 더 효율적으로 만들진 않습니다. 플래시는 어태션 타일을 온칩에 유지하고 데이터를 적극적으로 재사용하며, 전역 메모리에 전체 어텐션 매트릭스를 한 번에 구체화시키지 않기 위해 의도적으로 그 레지스터와 공유 메모리를 사용합니다.

### cuDNN 백엔드

```bash
uv run 04_c_sdpa_attention.py --backend cudnn
uvx trace-util -f traces -b <hf_uname>/traces
```


| ![Profiler trace of the SDPA cuDNN backend showing a single cudnn_generated attention kernel per forward](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cudnn-backend.png) |
| :--: |
| 그림 18: cuDNN 백엔드 트레이스, 순전파당 하나의 생성된 어텐션 커널 |

이 시점의 패턴은 익숙합니다. 플래시와 효율처럼 cuDNN은 순전파당 하나의 융합된, 플래시 스타일의 커널을 제공합니다(그림 18). 그래서 자연스러운 질문은: **플래시가 이미 어텐션을 융합하는데, 왜 파이토치는 또 다른 플래시 백엔드를 제공하는가?** 그 답은 _커널을 누가 쓰고 어떻게 빌드했는가_의 차이에 있으며, 그 차이가 트레이스를 다르게 보이게 만듭니다.

#### cuDNN 커널은 어떻게 다른가

Flash와 Efficient는 PyTorch에 벤더링된 **고정된, 미리 컴파일된 커널**입니다. 매번 같은 바이너리를 받는다. cuDNN은 NVIDIA의 자체 딥러닝 라이브러리이며, 그 주의 커널은 **주어진 문제에 맞춰 생성되고 튜닝됩니다**. 이는 고정된 cuBLAS 바이너리보다 더 코드 제너레이션에 가깝다. 그 매우 긴 커널 이름에서 그것을 바로 확인할 수 있습니다:

```bash
cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_6_128x64x64_4x1x1_cga1x1x1_kernel0_0
```


- `cudnn_generated`: 미리 배송되는 바이너리가 아니며 cuDNN에 의해 생성되었습니다.
- `flash_fprop`: 플래시 어텐션 스타일의 순전파다. 따라서 알고리즘은 플래시 백엔드의 계열과 같습니다.
- `wmma_f16`: 16비트 부동 소수 파이프라인에서 WMMA API를 사용하고, 텐서 코어 경로를 따릅니다.
- `knob_6`: cuDNN은 미리 조정된 구성("knobs") 세트에서 선택합니다. 다양한 형태가 서로 다른 knob을 선택하게 합니다. 이는 cuBLAS가 타일 변형을 선택하는 방식과 유사합니다.
- `128x64x64`: 그가 선택한 타일 차원.

그 하나의 사실, 문제당 생성된 것, 이 트레이스에서 보이는 다른 이상한 점들을 설명합니다.

1. No transposes: CPU 레인은 `_cudnn_attention_forward`에서 바로 곧바로 몇 개의 `aten::empty` 할당으로 이어진 다음 커널로 가며, `aten::transpose`는 전혀 없다(Figures 19, 20 및 21). Flash와 Efficient는 텐서를 재구성하기 위해 메타데이터 전치를 네 번 삽입하는 반면, cuDNN은 생성기가 그 레이아웃에 맞는 커널을 생성하기 때문에 네이티브 `[B, H, S, D]` 레이아웃을 직접 사용합니다.

   | Variant | Trace |
   | :--: | :--: |
   | Figure 19: Flash | ![CPU lane of the flash backend showing four aten::transpose ops before the fused attention kernel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/flash-transpose.png) |
   | Figure 20: Efficient | ![CPU lane of the efficient backend showing four aten::transpose ops before the fused attention kernel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/efficient-trasnpose.png) |
   | Figure 21: cuDNN | ![CPU lane of the cuDNN backend going straight to aten::empty allocations and the kernel, with no transpose ops](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cudnn-backend.png) |

2. It launches through `cuLaunchKernelEx`, not `cudaLaunchKernel`: Every other kernel in this whole series went through the runtime API `cudaLaunchKernel`. cuDNN uses the driver-level _extended_ launch, which carries launch attributes (Figure 22).

   | ![CPU lane of the cuDNN backend showing the cuLaunchKernelEx driver-level launch instead of cudaLaunchKernel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cudnn-launch.png) |
   | :--: |
   | Figure 22: CPU lane of the cuDNN backend showing the cuLaunchKernelEx driver-level launch instead of cudaLaunchKernel |

3. The profiler reports 0% achieved occupancy: Do not take that at face value, it is a measurement gap, not a stalled GPU. CUPTI (the profiling backend) cannot attribute occupancy to a driver-API (`cuLaunchKernelEx`) launch the way it does for `cudaLaunchKernel`, so the field reads 0. The footprint fills in the truth (Figure 23): `240 registers × 256 threads = 61,440` registers per block against the SM's 65,536, so only **one block** fits per SM (8 warps ≈ 12.5%), right in line with flash.

   | ![Perfetto footprint of the cuDNN kernel reporting 0% achieved occupancy, with 240 registers per thread and 256 threads per block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-attention-profile/cudnn-footprint.png) |
   | :--: |
   | Figure 23: cuDNN 커널이 0% 달성 점유율을 보고하고, 스레드당 240 레지스터, 블록당 256 스레드 |

#### 비용이 CPU로 이동했습니다

"전치 없음(no transposes)" 이야기는 CPU에서 cuDNN이 가장 '가볍다'는 백엔드일 것이라고 기대하게 만듭니다. 그러나 실제로는 정반대입니다.

| backend | CUDA 평균 시간 | CPU 평균 시간 |
| :--: | :--: | :--: |
| efficient | 277.9 µs | 117 µs |
| flash | 146.8 µs | 138 µs |
| cudnn | 186.3 µs | **214 µs** |

심지어 전치 연산이 하나도 없더라도 cuDNN은 CPU에서 순전파당 약 **214 µs**를 소비합니다. 이는 플래시(138)나 효율(117)보다 많습니다. 거의 전부가 `aten::scaled_dot_product_attention` 자체 시간(전체 실행의 26%)과 `_cudnn_attention_forward`에 있습니다. 이것은 cuDNN의 런타임 엔진이 매 호출마다 계획을 선택하고 준비하는(“knob” 탐색) 과정입니다.

더 적은 ATen 연산이 보이는 것이 CPU 작업이 적다는 뜻이 아닙니다. 그 작업은 라이브러리로 이동했고, 프로파일러는 이를 하나의 크고 불투명한 막대 하나로만 보여줄 뿐입니다. 트레이스가 갑자기 더 깨끗해지면, 작업이 사라진 것이 아니라 프로파일러가 분해할 수 없는 곳으로 이동한 것일 수 있습니다.

GPU에서 cuDNN(186.3 µs)은 효율성과 플래시 사이에 위치합니다. 이 아주 플래시 친화적인 형태에서 수작업 FlashAttention-2가 그 위를 약간 앞선다. cuDNN은 더 큰 head 차원이나 다른 시퀀스 길이의 _다른_ 형태에서 자주 이긴다. 다만 그 재조정은 당신이 CPU에서 지불한 비용이기도 합니다.

## Everything we covered, at a glance {#section-4}

마무리하기 전에, 우리가 프로파일링한 모든 어텐션 변형과 각 트레이스가 가르쳐 준 하나의 교훈을 정리한 하나의 표가 있습니다.

| Variant | What we changed | Kernels / forward | What the trace revealed |
| :-- | :-- | :--: | :-- |
| Naive attention | 원시 연산으로 직접 구성된 어텐션(matmul, mul, mask, softmax, matmul) | 6 | 외부의 `masked_fill`에서 온 숨겨진 `Memcpy`. |
| Naive in-place | `masked_fill` → `masked_fill_` | 5 | 한 줄로 `Memcpy` 커널을 완전히 제거합니다. |
| SDPA math | `F.scaled_dot_product_attention` 수학 백엔드에 고정 | 20 | 참조 구현: CUDA 코어의 FP32, 매 호출마다 마스크 재구성, `_safe_softmax`. 정확하지만 약 3.7배 느림. |
| SDPA efficient | Efficient(xformers) 백엔드 | 1 | 하나의 융합 `fmha_cutlassF` 커널, 텐서 코어에서 bf16 유지. |
| SDPA flash | Flash 백엔드 | 1 | 하나의 융합 `pytorch_flash` 커널(FlashAttention-2). 가장 빠르지만 13% 점유율은 "잘못 보이는" 경우. |
| SDPA cuDNN | cuDNN 백엔드 | 1 | 문제별로 생성된 커널: 전치 없음, `cuLaunchKernelEx`지만 비용이 무거운 CPU 바로 이동. |

## Concluding the series {#section-5}

이 시리즈에서 한 가지만 takeaway를 뽑자면, 모든 트레이스 전에 반복했던 습관인 **먼저 추측하고, 그다음 보자**가 되길 바란다.

트레이스에 무엇이 들어 있을지 소리 내어 예측하고, 트레이스를 열어 보며, 일치하지 않는 부분을 화면에서 가장 흥미로운 것으로 여기자. 이 세 편의 포스트에서 얻은 모든 실제 통찰은 숨겨진 `Memcpy`, `addmm` 에필로그, 20 커널 수학 백엔드, 플래시의 "잘못 보이는" 점유율, cuDNN의 두툼한 CPU 바에서 비롯된 것이며, 이는 트레이스와 일치하지 않는 추측에서 비롯됩니다.

프로파일링은 GPU 전문가를 위한 별도의, 위협적인 기술이 아닙니다. 그것은 단지 아주 면밀히 보고 “저게 왜 그런가?”를 묻고 대답이 떠오를 때까지 파고드는 훈련일 뿐입니다. 이제 당신은 자신이 다루는 모델에서 이를 스스로 할 수 있는 어휘와 반사적 반응을 갖추었다. 트레이스를 열고, 추측을 세우고, 불일치를 찾아보세요.

**Profiling in PyTorch** 시리즈를 읽어 주셔서 감사합니다. 이제 뭔가를 프로파일링해 봅시다. 🤗

초기 초안에 대한 리뷰를 남겨 주신 [Noe Flandre](https://huggingface.co/NoeFlandre)께 감사드립니다!

> [!NOTE]
> 이 블로그 포스트는 LLM을 이용해 다듬었습니다. 이것이 배경에서 에이전트가 포스트를 생성하도록 합니다는 뜻은 결코 아닙니다. 팀의 일부는 영어를 모국어로 하지 않으며, LLM(대부분 영어로 학습된 모델)이 어리석은 문법 실수를 바로잡거나 더 덜 위협적으로 들리고 깔끔하게 들리는 문장을 재구성할 수 있습니다고 생각합니다. 이 점이 "왜 읽어야 하나, 이것이 LLM으로 생성되었기 때문인가"라는 아이디어에 도움이 되길 바랍니다. 🤗
