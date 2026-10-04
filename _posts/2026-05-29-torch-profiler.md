---
layout: post
title: "PyTorch에서의 프로파일링(Part 1): torch.profiler에 대한 초보자 가이드"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-05-29-torch-profiler/thumbnail.png
authors:
  - user: ariG23498
  - user: sayakpaul
  - user: sergiopaniego
  - user: ror
  - user: pcuenq
slug: "torch-profiler"
source_url: "https://huggingface.co/blog/torch-profiler"
source_published_date: "2026-05-29"
source_published_at: "2026-05-29T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Profiling in PyTorch (Part 1): A Beginner's Guide to torch.profiler](https://huggingface.co/blog/torch-profiler)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/torch-profiler -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# PyTorch에서의 프로파일링(Part 1): torch.profiler에 대한 초보자 가이드

![Thumbnail of the blog post](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/thumbnail.png)

> *프로파일링할 수 없는 것은 최적화할 수 없다.*

대규모 언어 모델(LLM)에서 초당 더 많은 토큰 처리량을 얻으려 시도하든, 추론 시간을 밀리초 단위로 줄이려 하든, 혹은 학습 루프가 스펙 시트가 약속한 속도보다 느린 이유를 이해하려 하든, 그 경로는 결국 프로파일링을 거치게 됩니다.

문제는 프로파일링에 가파른 온-램프가 있다는 점입니다. 추적은 다채로운 색상의 직사각형으로 가득 찬 벽처럼 보입니다. 이벤트는 위협적으로 보이는 이름을 담고 있습니다. 대부분의 튜토리얼은 이미 그것들을 읽을 수 있다고 가정합니다. 그래서 우리가 *프로파일링을 해야 한다고 해도*, 추적을 여는 것은 나중에(또는 다른 사람에게) 맡겨두는 일이 되기 쉽습니다. 이 글과 시리즈는 그 온-램프를 낮추려는 시도입니다.

이 글은 **Profiling in PyTorch** 시리즈의 시작글로, 프로파일러 트레이스를 읽는 기술을 천천히 익히고 이를 최적화에 활용하는 방법을 다룹니다. 계획은 다음과 같습니다:

1. **Part 1 (본 글):** 가장 단순한 연산인 행렬 곱에 바이어스 추가를 시작으로, 프로파일러가 반환하는 것을 읽는 법을 배웁니다.
2. **Part 2:** `nn.Linear`와 작은 MLP로 확장하고, 추적을 통해 최적화를 이끌어내며 아래의 `kernels`를 살펴봅니다.
3. **Part 3:** Large Language Models에서 `transformers`와 함께 모든 것을 한데 엮습니다.

우리는 초보자의 시각에서 여정을 기록합니다. 기본적인 PyTorch 이외의 전제 조건은 없습니다. 이를 천천히 읽되 "아하!" 모멘트를 즐깁니다. 글의 구성은 의도적으로 질문 중심으로 구성되어 있습니다: 추적을 열고 "잠깐, 왜 *그 일이* 일어나지?"라고 묻고, 무언가가 이해될 때까지 그 이유를 쫓습니다. 끝에 가면 여러분은 다음을 알게 될 것입니다:

- `torch.profiler`를 설정하는 방법과 그것이 실제로 반환하는 내용,
- 프로파일러 표와 추적(CPU 레인, GPU 레인, 사이의 의심스러운 간격)을 읽는 방법,
- 파이썬 호출로부터 CUDA 커널까지의 이벤트 체인,
- 위에 `torch.compile`를 얹었을 때 무엇이 바뀌는지(그리고 더 흥미롭게도 무엇이 바뀌지 않는지)

시작하기에 앞서 아래의 두 정의가 아래의 모든 내용을 더 읽기 쉽도록 도와줄 것입니다:

1. GPU **커널**은 GPU의 다수 스레드에서 병렬로 실행되는 프로그램입니다.
2. CPU **스케줄링 및 런칭** 이 커널들을 수행합니다.

일반적으로 GPU 커널을 직접 작성할 필요는 없습니다. PyTorch 연산을 사용할 때, 그것은 자동으로 하나 이상의 커널로 GPU에서 작업을 수행하도록 변환됩니다.

두 가지 아이디어를 마음에 담고, 이제 질문을 시작해 봅시다.

> [!NOTE]
> 이 글에 사용할 전체 스크립트는 [`01_matmul_add.py`](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py)입니다. 이 스크립트를 별도의 탭에서 열어 코드를 단계별로 살펴보는 것을 권장합니다. 스크립트를 실행하는 GPU는 `NVIDIA A100-SXM4-80GB`를 사용합니다.

## 행렬 곱과 덧셈 연산 {#section-1}

정확히 [quipped by Dr. Sara Hooker](https://youtu.be/7knwihgj0fU?si=uvzGH-J9bsCHP4Nn&t=2199)처럼, 우리가 거의 물로 이루어진 존재인 것처럼, 심층 신경망도 주로 행렬 곱으로 구성됩니다. 이것들이 얼마나 기초적이든 간에, 프로파일링 여정을 다른 것과 시작하는 것은 아쉽습니다.

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)
```


> 행렬 곱과 함께 더해지는 행렬 덧셈은 가중치와 편향이 뉴런에서 어떻게 상호 작용하는지 모방합니다. 이 덧셈(의도된 말장난)은 컴파일로의 길을 열어 주는 방식을 이해하는 데 도움이 될 것입니다: [later in the post](#lets-see-some-torch-compile-at-work).

프로파일링하기 위해서는 `torch.profiler` 모듈을 사용할 것입니다. 포함되는 단계는:

1. [code to profile ready](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L26-L27)를 준비합니다(여기서는 `def fn`으로, 행렬 곱과 행렬 덧셈을 래핑합니다)
2. 알고리즘을 [Annotate](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L32)합니다. completely optional 이지만 권장합니다. `record_function`는 우리의 함수를 `matmul_add`로 주석 처리하여 추적에서 쉽게 탐색할 수 있게 합니다(나중에 설명합니다)
```py
def step():
  with torch.profiler.record_function("matmul_add"):
    return fn(x, w, b)
```

3. 코드를 `torch.profiler.profile` [context manager](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L53-L62)로 래핑합니다
```py
  with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,  # the cpu activities
        torch.profiler.ProfilerActivity.CUDA, # the gpu activities
    ],
  ) as prof:
    # it is recommended to run events multiple times to warm up the GPUs
    for _ in range(5):
      step()
      prof.step()
```

4. [profile](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L70)를 내보냅니다
```py
# the profiler table
prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

# the profiler trace
prof.export_chrome_trace(trace_path)
```


프로파일러는 두 가지 상이한 산출물을 내보냅니다:

1. 프로파일러 표: 알고리즘의 통계적 요약을 제공합니다. 무엇이 가장 많은 시간을 차지하는지에 대한 답을 주며 핫스팟 파악에 유용합니다. 핫스팟은 시간이 가장 많이 걸리는 이벤트일 수 있고, 파이프라인의 병목일 수 있으며, 여러 차례 호출되는 이벤트일 수 있습니다.
2. 프로파일러 추적: 시간적 실행 뷰를 제공합니다. 언제 어느 작업이 발생했는지, CPU와 GPU에서 발생하는 활동을 나타냅니다. 이는 런치된 커널, 런치 지연, CPU와 GPU 활동의 중첩 여부 등을 조사할 때 유용합니다.

처음 실행으로 두 가지를 실제로 확인해 봅시다. ([Here is the entire `01_matmul_add.py` script](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py))

> [!NOTE]
> GPU가 있는 머신에서 이 스크립트를 실행하는 것이 권장됩니다.

```bash
uv run 01_matmul_add.py --size 64
```


위의 스크립트를 실행하면(GPU 머신에서) 두 가지 산출물이 들어 있는 `traces/01_matmul_add` 폴더를 찾을 수 있습니다:

```bash
64_bf16_cold_eager.json
64_bf16_cold_eager.txt
```


| ![Profiler table for matmul add on 64 sized matrices](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profile-table-64.png) |
| :--: |
| 그림 1: 64 크기의 행렬에서 곱과 덧셈에 대한 프로파일러 표 |

`.txt` 파일에는 프로파일러 표가 들어 있습니다. 파일을 열면 그림 1과 같이 첫 번째 열에 프로파일링 범위 내에서 트리거된 이벤트들이 나열된 큰 표가 보입니다.

다른 열은 `activities` 내 `torch.profiler.profile`에서 CPU 또는 GPU 또는 다른 장치들에서 이벤트가 차지하는 시간과 관련이 있습니다. 어떤 이벤트가 가장 많은 시간인지를 살펴보고, 그 이벤트가 실제로 그 시간만큼 걸리는지 직관적으로 이해해 보세요. 또한 "호출 수(# of Calls)" 열은 이벤트가 얼마나 자주 트리거되었는지를 나타냅니다.

또한 "Self CPU/CUDA" 대 "CPU/CUDA 총합"에 대해 알아봅시다. "Self" 열은 오직 이벤트 자체에 소비된 시간을 측정하고 자식들을 제외합니다. "Total" 열은 이벤트와 모든 자식들을 함께 포함합니다. 따라서 `matmul_add`의 "CPU 총합"은 자기 자신이 소요된 시간에 자식 이벤트들이 트리거한 시간을 합친 것입니다. 이는 중요한 뉘앙스입니다.

표의 마지막 두 줄을 보면 프로파일러가 우리에게 다음과 같은 정보를 알려 주고 있음을 알 수 있습니다.

```bash
Self CPU time total: 2.314ms
Self CUDA time total: 23.104us
```


CPU 시간은 `ms`에서, GPU 시간은 `us`에서 나타납니다. 관점으로 보면 GPU에서 소요된 시간(커널 `ampere_bf16_s16816gemm...`)은 CPU에서 소요된 시간(`matmul_add` 연산)보다 1% 미만입니다. GPU는 대부분의 시간 동안 대기 상태이며, 이는 즉시 나타나는 적신호입니다. 이러한 현상은 GPU가 아주 작은 행렬 곱을 아주 빠르게 계산할 수 있기 때문이며, 우리의 코드는 커널을 준비하고, GPU에서 실행을 시작하고, 곱하기를 위해 데이터를 보내고 결과를 모으는 데 대부분의 시간을 소비합니다. 이 개념은 오버헤드-바운드(overhead-bound) 알고리즘으로 알려져 있습니다.

이런 환경에서 벗어나는 가장 쉬운 방법은 더 큰 행렬 곱을 사용하는 것입니다.

```bash
uv run 01_matmul_add.py --size 4096 
```


| ![Profiler table for matmul add algorithm on 4096 sized matrices](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/profiler-table-4096.png) |
| :--: |
| 그림 2: 4096 크기의 행렬에 대한 matmul 덧셈 프로파일러 표 |

그림 2의 마지막 두 줄은 다음과 같습니다:

```bash
Self CPU time total: 4.908ms
Self CUDA time total: 4.495ms
```


두 시간 모두 ms 단위이며, 행렬 곱의 크기를 키움으로써 더 많은 GPU 시간을 실제로 확보했습니다. 그림 2를 보면 이제 CUDA 시간이 대부분 GPU 커널(`ampere_bf16_s16816gemm_..`)에 의해 차지되고 이를 시작한 CPU 작업(`matmul_add`)의 시간은 줄어들었다는 것도 확인됩니다. 이는 우리가 실제로 오버헤드 바운드에서 계산 바운드로 이동했음을 의미합니다.

이제 `.json` 산출물 안에 있는 디스패치 체인(dispatch chain)을 시각화해 봅니다. 이를 [Perfetto UI](https://ui.perfetto.dev)에 업로드하거나, `uvx trace-util traces -b traces`를 사용해 Perfetto 링크를 직접 생성해 볼 수 있습니다.

## 64x64 traces {#section-2}

| ![PyTorch profiler trace of a 64×64 bf16 matmul followed by an add on a CUDA GPU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/64-matmul-add.png) |
| :--: |
| 그림 3: 64 크기의 행렬에서 곱과 덧셈에 대한 프로파일러 추적 |

그림 3에서 행렬 곱과 덧셈에 대한 프로파일러 추적을 볼 수 있습니다. 이때 막대의 너비는 이벤트의 지속 시간을, 수직 중첩은 호출 계층 구조를, CPU 레인은 CPU에서 일어나는 이벤트를, GPU 레인은 실제 커널 실행을 나타냅니다. 비어 있는 공간은 대기 시간이나 유휴 시간을 의미합니다.

기본 구성으로 이 스크립트를 실행했습니다(디폴트 설정):

- 크기 64: 입력, 가중치, 바이어스의 크기가 (64, 64)
- 데이터타입 bf16: 데이터 타입이 bf16
- 컴파일 없음: torch 연산을 컴파일하지 않음
- 워밍업 없음: 프로파일링 전에 GPU를 워밍업하지 않음

> Perfetto를 사용할 때 추적에 더 빨리 접근하려면 키보드 사용을 권장합니다. 추적을 탐색하려면 "W A S D"를 사용할 수 있습니다.

| ![PyTorch profiler trace with the CPU lane and GPU lane labelled side by side in Perfetto](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gpu-cpu-trace.png) |
| :--: |
| 그림 4: PyTorch 프로파일러 추적의 CPU 및 GPU 레인 |

그림 4에는 CPU 활동용 레인과 GPU 활동용 레인 두 개가 있습니다. CPU 레인에는 세 가지 프로파일 단계가 보이며( 시작은 `ProfilerStep#2`에서 시작합니다). 이는 `schedule`에서 비롯됩니다.

```py
schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
```


`wait`은 시끄러운 초기화를 건너뛰고( `ProfilerStep#0` ), `warmup`는 프로파일러를 거치지 않고 실행되며( `ProfilerStep#1` ), 그리고 `active`가 추적에 표시됩니다. 사용 중인 스케줄은 [script here](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L58)에서 확인할 수 있습니다.

탐정 모자를 쓰고 추적을 조사하며 몇 가지 질문을 던져 보자.

### Why does the ProfilerStep#2 take so long?

| ![ProfileStep#2 in a PyTorch profiler trace appears wider than ProfileStep#3 and ProfileStep#4](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/why-is-step-2-big.png) |
| :--: |
| 그림 5: `ProfileStep#2`가 뒤따르는 단계들보다 눈에 띄게 넓게 보임 |

그림 5에서 `ProfileStep#2`가 다른 단계들보다 시간이 더 걸리는 것을 확인할 수 있으며, 자세히 보면 `matmul_add` 주석에서도 비슷한 패턴이 보입니다. 결정적 단서는 주석 그 자체가 아니라 주석 안에 있습니다:

| 단계 | `matmul_add` 시작 | `aten::matmul` 시작 | 간격 |
| :--: | :--: | :--: | :--: |
| #2 | 138.736 | 366.493 | 227.757 µs |
| #3 | 517.926 | 523.447 | 5.521 µs |
| #4 | 610.039 | 614.527 | 4.488 µs |

| ![228 microsecond gap between record_function matmul_add and the aten::matmul dispatch in profile step 2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-227.png) |
| :--: |
| 그림 6: `record_function("matmul_add")`와 `aten::matmul` 사이의 약 228 µs의 데드 윈도우 |

그림 6에 보이는 약 228 µs는 `record_function("matmul_add")`에 진입한 뒤 PyTorch가 실제로 `aten::matmul`를 디스패치하기 전의 "데드 윈도우"입니다. 이는 버퍼의 오버플로우나 커널 실행 도중 GPU VRAM에 메모리를 할당하려는 버퍼 요청 등의 여러 이유로 발생할 수 있습니다. 우리는 추적을 보지 않거나(표준 방식은 추적 전 먼저 [some more warmup steps](https://huggingface.co/datasets/ariG23498/profiling-pytorch/blob/main/01_matmul_add.py#L35-L39)를 실행하는 것) 진행할 수 있습니다.

프로파일링 측면에서 워밍업은 실제로 프로파일링하기 전에 이벤트를 몇 차례 실행하는 것을 말합니다. 위 인자들을 포함한 GPU가 수행하는 사전 작업은 한 번의 노력으로 끝나며, 이를 프로파일링하고 싶지 않습니다. 예제에서는 두 번의 워밍업 단계가 있으며, 하나는 프로파일러에 진입하기 전에 함수 루프를 실제로 실행하는 것이고, 두 번째는 `warmup` 인수로 달성되는 프로파일러 내부의 루프입니다. 이 섹션에서는 실제 반복과 스케줄을 활성화했습니다.

```bash
uv run 01_matmul_add.py --warmup
```


[Perfetto Trace for 64x64 with Warmup](https://ui.perfetto.dev/#!/?url=https://huggingface.co/buckets/ariG23498/traces/resolve/01_matmul_add/64_bf16_warm_eager.json)

| ![PyTorch profiler trace after warmup steps where ProfileStep#2 no longer shows cold-start overhead](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/warmup.png)|
| :--: |
| 그림 7: 워밍업 후, 모든 프로파일 단계가 비슷한 시간 소요 |

그림 7에서 각 프로파일 단계가 비슷한 시간을 차지하는 것을 보지만, 이것이 한 번 오버헤드를 최적화했다는 뜻은 아닙니다. 오버헤드가 프로파일링되지 않도록 런을 워밍업했습니다. 이 부분을 단서 없이 마무리하는 것이 독자에게 불리하다고 생각해, 런치 오버헤드를 추가로 최적화하는 방법에 대해 읽을 수 있는 [link](https://pytorch.org/blog/accelerating-generative-ai-2/)를 제공합니다.

### CPU 및 GPU 레인 간 약 2.5 ms 오프셋의 이유는?

| ![2.32 millisecond offset between the CPU lane and the GPU lane in a PyTorch profiler trace](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/gap-bw-kernel-launch.png) |
| :--: |
| 그림 8: CPU와 GPU 레인의 약 2.5 ms 오프셋 |

그림 8에서 CPU 레인과 GPU 레인 사이에 약 2.5 ms의 오프설정이 존재함을 확인할 수 있습니다. 이는 CPU가 CUDA 커널을 제출하고 실제로 실행되기까지의 지연 시간입니다. 워밍업 단계와 스케줄의 `wait` 및 `warmup`이 합쳐져 GPU를 바쁘게 유지시켜 오프셋을 줄여줄 거라고 생각할 수 있습니다.

무슨 일이 실제로 벌어지는지 알아내기 위해 일정(schedule)을 조금 바꿔 봅시다:

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1)
```


| ![PyTorch profiler trace with wait=0 warmup=0 showing an Activity Buffer Request between steps](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/full-profile.png) |
| :--: |
| 그림 9: `wait=0` 및 `warmup=0`와 함께 추적은 `Activity Buffer Request`를 드러냅니다 |

그림 9는 GPU 레인에 작업 전 어떤 `Activity Buffer Request`가 있음을 보여줍니다. 조금 더 확대해 봅시다.

| ![gap between matmul and add CUDA kernels caused by profiler buffer request](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/mat-add-gap.png) |
| :--: |
| 그림 10: 프로파일 스텝 1에서 행렬 곱과 덧셈 커널 사이에 간격이 생깁니다 |

GPU 추적을 더 자세히 보면, `ProfileStep#0`의 행렬 곱과 덧셈 커널은 차례로 실행되고 있는 반면, `ProfileStep#1`의 커널 사이에 간격이 있습니다. 가장 그럴듯한 설명은 버퍼 오버플로우가 있었고 커널 실행 도중 GPU VRAM에 메모리 할당 요청이 발생했다는 것입니다.

다른 가능성을 배제하는 최선의 방법은 더 많은 반복을 프로파일링하고 추적의 다른 부분에서도 유사한 윈도가 나타나는지 확인하는 것입니다. 이를 위해 `active=20`로 실행합니다.

| ![PyTorch profiler trace of 20 active iterations confirming the buffer-request gap only appears once](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters.png) |
| :--: |
| 그림 11: 20개의 활성 스텝으로 실행할 때 간격은 한 번만 나타나며 버퍼 요청임을 확인합니다 |

그림 11에서 `ProfileStep#1`에서도 유사한 경향을 확인합니다. 이는 이전 결과와 일치하며, 이는 사실상 또 다른 버퍼 요청임을 안전하게 결론 내릴 수 있습니다.

### 이벤트 체인

| ![nested CPU dispatch chain in PyTorch profiler: ProfileStep, matmul_add, aten::matmul, aten::mm](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cpu-nests.png) |
| :--: |
| 그림 12: 디스패치의 체인 |

그림 12에서 중첩된 CPU 호출을 볼 수 있습니다. 이는 디스패치 체인이 실제로 어떻게 보이는지 이해하는 중요한 시각화입니다.

우리는 profiling 단위를 캡슐화하는 `ProfileStep#<id>`로 시작합니다. 단계에 주석을 달았기 때문에 `matmul_add` 행이 보이고, `matmul_add`는 행렬 곱과 행렬 덧셈의 두 개의 `aten` 호출로 구성됩니다.

`aten::matmul`은 사용자 facing PyTorch의 matmul 호출이 도착하는 [ATen-level](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen) 디스패치이며, `aten::mm`는 2D 매트릭스-매트릭스 곱 백엔드입니다.

매트릭스에 배치 축(batch axis)을 추가하면 PyTorch가 `aten::bmm`(배치된 매트릭스 곱)을 호출하는 방식이 매우 흥미롭습니다. 잠깐 벗어나 `aten::bmm`가 작동하는 것을 살펴봅시다.

```diff
- x = torch.randn(args.size, args.size, device=device, dtype=dtype)
- w = torch.randn( args.size, args.size, device=device, dtype=dtype)
- b = torch.randn(args.size, args.size, device=device, dtype=dtype)

+ # adding a batch size of 8
+ x = torch.randn(8, args.size, args.size, device=device, dtype=dtype)
+ w = torch.randn(8, args.size, args.size, device=device, dtype=dtype)
+ b = torch.randn(8, args.size, args.size, device=device, dtype=dtype)

```


| ![PyTorch profiler trace showing aten::matmul dispatching aten::bmm for 3D batched tensors](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/bmm.png) |
| :--: |
| 그림 13: 배치된 행렬 곱 |

그림 13에서 입력에 배치 축을 추가하면, `aten::matmul`은 이제 `aten::bmm`와 함께 여러 선행 CUDA 런타임 호출들을 포괄합니다(이전에는 `aten::mm`였음). 이는 cuBLAS가 프로그램에 맞는(가장 적합한) 커널을 디스패치하기 위해 필요한 휴리스틱을 암시합니다.

> 앞으로의 글에서 특별히 언급되지 않는 한 간단한 2D 행렬로 작업합니다.

### 왜 matmul에 추가 CUDA 런타임 호출이 있나요?

| ![CPU lane showing cudaOccupancyMaxActiveBlocksPerMultiprocessor preceding the matmul cudaLaunchKernel](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/cudaoccupancy.png) |
| :--: |
| 그림 14: matmul 커널 런치 전 CUDA 점유도 질의가 실행됩니다 |

우리는 `aten::mm`에 두 개의 CUDA 런타임 호출이 있고, 구체적으로 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`(그림 14에 박스 처리)와 `cudaLaunchKernel`가 있는 반면, `aten::add`에는 `cudaLaunchKernel`만 있다는 것을 확인합니다.

`cudaOccupancyMaxActiveBlocksPerMultiprocessor`은 계획(plan) 호출이며 순수하게 CPU 측에 있습니다. "주어진 커널 함수, 선택된 블록 크기, 선택된 동적 공유 메모리 크기"를 바탕으로 이 커널의 블록이 한 SM에 얼마나 동시에 존재할 수 있는지 묻습니다."

이로써 matmul에 대한 계획이 필요한 이유와 add에는 왜 필요한지 의문이 생깁니다.

이를 이해하려면 커널의 자원 footprint를 살펴봐야 합니다. GPU 커널을 클릭하면 해당 커널의 자원 발자국을 확인할 수 있습니다.

| ![cuBLAS matmul kernel resource footprint: registers, shared memory and block size in Perfetto](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/matmul-footprint.png) | ![elementwise add CUDA kernel resource footprint with 32 registers and zero shared memory](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/add-footprint.png) |
| :--: | :--: |
| 그림 15: 매트릭스 곱 footprint | 그림 16: 더하기 footprint |

그림 15에서 매트릭스 곱의 경우 `registers per thread`와 `shared memory`가 행렬 크기에 따라 동적입니다. cuBLAS는 수백 가지의 커널 변형을 제공하며, 각 변형은 런타임 정보가 필요합니다. 점유도(occupancy) 질의는 그 휴리스틱의 일부입니다. 개념적으로 GPU 가속 매트맷을 [working on independent tiles](https://alvinwan.com/how-to-tile-matrix-multiplication/)로 생각할 수 있습니다: 타일의 수와 각 타일의 크기는 행렬과 하드웨어에 따라 달라집니다. 현대 알고리즘은 더 복잡하지만 여전히 좋은 참조 프레임워크입니다.

그림 16에서 더하기의 발자국은 32개의 레지스터와 공유 메모리 0으로 되어 있습니다. 이는 직관적으로 맞아떨어집니다. 쿼리할 하드웨어 자원이 점유율을 제한하지 않기 때문에 조회할 것이 없습니다. 이 커널은 설계상 자원-가볍습니다.

> [!NOTE]
> 이 정보를 추적을 읽을 때 빠른 진단으로 사용할 수 있습니다. CPU 레인에서 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`를 찾아보세요. 각 발생은 일반적으로 GEMM(GEneral Matrix Multiplication), 컨볼루션(conv) 등의 대형 커널을 표시합니다. 선행 점유도 질의가 없는 커널은 PyTorch가 기계적으로 띄우는 원소별/감소 연산 커널들입니다.

### 왜 cudaDeviceSynchronize가 이렇게 크나요(~1.78 ms)?

`cudaDeviceSynchronize`은 이 디바이스에서 모든 GPU 작업이 완료될 때까지 CPU를 차단합니다. 프로파일러는 활성 창의 끝에서 이 동기화를 발생시켜 이벤트를 플러시합니다. 이 없으면 커널 시간은 누락될 수 있습니다.

1.78 ms의 동기화가 실제 GPU 작업 26 µs를 포함한다는 것은 이 런이 98% 아이들(idle)였음을 말합니다. 이는 전형적인 오버헤드-바운드 증상입니다.

## 4096x4096 추적 {#section-3}

위의 프로파일러 표 분석에서 이미 알 수 있듯이, 알고리즘에 더 큰 행렬을 제공하면 오버헤드-바운드 영역에서 벗어나 계산 바운드 상태로 이동합니다.

명령을 실행하고 추적을 더 깊이 파고들어 봅시다.

```bash
uv run 01_matmul_add.py --size 4096 --warmup
```


### 같은 커널이 다른 것들보다 더 오래 걸리는 이유는?

| ![4096x4096 bf16 matmul kernel timings varying across profiler steps on the same GPU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/kernel-time.png)|
| :--: |
| 그림 17: 입력이 같아도 하나의 matmul 커널이 다른 커널들보다 더 오래 실행됨 |

그림 17에서 `ProfileStep#3`의 matmul 커널이 GPU에서 다른 단계들보다 더 오래 걸리는 것을 확인합니다. 주목할 점은 다른 커널들은 동일하게 런치되었다는 점인데, 이는 cuBLAS 휴리스틱이 작용하지 않았다는 뜻입니다. 스케줄링 간격이 없고, CPU 런치도 정상이며, 이는 프로파일러의 인공물도 아닙니다.

그림 17의 이 추적은 이상화된 예에서 놓치기 쉬운 한 가지를 잘 보여 줍니다: 커널 런타임은 일정하지 않으며, 동일한 코드와 데이터에 대해서도 하드웨어 환경에 따라 다를 수 있습니다.

이를 좀 더 구체적으로 보이게 하기 위해 스크립트를 약간 수정합니다. 20번의 반복을 실행하고 각 단계의 시간을 기록합니다.

```diff
- schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
+ schedule = torch.profiler.schedule(wait=0, warmup=0, active=20, repeat=1)

- for _ in range(5):
+ for _ in range(20):
```


| ![PyTorch profiler trace of 20 matmul iterations showing kernel runtime variance](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/20-iters-kernels.png) |
| :--: |
| 그림 18: 20회 반복에서 같은 matmul 커널이 서로 다른 속도로 실행됨 |

그림 18에서도 비슷한 결과가 나타납니다. 각 커널이 정확히 같았음에도 시간은 다르게 측정됩니다. 서로 다른 계산 시간의 원인은 여러 가지로 돌릴 수 있습니다:

- GPU의 아이들 모드에서의 클럭 변화 및 부스트
- GPU의 발열
- GPU의 전력 관리
- 드라이버 측의 정리 작업

평균만 본 독자는 매트릭스 곱이 약 1 ms(5의 평균이 1084 µs) 걸린다고 결론지을 수 있습니다. 그러나 추적을 본 독자는 매트릭스 곱이 약 580 µs 정도 걸리다가 GPU가 간섭할 때만 늘어난다고 볼 수 있습니다. 이 두 가지는 아주 다른 사고 모델이며, 정답은 단 하나뿐입니다.

## Let’s see some torch compile at work {#section-4}

`torch.compile`와 함께 작업하는 것은 항상 우리를 놀라게 합니다. 보통은 일반적인 eager PyTorch 코드를 작성하지만, PyTorch는 텐서가 많은 영역을 포착해 그래프화하고 이를 최적화하고 생성된 코드를 실행합니다. 기본 백엔드는 보통 `TorchInductor`이고, 광범위한 파이프라인은 다음과 같습니다:

1. `TorchDynamo`가 Python 실행을 FX 그래프로 포착합니다
2. gradients가 관여될 때-forward/backward 그래프를 준비합니다 `AOTAutograd`
3. 그래프를 최적화된 CPU 또는 GPU 코드로 낮춥니다 `Inductor`

이 섹션에서는 컴파일에 대해 다루고 프로파일러 트레이스를 살펴봅니다.

```bash
uv run 01_matmul_add.py --size 4096 --warmup --compile
```


`args.compile` 플래그는 다음 코드를 트리거합니다:

```py
def fn(x, w, b):
  return torch.add(torch.matmul(x, w), b)

fn = torch.compile(fn) if args.compile else fn
```


| ![torch.compile region highlighted in a PyTorch profiler trace, showing TorchDynamo and Inductor frames](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/compilation-region.png) |
| :--: |
| 그림 19: 컴파일된 영역이 추적에서 TorchDynamo 및 Inductor 프레임으로 나타납니다 |

그림 19에서 새 CPU 행들이 `Torch-Compiled Region: 0/0`로 명명되어 있으며, 이는 사용 중인 컴파일된 함수들을 가리킵니다.

### matmul과 add 커널을 하나로 융합했습니다(strip)?

| ![Compiled trace showing aten::addmm replacing the eager aten::add and aten::mm pair](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/fused-ops.png) |
| :--: |
| 그림 20: 컴파일된 실행이 단일 `aten::addmm`를 디스패치합니다 |

그림 20을 보며 실제로 곱셈과 덧셈 연산이 하나로 융합됐는지 묻습니다.

이는 그래프 수준의 연산 융합입니다. Inductor가 우리의 `torch.add(torch.matmul(x, w), b)`를 받아 하나의 `aten::addmm(b, x, w)` 호출로 재작성했습니다. 주의할 점은 새로운 융합된 CUDA 커널을 만들어 내지 않았다는 점입니다. 실제 GPU 작업은 여전히 `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_nn`이며, 같은 eager 모드의 cuBLAS 커널을 사용합니다. 따라서 이 '융합'은 디스패처 수준의 것이지 커널 수준의 융합은 아닙니다.

> [!NOTE]
> PyTorch는 두 단계로 수행한 것을 하나의 곱셈-더하기로 바꾸는 [`torch.addmm`](https://docs.pytorch.org/docs/2.12/generated/torch.addmm.html) 함수를 제공합니다. 이 함수의 추적을 살펴보고 아래 코멘트에 관찰 내용을 남겨 주세요!

### torch.compile의 런타임 아키텍처

이론적으로 함수들을 컴파일하면 어떤 일이 일어나는지 알고 있는 것도 중요하지만, 이를 실제로 보는 것도 중요합니다. `torch.compile`의 런타임 아키텍처를 반영하는 CPU 측 계층 구조를 살펴봅시다.

**TorchDynamo Cache Lookup**은 Dynamo가 현재 호출이 동일한 입력 형태, 데이터타입, 디바이스, 텐서 메타데이터로 컴파일된 것과 여전히 일치하는지 확인하는 지점입니다. 만약 무엇인가가 다르면 Dynamo는 다시 컴파일합니다. 이 비용은 매 호출마다 지불됩니다, 컴파일 이후에도 말이죠.

**Torch-Compiled Region**은 컴파일된 버전에 '들어가는' 래퍼입니다. **AOTDispatcher Runtime Wrapper Prologue**는 AOT Autograd의 런타임 래퍼입니다. 여기서는 그래디언트가 필요 없더라도, AOTDispatcher가 텐서 메타데이터와 뷰 추적을 처리하며 `requires_grad`가 참일 경우 역전파를 설정합니다.

**## Call CompiledFxGraph <hash>**는 실제로 생성된 코드가 실행되는 위치입니다. 'CompiledFxGraph' 뒤의 문자열은 FX 그래프의 콘텐츠 해시이며, 세 단계 모두에서 동일하므로 캐시 적중을 확인할 수 있습니다.

> [!TIP]
> 이 해시로 키가 부여된 디스크상의 생성된 코드를 찾아볼 수 있으며, Inductor가 실제로 생성한 Triton/C++를 읽고 싶을 때 유용합니다.

### CUDA 런치가 절반으로 줄어들까요?

| ![compiled matmul trace showing Memcpy DtoD and GEMM kernels launched per step](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/torch-profiler/memcpy.png) |
| :--: |
| 그림 21: 각 컴파일 단계는 여전히 두 개의 GPU 커널을 런치하며, 디바이스-투-디바이스 memcpy와 GEMM을 포함합니다 |

그림 21의 추적에서 단계당 하나의 `cudaLaunchKernel`만 보인다는 사실에 매우 기뻤습니다. 이 관찰은 GPU 추적에서 보던 것과 정면으로 모순되었습니다. 단계당 두 개의 커널이 런치되었으며, 이는 `Memcpy DtoD (Device -> Device)`와 GEMM이었습니다. CPU 추적으로 돌아가 보니 `cudaMemcpyAsync` 디스패치를 전혀 놓쳤다는 것을 알았습니다.

`addmm`는 `out = α·A·B + β·C`를 계산하고, cuBLAS의 GEMM-편향 추가 에필로그가 바이어스가 이미 포함된 대상 버퍼에 기록합니다. 에필로그는 GEMM 뒤에 일어나는 모든 작업으로 생각할 수 있습니다. 딥러닝 세계에서는 활성화, 바이어스 추가, 정규화 등과 같은 GEMM-에필로그를 지속적으로 만들어 냅니다. 이로 인해 cuBLAS GEMM-with-<specific epilogue> 커널 변형이 존재합니다.

> [!NOTE]
> 서로 다른 `mode`를 `torch.compile`에 사용하면 다른 커널 변형이 런치되는 것을 발견할 수 있습니다. 직접 시도해 보고 아래에 관찰 내용을 남겨 주세요!

그래서 Inductor가 생성한 코드는 다음을 수행합니다:

- `out = copy(C)` ← DtoD memcpy(32 MB, 약 33 µs 소요)
- `out = α·(A·B) + β·out` ← `α=β=1`의 GEMM에 바이어스 추가를 쓰기 단계로 융합

결과는 수학적으로 여전히 동일합니다. 바이어스 추가는 무료가 아니며, 먼저 memcpy를 한 번 수행하고 약간 더 많은 비용의 GEMM 에필로그를 수행합니다.

정말로 원하던 융합은 `x·w + b`(여기서는 `out = α·A·B + β·C`가) 추가 메모리 트래픽 없이 단일 커널로 축소되는 것이었지만 그렇지 않았습니다. Inductor는 두 개의 메모리 접촉 작업을 보존했으며, 바이어스 복사를 memcpy로, 덧셈을 GEMM 에필로그로 라벨링만 바꿨습니다.

완전히 융합된 구현은 memcpy를 건너뛰게 합니다. 이는 FlashAttention 스타일의 직작성 커널이 하는 일이며, Inductor가 Triton 코드생성을 통해 할 수 있는 일입니다. 그러나 `4096×4096 bf16 matmul`의 경우 Inductor는 'cuBLAS를 사용하고 바이어스를 에필로그로 처리'하는 것이 최선의 경로라고 판단한 것으로 보입니다.

### CPU 오버헤드가 올라갔고, 내려가지 않았다

 eager 실행과 컴파일된 실행을 비교할 때 놓치기 쉬운 점은 다음과 같습니다:

| 단계 | eager 소요(ms) | 컴파일 소요(ms) |
| :--: | :--: | :--: |
| #2 | 0.1 | 0.2 |
| #3 | 0.07 | 0.1 |
| #4 | 0.07 | 0.1 |

컴파일은 매 단계 CPU 측에서 대략 두 배 더 비쌉니다. 이는 Dynamo > AOTAutograd > Inductor 스택 전체를 거치고, 이미 존재하는 `aten::addmm` 디스패치를 추가로 거치기 때문입니다. 컴파일 파이프라인은 수십 개의 연산(op)을 가진 ML 모델에 맞춰 설계되어 있어, 호출당 오버헤드가 다년간 amortize 되지만(단일 op에 대해서는 비용이 증가합니다).

> [!TIP]
> `torch.compile`에는 `mode` 인수가 있습니다. 독자가 문서를 읽고 CPU 오버헤드를 낮출 수 있는 `mode`를 찾아보는 과제를 떠안아 보기를 권합니다. 🤗

## Trace reading cheatsheet {#section-5}

우리가 살펴본 패턴에 대한 빠른 참고입니다. 아이디어는 이 패턴을 추적에서 보면 보통 이런 뜻이라는 것입니다.

### Profiler table

| What you see | What it usually means |
| :-- | :-- |
| `Self CPU time total` ≫ `Self CUDA time total` (CPU in ms, GPU in µs) | 오버헤드-바운드. CPU가 디스패칭에 더 많은 시간을 쓰는 반면 GPU가 계산에 소비하는 시간은 작습니다. 작업 규모를 키우거나 호출을 융합하세요. |
| `Self CPU time total` ≈ `Self CUDA time total`, both in ms | 계산-바운드. GPU가 병목이며, 일반적으로 원하는 상태입니다. |
| 한 이벤트가 `CUDA total`를 지배 | 핫스팟입니다. 최적화 시작점으로 삼으세요. |
| 한 이벤트가 큰 `# of Calls`를 가짐 | 각 호출이 저렴하더라도 잠재적 병목일 수 있습니다. 융합 또는 배치 가능한지 확인하세요. |
| `CPU total` ≫ `Self CPU` 한 줄에 | 자식들에 비용의 대부분이 있습니다. 중첩된 이벤트를 파고드세요, 부모를 넘어서.

### CPU lane

| What you see | What it usually means |
| :-- | :-- |
| First `ProfileStep` 훨씬 폭이 큼 | 콜드 스타트 오버헤드: 작업 공간 할당, cuBLAS 휴리스틱, 느린 모듈 로딩. 워밍업 반복과/또는 스케줄러의 `warmup` 인자를 추가하세요. |
| `record_function("...")` 시작과 내부의 첫 번째 `aten::*` 사이의 큰 간격 | 같은 콜드 스타트 비용, 확대해 본 것일 뿐. 주석이 들어갔지만 디스패치는 아직 일어나지 않았습니다. |
| `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 전에 `cudaLaunchKernel` | 무거운, 적응적으로 시작되는 커널(GEMM, conv 등). cuBLAS가 SM에 몇 개의 블록이 들어갈 수 있는지 Driver에 묻고 커널 변형을 고릅니다. |
| 선행 점유도 질의 없이 `cudaLaunchKernel`가 있음 | 원소별 또는 축소 커널로, 고정된 자원-경량한 발자국을 가집니다. 계획할 필요가 없습니다. |
| 활성 창 끝에서 길게 남는 `cudaDeviceSynchronize` | 프로파일러가 이벤트를 플러시하는 시간. 시간은 주로 GPU가 대기하느라 걸리는 시간이지 실제 CPU 비용은 아닙니다. 아주 작은 GPU 작업에 대한 동기화(sync)도 전형적인 오버헤드-바운드 증상입니다. |
| 당신이 작성하지 않은 `cudaMemcpyAsync` | 보통은 은밀한 디바이스-투-디바이스 복사입니다. `addmm`가 바이어스로 버퍼의 대상 버퍼를 채울 때 생깁니다. |

### GPU lane

| What you see | What it usually means |
| :-- | :-- |
| GPU 레인에서 `Activity Buffer Request` | 프로파일러가 자체 이벤트 버퍼를 할당/재충전하는 중입니다. 첫 번째 것은 보통 초기 CPU↔GPU 레인 오프셋을 차지합니다. |
| 하나의 스텝 안에서 두 커널 사이의 간격 | 실행 중 추가 버퍼 요청일 가능성이 큽니다. 더 많은 반복을 실행해 확인하세요: 한 번만 나타나면 프로파일러 문제이고 코드의 문제가 아닙니다. |
| 여러 스텝에서 같은 커널의 타이밍이 다름 | GPU 클록, 열 관리, 전력 관리, 드라이버 정리 작업 등으로 인한 차이일 수 있습니다. 평균만 보지 말고 추적을 읽으세요. |
| `ampere_bf16_s16816gemm_...` 같은 커널 이름 | 실제 cuBLAS의 매트릭스 곱 작업입니다. 같은 모양/데이터타입에서 커널 이름은 일반적으로 eager 모드와 compiled 모드에서 동일합니다. |
| `Memcpy DtoD`가 GEMM 직전에 바로 있음 | 바이어스 복사를 위한 에필로그, 즉 에필로그 설정으로의 처리입니다. "융합"은 디스패처 수준의 것이고 커널 자체의 융합은 아닙니다. |

### Dispatch chain

| What you see | What it usually means |
| :-- | :-- |
| `ProfileStep#N` → `<record_function name>` → `aten::*` → `aten::mm` / `aten::bmm` / `aten::add` | 전형적인 중첩 호출 계층. Self 시간은 자식을 제외하고, Total 시간은 자식을 포함합니다. |
| `aten::matmul`가 `aten::mm`로 해석 | 2D × 2D 매트릭스 곱. |
| `aten::matmul`가 `aten::bmm`(추가 CUDA 런타임 호출 포함)로 해석 | 3D+ 텐서의 배치 매트릭스 곱. cuBLAS가 변형을 고르는 데 더 많은 휴리스틱 작업을 수행합니다. |
| `aten::addmm(b, x, w)`가 별도의 `aten::add` + `aten::mm` 쌍 대신 | 디스패처 수준의 Operator 융합. GPU 커널은 여전히 같은 GEMM이며 바이어스 추가는 에필로그로 접히고 있습니다. |

### torch.compile

| What you see | What it usually means |
| :-- | :-- |
| CPU 레인에 있는 한 줄의 `Torch-Compiled Region: K/M` | 컴파일된 함수 안에 있습니다. |
| 매 스텝마다 `TorchDynamo Cache Lookup` | Dynamo가 캐시된 컴파일과 입력 형태/데이터타입/디바이스가 일치하는지 확인합니다. 컴파일 이후에도 매 호출마다 비용이 발생합니다. |
| gradients가 없어도 `AOTDispatcher Runtime Wrapper Prologue` | AOTAutograd의 런타임 래퍼가 항상 스택에 있으며 텐서 메타데이터 및 뷰 추적을 처리합니다. |
| 같은 해시로 스텝 간 `## Call CompiledFxGraph <hash>`가 이어짐 | 생성된 코드의 캐시 적중. 생성된 소스는 `/tmp/torchinductor_<user>/fxgraph/<hash>` 아래에 있습니다. |
| 작은 연산에 대해 규모가 큰 Op에서의 per-step CPU 시간이 eager보다 높음 | 예상됩니다. Dynamo → AOTAutograd → Inductor 스택은 다수의 연산에 대해 보상을 받는 과세입니다. |

## 결론 {#section-6}

우리는 아주 작은 `matmul + add`로 시작해 PyTorch 프로파일러를 읽는 방법을 배우는 핑계로 삼았습니다. 이 과정을 통해 더 큰 작업에 잘 적용되는 몇 가지 사고 모델을 얻었습니다. 이것은 **Profiling PyTorch** 시리즈의 첫 번째 글이었습니다. 이어지는 글들에서 이 두-연산 토이를 점차 벗어나 복잡도의 사다리를 올라가며 더 큰 구성 요소들을 살펴보고 궁극적으로 실제 모델까지 다룰 예정입니다.

마지막으로, 초안 초기에 리뷰를 해주신 [Noe Flandre](https://huggingface.co/NoeFlandre), [Suvaditya Mukherjee](https://huggingface.co/suvadityamuk), 그리고 [Vidit Ostwal](https://huggingface.co/ViditOstwal)께 감사드립니다!
