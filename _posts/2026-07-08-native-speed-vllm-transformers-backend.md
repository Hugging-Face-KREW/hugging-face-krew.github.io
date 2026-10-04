---
layout: post
title: "네이티브-스피드 vLLM transformers 모델링 백엔드"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-07-08-native-speed-vllm-transformers-backend/thumbnail.png
authors:
  - user: hmellor
  - user: lysandre
slug: "native-speed-vllm-transformers-backend"
source_url: "https://huggingface.co/blog/native-speed-vllm-transformers-backend"
source_published_date: "2026-07-08"
source_published_at: "2026-07-08T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Native-speed vLLM transformers modeling backend](https://huggingface.co/blog/native-speed-vllm-transformers-backend)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/native-speed-vllm-transformers-backend -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 네이티브-스피드 vLLM transformers 모델링 백엔드

**TL;DR**: 다수의 LLM 아키텍처에서 이제 transformers vLLM 백엔드가 맞춤형 vLLM 구현만큼 빠르거나 더 빠릅니다. 모델 저자들이 그들의 transformers 구현을 자동으로 활용해 초고속 vLLM 추론을 무료로 얻을 수 있습니다.

```bash
# Upgrade the vllm pip package
uv pip install --upgrade vllm --torch-backend auto
```


transformers 라이브러리는 머신 러닝을 위한 **참조 모델링 라이브러리**가 되었습니다. 일관된 API를 통해 450개가 넘는 아키텍처를 지원하며, 모델 구현이 _자체 포함_이고 _이해하기 쉬운_ 것이 주된 목표로 설계되었습니다. transformers 코드의 흐름을 따라가면 기여자들이 아키텍처가 어떻게 작동하는지 배우고, 이를 vLLM, SGLang, MLX, llama.cpp 및 그 밖의 많은 프레임워크로 이식하는 일이 수월해집니다.

우리는 이 생태계에서 이 역할을 완전히 수용했고, 이를 더 쉽게 만들기 위해 많은 노력을 기울이고 있습니다. 이 방향의 큰 걸음은 작년에 vLLM에 모델링 백엔드로서 transformers를 통합한 것이었습니다. 이를 통해 모델 저자들은 포팅 없이도 vLLM 내부에서 transformers 모델(LLMs와 VLM 모두)을 실행할 수 있습니다. transformers는 모델링 코드를 제공하고, vLLM은 연속 배치 처리 및 커스텀 어텐션 커널과 같은 극도로 최적화된 추론 기법을 제공합니다.

이 통합은 이제 더 좋아집니다 🚀!

## 쇼케이스 {#section-1}

우리는 vLLM용 transformers 모델링 백엔드를 세 가지 매우 다른 Qwen3 모델에 대해 vLLM의 수작업으로 작성된 네이티브 구현과 정면 대결시키고 비교했습니다:

* 단일 GPU에서 4B 밀집 모델
* 텐서 병렬화를 통한 32B 밀집 모델
* 같은 8×H100 노드에서 데이터+전문가 병렬화를 통해 235B 매개변수 FP8 Mixture-of-Experts

| ![Pre and Post PR benchmarks with trasnformers vllm backend](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-backend/pre-post-pr.png) |
| :--: |
| 결과: 이제 transformers 모델링 백엔드는 이들 각각에서 네이티브 처리량에 필적하거나 이를 능가합니다. |

어떤 Hugging Face 모델이든 transformers 모델링 백엔드를 통해 실행하는 것은 단 하나의 플래그 — `--model-impl transformers`입니다. 이는 일반적인 병렬성 옵션과 함께 작동하므로 서빙 설정에는 아무 것도 변경되지 않습니다:

```bash
# Qwen3-4B dense, single GPU
vllm serve Qwen/Qwen3-4B --model-impl transformers

# Qwen3-32B dense, tensor-parallel across 2 GPUs
vllm serve Qwen/Qwen3-32B --model-impl transformers --tensor-parallel-size 2

# Qwen3-235B-A22B-FP8 MoE, data-parallel + expert-parallel across 8 GPUs
vllm serve Qwen/Qwen3-235B-A22B-FP8 --model-impl transformers --data-parallel-size 8 --enable-expert-parallel
# add --max-model-len 8192 if your node is memory constrained
```


_선형 어텐션을 사용하는 모델은 현재 지원되지 않지만 곧 지원될 예정입니다! Hub 저장소의 코드가 있는 커스텀 모델은 올바르게 작성되지 않았을 가능성이 있어 작동하지 않을 확률이 큽니다._

### 어떻게 측정했나

각 모델은 코드 경로 외에는 모든 면에서 동일한 세 가지 조건에서 비교됩니다:

1. **네이티브** — `--model-impl vllm`, vLLM의 수작업으로 작성된 모델(일치 기준치)
2. **이후** — `--model-impl transformers` _PR과 함께_
3. **이전** — `--model-impl transformers` _PR 없이_

전체 재현 가능한 런너는 gist로 제공됩니다: [`benchmark.sh`](https://huggingface.co/datasets/ariG23498/useful-scripts/blob/main/transformers-backend-vllm-benchmark.sh)

## 그래서, 새로 바뀐 점은? {#section-2}

vLLM용 transformers 모델링 백엔드는 추론의 병목으로 주로 _어텐션_에 집중했습니다. 런타임에 vLLM의 어텐션 구현을 주입함으로써, transformers 모델이 vLLM 엔진 내부에서 효율적으로 실행되도록 만들 수 있었습니다. 그러나 최대 추론 성능을 끌어내려면 배포의 차원이 많습니다. GPU 간 병렬화, 컴파일, 융합 커널 등 수많은 요소가 하드웨어를 최대한 활용해 초고속 추론을 달성하는 데 기여합니다.

| ![New model integration to transformers and vLLM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-backend/previous-pipeline.png) |
| :--: |
| 새 모델은 한때 transformers에 한 번, vLLM에는 커스텀 최적화와 함께 한 번 통합되었습니다. |

모델 저자들이 가장 최상의 성능을 원할 때는 여전히 커스텀 vLLM 구현을 작성하고 있었습니다.

| ![New model integrates to transformers, and is immediately available to vLLM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/vllm-backend/current-pipeline.png) |
| :--: |
| 새 모델은 한때 transformers에 통합되었고, 이제는 vLLM에서 네이티브 vLLM 구현 속도로 즉시 사용할 수 있습니다. |

vLLM용 최신 버전의 transformers 모델링 백엔드는 호환 가능한 아키텍처에 대해 런타임에 추론 특화 계층 융합을 동적으로 적용하여, 커스텀 코드 구현의 속도에 맞춥니다.

## 어떻게 작동하나요? {#section-3}

vLLM용 transformers 모델링 백엔드는 이제 `torch.fx`을 사용해 모델 그래프에 대한 정적 분석을 수행합니다. 이 과정은 최적화할 수 있는 알려진 패턴을 검색합니다. 패턴이 식별된 후에는 ast(추상 구문 트리)를 사용해 소스 코드를 조작하고 일부 연산을 제자리에 다시 작성합니다.

**무엇을 달성할 수 있을까요?**

* 다대일 매핑된 융합 연산은 (울트라) 최적화된 vLLM 커널로 매핑되며, Mixture-of-Experts(MoE) 모델의 Expert Parallelization(EP)에서 사용되는 연산들처럼 동작합니다.
* 다른 주요 융합 연산은 vLLM의 `MergedColumnParallelLinear` 및 `QKVParallelLinear`입니다. 이 블록들은 TP(텐서 병렬화)에 대한 병렬 계획을 추론할 수 있게 해 줍니다. 이 블록들이 decoder 블록 목록이 쉽게 식별되면 PP(파이프라인 병렬화) 계획도 추론될 수 있습니다.
* 조작된 모델은 여전히 완전하게 (Torch)로 컴파일 가능하며, `torch.compile` 및 CUDA Graphs를 거쳐 처리되며, 이는 전용 vLLM 모델 구현과 똑같은 방식으로 작동합니다.
* vLLM 모델 구현과 달리, Transformers 모델 구현은 학습에도 사용할 수 있습니다. 따라서 동일한 모델 코드를 학습/평가/RL 롤아웃에 사용할 수 있습니다.

위에서 보듯이, 이는 호환 가능한 모델에 대해 네이티브 vLLM 추론 속도를 제공합니다. 추론 최적화를 위한 한 줄의 코드도 작성할 필요가 없습니다.

> [참고]
> 이 최적화된 추론 방법들에 대해 자세히 다루고, 어떻게 모델을 이들에 적응시키는지 자세히 설명하는 상세한 블로그 글을 작성 중에 있습니다.

## 자료 {#section-4}

* [Transformers model definition](https://huggingface.co/blog/transformers-model-definition#a-model-definition-library)
* [Transformers modeling backend in vLLM](https://vllm.ai/blog/2025-04-11-transformers-backend)
* [Large scale serving](https://vllm.ai/blog/2025-12-17-large-scale-serving)
* [Torch FX](https://docs.pytorch.org/docs/2.12/fx.html)
* [Abstract syntax tree](https://docs.python.org/3/library/ast.html)
