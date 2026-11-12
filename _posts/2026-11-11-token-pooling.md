---
layout: post
title: "멀티 벡터 임베딩 모델을 위한 토큰 풀링 인식 학습"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/train-sentence-transformers/st-hf-thumbnail.png
image: assets/images/blog/posts/2026-11-11-token-pooling/thumbnail.png
authors:
  - user: tomaarsen
slug: "token-pooling"
source_url: "https://huggingface.co/blog/token-pooling"
source_published_date: "2026-11-11"
source_published_at: "2026-11-11T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Token Pooling-Aware Training for Multi-Vector Embedding Models](https://huggingface.co/blog/token-pooling)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/token-pooling -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 멀티 벡터 임베딩 모델을 위한 토큰 풀링 인식 학습

멀티 벡터 임베딩 모델은 쿼리의 각 토큰에 문서에서 일치하는 항목을 찾을 수 있는 기회를 제공합니다. 따라서 검색에서는 매우 표현력이 뛰어나지만, 문서 벡터는 빠르게 늘어납니다. 수백만 개의 문서로 구성된 코퍼스에는 수억 개의 벡터가 포함될 수 있습니다!

토큰 풀링은 유사한 토큰 벡터를 병합하여 이러한 표현을 더 작게 만듭니다. 하지만 더 공격적으로 압축할수록 유용한 정보를 잃기 쉬워집니다. 풀링 인식 학습은 벡터가 병합될 때 중요한 정보를 보존하도록 인코더를 학습시켜, 동일한 저장공간 예산으로 더 나은 검색을 수행할 수 있도록 합니다.

이 글에서는 풀링의 작동 방식과 Sentence Transformers를 사용해 압축 검색을 위한 모델을 학습하는 방법을 설명하고, 그 결과로 나타나는 품질과 저장공간 간의 trade-off를 살펴보겠습니다. `pool_factor=4`을 사용하면, 일반적으로 학습한 모델은 문서 벡터를 **4x** 줄이면서 풀링하지 않은 NanoBEIR nDCG@10의 94.6%를 유지합니다. 토큰 풀링 인식 학습을 사용하면 동일한 저장공간 예산에서 이 유지율을 97.8%까지 높일 수 있습니다! 방법을 살펴보겠습니다.

> [!NOTE]
> 이 글은 Stefan Josef의 [Learn to Pool](https://stefan-jo.github.io/learn-to-pool/downloads/paper.pdf)과 [LightOn의 pooling regularization study](https://huggingface.co/blog/lightonai/lateon-hpool-regularization)에서 연구된 풀링 인식 학습을 기반으로 합니다. 두 연구에서는 대상 지정 학습과 adaptive budget도 다룹니다. 여기서는 방법을 이해하고 짧고 재현 가능한 Sentence Transformers 예제로 적용하는 데 초점을 맞추겠습니다.

![Mean NanoBEIR nDCG@10 versus stored document vectors for pool factors 2, 3, 4, and 6 and unpooled retrieval, comparing ordinary and pooling-aware training.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/token-pooling/mve_token_pooling_results.png)

<p align="center"><em><small>Pooling-aware training improves retrieval quality at the same vector budget. Details below.</small></em></p>

## 목차 {#section-1}

* [멀티 벡터 검색에 이렇게 많은 벡터가 필요한 이유](#section-2)
* [토큰 풀링](#section-3)
* [토큰 풀링 인식 학습](#section-4)
* [학습한 모델 사용](#section-5)
* [평가 결과](#section-6)
* [압축 기법 결합](#section-7)
* [참고 문헌](#section-8)
* [코드 및 모델](#section-9)

## 멀티 벡터 검색에 이렇게 많은 벡터가 필요한 이유 {#section-2}

단일 벡터 임베딩 모델은 전체 문서를 하나의 벡터로 압축합니다. 반면 ColBERT와 같은 [멀티 벡터 모델](https://huggingface.co/blog/multi-vector-encoder)은 문맥화된 토큰 벡터 모음을 그대로 유지하여, 쿼리의 서로 다른 부분이 문서의 서로 다른 부분과 일치할 수 있도록 합니다.

예를 들어, 배터리 수명이 긴 노트북에 대한 쿼리는 문서의 한 부분에서 노트북에 관한 근거를 찾고, 다른 부분에서 배터리 지속 시간에 관한 근거를 찾을 수 있습니다. MaxSim 점수 계산에서는 각 쿼리 벡터가 가장 잘 일치하는 문서 벡터를 선택하고, 해당 점수를 모두 더합니다.

$$
\operatorname{MaxSim}(Q, D) = \sum_{q \in Q} \max_{d \in D} q^\top d
$$

[Fast-PLAID](https://github.com/lightonai/fast-plaid)에서 지원하는 2비트 또는 4비트 residual quantization을 사용하더라도 문서당 수백 개의 벡터를 저장하면 용량이 커집니다. residual coordinate당 4비트일 때 128차원 벡터 200개에는 문서당 12.8 KB의 양자화된 residual이 필요하며, 문서 100만 개에서는 12.8 GB가 됩니다. 각 문서를 50개 벡터로 풀링하면 동일한 차원과 정밀도에서 3.2 GB로 줄어듭니다.

중요한 점은 많은 문서 벡터가 서로 유사할 수 있다는 것입니다. 여러 토큰이 밀접하게 관련된 개념을 설명할 수 있으므로, 이들을 함께 표현할 기회가 생깁니다.

## 토큰 풀링 {#section-3}

[HierarchicalTokenPooling](https://sbert.net/docs/package_reference/multi_vector_encoder/modules.html#hierarchicaltokenpooling)은 [Ward hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)을 사용해 유사한 토큰 임베딩을 그룹화합니다. 토큰 벡터마다 하나의 그룹으로 시작한 뒤, 그룹 내 벡터가 얼마나 퍼져 있는지를 증가시키는 정도가 가장 작은 두 그룹을 반복해서 병합합니다. 이 퍼짐 정도는 각 벡터와 해당 그룹의 평균 간 제곱 거리의 합으로 측정되므로, Ward 방법은 유사한 벡터로 구성된 조밀한 그룹을 선호합니다.

이러한 병합은 점점 더 큰 그룹으로 이루어진 계층 구조를 만들며, 예산에 맞는 압축 수준을 선택할 수 있게 합니다. 그런 다음 각 그룹을 평균으로 표현하고, 결과 벡터를 다시 정규화하여 MaxSim 점수 계산에 사용할 수 있도록 합니다.

문서 길이에 비례하도록 압축을 수행할 수 있습니다.

```python
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

pooling = HierarchicalTokenPooling(pool_factor=4)
```


`pool_factor=4`을 사용하면 문서는 원래 벡터의 약 4분의 1을 유지하므로, 긴 문서는 비례적으로 더 많은 용량을 유지할 수 있습니다. 기본적으로 첫 번째 토큰은 병합 대상에서 보호되며, 풀링은 문서 임베딩에만 적용되고 쿼리 임베딩은 변경되지 않습니다.

### 풀링 예산 선택

`HierarchicalTokenPooling`은 세 가지 대체 제어 방식을 제공합니다.

| 설정 | 의미 |
| --- | --- |
| `max_tokens=16` | 보호된 토큰을 포함하여 문서당 최대 16개의 벡터를 사용합니다. |
| `pool_factor=4` | 각 문서의 원래 벡터 중 약 4분의 1을 사용합니다. |
| `distance_threshold=...` | 공유되는 Ward 병합 거리 임계값으로, 임베딩 geometry에 따라 벡터 수가 달라질 수 있습니다. |

문서 길이가 서로 다를 때 이 차이가 중요합니다. `pool_factor=4`을 사용하면 긴 문서는 벡터의 약 4분의 1을 유지할 수 있으므로, 콘텐츠에 따라 표현의 크기도 커질 수 있습니다. `distance_threshold`을 사용하면 더 많은 서로 다른 임베딩을 보존해야 할 때 벡터 수가 증가할 수 있습니다.

### Adaptive Pooling 보정

Adaptive pooling은 공유 거리 임계값을 사용하여 유사한 임베딩은 더 적극적으로 병합하고, 서로 다른 콘텐츠를 가진 문서에는 더 많은 벡터를 보존합니다. 이는 _풀링 인식 objective 없이 학습한 모델을 포함하여_ 모든 멀티 벡터 모델에서 사용할 수 있습니다.

임계값을 수동으로 선택하는 대신, 목표 평균 벡터 수에 맞게 보정할 수 있습니다. `calibration_documents`에 코퍼스를 대표하는 샘플이 있다면 다음과 같이 수행합니다.

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

model = MultiVectorEncoder("lightonai/LateOn")
calibration_embeddings = model.encode_document(calibration_documents)
threshold = HierarchicalTokenPooling.calibrate_distance_threshold(calibration_embeddings, target_tokens=16)
pooling = HierarchicalTokenPooling(distance_threshold=threshold)

# Apply pooling, targeting an average of 16 vectors per document
document_embeddings = model.encode_document(documents_to_index, token_pooling=pooling)
```


보정에는 문서 임베딩만 필요하며, 인덱싱 시 재사용할 수 있는 임계값이 반환됩니다. 목표는 평균값이므로 개별 문서에는 16개보다 많거나 적은 벡터가 할당될 수 있습니다. 다른 코퍼스에서는 실제 평균도 달라질 수 있습니다.

## 토큰 풀링 인식 학습 {#section-4}

일반적인 멀티 벡터 모델에서도 풀링은 유용할 수 있습니다. 문제는 문서에 검색에 중요한 구분이 포함되어 있지만, 벡터를 병합하면 그 구분을 복원하기 어려워질 때 발생합니다. 기존 학습 objective는 모델이 이러한 압축 표현으로 작동하도록 요구하지 않았습니다. 하지만 정확히 그렇게 학습시킬 수 있습니다.

[Matryoshka 임베딩](https://huggingface.co/blog/matryoshka)을 사용해 본 적이 있다면 이 학습 원리가 익숙하게 느껴질 것입니다. Matryoshka는 여러 임베딩 폭에서 retrieval objective를 적용하여 더 작은 prefix도 유용하게 유지되도록 학습합니다. 풀링 인식 학습은 토큰 벡터 수가 더 적은 표현에 objective를 적용합니다.

이 예제에서는 각 학습 배치가 두 가지 loss에 기여합니다.

1. 원래 문서 임베딩을 사용한 검색.
2. 문서를 원래 벡터의 약 4분의 1로 압축한 후의 검색.

두 loss를 평균냅니다.

$$
\mathcal{L} = \tfrac{1}{2}\mathcal{L}_{\mathrm{unpooled}} + \tfrac{1}{2}\mathcal{L}_{\mathrm{pooled}}
$$

첫 번째 항은 모델이 전체 표현으로 검색하도록 학습합니다. 두 번째 항은 풀링 후에도 유용한 일치가 남도록 토큰 임베딩을 배치하는 방법을 학습합니다. 두 objective는 동일한 인코더를 업데이트하므로, 압축 전후의 검색 결과에 대한 피드백을 모두 제공합니다.

Ward clustering은 어떤 토큰이 함께 속하는지에 대해 이산적인 결정을 내립니다. 학습 중 `HierarchicalTokenPooling`은 straight-through estimator를 사용하여 풀링된 벡터의 gradient를 해당 구성 토큰으로 전달하되, 클러스터 할당 자체에 대해서는 미분하지 않습니다. 그러면 인코더는 풀링된 표현으로 수행한 검색에서 발생한 오류를 통해 학습할 수 있습니다.

### Sentence Transformers의 Loss Wrapper

[`MultiVectorTokenPoolingLoss`](https://sbert.net/docs/package_reference/multi_vector_encoder/losses.html#multivectortokenpoolingloss)은 기존 멀티 벡터 검색 loss를 감싸고, 사용자가 선택한 pooler를 사용해 적용합니다.

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.losses import (
    CachedMultiVectorMultipleNegativesRankingLoss,
    MultiVectorTokenPoolingLoss,
)
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

model = MultiVectorEncoder("answerdotai/ModernBERT-base", similarity_fn_name="maxsim")
pooling = HierarchicalTokenPooling(pool_factor=4)

base_loss = CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=64)
loss = MultiVectorTokenPoolingLoss(
    model,
    base_loss,
    token_poolings=[None, pooling],
    weights=[0.5, 0.5],
)
```


`CachedMultiVectorMultipleNegativesRankingLoss`은 retrieval objective를 제공합니다. 각 쿼리가 배치 내 negative 문서보다 관련 문서에 더 높은 점수를 주도록 학습합니다. 기본적으로 `model.similarity`을 사용하며, 모델의 similarity 설정(이 예제에서는 MaxSim)과 kernel 설정을 따릅니다. `similarity_fn_name="meanmaxsim"`로 구성된 모델은 학습 중 MeanMaxSim도 사용합니다.

Gradient caching을 사용하면 더 많은 in-batch negative를 유지하면서 인코더가 작은 mini-batch를 처리할 수 있습니다. 제 학습 스크립트는 mini-batch 64개와 학습 batch size 256을 사용합니다.

`MultiVectorTokenPoolingLoss`은 compression objective를 추가합니다. 여기서 `None`은 한 변형이 원래 임베딩을 사용하고, `pooling`는 factor-4 변형을 제공한다는 의미입니다. 두 weight가 loss의 평균을 계산합니다. 풀링 변형은 계산된 토큰 임베딩을 재사용하므로, 각 추가 pooler는 해당 변형을 위해 별도의 encoder pass를 수행하지 않고 clustering 및 scoring 작업만 추가합니다.

pooler는 입력 임베딩을 통한 gradient를 보존하는 사용자 정의 `BaseTokenPooling` 구현일 수도 있습니다. 배포하려는 표현에 맞게 학습 objective를 조정할 수 있습니다.

factor-4 pooler를 선택한 배포 구성으로 교체하거나, 여러 pooler를 포함하고 각 loss에 대한 weight를 선택할 수 있습니다.

전체 [학습 스크립트](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py)에는 데이터셋, trainer, validation loss, 학습 중 수행하는 간단한 검색 평가가 포함되어 있습니다. repository를 checkout한 뒤 [uv](https://docs.astral.sh/uv/getting-started/installation/)로 두 변형을 로컬에서 모두 실행합니다.

```bash
uv run examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py --baseline
uv run examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py
```


스크립트가 학습 및 kernel dependency를 선언하므로 uv가 이를 대신 설치합니다. 호환되는 kernel은 자동으로 활성화됩니다.

[Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-quickstart)를 사용해 cloud GPU에서 학습할 수도 있습니다. CLI를 설치하고 Jobs를 실행하고 모델을 업로드할 수 있는 token으로 로그인합니다.

```bash
pip install -U huggingface_hub
hf auth login

hf jobs uv run \
    --flavor a10g-small \
    --timeout 4h \
    --secrets HF_TOKEN \
    -- https://raw.githubusercontent.com/huggingface/sentence-transformers/main/examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py
```


이 명령은 A10G GPU에서 4시간 제한으로 pooling-aware 변형을 실행합니다. `HF_TOKEN`을 secret으로 전달하면 학습이 끝났을 때 스크립트가 학습한 모델을 사용자의 Hub account에 업로드할 수 있습니다. 일반 loss baseline도 실행하려면 script URL 뒤에 `--baseline`을 추가하여 명령을 반복합니다.

### 선택적 커널 가속

[kernels](https://sbert.net/docs/multi_vector_encoder/usage/kernels.html)를 사용해 MaxSim scoring, Ward clustering 및 지원되는 backbone 연산을 가속할 수 있습니다. 학습 및 kernel dependency를 설치합니다.

```bash
pip install -U "sentence-transformers[train,kernels]"
```


위 학습 예제에서 loss를 구성하기 전에 model 및 pooler initialization을 교체합니다.

```python
model = MultiVectorEncoder(
    "answerdotai/ModernBERT-base",
    device="cuda",
    similarity_fn_name="maxsim",
    use_kernels=True,
)
pooling = HierarchicalTokenPooling(pool_factor=4)
```


계층적 pooler는 기본적으로 모델의 kernel 설정을 상속하므로, 모델에서만 kernel을 활성화하면 됩니다.

Sentence Transformers는 설정에 맞는 호환 kernel을 선택하고, 대체 구현을 사용할 수 없는 경우에는 기존 구현을 사용합니다. 예를 들어 [기본 MaxSim kernel](https://huggingface.co/kernels/sentence-transformers/late-interaction-kernels)은 메모리에 전체 token similarity matrix를 생성하지 않고 쿼리와 문서의 점수를 계산하지만, CUDA와 Triton이 필요합니다.

기본 [Ward clustering kernel](https://huggingface.co/kernels/sentence-transformers/ward-clustering)은 CPU 대신 CUDA에서 실행하여 계층적 clustering을 크게 가속합니다. 설정에 따라 SciPy보다 30~150배 빠릅니다.

## 학습한 모델 사용 {#section-5}

pooling-aware 모델을 로드하고 압축된 문서 임베딩으로 검색해 보겠습니다. kernel 설정은 모델과 함께 저장되지 않으므로, inference에서도 사용하려면 로드할 때 `use_kernels=True`을 추가합니다.

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

model = MultiVectorEncoder("tomaarsen/mve-ModernBERT-base-token-pooling")
pooling = HierarchicalTokenPooling(pool_factor=4)

queries = ["Which planet is known as the Red Planet?"]
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents, token_pooling=pooling)

print([len(embedding) for embedding in document_embeddings])
# [5, 6, 5]
print(model.similarity(query_embeddings, document_embeddings))
# tensor([[16.0037, 25.9879, 22.8159]], device='cuda:0')

# For comparison, the same model without token pooling:
# [18, 21, 19]
# tensor([[16.6115, 27.6060, 24.1766]], device='cuda:0')
```


pooler는 encoder의 forward pipeline 이후 문서 임베딩을 압축하여 전체 벡터 수를 58개에서 16개로 줄이고, 쿼리 임베딩은 변경하지 않습니다. Mars가 여전히 가장 높은 similarity score를 받고 Jupiter와 Venus가 그 뒤를 따르므로, 훨씬 적은 수의 문서 벡터를 저장해도 순위는 유지됩니다.

추론 시 `pool_factor`을 변경하여 동일한 학습 모델로 다른 저장공간 예산을 시험할 수 있습니다. 반복해서 사용하려면 `MultiVectorEncoder`을 초기화할 때 `token_pooling=pooling`을 전달하여 runtime default로 설정할 수도 있습니다. pooler의 `tasks`은 어떤 입력을 압축할지(예: 문서만) 계속 제어합니다.

## 평가 결과 {#section-6}

이를 조사하기 위해 `answerdotai/ModernBERT-base`에서 두 모델을 학습했으며, 각 모델은 새로운 128차원 토큰 projection을 사용했습니다.

- [tomaarsen/mve-ModernBERT-base-token-pooling-baseline](https://huggingface.co/tomaarsen/mve-ModernBERT-base-token-pooling-baseline): cached contrastive loss만 사용합니다.
- [tomaarsen/mve-ModernBERT-base-token-pooling](https://huggingface.co/tomaarsen/mve-ModernBERT-base-token-pooling): 위의 unpooled objective와 factor-4 objective로 cached contrastive loss를 감싸서 사용합니다.

두 실행 모두 동일한 120,000개의 MS MARCO triplet에 대해 1 epoch를 사용하고, seed 42와 동일한 initialization 및 optimization 설정을 적용했습니다. 학습 token cap은 300이며, 짧은 쿼리는 추가적인 문맥 쿼리 벡터를 제공하기 위해 최소 32개의 token position으로 확장됩니다. 두 실험에서 달라지는 것은 loss뿐이며, hyperparameter search는 수행하지 않았습니다.

두 모델을 NanoBEIR@13에서 평가하고 평균 nDCG@10(%)을 보고했습니다. 그림은 pool factor 2, 3, 4, 6에서 일반 학습과 pooling-aware 학습을 비교하고, 풀링하지 않은 검색도 함께 보여줍니다. 표에는 fixed cap과 adaptive threshold도 포함하여 학습의 이점이 다른 풀링 설정으로 어떻게 전이되는지 보여줍니다.

![Mean NanoBEIR nDCG@10 versus stored document vectors for pool factors 2, 3, 4, and 6 and unpooled retrieval, comparing ordinary and pooling-aware training.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/token-pooling/mve_token_pooling_results.png)

| 추론 시 풀링 | 평균 벡터 수 | 일반 모델 | Pooling-aware 모델 | 향상폭 (points) |
| --- | ---: | ---: | ---: | ---: |
| Unpooled | 193.22 | 51.84 | 52.39 | +0.55 |
| `pool_factor=2` | 96.86 | 50.10 | 51.74 | +1.64 |
| `pool_factor=3` | 64.74 | 49.70 | 51.45 | +1.76 |
| `pool_factor=4` | 48.68 | 49.03 | 51.23 | +2.20 |
| `pool_factor=5` | 39.04 | 48.13 | 50.28 | +2.15 |
| `pool_factor=6` | 32.62 | 47.07 | 49.85 | +2.78 |
| `max_tokens=64` | 54.76 | 49.27 | 50.89 | +1.63 |
| `max_tokens=32` | 29.58 | 48.25 | 50.12 | +1.87 |
| `max_tokens=16` | 15.60 | 46.61 | 49.26 | +2.65 |
| `max_tokens=8` | 7.99 | 43.12 | 46.61 | +3.48 |
| `max_tokens=4` | 4.00 | 37.22 | 43.00 | +5.78 |
| `distance_threshold` (보정됨) | 54.76 | 50.10 | 51.24 | +1.14 |
| `distance_threshold` (보정됨) | 29.58 | 48.38 | 50.96 | +2.58 |
| `distance_threshold` (보정됨) | 15.60 | 47.23 | 49.63 | +2.41 |

Adaptive pooling의 경우 각 encoder와 NanoBEIR corpus에 대해 별도의 threshold를 보정했습니다. unpooled 문서 임베딩을 사용하여 cap 64, 32, 16에서 평균 벡터 수가 일치하도록 했습니다. 짧은 문서로 인해 실제 평균은 해당 cap보다 낮아집니다.

학습한 `pool_factor=4`에서 nDCG@10은 49.03에서 51.23으로 향상되었으며, unpooled 검색보다 문서 벡터를 74.8% 적게 저장합니다. 이에 따라 score retention은 94.6%에서 97.8%로 증가합니다. 여기서 retention은 pooled 평균 nDCG@10을 각 모델 자체의 unpooled 평균으로 나눈 값입니다. Unpooled 검색도 51.84에서 52.39로 향상됩니다.

학습에는 전체 표현과 비례 풀링만 사용했지만, 이러한 향상은 fixed cap과 adaptive pooling에도 이어집니다. 문서당 평균 29.58개의 벡터를 사용하는 adaptive pooling에서는 문서 벡터를 84.7% 적게 저장하면서 pooling-aware 모델의 unpooled score 중 97.3%를 유지합니다. 일반적으로 학습한 모델은 동일한 예산에서 93.3%를 유지합니다.

## 압축 기법 결합 {#section-7}

인덱스 크기가 가장 큰 관심사라면 다음 옵션을 이 순서로 적용하겠습니다.

1. [Fast-PLAID](https://github.com/lightonai/fast-plaid) 등을 사용해 임베딩을 4, 2 또는 1비트로 양자화하여 coordinate당 저장공간을 줄입니다.
2. 토큰 풀링으로 문서 벡터 수를 줄입니다. 이 글에서 설명한 것처럼 pooling-aware fine-tuning을 선택적으로 적용할 수 있습니다.
3. [Matryoshka token embeddings](https://sbert.net/examples/multi_vector_encoder/training/matryoshka/README.html)으로 각 토큰 벡터의 차원을 줄입니다. 더 작은 폭에서도 검색 품질을 유지하도록 fine-tuning을 선택적으로 적용할 수 있습니다.

모델의 기본 차원이 128보다 크다면 Matryoshka truncation과 fine-tuning을 더 일찍 고려할 수도 있습니다. 이러한 접근 방식은 결합할 수 있으므로, 검색 엔진과 코퍼스를 사용해 결과 인덱스 크기와 검색 품질을 평가해 보세요.

`MultiVectorMatryoshkaLoss`은 truncation 이후에도 유용하게 유지되도록 토큰 임베딩을 학습합니다. 이를 `MultiVectorTokenPoolingLoss`과 결합하면 더 작은 차원과 더 적은 문서 벡터를 모두 학습할 수 있으며, 압축 설정 전반에서 encoder의 임베딩을 재사용할 수 있습니다.

[전체 학습 예제](https://sbert.net/examples/multi_vector_encoder/training/token_pooling/README.html)를 실행하고, 애플리케이션에서 중요한 예산을 선택한 뒤 동일한 조건의 일반 loss 실행과 비교해 보세요. 함께 제공되는 [evaluation script](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/training/token_pooling/evaluate_token_pooling.py)는 unpooled 검색, 64·32·16·8·4 벡터 cap, 그리고 2부터 6까지의 pool factor를 측정하므로, 자체 모델의 품질과 저장공간 간 trade-off를 그래프로 나타낼 수 있습니다.

학습 objective에 몇 줄을 추가하는 것만으로 더 작은 문서 표현을 모델이 적극적으로 사용하는 대상으로 만들 수 있습니다. 이를 통해 멀티 벡터 검색을 더욱 실용적으로 만들 수 있는 상당한 여지가 열립니다!

## 참고 문헌 {#section-8}

- Stefan Josef. [Learn to Pool: Lightweight Fine-Tuning for Flexible Multi-Vector Compression](https://stefan-jo.github.io/learn-to-pool).
- Benjamin Clavié, Antoine Chaffin, and Griffin Adams. [Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling](https://huggingface.co/papers/2409.14683).
- Antoine Chaffin. [After the party comes the free lunch: regularizing ColBERT models to enhance pooling capabilities and reduce index footprint](https://huggingface.co/blog/lightonai/lateon-hpool-regularization).
- Luyu Gao, Yunyi Zhang, Jiawei Han, and Jamie Callan. [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://huggingface.co/papers/2101.06983).
- [Matryoshka Embedding Models 소개](https://huggingface.co/blog/matryoshka).

## 코드 및 모델 {#section-9}

- [토큰 풀링 인식 학습 예제 및 README](https://sbert.net/examples/multi_vector_encoder/training/token_pooling/README.html)
- [전체 학습 스크립트](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/training/token_pooling/training_cached_contrastive.py)
- [NanoBEIR 평가 스크립트](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/training/token_pooling/evaluate_token_pooling.py)
- 예제 checkpoint:
    - [tomaarsen/mve-ModernBERT-base-token-pooling](https://huggingface.co/tomaarsen/mve-ModernBERT-base-token-pooling)
    - [tomaarsen/mve-ModernBERT-base-token-pooling-baseline](https://huggingface.co/tomaarsen/mve-ModernBERT-base-token-pooling-baseline)
- [Sentence Transformers와 호환되는 사전 학습 멀티 벡터 모델](https://huggingface.co/collections/sentence-transformers/multivectorencoder-integrations)
- [멀티 벡터 모델을 위한 Matryoshka 인식 학습](https://sbert.net/examples/multi_vector_encoder/training/matryoshka/README.html)
- [Fast-PLAID](https://github.com/lightonai/fast-plaid)
