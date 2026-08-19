---
layout: post
title: "멀티 벡터 (Late Interaction) 임베딩 모델과 Sentence Transformers"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/multi-vector-encoder/st-hf-lighton-thumbnail.png
image: assets/images/blog/posts/2026-08-18-multi-vector-encoder/thumbnail.png
authors:
  - user: tomaarsen
  - user: NohTow
slug: "multi-vector-encoder"
source_url: "https://huggingface.co/blog/multi-vector-encoder"
source_published_date: "2026-08-18"
source_published_at: "2026-08-18T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Multi-Vector (Late Interaction) Embedding Models with Sentence Transformers](https://huggingface.co/blog/multi-vector-encoder)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/multi-vector-encoder -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 멀티 벡터 (Late Interaction) 임베딩 모델과 Sentence Transformers

[Sentence Transformers](https://sbert.net/) 는 검색 보강 생성, 시맨틱 검색 등과 같은 응용에 사용하고 학습하는 임베딩 및 reranker 모델을 위한 Python 라이브러리입니다. v6.0 업데이트로 네 번째 모델 유형 `MultiVectorEncoder`이 추가되었으며, ColBERT 스타일의 Late Interaction 검색용입니다. 어떠한 [PyLate](https://github.com/lightonai/pylate) 체크포인트와 어떤 [Stanford-NLP ColBERT](https://github.com/stanford-futuredata/ColBERT) 체크포인트도 그대로 로드되며, 시각적 문서 검색용 [colpali-engine](https://github.com/illuin-tech/colpali) 모델도 Dense, Sparse, 그리고 reranker 모델에 대해 이미 사용하는 같은 익숙한 API를 통해 사용할 수 있습니다.

일반 임베딩 모델이 텍스트 전체를 하나의 벡터로 압축하는 반면, 멀티-벡터 모델은 **토큰당 하나의 벡터**를 유지하고 쿼리를 문서에 대해 MaxSim 연산자로 매칭합니다. 이는 단일 벡터가 평균화해 버려야 하는 토큰 수준의 매칭 정보를 보존하므로 일반적으로 더 강력한 검색을 제공하지만, 인덱스 크기가 커지는 대가가 있습니다. 또한 문장 텍스트 검색의 최첨단으로도 작용하며, 여기에 포함된 텍스트 쿼리가 페이지 이미지에 직접 매칭되고 OCR 단계가 앞뒤로 없다는 점이 특징입니다.

이 블로그 포스트에서는 이들 모델의 사용 방법을 보여드립니다: 다양한 체크포인트 포맷 로딩, 인코딩 및 스코어링, 검색 스택에의 연결, 페이지 이미지에서의 실행, 그리고 인덱스를 합리적으로 유지하는 방법. 아래 모든 내용은 일반 `pip install -U sentence-transformers`에서 실행됩니다.

<!--
> [!TIP]
> If you want to train your own multi-vector models, check out the companion blogpost: [Training and Finetuning Multi-Vector Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-multi-vector-encoder).
-->

## 목차 {#section-1}

* [멀티-벡터 모델이란 무엇인가?](#section-2)
* [설치](#section-3)
* [모델 로딩](#section-4)
* [질의 및 문서 인코딩](#section-5)
* [MaxSim으로 점수 산정](#section-6)
* [시맨틱 검색](#section-7)
* [검색 및 재정렬](#section-8)
* [인덱싱](#section-9)
* [시각적 문서 검색](#section-10)
* [오디오 검색](#section-11)
* [비디오 검색](#section-12)
* [해석 가능성](#section-13)
* [토큰 풀링](#section-14)
* [추론 속도 향상](#section-15)
* [모델 평가](#section-16)
* [PyLate 또는 colpali-engine에서 오신 분들](#section-17)
* [지원하는 모델](#section-18)
* [감사의 말씀](#section-19)
* [추가 리소스](#section-20)

## 멀티-벡터 모델이란 무엇인가? {#section-2}

밀집 임베딩 모델은 텍스트를 읽고 하나의 고정 크기 벡터를 반환합니다. 모델이 포착한 모든 정보는 그 벡터에 들어가야 하며, 유사도는 두 요약 간의 하나의 점곱으로 계산됩니다. 이 방식은 매우 잘 작동하지만, 특정 방식으로 손실이 발생하는 압축이 존재합니다: 희귀한 엔티티, 정확한 식별자, 또는 긴 구절의 핵심 조항 하나가 동일 벡터 내에서 공간을 차지하기 위해 경쟁합니다. 여러 요구사항이 한 번에 주어지는 질의는 동일한 한계에 부딪힙니다. 예를 들어 “나무 다리가 있는 초록 소파와 둥근 쿠션”의 경우, 네 가지 요소를 하나의 점으로 혼합해야 하므로 다리가 잘못된 초록 소파가 사실 요청한 소파와 가까이 매칭될 수 있습니다.

멀티-벡터 모델(일명 늦은 인터랙션 또는 ColBERT 스타일 모델, [ColBERT paper](https://arxiv.org/abs/2004.12832) 이후)은 그 압축을 건너뜁니다. 같은 트랜스포머를 실행하되 토큰 임베딩을 하나의 벡터로 풀링하는 대신 각 토큰 임베딩을 작은 차원으로 투영하고 모두를 유지합니다. 보통 128 차원이 되고, 9토큰 문서는 9x128 매트릭스가 됩니다.

질의와 문서 간의 상호작용은 스코어링 시점까지 연기되며, 이는 이름의 "늦은 인터랙션"의 근거가 됩니다. 크로스-인코더는 초기에 상호작용합니다: 두 텍스트가 함께 모델을 통과하므로 정확하지만, 새로운 질의마다 각 문서를 다시 인코딩해야 하므로 사전 계산이 거의 불가능합니다. 위의 Dense 임베딩 모델이 다루는 바이-인코더는 상호작용이 거의 없고(완성된 두 요약 간의 하나의 점곱), 이로 인해 컬렉션을 한 번 인코딩하고 빠르게 질의할 수 있습니다. 늦은 인터랙션은 그 사이에 위치합니다: 문서는 여전히 독립적으로 인코딩되어 오프라인 인덱싱이 가능하지만, 점수 산정은 모든 질의 토큰을 모든 문서 토큰에 대해 비교합니다. 이로써 두 요소가 더 많이 상호작용할 여지가 남습니다.

![Dense embedding versus multi-vector late interaction: a dense model encodes each text into one vector and scores with cosine similarity, while a multi-vector model keeps one vector per token and scores every query token against every document token with MaxSim](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/maxsim_explainer.gif)

### MaxSim 연산자

점수 산정은 MaxSim을 사용합니다: 각 질의 토큰에 대해 문서 토큰에 대한 가장 높은 유사도를 취한 후, 그 최대값들을 질의 전체에 걸쳐 합산합니다.

$$\text{MaxSim}(Q, D) = \sum_{Q_i \in Q} \max_{D_j \in D} Q_i \cdot D_j$$

토큰 임베딩이 L2로 정규화되어 있기 때문에, 위의 각 점곱은 `[-1, 1]`에서의 코사인 유사도이며, 전체 합은 `[-num_query_tokens, num_query_tokens]` 범위 안에 들어옵니다.

연산자를 소프트 얼라인먼트로 읽을 수 있습니다: 모든 질의 토큰은 가장 잘 설명해 주는 하나의 문서 토큰을 가리키며, 점수는 문서가 질의를 전반적으로 얼마나 잘 뒷받침하는지에 달려 있습니다.

정합은 어휘적일 필요가 없으며, 토큰 임베딩은 맥락화되어 있기 때문입니다. [`lightonai/mLateOn`](https://huggingface.co/lightonai/mLateOn)로 "Where do penguins live?"를 인코드하고 질의 토큰 `live`가 최상의 매치를 `inhabit`에서 0.94로 찾는 예를 보십시오. 이 토큰은 서로 문자도 공유하지 않는 단어입니다! 이는 어휘 기반 검색이 할 수 없는 점입니다. BM25 및 그 유사체는 용어 자체를 필요로 하므로 동의어와 의역은 그들을 지나갑니다. 물론 Dense 임베딩 모델도 그 격차를 메웁니다. 늦은 인터랙션이 추가하는 점은, 정확한 매치가 중요한 경우에도(Maximum 같은 경우) 그 토큰을 단독으로 남겨두는 것이며, 단일 벡터 모델이 모든 것을 다른 것으로 평균해 버려야 하는 상황에서도 동일합니다. 또한 1:N 관계가 아니라는 점도 흔합니다. 여러 질의 토큰이 자주 같은 문서 토큰에 정착하기 때문입니다.

### 얻는 이점과 비용

검색 품질이 향상됩니다. 특히 문서의 특정 부분이 관련성을 결정하는 질의에서, 위의 소파 예시처럼 각 요구가 고유의 증거를 찾는 다중 요구 질의에서, Dense 모델의 압축이 다른 분포에 대해 학습된 데이터에서 작동하는 경우 등에서 말이죠. 이 압축은 학습 데이터의 질의에서 배운 것이므로 모델은 필요한 부분만 남기고 나머지는 제거합니다. 생산 질의가 정확히 어떤 정보를 요구하는지 포함될 수 있습니다. 효과는 문서 길이가 길어질수록 커지는데, 더 많은 텍스트를 같은 고정 벡터에 담아야 하기 때문입니다.

비용은 인덱스 크기입니다. 토큰 하나당 벡터 하나를 저장하는 것은 문서당 벡터 하나를 저장하는 것보다 훨씬 더 많은 벡터를 필요로 하며, 차원 수가 작아진 것만으로 충분히 보정되지 않습니다. [`lightonai/LateOn`](https://huggingface.co/lightonai/LateOn)로 4,874개의 Natural Questions 패시지를 인코딩하면 608,414개의 토큰 벡터가 생성되었고 패시지당 평균 124.8개였습니다:

| 표현 방식 | 벡터 수 | 차원 수 | float32 크기 |
| --- | ---: | ---: | ---: |
| Dense, [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 4,874 | 384 | 7.5 MB |
| Dense, [`gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 4,874 | 768 | 15.0 MB |
| 다중 벡터, `LateOn` | 608,414 | 128 | 311.5 MB |

그 저장 용량은 MiniLM 인덱스의 약 42배에 달하며 패시지당 약 62 KiB에 이릅니다. 다만 인덱스는 보통 압축되며, 예를 들어 동일한 608,414 벡터가 [fast-plaid](#indexing) 인덱스로 92 MB를 차지합니다. PLAID는 벡터 자체 대신 중심 벡터와 양자화된 잔차를 저장하기 때문입니다. 규모를 보면, 4,874 패시지를 대상으로 한 4096차원의 Dense 모델인 [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) 은 약 80 MB가 필요하므로, 압축된 멀티-벡터 인덱스는 이미 운영 중인 Dense 인덱스와 같은 범위에 속합니다. 벡터 수를 줄이기 전에 [Token Pooling](#token-pooling)이 먼저 작동하고, [Retrieve and Rerank](#retrieve-and-rerank) 은 인덱스 자체를 만들지 않습니다.

[PyLate](https://github.com/lightonai/pylate) 는 이 포스트 전반에 걸쳐 등장하므로 간단히 정리합니다: Sentence Transformers 는 Dense 및 Sparse 모델을 다루었지만 Late Interaction 은 다루지 못했고, 그래서 [LightOn](https://huggingface.co/lightonai) 이 이를 보완하기 위해 PyLate 를 위에 구축하여 이 모델들이 필요로 하는 학습, 추론, 검색 파트를 더했습니다. 아래에서 로드하는 많은 내용은 그와 함께 학습되었고, LightOn 도 그 주위를 둘러싼 생태계를 구축하여 [fast-plaid](https://github.com/lightonai/fast-plaid)를 포함시키고, [Indexing](#indexing) 에서 Late-Interaction 인덱스를 제공합니다. v6.0 부터는 이 기능들이 Sentence Transformers 자체에 내재되어 있습니다.

트레이드오프를 염두에 두고, 이제 모델을 작동시켜 봅시다.

## 설치 {#section-3}

멀티-벡터 모델은 간단한 설치로 작동합니다:

```bash
pip install -U sentence-transformers
```


ColPali 스타일의 시각적 문서 검색을 위해서는 이미지 의존성도 필요합니다(모든 확장 기능은 [Installation](https://sbert.net/docs/installation.html) 를 참조하고, 멀티모달 지원은 일반적으로 [Multimodal Embedding & Reranker Models](https://huggingface.co/blog/multimodal-sentence-transformers) 를 참조하십시오):

```bash
pip install -U "sentence-transformers[image]"
```


> [!NOTE]
> Sentence Transformers v6.0은 `transformers` v5.x, `torch` 2.2+, 그리고 `huggingface-hub` v1.x를 필요로 합니다. 이들 중 어느 하나라도 더 낮은 버전으로 고정하면 먼저 업그레이드를 계획하십시오. 전체 변경 목록은 [Migration Guide](https://sbert.net/docs/migration_guide.html)를 참고하십시오.

## 모델 로딩 {#section-4}

멀티-벡터 모델 로딩은 다른 Sentence Transformers 모델 로딩과 똑같이 보입니다:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("lightonai/LateOn")
```


작동하는 모델을 찾으려면 Hub의 [`multi-vector` and `sentence-transformers` tags](https://huggingface.co/models?library=sentence-transformers&other=multi-vector) 태그를 찾으세요. 이 태그가 달린 어떤 모델이든 위의 로딩 방식으로 로드되며, PyLate 체크포인트로 시작했든, Stanford-NLP ColBERT 체크포인트였든, 또는 ColPali 계열의 시각적 문서 검색용 모델이었든 상관없이 동일하게 동작합니다. 우리는 이 태그를 작동하는 모든 모델에 붙이기 위해 생태계를 정비 중이며 목록은 계속 확장 중입니다.

그 아래에서, `MultiVectorEncoder` 는 수년간 발표된 포맷 각각을 읽습니다. 그래서 PyLate와 Stanford-NLP 체크포인트는 태그가 아직 추가되지 않았더라도 바로 로드됩니다:

```python
from sentence_transformers import MultiVectorEncoder

# Native Sentence Transformers checkpoints. PyLate builds on the same schema,
# so any PyLate checkpoint loads identically
model = MultiVectorEncoder("lightonai/LateOn")
model = MultiVectorEncoder("mixedbread-ai/mxbai-edge-colbert-v0-17m")
model = MultiVectorEncoder("LiquidAI/LFM2.5-ColBERT-350M", trust_remote_code=True)

# Any Stanford-NLP ColBERT checkpoint, detected via the `HF_ColBERT` architecture
# marker. The inline projection weight and the recipe come from `artifact.metadata`
model = MultiVectorEncoder("colbert-ir/colbertv2.0")
model = MultiVectorEncoder("answerdotai/answerai-colbert-small-v1")

# A bare transformer: a fresh random projection is appended, so training is required
model = MultiVectorEncoder("answerdotai/ModernBERT-base")
```


시각적 문서 검색 모델은 예외입니다. ColPali 계열 체크포인트는 colpali-engine 고유 포맷으로 제공되며 이는 Sentence Transformers가 활용할 수 있는 정보를 담고 있지 않으므로 로드하기 전에 저장소에 작은 구성이 필요합니다. 대부분의 작업은 이미 완료되어 병합되기를 기다리고 있습니다. 현재 상태와 오늘날 로드하는 방법은 [Supported Models](#supported-models)를 참조하십시오.

### 체크포인트 구성 내용 점검

멀티-벡터 모델은 체크포인트마다 다른 조합 매개변수를 담고 있습니다: 질의와 문서에 대한 마커 접두사, 길이 상한, 질의를 `[MASK]` 토큰으로 패딩하는지 여부, 그리고 점수 산정 시 건너뛸 토큰을 결정하는 토큰들. 이 모든 설정은 모듈 구성에 포함되어 있으므로 `print(model)` 은 로드한 내용을 정확히 보여줍니다. 아래는 원래 ColBERTv2 체크포인트로, 모든 질의를 정확히 32토큰으로 패딩하고 문서를 180에서 잘라냅니다:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("colbert-ir/colbertv2.0")
print(model)
"""
MultiVectorEncoder(
  (0): Transformer({..., 'document_length': 180,
                    'query_expansion': {'strategy': 'fixed', 'attend': False, 'token': None, 'length': 32}})
  (1): Dense({'in_features': 768, 'out_features': 128, 'bias': False, ...})
  (2): MultiVectorMask({'skiplist_words': ['!', '"', '#', ...], 'skiplist_tasks': ['document'], ...})
  (3): Normalize({...})
)
"""
print(model.prompts)
# {'query': '[unused0] ', 'document': '[unused1] '}
```


그건 전형적인 ColBERT 파이프라인입니다: 컨텍스트화된 토큰 임베딩을 생성하는 `Transformer`, 각 임베딩을 128차원으로 투영하는 토큰-레벨 `Dense`, 점수 산정 시 계산에 포함될 토큰을 결정하는 `MultiVectorMask`, 그리고 토큰-레벨 `Normalize`입니다. 다른 체크포인트들은 서로 다른 값을 채웁니다. `lightonai/GTE-ModernColBERT-v1` 은 같은 네 모듈을 `[Q] ` 및 `[D] ` 프롬프트와 함께 사용하며, 질의 확장 없이 상한은 48 및 300입니다.

대부분의 경우, 출시된 체크포인트 각각이 자체 구성을 갖추고 있어 이 부분을 건드릴 필요가 거의 없습니다. bare 백본으로 모델을 구성할 때에만 다루게 되는데, 이는 [Creating Custom Models](https://sbert.net/docs/multi_vector_encoder/usage/custom_models.html)에서 다루고 있습니다.

다만 한 가지 값은 당신의 데이터에 대해 확인해 보는 것이 가치가 있습니다. `document_length` 은 잘라내므로 그 기준을 넘는 것은 인덱스에 도달하지 못합니다. 예를 들어 LateOn의 상한이 300인 경우 662토큰 패시지는 273 벡터로 반환되고, 남은 부분은 그대로 사라진 셈입니다. 이들 체크포인트의 대다수는 짧은 패시지를 대상으로 훈련되었으므로 cap보다 긴 청크를 한 번의 호출에서 늘릴 수 있지만, 그 경우 학습 당시의 길이를 넘어 작동하게 되어 인덱스 크기가 대략 비례적으로 증가합니다. 멀티-벡터 모델은 이를 대체로 잘 견딥니다. 상위 수준의 장문의 검색 벤치마크에서의 multilingual 형제들 간의 차이는 아래와 같습니다: [mLateOn scores 77.92 against mDenseOn's 51.59](https://huggingface.co/blog/lightonai/mdenseon-mlateon#long-document-retrieval-mldr).

## 질의 및 문서 인코딩 {#section-5}

멀티-벡터 모델은 비대칭적입니다: 질의와 문서는 서로 다른 접두사, 서로 다른 길이 상한, 서로 다른 점수 산정 마스크를 거칩니다. Dense 모델이 서로 교환 가능하다는 것과 달리, 정확한 임베딩을 얻으려면 [`encode_query()`](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html#sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_query)와 [`encode_document()`](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html#sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_document)가 필요합니다:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("lightonai/mLateOn")

queries = ["What is the capital of France?"]
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital and largest city of Germany, by both area and population.",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)

print(query_embeddings[0].shape)
# (10, 128)
print(document_embeddings[0].shape, document_embeddings[1].shape)
# (10, 128) (19, 128)
```


되돌아오는 것을 주목하세요: 입력마다 모양이 다른 2D 텐서의 *목록*입니다. 각 텐서의 모양은 `(num_tokens, embedding_dim)`입니다. Dense 임베딩과 달리 이를 하나의 직사각형 텐서로 쌓을 수 없으므로 각 입력마다 토큰 수가 다릅니다. 두 번째 문서는 첫 번째 문서보다 길어서 더 높은 매트릭스로 반환됩니다.

각 호출은 모델의 고유 조합법칙을 적용합니다. `encode_query` 은 질의 마커를 앞에 붙이고, 체크포인트가 요구하면 질의를 고정 길이로 확장하며, 질의 길이에 맞춥니다. `encode_document` 은 문서 마커를 앞에 붙이고 문서 길이에서 캡하며, 점수 산정 마스크에서 건너뛴 토큰(대부분의 체크포인트에서 구두점)을 제거합니다.

일반적인 `encode()` 인수는 여전히 적용되므로, `batch_size`, `show_progress_bar`, `convert_to_numpy`, `device`, 그리고 다중 프로세스 풀도 기대하는 대로 작동합니다:

```python
document_embeddings = model.encode_document(
    documents,
    batch_size=64,
    show_progress_bar=True,
)
```


## MaxSim으로 점수 산정 {#section-6}

[`model.similarity()`](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html#sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.similarity) 는 모든 페어의 MaxSim 매트릭스를 계산합니다:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("lightonai/LateOn")

query_embeddings = model.encode_query(["Which planet is known as the Red Planet?"])
document_embeddings = model.encode_document([
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
])

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[10.7942, 11.1104, 10.9743, 11.0811]])
```


Mars가 마땅히 이깁니다. 근소한 차이가 난 후보들을 주목해 보십시오: Saturn은 "the Red Planet"이라는 구절을 정확히 포함하고 있으며, Jupiter는 붉은 반점이 있는 행성이라 세 문서 모두에서 토큰 단위 연산자가 활용될 여지가 충분합니다. 순서가 중요한 점은 바로 이 때문입니다.

점수는 자주 이렇게 근접하게 위치합니다. [GLInt](https://huggingface.co/blog/chungimungi/glint#1-mining-in-maxsim-space) 가 전체 후보 풀에서 분산을 측정하는 방식으로 이를 보여주며, MaxSim 은 질의 토큰당 최대값을 취하므로 문서는 보통 각 질의 토큰에 대해 괜찮은 최적 매치를 제공하고 점수는 바닥에서 시작합니다. 맥락화된 토큰 임베딩은 비등방성으로 굴절하는 쐐기형 콘에 모여 분포를 퍼뜨리지 않기 때문에 임의의 토큰 쌍도 높은 점수를 얻는 경향이 있습니다.

또한 [`model.similarity_pairwise()`](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html#sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.similarity_pairwise) 가 있습니다. 이미 매칭된 쌍이 있고 전체 유사도 행렬 대신 쌍 점수만 필요할 때 사용합니다:

```python
scores = model.similarity_pairwise(query_embeddings, document_embeddings[:1])
print(scores)
# tensor([10.7942])
```


### 점수의 크기와 MeanMaxSim

MaxSim 은 질의 토큰 수에 따라 합산되므로, 질의 구성에 따라 점수의 규모가 달라집니다. 따라서 서로 다른 질의 구성의 모델 간에 점수를 비교할 수 없습니다. 위의 Red Planet 질의를 LateOn은 12개의 토큰으로 인코드합니다. 동일한 질의와 문서를 ColBERTv2에 넣으면 모든 질의가 정확히 32토큰으로 패딩되고 자르는 방식으로 점수를 내는데, 점수 범위가 완전히 다르게 나타납니다:

```python
model = MultiVectorEncoder("colbert-ir/colbertv2.0")
# ... same encode_query / encode_document / similarity calls ...
print(scores)
# tensor([[12.7970, 27.1945, 23.8495, 24.5656]])
```


한 모델 내의 순서만 보면 충분하지만, 점수를 한정된 스케일로 보고 싶다면 모델의 유사도 함수를 MeanMaxSim으로 바꿔서 질의 토큰 수로 나누게 하십시오. LateOn에서 다시 보자면:

```python
model = MultiVectorEncoder("lightonai/LateOn", similarity_fn_name="meanmaxsim")
# or on an already-loaded model: model.similarity_fn_name = "meanmaxsim"

print(model.similarity(query_embeddings, document_embeddings))
# tensor([[0.8995, 0.9259, 0.9145, 0.9234]])
```


이제 모든 점수는 `[-1, 1]`에서의 평균 코사인 유사도이며, 실제로는 `[0, 1]`만 보게 됩니다.

## 시맨틱 검색 {#section-7}

코퍼스가 작다면 전체에 대한 exhaustive MaxSim 이 가장 간단하게 작동합니다. 코퍼스를 한 번 인코딩하고 모든 질의에 대해 점수화합니다:

```python
import time

from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
# Several questions share an answer passage, so drop repeats but keep the order
corpus = list(dict.fromkeys(dataset["answer"]))  # 5,000 rows -> 4,874 passages

model = MultiVectorEncoder("lightonai/LateOn")
corpus_embeddings = model.encode_document(corpus, show_progress_bar=True)

query = "when did richmond last play in a preliminary final"
start = time.perf_counter()
query_embeddings = model.encode_query([query])
scores = model.similarity(query_embeddings, corpus_embeddings)[0]  # 98ms
top_scores, top_indices = scores.topk(3)
print(f"Search took {(time.perf_counter() - start) * 1000:.1f}ms")

for score, index in zip(top_scores.tolist(), top_indices.tolist()):
    print(f"{score:.4f}  {corpus[index][:100]}")
"""
Search took 122.7ms
11.9192  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieved
11.7591  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contest
11.6710  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fou
"""
```


그 4,874개의 패시지는 RTX 3090에서 20초 만에 인코딩되었고, 각 검색은 전체적으로 약 120ms 정도 걸리며 그 대부분은 608,414 토큰 벡터에 대한 MaxSim 점수 산정 때문입니다. 이것은 정확하지만, 전체 코퍼스 토큰 수에 선형적으로 비례하여 확장되며 모든 토큰 벡터를 메모리에 보관하므로 수천 개의 문서 정도가 있을 때 사용하십시오. 이 스크립트의 실행 가능 버전은 [semantic_search.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/applications/semantic_search.py)입니다.

그 크기 이상으로 가면 실제 늦은 인터랙션 인덱스가 필요합니다. 이는 Sentence Transformers에서 제공하지 않지만 필요하지 않기도 합니다. 이 인덱스는 `encode_document`가 생성한 것을 저장하므로 여기에서 인코딩하고 토큰 임베딩을 이를 다룰 수 있는 다른 엔진에 넘깁니다. [Indexing](#indexing) 는 네 가지 옵션에 대한 작동 중인 스니펫을 제공하며, 바로 아래 섹션은 인덱스를 건너뛰는 방법을 다룹니다.

## 검색 및 재정렬 {#section-8}

늦은 상호작용의 품질은 늦은 상호작용 인덱스를 유지하지 않고도 얻을 수 있습니다. 멀티-벡터 모델을 your reranker로 사용하면 빠른 바이-인코더가 대형 코퍼스를 소수의 후보로 축소한 뒤 멀티-벡터 모델이 그 후보들만 재점수합니다:

```python
from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder, SentenceTransformer
from sentence_transformers.util import semantic_search

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:50000]")
corpus = list(dict.fromkeys(dataset["answer"]))

retriever = SentenceTransformer("jinaai/jina-embeddings-v5-text-nano-retrieval")
reranker = MultiVectorEncoder("perplexity-ai/pplx-embed-v1-late-0.6b", trust_remote_code=True)

# First stage: index the corpus once with a fast bi-encoder
corpus_embeddings = retriever.encode_document(corpus, convert_to_tensor=True, show_progress_bar=True)

# Retrieve the top 50
query = "when did richmond last play in a preliminary final"
hits = semantic_search(retriever.encode_query([query], convert_to_tensor=True), corpus_embeddings, top_k=50)[0]
candidates = [corpus[hit["corpus_id"]] for hit in hits]

# Second stage: rescore just those candidates with MaxSim
query_embeddings = reranker.encode_query([query])
document_embeddings = reranker.encode_document(candidates)
scores = reranker.similarity(query_embeddings, document_embeddings)[0]

for index in scores.argsort(descending=True)[:3].tolist():
    print(f"{scores[index].item():.4f}  {candidates[index][:100]}")
```


후보 50개만이 멀티-벡터로 인코딩되므로 인덱스는 일반 Dense 인덱스로 유지되고 토큰 벡터는 일시적입니다. 이는 크로스-인코더가 Retrieve-and-Rerank 스택에서 하는 역할과 같지만, 멀티-벡터 모델은 후보 하나당 상당히 저렴합니다. 문서를 한 배치로 인코딩하고 행렬 곱으로 점수를 계산하며 질의-문서 쌍마다 한 번의 순전파를 수행하는 대신에 그리하세요. 실행 가능한 스크립트는 [retrieve_rerank.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/applications/retrieve_rerank.py)이며 두 단계의 시간 측정을 출력합니다.

## 인덱싱 {#section-9}

여러 벡터 데이터베이스가 멀티-벡터를 네이티브로 인덱싱하고 점수 산정합니다: v1.10 이후 [Qdrant](https://qdrant.tech/documentation/concepts/vectors/), v1.29 이후 [Weaviate](https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings), 수년간 [Vespa](https://blog.vespa.ai/announcing-long-context-colbert-in-vespa/), v0.15.0 이후 [LanceDB](https://docs.lancedb.com/search/multivector-search), 그리고 Postgres에 MaxSim 연산자를 추가하는 [VectorChord](https://docs.vectorchord.ai/vectorchord/usage/indexing-with-maxsim-operators.html)가 있습니다. PyLate의 멀티-벡터 검색을 위한 비연관 기능으로 인덱스를 더했으며, [Milvus](https://milvus.io/docs/array-of-structs.md) 는 v2.6.4에서 합류했습니다. 서버를 전혀 실행하고 싶지 않다면 LightOn의 [fast-plaid](https://github.com/lightonai/fast-plaid) 은 `pip install`만으로 가능하고 PLAID를 직접 구현하며, [PyLate](https://github.com/lightonai/pylate) 은 이를 더 완전한 검색 스택으로 래핑합니다.

다른 몇 가지는 부분적으로 제공합니다. [OpenSearch](https://docs.opensearch.org/latest/search-plugins/search-relevance/rerank-by-field-late-interaction/) 및 [Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/rank-vectors) 은 MaxSim으로 후보를 재점수화할 수는 있지만 그 위에서 검색은 지원하지 않으며, Elasticsearch 필드는 기술 프리뷰 및 엔터프라이즈급으로 제공됩니다. [turbopuffer](https://turbopuffer.com/docs/schema) 는 비공개 베타에서 늦은 인터랙션 인덱싱을 제공합니다.

아래의 스니펫들은 텍스트를 인덱싱하지만 텍스트에 특화된 내용은 없습니다. `encode_document` 는 문서가 패시지이든 페이지 이미지이든 오디오 클립이든 비디오이든 간에 같은 토큰-벡터 매트릭스 목록을 반환합니다. 따라서 [Visual Document Retrieval](#visual-document-retrieval)의 ColPali 스타일 모델은 이러한 형태로 그대로 적용됩니다. 문서당 벡터 수가 늘어나므로 [Token Pooling](#token-pooling) 를 더 빨리 활용하게 됩니다.

fast-plaid, Qdrant, Weaviate, Vespa 는 모두 `encode_document`가 반환하는 것을 정확히 받아들이므로 코드 구조는 클라이언트 라이브러리까지 동일합니다. 아래는 4,874개의 패시지와 [Semantic Search](#semantic-search) 예시의 608,414 토큰 벡터를 대상으로 실행된 각 기술의 동작 예시 스니펫입니다. 각 스니펫은 한 대의 머신(RTX 3090, i7-13700K)에서 생성된 인제스션 및 질의 시간을 담고 있으며, 코드가 보여주는 것 외에는 추가 튜닝이 없습니다. 네 기술 모두 이 섹션에서 98ms 걸린 `model.similarity`보다 더 빠르게 질의에 응답했고, 세 가지는 CPU에서 실행되며, fast-plaid만 유일하게 GPU를 사용합니다.

네 기술 모두 이 글의 앞부분에서 포괄적으로 계산된 PyTorch MaxSim과 동일한 순서로 세 개의 패시지를 반환했고, 세 데이터베이스는 그 점수를 소수점 4자리까지 재현합니다! 이는 이들의 스니펫이 모든 문서를 점수화하기 때문이며, 이 규모에서 실행이 가능하고 정확도에서 근사치를 제거합니다. design: fast-plaid 는 설계상 근사이므로 점수가 약간 다를 수 있습니다. 각 항목 아래의 주석은 근사 인덱스로 바꿀 때의 차이를 설명하며, 이것이 랭킹이 드리프트하기 시작하는 지점입니다.

<details>
<summary><b>fast-plaid</b></summary>

[fast-plaid](https://github.com/lightonai/fast-plaid) 는 LightOn의 PLAID의 Rust 구현으로, ColBERT가 원래 그것을 중심으로 구축되었는 인덱스입니다. 서버를 시작할 필요 없고, `encode_document` 가 반환하는 텐서를 변환 없이 읽습니다.

```python
# pip install sentence-transformers datasets fast-plaid
from datasets import load_dataset
from fast_plaid import search
from sentence_transformers import MultiVectorEncoder

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
corpus = list(dict.fromkeys(dataset["answer"]))
model = MultiVectorEncoder("lightonai/LateOn")
query = "when did richmond last play in a preliminary final"

document_embeddings = model.encode_document(corpus, batch_size=32)
query_embedding = model.encode_query(query)

fast_plaid = search.FastPlaid(index="natural-questions", device="cuda")

# 4,874 documents (608,414 token vectors) indexed in 5s
fast_plaid.create(documents_embeddings=document_embeddings)

results = fast_plaid.search(queries_embeddings=query_embedding.unsqueeze(0), top_k=3)  # 11ms

for index, score in results[0]:
    print(f"{score:.4f}  {corpus[index][:90]}")
"""
11.8828  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieve
11.7676  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contes
11.6758  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fo
"""
```


`index` 인자는 라벨이 아니라 디렉토리이므로 인덱스가 구성되는 대로 디스크에 저장됩니다. 같은 경로에 새로운 `FastPlaid` 를 가리키면 매번 임베딩에서 다시 빌드하지 않고 검색용으로 열거나 문서를 추가하도록 열 수 있습니다. 이 코퍼스에서 이는 92 MB를 차지하며, 원시 float32 벡터의 311.5 MB에 비해 작습니다.

이 네 가지 중 근사적 인덱스인 것은 단 하나이며, 이 섹션에서의 점수는 exhaustive MaxSim과 일치하지 않는 곳이기도 합니다. PLAID 는 중심 벡터를 사용해 잔차를 양자화하고 저장하므로, 세 점수는 앞서 계산된 11.9192 / 11.7591 / 11.6710에 대해 양 방향으로 미세하게 차이가 납니다. 이 순위는 여기서는 영향을 받지 않으며, 이것이 PLAID가 감수하는 타협입니다. 이 인덱스는 이보다 훨씬 큰 코퍼라를 대상으로 설계되어 모든 것을 스캔하는 옵션이 아니라는 점이 특징입니다.

</details>

<details>
<summary><b>Qdrant</b></summary>

[Qdrant](https://qdrant.tech/documentation/concepts/vectors/) 는 서버가 필요합니다: `docker run -p 6333:6333 qdrant/qdrant`. 클라이언트에는 서버가 필요 없는 로컬 모드(`QdrantClient(":memory:")`)도 있지만, 이는 순수 파이썬 재구현이므로 실험 용도에 더 적합합니다.

```python
# pip install sentence-transformers datasets qdrant-client
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from sentence_transformers import MultiVectorEncoder

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
corpus = list(dict.fromkeys(dataset["answer"]))
model = MultiVectorEncoder("lightonai/LateOn")
query = "when did richmond last play in a preliminary final"

document_embeddings = model.encode_document(corpus, batch_size=32)
query_embedding = model.encode_query(query)

client = QdrantClient("http://localhost:6333")
client.create_collection(
    collection_name="natural-questions",
    vectors_config=models.VectorParams(
        size=model.get_embedding_dimension(),
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        # MaxSim never walks the HNSW graph, so skip building one
        hnsw_config=models.HnswConfigDiff(m=0),
    ),
)

# 4,874 documents (608,414 token vectors) ingested in 26.3s
client.upload_points(
    collection_name="natural-questions",
    points=[
        models.PointStruct(id=idx, vector=embedding, payload={"text": text})
        for idx, (embedding, text) in enumerate(zip(document_embeddings, corpus))
    ],
    batch_size=64,
)

results = client.query_points(
    collection_name="natural-questions",
    query=query_embedding,
    limit=3,
    with_payload=True,
).points  # 18ms

for result in results:
    print(f"{result.score:.4f}  {result.payload['text'][:90]}")
"""
11.9192  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieve
11.7591  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contes
11.6710  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fo
"""
```


`MAX_SIM` 은 Qdrant가 제공하는 유일한 비교 수단이며, `hnsw_config=HnswConfigDiff(m=0)` 은 레이트 인터랙션 필드에 대한 그들의 권고안으로, 벡터가 그래프 탐색이 아닌 재점수화에 사용되기 때문입니다. Qdrant 측에서도 레이트 인터랙션은 후보 수백 개를 재정렬하는 용도로만 사용하는 것을 권장하며, 이는 [Retrieve and Rerank](#retrieve-and-rerank) 패턴입니다. 4,874 문서에서 전체 스캔은 18ms로 정확하지만, 이를 일반화해서는 안 됩니다.

</details>

<details>
<summary><b>Weaviate</b></summary>

[Weaviate](https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings) 역시 서버가 필요합니다: `docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.34.0`. 멀티-벡터 지원은 1.29 이상이 필요하며, 임베디드 모드는 Windows에서 사용할 수 없습니다.

```python
# pip install sentence-transformers datasets weaviate-client
import weaviate
from datasets import load_dataset
from sentence_transformers import MultiVectorEncoder
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
corpus = list(dict.fromkeys(dataset["answer"]))
model = MultiVectorEncoder("lightonai/LateOn")
query = "when did richmond last play in a preliminary final"

document_embeddings = model.encode_document(corpus, batch_size=32)
query_embedding = model.encode_query(query)

client = weaviate.connect_to_local()
collection = client.collections.create(
    "Documents",
    # self_provided turns on MaxSim late interaction
    vector_config=[Configure.MultiVectors.self_provided(name="colbert")],
    properties=[Property(name="text", data_type=DataType.TEXT)],
)

# 4,874 documents (608,414 token vectors) ingested in 41s
with collection.batch.fixed_size(batch_size=64) as batch:
    for text, embedding in zip(corpus, document_embeddings):
        batch.add_object(properties={"text": text}, vector={"colbert": embedding.tolist()})

results = collection.query.near_vector(
    near_vector=query_embedding.tolist(),
    target_vector="colbert",
    limit=3,
    return_metadata=MetadataQuery(distance=True),
)  # 17ms

for result in results.objects:
    # Weaviate reports the MaxSim score as a negated distance
    print(f"{-result.metadata.distance:.4f}  {result.properties['text'][:90]}")
"""
11.9192  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieve
11.7591  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contes
11.6710  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fo
"""

client.close()
```


Defaults are enough here: Weaviate의 동적 `ef` 는 상위 3 질의에서 100으로 수렴하며, 이 랭킹은 32 이상부터 이미 정확합니다. 이 여백은 Weaviate의 문제가 아니라 임베딩의 특성에 속하므로 기본값이 유지된다고 가정하기보다 자신의 모델에서 확인하는 편이 좋습니다.

Weaviate는 MUVERA 인코딩도 지원하여, 우리 테스트에서 수집 시간이 3배, 질의 속도가 1.8배 빨라졌습니다. 그러나 이 규모에서 이 속도 향상의 대가로 얻는 정확도 손실은 큽니다: 상위 50 내에도 실제로 올바른 세 번째 패시지가 나타나지 않았습니다.

</details>

<details>
<summary><b>Vespa</b></summary>

[Vespa](https://docs.vespa.ai/en/tensor-user-guide.html) 역시 컨테이너에서 실행되지만 `pyvespa`가 이를 시작해 주므로 별도의 `docker run`가 필요 없습니다.

```python
# pip install sentence-transformers datasets pyvespa
from datasets import load_dataset
from sentence_transformers import MultiVectorEncoder
from vespa.deployment import VespaDocker
from vespa.package import (
    ApplicationPackage, Document, Field, FirstPhaseRanking, Function, RankProfile, Schema,
)

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
corpus = list(dict.fromkeys(dataset["answer"]))
model = MultiVectorEncoder("lightonai/LateOn")
query = "when did richmond last play in a preliminary final"

document_embeddings = model.encode_document(corpus, batch_size=32)
query_embedding = model.encode_query(query)

# "dt" is a mapped dimension over the variable token count, "x" the dense 128-dim vector
package = ApplicationPackage(
    name="colbert",
    schema=[
        Schema(
            name="doc",
            document=Document(fields=[
                Field(name="text", type="string", indexing=["summary"]),
                Field(name="colbert", type="tensor<float>(dt{}, x[128])", indexing=["attribute"]),
            ]),
            rank_profiles=[
                RankProfile(
                    name="colbert",
                    inputs=[("query(qt)", "tensor<float>(qt{}, x[128])")],
                    functions=[Function(
                        name="max_sim",  # per query token take the best document token, then sum
                        expression="sum(reduce(sum(query(qt) * attribute(colbert), x), max, dt), qt)",
                    )],
                    first_phase=FirstPhaseRanking(expression="max_sim"),
                )
            ],
        )
    ],
)
app = VespaDocker(port=8080).deploy(application_package=package)  # ~40s to boot

# Vespa reads a mixed tensor as {token index: vector}, for documents and queries alike
def to_tensor(embedding):
    return {str(token): vector for token, vector in enumerate(embedding.tolist())}

# 4,874 documents (608,414 token vectors) ingested in ~80s
app.feed_iterable(
    ({"id": str(idx), "fields": {"text": text, "colbert": to_tensor(embedding)}}
     for idx, (text, embedding) in enumerate(zip(corpus, document_embeddings))),
    schema="doc",
)

response = app.query(body={
    "yql": "select text from doc where true",
    "ranking.profile": "colbert",
    "hits": 3,
    "input.query(qt)": to_tensor(query_embedding),
})  # ~75ms warm, ~115ms on the first call

for hit in response.hits:
    print(f"{hit['relevance']:.4f}  {hit['fields']['text'][:90]}")
"""
11.9192  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieve
11.7591  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contes
11.6710  Battle of Appomattox Court House The Battle of Appomattox Court House (Virginia, U.S.), fo
"""
```


Vespa는 네 가지 중 가장 구조를 미리 요구합니다. 이는 순위 파이프라인을 선언하는 것이지 단지 인덱스만을 위한 것이 아니기 때문입니다. 그 대가로 MaxSim을 텐서 식으로 작성하고 정확히 무엇을 계산하는지 확인할 수 있습니다. 이 버전은 `first-phase`의 MaxSim을 `where true`보다 앞에 두고, 4,874 문서를 모두 점수화하며 출력이 exhaustive MaxSim과 정확히 일치하는 이유입니다. 이는 규모가 큰 Vespa의 권장 방식은 아니며, 그들의 [ColBERT sample app](https://github.com/vespa-engine/sample-apps/tree/master/colbert)는 int8-바이너리 벡터를 저장하고 MaxSim을 `second-phase`로 이동시켜 더 저렴한 초기 단계를 재정렬합니다.

그러한 단계적 설정으로의 이동은 주의가 필요합니다: `second-phase` 은 기본적으로 상위 100개 후보만 재점수화하며, 여기서는 그 창에서 세 개의 올바른 패시지가 두 개만 점수화되지 않는 결과를 냈습니다. 후보 세트를 커버하도록 `rerank-count` 을 올리면 이 문제가 해결되지만, 이 크기에서는 단계적 버전이 단순히 모든 것을 스캔하는 것보다 여전히 느립니다.

</details>

## 시각적 문서 검색 {#section-10}

늦은 인터랙션은 시각적 문서 검색의 최첨단입니다: 텍스트 쿼리를 페이지 이미지와 매칭하고, 표와 차트의 레이아웃을 유지하며 OCR 단계가 필요 없는 방식입니다. 이것이 [ColPali](https://arxiv.org/abs/2407.01449) 가족 모델이 하는 일이며, 체크포인트들은 동일 API를 통해 로드하고 실행되며, `revision` 핀으로 이 모델의 Sentence Transformers 구성이 추가된 오픈 풀 요청이 고정되어 있습니다([Supported Models](#visual-document-retrieval-models) 에 전체 목록이 있습니다). 이미지 문서는 URL, 로컬 경로, 또는 PIL 이미지로 전달됩니다:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
images = [
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc1.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc2.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc3.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc4.jpg",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(images)
print(query_embeddings[0].shape, document_embeddings[0].shape)
# (25, 128) (755, 128)

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[13.8672, 12.3115, 12.1670, 11.0293],
#         [ 7.2012, 14.7207,  6.9414,  6.9746]])
```


각 질의는 고유의 페이지(대각선)를 검색하고, 두 번째 질의는 첫 번째에 비해 훨씬 더 명확하게 구분되며, 왜냐하면 네 페이지 중 하나만 시간이 소요되는 비용에 관한 것이기 때문입니다.

코드는 바뀌지 않습니다. 그 아래에서 프로세서는 시각적 프롬프트와 이미지 패치를 처리하고, MaxSim은 질의 텍스트 토큰을 문서 이미지 패치에 대해 점수 매깁니다. 페이지는 여러 구역으로 나뉘므로, 단일 벡터가 차트, 표, 세 문단을 하나의 요약으로 평균해야 하는 상황에서 늦은 인터랙션이 자연스럽게 맞아떨어집니다. 다만 그 충실도는 인덱스 공간을 차지합니다. 위의 모양은 한 페이지당 755개의 토큰 벡터이고 질의 당 25개이며, 앞서 언급한 Natural Questions 패시지는 대략 125개를 평균했습니다. 따라서 [token pooling](#token-pooling) 은 텍스트보다 여기에선 더 일찍 활용하는 것이 가치가 있습니다.

이들은 VLMs이므로 필요한 메모리를 고려해야 합니다. [The table in Supported Models](#visual-document-retrieval-models) 는 252M에서 8.8B 파라미터까지 동작하며, 작은 쪽은 CPU에서도 실용적일 수 있습니다.

페이지 이미지는 일반적인 경우지만 텍스트가 아닌 모달리티가 전부는 아닙니다. Sentence Transformers는 텍스트, 이미지, 오디오, 비디오를 받아들이고, 체크포인트는 해당 프로세서가 다루는 모달리티를 지원합니다. `model.modalities` 는 이를 보고합니다. 단일 문서에서도 모달리티를 결합할 수 있으며, 예를 들어 `{"text": ..., "image": ...}` 과 같은 dict 를 bare 값 대신 전달합니다. [Multimodal Embedding & Reranker Models](https://huggingface.co/blog/multimodal-sentence-transformers) 는 Sentence Transformers에서 멀티모달 모델을 더 넓게 다루고, [Usage documentation](https://sbert.net/docs/sentence_transformer/usage/usage.html) 는 각 모달리티가 어떤 입력 형식을 수용하는지 정확히 나열합니다.

## 오디오 검색 {#section-11}

[vidore/colqwen-omni-v0.1](https://huggingface.co/vidore/colqwen-omni-v0.1) 는 Qwen2.5-Omni 를 기반으로 하며 네 가지 모달리티를 모두 다룹니다. 이를 사용해 녹음 대화를 검색하는 것은 페이지를 검색하는 것과 동일한 두 번의 호출로 수행됩니다:

```python
# pip install -U "sentence-transformers[audio,video]"
import torch
from datasets import Audio, load_dataset

from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder(
    "vidore/colqwen-omni-v0.1",
    model_kwargs={"dtype": torch.bfloat16},
)
print(model.modalities)
# ['text', 'image', 'audio', 'video', 'message']

# 20 recorded conversations, averaging 28 seconds each
dataset = load_dataset("eustlb/dailytalk-conversations-grouped", split="train[:20]")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
audio = [row["array"] for row in dataset["audio"]]  # raw mono waveforms, float32 at 16 kHz

query_embeddings = model.encode_query(["medicine for car nausea"])
document_embeddings = model.encode_document(audio, batch_size=2)
scores = model.similarity(query_embeddings, document_embeddings)[0]

top_scores, top_indices = scores.topk(3)
for score, index in zip(top_scores.tolist(), top_indices.tolist()):
    print(f"{score:.4f}  {' / '.join(dataset[index]['texts'][:2])}")
"""
50.8902  Excuse me? Do you have anything for a carsickness? / Yes, but you look fine.
46.1028  Excuse me, could you tell me where you have got that music book? / Certainly. Let me see. Oh, it's on that shelf.
46.0514  Jeff, I'm going to the supermarket. Do you want to come with me? / I think the supermarket is closed now.
"""
```


ColQwen-Omni 는 이미지-텍스트 쌍으로만 학습되어 왔으므로 오디오 검색은 제로샷입니다: 학습 예제를 본 적이 없고 파이프라인 어디에도 전사 단계가 없습니다. 질의가 녹음이 말하는 `nausea` 를 지시하고, 녹음은 `carsickness` 를 말하는 곳에서 여전히 20개 중에서 약간의 큰 차이로 약국 대화를 골라냅니다.

## 비디오 검색 {#section-12}

비디오도 동일한 방식으로 작동하지만 프레임을 샘플링해야 하며 프레임 샘플링을 하지 않으면 VRAM 을 빨아들입니다. 그 모델의 [release blogpost](https://huggingface.co/blog/manu/colqwen-omni-omnimodal-retrieval) 는 이것에 대해 다소 직설적이며, 비디오는 "메모리 집약적이므로 짧은 클립에 가장 적합하다"고 말합니다:

```python
import torch

from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder(
    "vidore/colqwen-omni-v0.1",
    model_kwargs={"dtype": torch.bfloat16},
)

# Sparse, low-resolution frames: 0.5 fps rather than the full frame rate
model[0].processing_kwargs.update(
    {"video": {"max_pixels": 32 * 28 * 28, "do_sample_frames": True, "fps": 0.5}}
)

query_embeddings = model.encode_query(["How to cook Mapo Tofu?"])
document_embeddings = model.encode_document([
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/mapo_tofu.mp4",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/zhajiang_noodle.mp4",
], batch_size=1)
print(model.similarity(query_embeddings, document_embeddings))
# tensor([[53.3100, 51.0561]])
```


1 FPS, 전체 해상도에서 같은 두 비디오 쌍은 8,426개와 5,137개의 토큰 벡터를 생성하고 VRAM 최대 20.8GB를 사용합니다. 이는 여기서의 4,240개와 2,446개의 벡터, 12.5GB와 비교됩니다. 이 모델은 자체적으로 약 9.0 GB를 차지합니다. 랭킹은 어느 쪽이든 동일합니다. 긴 오디오도 같은 처리가 필요하며, 출시 블로그 포스트는 30초 단위의 조각을 권장하며 각각 약 800 토큰에 해당합니다.

## 해석 가능성 {#section-13}

MaxSim 은 질의 토큰별 최대값의 합이므로, 랭킹은 정확히 분해됩니다: 문서 점수의 모든 포인트는 하나의 질의 토큰과 하나의 문서 토큰에 소속됩니다. 이를 통해 "왜 이 랭크가 이렇지"에 대해 눈으로 보지 않고도 정확히 대답할 수 있습니다.

이미지 문서의 경우, `sentence_transformers.multi_vector_encoder.interpretability` 는 이 분해를 페이지 위에 ColPali 열지도 형태로 중첩합니다. 질의에 따라 집계하거나 질의 토큰당 하나의 맵을 생성할 수 있습니다. 위의 지출 페이지를 기준으로 "물 자원과 전기에 얼마나 지출되었나?"를 묻는다면, 여기에서 `water` 토큰이 들어갑니다:

![MaxSim heatmap of the query token "water" overlaid on a 1971 US budget outlays page, with the brightest patch on the "Water Resources & Power" bar of the lower chart](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/maxsim_heatmap.png)

[heatmap.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/interpretability/heatmap.py) 은 실행 가능한 버전으로, 문서 임베딩을 패치 격자와 정렬하는 마스킹 단계를 포함합니다.

텍스트 문서에는 패치 격자가 없지만 동일한 분해가 적용됩니다. [text_similarity_map.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/interpretability/text_similarity_map.py) 는 코퍼스를 랭크하고 최상위 히트를 토큰별로 할당하는데, 여기서는 앞에서 다룬 Natural Questions 코퍼스의 32M 파라미터 [mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) 를 사용하는 예를 보여줍니다:

```
Query: when did richmond last play in a preliminary final
Top 3 of 4874 documents by exhaustive MaxSim (191.0ms):
  12.3489  Richmond Football Club Richmond began 2017 with 5 straight wins, a feat it had not achieved since 19
  12.1771  2017 AFL Grand Final The 2017 AFL Grand Final was an Australian rules football game contested betwee
  12.0591  2018 UEFA Champions League Final The 2018 UEFA Champions League Final was the final match of the 201

  query token       best document token      sim   share
  when              since                 0.9154    7.4%
  did               had                   0.9675    7.8%
  rich              rich                  0.9764    7.9%
  mond              mond                  0.9856    8.0%
  last              to                    0.9249    7.5%
  play              game                  0.9384    7.6%
  in                the                   0.9732    7.9%
  a                 a                     0.9587    7.8%
  preliminary       preliminary           0.9394    7.6%
  final             final                 0.9654    7.8%
  --------------------------------------------------------
  3 special tokens                        2.8038   22.7%
  MaxSim score                           12.3489  100.0%
```


`rich`, `mond`, `preliminary`, 및 `final` 은 서로 매칭되었고, `when` 는 `since` 를, `play` 은 `game` 를 선택했습니다. 특별한 토큰들 역시 주목할 만합니다: 이들 중 세 개는 질의 내용에 전혀 의존하지 않으면서도 점수의 22.7%를 기여합니다. 이 표 아래의 스크립트는 승리한 토큰이 제자리에 강조된 패시지를 출력합니다.

## 토큰 풀링 {#section-14}

인덱스 풋프린트가 걱정된다면 가장 효과적인 조정은 더 적은 토큰 벡터를 저장하는 것입니다. `HierarchicalTokenPooling` 은 Clavié, Chaffin, Adams 의 [token pooling](https://arxiv.org/abs/2409.14683v1) 기법을 구현합니다: 각 문서의 토큰 벡터를 코사인 거리의 Ward 연결로 클러스터링하고 각 클러스터를 그 평균으로 대체하며, 약 `1 / pool_factor` 만큼의 토큰을 남깁니다. 한 문서 내에서 많은 토큰 벡터가 서로 가깝게 몰리므로 제거하는 많은 부분은 신호 보다는 중복성입니다:

```python
from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
documents = list(dict.fromkeys(dataset["answer"]))

model = MultiVectorEncoder("lightonai/LateOn")

pooling = HierarchicalTokenPooling(pool_factor=2)
document_embeddings = model.encode_document(documents, token_pooling=pooling)
```


적용할 위치는 세 가지가 있으며, 적용 시점에 따라 다릅니다:

```python
# 1. Per encode call, as above
document_embeddings = model.encode_document(documents, token_pooling=pooling)

# 2. Standalone, on embeddings you already have saved (e.g. list of [num_tokens, num_dims] tensors)
pooled = pooling.pool(document_embeddings)

# 3. Baked into the model, so every consumer of the checkpoint gets pooled documents
model.append(HierarchicalTokenPooling(pool_factor=2))
model.save_pretrained("my-pooled-colbert")
```


기본적으로 풀링은 문서에만 적용되며, 질의는 짧아 변형될 여지가 없기 때문입니다. 앞에서 다룬 Natural Questions 코퍼스에서 감소 효과는 `pool_factor` 와 근접했고, 608k 토큰 벡터를 모두 풀링하는 데 약 6초가 걸렸습니다:

| `pool_factor` | 토큰 벡터 | 감소율 | float32 인덱스 |
| :---: | ---: | :---: | ---: |
| 1 (끄짐) | 608,414 | 1.00x | 311.5 MB |
| 2 | 305,438 | 1.99x | 156.4 MB |
| 3 | 204,407 | 2.98x | 104.7 MB |
| 4 | 153,936 | 3.95x | 78.8 MB |

클러스터 평균은 쿼리 토큰에 대해 최상의 멤버 중 하나보다 매칭이 떨어지며, 클러스터가 더 거칠수록 그 차이는 더 크게 드러납니다. BEIR에서 그 비용을 측정한 [original experiments](https://arxiv.org/abs/2409.14683v1) 은 거의 영향을 보지 않았습니다: 비풀링된 검색 성능 대비 평균 100.6% 수준에서 `pool_factor=2`, 그리고 `pool_factor=3` 에서는 99.0% 수준에서 나타났습니다. 인덱스를 무료로 절반으로 줄이는 것은 좋은 거래이므로 2를 시작 지점으로 삼는 것이 합리적입니다. 데이터에 따른 비용은 코퍼스별로 다르므로, 결정을 내리기 전에 [evaluator](#evaluating-a-model) 로 측정해 보십시오. 실행 가능 비교는 [token_pooling.py](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/compression/token_pooling.py) 입니다.

`pool_factor` 를 얼마나 멀리 밀어붙일 수 있는지는 모델의 속성에 부분적으로 좌우됩니다. LightOn의 [hierarchical pooling regularization](https://huggingface.co/blog/lightonai/lateon-hpool-regularization) 은 정확히 그 목표를 겨냥해 임베딩 공간을 형성하고 풀링 비용을 줄이며, 5배 압축에서도 99.4% retention 을 달성합니다. 그 정규화로의 학습은 아직 Sentence Transformers 에 포함되지는 않았지만, 결과 체크포인트들은 일반 PyLate 모델이므로 [`lightonai/LateOn-hpool-regularized`](https://huggingface.co/lightonai/LateOn-hpool-regularized) 는 다른 모델들처럼 로드되고 풀링됩니다.

## 추론 속도 향상 {#section-15}

멀티-벡터 모델은 나머지 Sentence Transformers 와 동일한 백엔드 기계를 통해 작동하므로 기본적으로 `torch`(기본값), `onnx`, 및 `openvino` 를 얻고, 반정밀도, Flash Attention, 그리고 `torch.compile` 와 함께 제공됩니다.

GPU에서 Flash Attention이 적용된 fp16 이 최적의 구성으로 측정되었고, fp32 대비 처리량이 2.44배이며 검색 품질 저하 없이 측정되었습니다. Flash Attention 은 문서가 공유 길이로 패딩되지 않고 잘려나가므로 배치의 시퀀스 길이가 크게 다르며 언패딩이 그 이점을 활용할 수 있어 멀티-벡터 모델에 특히 도움이 됩니다:

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder(
    "lightonai/GTE-ModernColBERT-v1",
    model_kwargs={"attn_implementation": "flash_attention_2", "dtype": "float16"},
)
```


<div style="display: flex; flex-wrap: wrap; gap: 16px; justify-content: center;">
  <figure style="flex: 1 1 300px; min-width: 0; margin: 0; text-align: center;">
    <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/mve_backends_benchmark_gpu.png"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/mve_backends_benchmark_gpu.png" alt="Multi-vector backend benchmarks on GPU" style="width: 100%;" /></a>
    <figcaption>GPU</figcaption>
  </figure>
  <figure style="flex: 1 1 300px; min-width: 0; margin: 0; text-align: center;">
    <a href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/mve_backends_benchmark_cpu.png"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/multi-vector-encoder/mve_backends_benchmark_cpu.png" alt="Multi-vector backend benchmarks on CPU" style="width: 100%;" /></a>
    <figcaption>CPU</figcaption>
  </figure>
</div>

> [경고]
> 비-attend 질의 확장 모델(`attend=False`으로, Stanford-NLP 체크포인트인 `colbert-ir/colbertv2.0` 및 `answerdotai/answerai-colbert-small-v1`를 포함)을 로드 시 Flash Attention 을 거부합니다. Flash Attention 은 `attention_mask=0` 위치를 제거하므로 MaxSim 이 점수화하는 `[MASK]` 확장 토큰은 주의 업데이트를 받지 못합니다. 이러한 모델에는 `"sdpa"` 를 사용하세요.

CPU에서는 아키텍처가 지원되는 경우 OpenVINO가 더 낫고, int8 양자화는 정확도 약 0.4% 손실로 속도를 더 끌어올립니다. 전체 벤치마크 세부 정보, 내보내기 및 양자화 도구, 백엔드 선택 흐름도는 [Speeding up Inference](https://sbert.net/docs/multi_vector_encoder/usage/efficiency.html)를 참고하십시오.

## 모델 평가 {#section-16}

`MultiVectorNanoBEIREvaluator` 는 MaxSim 점수화가 포함된 13개의 작은 BEIR 하위 집합에 대해 [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6) 스위트를 실행하며, 사용자의 데이터 전처리가 필요하지 않습니다:

```python
from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator

model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")
evaluator = MultiVectorNanoBEIREvaluator(batch_size=16)
results = evaluator(model)
print(f"{evaluator.primary_metric}: {results[evaluator.primary_metric]:.4f}")
```


또한 이 포스트 맨 위의 주장도 확인하기 쉽습니다. [`lightonai/LateOn`](https://huggingface.co/lightonai/LateOn) 와 [`lightonai/DenseOn`](https://huggingface.co/lightonai/DenseOn) 은 LightOn이 동일한 데이터와 ModernBERT 백본, 동일한 149M 파라미터로 학습했으며, 토큰 하나당 하나의 벡터를 유지하는지 아니면 문서당 하나로 풀링하는지에 따라 달랐습니다. 두 모델을 모두 13개의 NanoBEIR 데이터 세트에 걸쳐 실행하면 그 선택이 무엇을 가져오는지 분리해서 확인할 수 있습니다:

| NanoBEIR 데이터 세트 | LateOn (멀티-벡터, 128d) | DenseOn (dense, 768d) |
| --- | :---: | :---: |
| MSMARCO | **0.7194** | 0.6517 |
| NQ | **0.7810** | 0.7511 |
| HotpotQA | **0.9295** | 0.8802 |
| FEVER | **0.9702** | 0.9612 |
| ClimateFEVER | **0.4887** | 0.4846 |
| DBPedia | **0.6836** | 0.6748 |
| QuoraRetrieval | **0.9795** | 0.9687 |
| Touche2020 | **0.5938** | 0.5673 |
| ArguAna | 0.5562 | **0.5660** |
| NFCorpus | **0.3949** | 0.3851 |
| SciFact | 0.7978 | **0.8057** |
| SCIDOCS | 0.4469 | **0.4484** |
| FiQA2018 | 0.5871 | **0.6491** |
| **Mean** | **0.6868** | 0.6764 |

늦은 상호작용은 13개 데이터 세트 중 9개에서, 그리고 평균에서 대략 한 NDCG 포인트 차이로 우세합니다. 패배한 네 가지(ArguAna, FiQA2018, SCIDOCS, SciFact)는 동일한 트레이드오프의 형태를 보여주는데, 같은 모델 크기에서 검색 품질의 실질적 향상을 얻되 인덱스 공간의 비용을 치르는 것이지 모든 데이터 세트에서 만능 승리가 아닙니다. 같은 쌍이 BEIR의 전체 15-데이터 세트에서도 57.22 대 56.20으로 비슷한 차이를 보이므로, 이 차이는 작은 벤치마크의 산물이 아닙니다.

NanoBEIR 와 함께, `MultiVectorInformationRetrievalEvaluator`, `MultiVectorRerankingEvaluator`, `MultiVectorTripletEvaluator`, 및 `MultiVectorDistillationEvaluator` 는 자신의 데이터에 대한 일반적인 평가 구성도 다룹니다. 이들은 [Evaluation API Reference](https://sbert.net/docs/package_reference/multi_vector_encoder/evaluation.html) 에 문서화되어 있습니다.

## PyLate 또는 colpali-engine에서 오신 분들 {#section-17}

`MultiVectorEncoder` 는 두 라이브러리의 모델링, 추론, 학습 및 평가를 흡수합니다. 모든 PyLate 체크포인트는 직접 로드되며, [Supported Models](#supported-models) 은 colpali-engine 체크포인트와 함께 `revision` 를 통과시켜야 하는 경우를 나열합니다. 마이그레이션 중이라면 변경되는 호출은 아래와 같습니다:

| PyLate | Sentence Transformers |
|---|---|
| `pylate.models.ColBERT(model_name_or_path=...)` | `MultiVectorEncoder(...)` |
| `model.encode(..., is_query=True)` | `model.encode_query(...)` |
| `model.encode(..., is_query=False)` | `model.encode_document(...)` |
| `pylate.scores.colbert_scores` | `model.similarity` |
| `pylate.indexes.PLAID` / `pylate.retrieve.ColBERT` | 대응하는 것이 없으며, PyLate의 PLAID를 유지하거나 [Indexing](#indexing)를 참조하십시오 |

| colpali-engine | Sentence Transformers |
|---|---|
| `ColQwen2.from_pretrained(...)` + `ColQwen2Processor` | `MultiVectorEncoder(...)` |
| `processor.process_queries(...)` + `model(**batch)` | `model.encode_query(queries)` |
| `processor.process_images(...)` + `model(**batch)` | `model.encode_document(images)` |
| `processor.score_multi_vector(qs, ds)` | `model.similarity(query_embeddings, document_embeddings)` |
| `mask_non_image_embeddings=True` | `MultiVectorMask(keep_only_token_ids=[...])` |
| `HierarchicalTokenPooler` | `HierarchicalTokenPooling` |
| `colpali_engine.interpretability` | `sentence_transformers.multi_vector_encoder.interpretability` |

한 가지 차이점 주목: 베어(bare) 체크포인트의 경우 PyLate의 `ColBERT("bert-base-uncased")` 는 기본적으로 고전적인 구성(recipe)을 적용하는 반면, `MultiVectorEncoder("bert-base-uncased")` 는 일반 스택을 구성하고 접두사, 질의 확장, 및 스킵리스트를 명시적으로 선택합니다. 학습 손실 및 평가자 대응, 데이터 처리 차이는 [Migration Guide](https://sbert.net/docs/migration_guide.html#migrating-from-pylate)에 있습니다.

저장 호환성은 모든 경우에 단방향임에 유의하십시오: PyLate, Stanford-NLP ColBERT, 그리고 colpali-engine 체크포인트는 모두 `MultiVectorEncoder`으로 로드되지만, `MultiVectorEncoder.save_pretrained` 의 출력은 이들 중 어느 것도 로드될 수 없습니다.

## 지원하는 모델 {#section-18}

Hub에서 [`multi-vector` and `sentence-transformers` tags](https://huggingface.co/models?library=sentence-transformers&other=multi-vector) 태그를 가진 모델은 최신 상태를 유지하는 목록이며, 작동하는 모든 모델에 이 태그를 달 수 있도록 작업 중입니다. 아래 표는 우리가 직접 테스트하는 대상이며, 전체 세트의 시작점으로 보시되, 완전한 목록은 아닙니다. 특히 텍스트 검색의 경우, 태그가 달려 있든 없든 PyLate 혹은 Stanford-NLP ColBERT 체크포인트는 로드됩니다.

일부 항목은 저장소에 작은 Sentence Transformers 구성 추가가 필요하며, 이 중 다수는 작성 시점에 아직 열려 있는 풀 리퀘스트입니다. 아래에 `revision` 가 나열된 경우, 해당 풀 리퀘스트가 병합될 때까지 이를 넘겨주고, 그 후에는 일반 모델 이름만으로 충분합니다:

```python
model = MultiVectorEncoder("vidore/colqwen-omni-v0.1", revision="refs/pr/N")
```


### Text Retrieval Models

이 모델들은 저장된 구성에서 회수된 학습된 접두 토큰, 질의 확장 및 구두점 건너뛰기 리스트와 함께 로드됩니다.

NanoBEIR 열은 13개의 [NanoBEIR datasets](https://huggingface.co/datasets/sentence-transformers/NanoBEIR-en) 전체에서 평균 NDCG@10를 보고합니다(높을수록 좋음). 이는 BEIR 데이터셋의 50쿼리 부분 샘플에 대한 영어 텍스트 검색 품질의 빠른 지표로 사용됩니다. 우리는 주로 영어 모델의 점수를 계산하기 위해 `MultiVectorNanoBEIREvaluator` 을 사용했습니다. `-` 는 모델이 이에 대해 평가되지 않았음을 의미합니다. NanoBEIR 는 작은 벤치마크이며, 그 점수들은 자체 데이터에 대해 평가하는 것을 대체하지 못합니다. 모델을 선택하는 올바른 방법은 항상 자신의 데이터로 평가하는 것입니다.

| Model | Parameters | Dimensionality | NanoBEIR | Notes |
| --- | :---: | :---: | :---: | --- |
| [lightonai/LateOn-regularized](https://huggingface.co/lightonai/LateOn-regularized) | 149M | 128 | 0.6897 | - |
| [lightonai/LateOn-hpool-regularized](https://huggingface.co/lightonai/LateOn-hpool-regularized) | 149M | 128 | 0.6876 | - |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) | 149M | 128 | 0.6868 | - |
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | 353M | 128 | 0.6864 | needs `trust_remote_code=True` |
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | 307M | 128 | 0.6851 | - |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 149M | 128 | 0.6720 | - |
| [topk-io/Iso-ModernColBERT](https://huggingface.co/topk-io/Iso-ModernColBERT) | 149M | 128 | 0.6687 | - |
| [perplexity-ai/pplx-embed-v1-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b) | 596M | 128 | 0.6662 | needs `trust_remote_code=True` |
| [lightonai/ColBERT-Zero](https://huggingface.co/lightonai/ColBERT-Zero) | 149M | 128 | 0.6569 | - |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 33M | 96 | 0.6550 | - |
| [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | 32M | 64 | 0.6524 | - |
| [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 353M | 128 | 0.6441 | - |
| [mixedbread-ai/mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) | 17M | 48 | 0.6407 | - |
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0) | 110M | 128 | 0.6201 | - |
| [lightonai/LateOn-Code](https://huggingface.co/lightonai/LateOn-Code) | 149M | 128 | 0.6169 | - |
| [lightonai/Agent-ModernColBERT](https://huggingface.co/lightonai/Agent-ModernColBERT) | 149M | 128 | 0.6164 | - |
| [lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT) | 149M | 128 | 0.6078 | - |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | 110M | 128 | 0.6053 | - |
| [VAGOsolutions/SauerkrautLM-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-EuroColBERT) | 212M | 128 | 0.5982 | - |
| [antoinelouis/colbert-xm](https://huggingface.co/antoinelouis/colbert-xm) | 853M | 128 | 0.5915 | - |
| [VAGOsolutions/SauerkrautLM-Multi-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ModernColBERT) | 149M | 128 | 0.5886 | - |
| [mixedbread-ai/mxbai-colbert-large-v1](https://huggingface.co/mixedbread-ai/mxbai-colbert-large-v1) | 335M | 128 | 0.5733 | `revision="refs/pr/4"` |
| [lightonai/LateOn-Code-edge](https://huggingface.co/lightonai/LateOn-Code-edge) | 17M | 48 | 0.5274 | - |
| [VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT) | 149M | 128 | 0.5267 | - |
| [VAGOsolutions/SauerkrautLM-Reason-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Reason-EuroColBERT) | 212M | 128 | 0.4479 | - |
| [NeuML/biomedbert-base-colbert](https://huggingface.co/NeuML/biomedbert-base-colbert) | 110M | 128 | 0.4320 | - |
| [yjoonjang/colbert-ko-v1](https://huggingface.co/yjoonjang/colbert-ko-v1) | 149M | 128 | - | - |
| [ytu-ce-cosmos/turkish-colbert](https://huggingface.co/ytu-ce-cosmos/turkish-colbert) | 111M | 256 | - | - |
| [samheym/GerColBERT](https://huggingface.co/samheym/GerColBERT) | 110M | 128 | - | - |

### Visual Document Retrieval Models

ColPali 스타일 모델은 페이지 이미지를 문서로, 텍스트를 질의로 임베딩합니다.

NanoViDoRe 열은 [NanoViDoRe v3](https://huggingface.co/datasets/lightonai/NanoViDoRe_v3) 전체에서 평균 NDCG@10를 보고합니다. 이 벤치마크는 8개 부분집합(컴퓨터 과학, 에너지, 영어 및 프랑스어 재정, HR, 산업, 제약, 물리학)을 아우르는compact한 시각적 문서 검색 벤치마크입니다. NanoBEIR처럼 NanoViDoRe 역시 작기 때문에 자체 데이터에 대한 평가를 대체하지 않습니다.

| Model | Parameters | Dimensionality | NanoViDoRe | Notes |
| --- | :---: | :---: | :---: | --- |
| [webAI-Official/webAI-ColVec1.1-8b](https://huggingface.co/webAI-Official/webAI-ColVec1.1-8b) | 8.4B | 640 | 0.6580 | needs `trust_remote_code=True` |
| [webAI-Official/webAI-ColVec1.1-4b](https://huggingface.co/webAI-Official/webAI-ColVec1.1-4b) | 4.5B | 640 | 0.6520 | needs `trust_remote_code=True` |
| [tencent/EVIE-Preview-4.5B](https://huggingface.co/tencent/EVIE-Preview-4.5B) | 4.54B | 128 | 0.6405 | - |
| [TomoroAI/tomoro-colqwen3-embed-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b) | 8.8B | 320 | 0.6206 | needs `trust_remote_code=True` |
| [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 4.4B | 320 | 0.6019 | needs `trust_remote_code=True` |
| [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | 3.8B | 128 | 0.5402 | - |
| [vidore/colqwen2.5-v0.1](https://huggingface.co/vidore/colqwen2.5-v0.1) | 3.8B | 128 | 0.5395 | - |
| [vidore/colqwen-omni-v0.1](https://huggingface.co/vidore/colqwen-omni-v0.1) | 4.4B | 128 | 0.5309 | - |
| [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) | 2.9B | 128 | 0.4802 | - |
| [vidore/colpali-v1.3-hf](https://huggingface.co/vidore/colpali-v1.3-hf) | 2.9B | 128 | 0.4793 | - |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) | 2.9B | 128 | 0.4691 | - |
| [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) | 2.2B | 128 | 0.4685 | - |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 2.2B | 128 | 0.4526 | - |
| [vidore/colpali](https://huggingface.co/vidore/colpali) | 2.9B | 128 | 0.4516 | - |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1) | 2.9B | 128 | 0.4314 | - |
| [vidore/colsmolvlm-v0.1](https://huggingface.co/vidore/colsmolvlm-v0.1) | 2.1B | 128 | 0.4054 | - |
| [vidore/colpali-hard-v1.1](https://huggingface.co/vidore/colpali-hard-v1.1) | 2.9B | 128 | 0.3949 | - |
| [vidore/colSmol-500M](https://huggingface.co/vidore/colSmol-500M) | 507M | 128 | 0.3459 | - |
| [vidore/colSmol-256M](https://huggingface.co/vidore/colSmol-256M) | 256M | 128 | 0.2673 | - |
| [ModernVBERT/colmodernvbert](https://huggingface.co/ModernVBERT/colmodernvbert) | 252M | 128 | 0.2632 | - |
| [vidore/colpali-v1.2-hf](https://huggingface.co/vidore/colpali-v1.2-hf) | 2.9B | 128 | - | - |
| [vidore/colqwen2-v1.0-hf](https://huggingface.co/vidore/colqwen2-v1.0-hf) | 2.2B | 128 | - | - |

대부분의 이 저장소들은 LoRA 어댑터 저장소이며, 로드 시 베이스에 어댑터를 직접 적용합니다. 일부는 HUB 에도 `-merged` 형제({예: [vidore/colpali-v1.3-merged](https://huggingface.co/vidore/colpali-v1.3-merged))가 있어 가중치에 이미 어댑터가 접혀 있습니다.

세 가지 `-hf` 항목은 트랜스포머 네이티브 `*ForRetrieval` 포트입니다. 구성 없이 로드되지만, `transformers`의 모델링을 더 많이 활용하고 `sentence_transformers`의 활용은 적습니다. 일반적으로 원래 모델을 사용하는 편이 좋으며, 포트의 점수는 대략 동일합니다.

## 감사의 말씀 {#section-19}

Sentence Transformers의 늦은 상호작용은 다수의 선행 연구에 의지합니다. 이 글의 기반이 된 [ColBERT](https://arxiv.org/abs/2004.12832) 를 만든 Omar Khattab과 Matei Zaharia 및 수년간 늦은 인터랙션을 이끌고 API의 상당 부분을 형성한 LightOn 팀(Antoine Chaffin, Raphael Sourty, Paulo Moura, Amélie Chatelain) 에 깊은 감사의 뜻을 전합니다.

ColPali 팀(Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo)과 colpali-engine 에 대해 감사드리며, 페이지 이미지에 늦은 인터랙션을 가져온 것에 대해, 그리고 [token pooling](https://arxiv.org/abs/2409.14683v1) 을 제공해 준 Benjamin Clavié, Antoine Chaffin, Griffin Adams 에도 감사합니다.

또한 다수의 정보 검색 연구를 계속 이끄는 숨은 작업에 기여한 핵심 MTEB 팀의 Kenneth Enevoldsen 및 Roman Solomatin 등 모든 분들께 감사드립니다. [MTEB](https://github.com/embeddings-benchmark/mteb) 와 체크포인트를 공개해 주신 분들에게도 감사합니다.

또한 이 포스트의 체크포인트를 학습하고 공개해 주신 모든 분들께 감사합니다. 이들이 없었다면 이 글은 측정할 것이 전혀 없었을 것입니다.

## 추가 리소스 {#section-20}

### 문서

- [Multi-Vector Encoder > Usage](https://sbert.net/docs/multi_vector_encoder/usage/usage.html)
- [Multi-Vector Encoder > Pretrained Models](https://sbert.net/docs/multi_vector_encoder/pretrained_models.html)
- [Multi-Vector Encoder > Creating Custom Models](https://sbert.net/docs/multi_vector_encoder/usage/custom_models.html)
- [Multi-Vector Encoder > Speeding up Inference](https://sbert.net/docs/multi_vector_encoder/usage/efficiency.html)
- [Multi-Vector Encoder > API Reference](https://sbert.net/docs/package_reference/multi_vector_encoder/index.html)
- [Installation](https://sbert.net/docs/installation.html)
- [Migration Guide](https://sbert.net/docs/migration_guide.html)

### 예제 스크립트

- [Semantic Search](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/applications/semantic_search.py)
- [Retrieve and Rerank](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/applications/retrieve_rerank.py)
- [Token Pooling](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/compression/token_pooling.py)
- [ColPali Heatmaps](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/interpretability/heatmap.py)
- [Text Similarity Maps](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/interpretability/text_similarity_map.py)
- [NanoBEIR Evaluation](https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/evaluation/nano_beir.py)

### 학습

이 모델을 자신의 데이터로 학습하거나 미세조정하는 방법을 배우려면:

<!--
See the companion blogpost: [Training and Finetuning Multi-Vector Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-multi-vector-encoder).
-->

- [Multi-Vector Encoder > Training Overview](https://sbert.net/docs/multi_vector_encoder/training_overview.html)
- [Multi-Vector Encoder > Loss Overview](https://sbert.net/docs/multi_vector_encoder/loss_overview.html)
- [Multi-Vector Encoder > Training Examples](https://sbert.net/docs/multi_vector_encoder/training/examples.html)
- [LateOn and mLateOn training scripts](https://github.com/lightonai/mdenseon-mlateon): LightOn의 PyLate 레시피로 LateOn, mLateOn, DenseOn, 및 mDenseOn에 대한 미세조정 스크립트에서 16,384 예제의 배치를 16개씩의 미니배치로 분할하는 등의 실용적 디테일을 보여줍니다.

### Hugging Face Hub

- [Multi-vector models on the Hub](https://huggingface.co/models?library=sentence-transformers&other=multi-vector)
- [Sentence Transformers datasets on the Hub](https://huggingface.co/datasets?other=sentence-transformers)

### Companion Blogposts

<!--
- [Training and Finetuning Multi-Vector Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-multi-vector-encoder): the direct training companion to this post.
-->

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): 텍스트 전용 Dense 임베딩 모델의 일반 학습 가이드.
- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): Cross Encoder 학습, 정확한 두 번째 단계를 추가하는 다른 방법.
- [Training and Finetuning Sparse Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sparse-encoder): SPLADE 및 기타 Sparse 인코더, 하이브리드 검색에서 늦은 인터랙션과 잘 조합됩니다.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): 단일 벡터 멀티모달 모델, ColPali 스타일 검색의 Dense 버전.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): 단일 벡터 모델과 함께 하는 Visual Document Retrieval 워크스루 포함.
- [🪆 Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): 차원별로 Dense 임베딩 축소, 토큰 풀링이 멀티-벡터를 개수로 축소하는 방식.
