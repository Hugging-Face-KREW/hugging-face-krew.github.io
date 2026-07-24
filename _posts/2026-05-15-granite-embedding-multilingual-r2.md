---
layout: post
title: "Granite Embedding Multilingual R2: 32K 컨텍스트를 갖춘 Apache 2.0 다국어 임베딩 — 100M 미만 중 최상의 검색 품질"
author: dailybot
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-05-15-granite-embedding-multilingual-r2/thumbnail.png
slug: "granite-embedding-multilingual-r2"
source_url: "https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2"
source_published_date: "2026-05-14"
source_published_at: "2026-05-14T18:55:01+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Granite Embedding Multilingual R2: Open Apache 2.0 Multilingual Embeddings with 32K Context — Best Sub-100M Retrieval Quality](https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-granite/granite-embedding-multilingual-r2 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Granite Embedding Multilingual R2: 32K 컨텍스트를 갖춘 Apache 2.0 다국어 임베딩 — 100M 미만 중 최상의 검색 품질

TL;DR: ModernBERT 기반의 두 개의 신규 Apache 2.0 다국어 임베딩 모델 — 97M 매개변수의 컴팩트 모델은 MTEB 다국어 검색에서 모든 오픈 100M 미만 다국어 임베더를 이기는 60.3의 성능을 보이고, 311M 풀사이즈 모델은 MTEB 다국어 검색에서 65.2점으로 오픈 모델 중 500M 매개변수 이하에서 2위에 해당하며 Matryoshka 지원이 더해져 있습니다. 두 모델 모두 200개가 넘는 언어를 아우르고, 52개 언어에 대해 튜닝되었으며, 32,768 토큰의 컨텍스트를 처리하고(이전 R1 대비 64배 증가), 9개 프로그래밍 언어에 대한 코드 검색도 추가합니다.

이번 포스트에서: [Enterprise-Ready by Design](#enterprise-ready-by-design) · [A Strong Sub-100M Multilingual Model](#a-strong-sub-100m-multilingual-model) · [What Changed from R1](#what-changed-from-r1) · [Training the Full-Size 311M Model](#training-the-full-size-311m-model) · [Building the compact 97M Multilingual Model](#building-the-compact-97m-multilingual-model) · [Benchmark Results](#benchmark-results) · [Matryoshka Embeddings](#matryoshka-embeddings-311m) · [Deployment Options](#deployment-options) · [For Framework Integrators](#for-framework-integrators) · [Which Model Should You Use?](#which-model-should-you-use) · [Try The Models](#try-the-models)

다국어 임베딩 모델은 항상 긴장 상태에 놓여 있습니다. 넓은 언어 커버리지는 보통 모델 크기의 증가를 동반하고, 작은 모델은 일반적으로 언어 수를 포기하곤 합니다. 다국어 코퍼스 기반의 검색-확장 생성, 다국어 간 검색, 국제 팀에서의 코드 검색 등 다양한 언어를 다루는 작업을 한다면 — 빠른 모델과 충분한 품질 사이에서 선택해야 하는 경우가 많습니다.

Granite Embedding Multilingual R2 릴리스는 그 격차를 상당히 좁힙니다. 두 개의 신규 다국어 임베딩 모델을 출시합니다:

- [granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) — 768 차원의 임베딩, Matryoshka 차원 지원, 그리고 최상위 다국어 검색 품질의 311M 매개변수 풀사이즈 모델.

- [granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) — 384 차원 임베딩을 가지는 97M 매개변수의 컴팩트 모델로, 그 규모에 비해 강력한 검색 품질을 제공합니다.

두 모델 모두 200+개 언어를 지원하고, 52개 언어 및 프로그래밍 코드에 대해 향상된 검색 품질, 32,768 토큰까지의 컨텍스트 처리(이전 모델 대비 64배 증가), Apache 2.0 라이선스 하에 배포됩니다. 기본적으로 `sentence-transformers`와 `transformers`에서 바로 작동하며, 과제별 특별한 지시가 필요 없고 LangChain, LlamaIndex, Haystack, Milvus에서 모델 이름 한 줄의 변경으로 드롭인 대체로 사용할 수 있습니다. 현재 영어 전용 기본을 사용하는 프레임워크의 경우 한 줄 변경으로 커뮤니티의 모든 사용자가 200+개 언어를 지원하게 됩니다 — API 변경, 새로운 의존성, 또는 End에서의 코드 변경이 필요 없습니다. 두 모델 모두 CPU 최적화 추론용 ONNX 및 OpenVINO 가중치를 함께 제공합니다.

기저 인코더는 200+개 언어의 텍스트로 사전 학습되어, 그 중 어떤 언어에 대해서도 일반 목적의 임베딩을 생성합니다. 아래의 52개 언어는 명시적 검색 페어 및 교차 언어 훈련이 적용되어 더 높은 품질의 검색을 제공합니다:

알바니아어(sq), 아랍어(ar), 아제르바이잔어(az), 벵골어(bn), 불가리아어(bg), 카탈루냐어(ca), 중국어(zh), 크로아티아어(hr), 체코어(cs), 덴마크어(da), 네덜란드어(nl), 영어(en), 에스토니아어(et), 핀란드어(fi), 프랑스어(fr), 조지아어(ka), 독일어(de), 그리스어(el), 히브리어(he), 힌디어(hi), 헝가리어(hu), 아이슬란드어(is), 인도네시아어(id), 이탈리아어(it), 일본어(ja), 카자흐어(kk), 크메르어(km), 한국어(ko), 라트비아어(lv), 리투아니아어(lt), 말레이어(ms), 마라티어(mr), 노르웨이어(no), 페르시아어(fa), 폴란드어(pl), 포르투갈어(pt), 루마니아어(ro), 러시아어(ru), 세르비아어(sr), 슬로바키아어(sk), 슬로베니아어(sl), 스페인어(es), 스와힐리어(sw), 스웨덴어(sv), 타갈로그어(tl), 텔루구어(te), 태국어(th), 터키어(tr), 우크라이나어(uk), 우르두어(ur), 우즈베크어(uz), 베트남어(vi).

또한, 이 모델들은 프로그래밍 코드(Python, Go, Java, JavaScript, PHP, Ruby, SQL, C, C++)로도 학습되어 교차 언어 코드 검색도 지원합니다.

## Enterprise-Ready by Design

두 임베딩 모델은 IBM이 큐레이션한 데이터셋, 공개 데이터, 내부 생성 데이터(또는 합성 데이터)를 혼합하여 학습했습니다. 학습 데이터에 사용된 공개 웹 기반 데이터는 IBM이 개발한 품질 관리, 중복 제거 및 거버넌스 프로세스를 통해 선정 및 필터링되어 다운스트림 상용 사용에서의 위험을 줄이고자 합니다. MS-MARCO 학습 데이터와 명시적으로 비상업적 라이선스 제한이 있는 데이터의 사용은 의도적으로 피했습니다. 이 모델들은 [GneissWeb](https://huggingface.co/datasets/ibm-granite/GneissWeb)으로 사전 학습되었으며, 공개 웹 콘텐츠에서 파생된 IBM이 관리하는 데이터 세트와 IBM의 데이터 준비 및 거버넌스 도구를 통해 처리된 추가 IBM 관리 및 기타 공개 소스를 포함합니다. 데이터 세트는 라이선스 고려사항, 소유권 신호 및 개인 데이터 위험 등을 평가하기 위한 IBM 거버넌스 심사를 거칩니다. 이러한 프로세스는 책임 있는 사용 및 기업 배포에 기여하도록 설계되었습니다.

## A Strong Sub-100M Multilingual Model

주목할 만한 점은 gran ite-embedding-97m-multilingual-r2입니다. 9700만 매개변수로, 18개 언어에 걸친 Multilingual MTEB Retrieval에서 60.3점을 기록해, 100M 매개변수 이하의 오픈 다국어 임베딩 모델 중에서 가장 높은 검색 품질을 보여주었습니다. 동일한 규모 클래스의 다음으로 좋은 모델인 multilingual-e5-small은 같은 벤치마크에서 50.9점을 기록합니다 — 성숙한 벤치마크에서 +9.4 포인트 차이입니다.

311M 풀사이즈 모델의 약 1/3 규모임에도 다국어, 코드, 장문 벤치마크에서 다수의 품질을 보유하고 있으며, 다국어 다변성 대비 영어 심도에서의 보강 등으로 인해 직접 선행 모델보다 +12.2 포인트의 MTEB 다국어 검색 향상을 보입니다. 풀사이즈 granite-embedding-311m-multilingual-r2는 동일 벤치마크에서 65.2를 기록하고, R1 선행 모델에 비해 평균 +14.5 포인트의 향상을 제공합니다.

## What Changed from R1

Granite Embedding Multilingual R1 모델은 512 토큰 컨텍스트 윈도우를 가진 XLM-RoBERTa 인코더를 기반으로 했습니다. R2 생애 주기는 처음부터 재구축한 버전입니다:

[ModernBERT](https://huggingface.co/blog/modernbert)는 최근의 인코더 아키텍처로, 지난 5년간의 트랜스포머 연구의 기법들을 재고합니다. 실용적 이점으로는 긴 시퀀스에서의 주의 집중 길이를 번갈아 조정하여 계산량을 줄이고(긴 시퀀스 처리 속도 크게 향상), 32K 컨텍스트 윈도우를 가능하게 하는 로터리 포지션 임베딩, 이전 아키텍처에서 문제가 되었던 위치 보간 해킹 없이도 32K 컨텍스트를 활용할 수 있는 구조, 그리고 Flash Attention 2.0 지원으로 현대 GPU에서 인코딩이 빨라지는 점 등이 있습니다.

새로운 다국어 토크나이저도 주목할 만합니다. XLM-RoBERTa의 250K 토큰 어휘를 재사용하기보다, 강력한 다국어 및 코드 커버리지를 가진 기존 토크나이저를 채택했습니다. 311M 모델은 Gemma 3 토크나이저(262K 토큰)를 사용하고, 97M 모델은 GPT-OSS 토크나이저에서 시작해 180K 토큰 어휘로 축소하여 광범위한 다국어 커버리지를 유지하면서 임베딩 테이블의 매개변수 규모를 줄였습니다. 토크나이저의 효율성은 우리가 생각하는 것보다 더 중요합니다 — 32K 컨텍스트 윈도우가 인상적으로 들리더라도, 토크나이저가 태국어 한 문단을 인코딩하는 데 절반을 소모하면 의미가 반감됩니다.

## Training the Full-Size 311M Model

311M 모델은 22-layer ModernBERT 인코더와 262K 토큰 다국어 어휘를 갖고, 다단계 파이프라인으로 학습됩니다:

- 지식 증류(Knowledge distillation): 모델은 여러 교사 모델로부터 동시에 학습합니다. 교사는 Granite 3.3 Instruct 및 Mistral v0.2 Instruct 디코더 기반 모델이며, 텍스트 임베딩에 맞춰 추가로 미세 조정되어 검색 전용 지식을 311M 인코더 아키텍처로 전달합니다.

- 대조 정합(Contrastive fine-tuning): 52개 언어 및 코드에 걸친 다국어 검색 쌍에 대해 표준 대조 학습을 수행하여 관련 결과와 무관한 결과를 구분하는 능력을 향상시킵니다.

- 모델 병합(Model merging): 학습 후, 다양한 학습 단계 및 구성의 체크포인트를 병합합니다. 이는 다국어 폭넓은 모델과 영어 심층 모델처럼 서로 다른 목표를 최적화한 모델들의 강점을 하나의 가중치 세트로 합쳐 추가 학습 계산 없이 사용할 수 있게 합니다.

- Matryoshka 표현 학습(Matryoshka Representation Learning): 768 차원의 임베딩을 512, 384, 256, 또는 128 차원으로 잘라내더라도 손실이 최소화되도록 Matryoshka 목표로 학습합니다(아래의 [Matryoshka Embeddings](#matryoshka-embeddings-311m) 참조).

그 결과 MTEB 다국어 검색에서 65.2, 전체 평균에서 56.3의 점수를 얻어 R1 선행 모델에 비해 평균 +14.5 포인트의 이득을 제공합니다.

## Building the compact 97M Multilingual model

97M 모델은 어휘 선택과 지식 증류의 조합으로 학습됩니다:

- 어휘 선택(Vocabulary selection): 262K 토큰 어휘를 목적에 맞게 학습된 180K 토큰 어휘로 축소하여 광범위한 다국어 커버리지를 유지하면서 임베딩 테이블의 크기를 크게 줄입니다.

- 지식 증류(Knowledge distillation): 잘라낸 모델은 여러 교사 모델(그레이나이트 4.1 8B 및 Mistral Instruct 디코더 기반 교사 포함)로부터 지식 증류와 대조 학습을 통해 검색 품질을 개선합니다.

이 접근 방식은 다수의 강력한 교사들로부터 검색 전용 지식을 전달받으면서도 언어 커버리지를 해치지 않고 모델 매개변수를 줄이는 효과를 냅니다. 결과적으로 높은 효율의 컴팩트 모델이 나오는데, 다국어 검색에서 60.3점을 기록하고(97M 대비) 풀사이즈 모델은 65.2점을 기록하며 약 3배 정도 더 작습니다.

## Benchmark Results

### Multilingual Retrieval

모델 크기에 따라 정렬된 주요 벤치마크의 성능. 점수는 각 벤치마크 내 태스크들의 평균(높을수록 좋음):

몇 가지 특징이 눈에 띕니다:

- 97M R2 모델은 multilingual-e5-base 및 gte-multilingual-base(약 300M 매개변수 모델들)보다 평균 및 대부분의 개별 벤치마크에서 더 좋으며, 약 3배 작은 크기에도 불구하고 우수한 성능을 보입니다.

- `paraphrase-multilingual-MiniLM-L12-v2` — 널리 사용되는 프레임워크 기본값 — 36.6으로, 97M R2 모델보다 23.7 포인트 낮습니다. 같은 384차원 출력임에도 매개변수 수는 97M과 110M으로 차이가 있습니다.

- LongEmbed은 R1에서 R2로의 가장 큰 증가폭입니다: 97M 모델에서 +31.3 포인트, 311M에서 +34.0 포인트. 이는 32K 컨텍스트 윈도우의 직접적인 보상으로, R1의 512 토큰 한계가 처음 페이지의 내용으로만 계약서를 판단하던 시절을 반영합니다. 다국어 워크로드 중 다수는 긴 문서(법적 계약서, 기술 매뉴얼, 연구 논문, 다페이지 보고서)를 다루므로 R1로는 전체를 볼 수 없었습니다.

- 코드 검색은 급격히 향상됩니다: R1 대비 +19.7(97M), +15.3(311M). 새로운 코드 학습 세트, 더 큰 컨텍스트 윈도우, 더 나은 학습 방법론의 효과입니다.

- 더 넓은 경쟁 구도에서 harrier-oss-v1-270m가 MTEB 다국어 검색에서 선두를 달리고(RaR-b 32.9 포함) 있고, jina-embeddings-v5-text-nano가 코드(71.2)와 영어 검색(58.8)에서 선두를 달리는 가운데, 311M Granite 모델은 평균적으로는 경쟁력이 있으며(LongEmbed 71.7에서 이점을 보이고), jina-embeddings-v5-text-nano보다 인코딩 처리량이 훨씬 높습니다(아래 속도 표 참조).

### 속도 및 처리량

생산 워크로드에서 인코딩 속도는 중요합니다. 특히 수백만 개 문서를 인덱싱하거나 낮은 지연의 쿼리 인코딩이 필요한 경우 더욱 그렇습니다. 우리는 512-token 청크를 사용한 단일 NVIDIA H100 GPU에서 지연 시간과 처리량을 측정했습니다:

97M 모델은 초당 2,500건 이상의 문서를 인코딩 — 다국어-e5-small와 비슷한 처리량이며, 검색 품질은 훨씬 우수합니다. 311M 모델은 약 1,800건/초로, 검색 품질 측면에서 jina-embeddings-v5-text-nano보다 더 좋으며(65.2 대 63.3), 인코딩 속도는 5.5배 이상 빠릅니다(참고: 속도 수치는 최신 트랜스포머 코드로 계산되었으며, Jina 및 Granite 모델 모두에서 4.57 버전과의 속도 저하가 있습니다. 자세한 내용은 기술 보고서를 참조하시기 바랍니다). 이 목록에 있는 경쟁 모델 중 harrier-oss-v1-270m가 가장 빠른 속도와 검색 점수의 조합을 제공합니다.

## Matryoshka Embeddings (311M)

311M 모델은 [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)을 지원하여 768 차원의 전체 임베딩을 512, 384, 256, 또는 128 차원으로 손실 없이 축소할 수 있습니다. 이는 저장소, 메모리, 유사도 계산 비용이 부담될 때 유용합니다 — 예를 들어 256 차원 임베딩은 768 차원 임베딩의 저장 공간의 3분의 1을 차지하고, 코사인 유사도 계산 비용도 비례하여 저렴해집니다.

다음은 임베딩 차원에 따른 검색 품질의 변화입니다:

차원 축소로 인한 품질 손실은 놀랄 정도로 작습니다. 768에서 256 차원으로 축소하면 저장 및 유사도 계산 비용이 3배 감소하지만 MTEB 다국어 검색은 65.2에서 64.7로 단 0.5포인트 하락하고, 코드 검색도 63.9에서 63.4로 0.5포인트 하락합니다. 128 차원(6배 축소)에서도 다국어 검색은 63.7, 코드 검색은 62.3으로 여전히 전체 차원 성능의 97% 이상을 유지합니다. 실무적으로는 인덱스 크기를 크게 줄이고 검색 지연 시간을 감소시키되 결과 품질에 미치는 영향은 최소화할 수 있습니다. (참고: 위 그림의 결과는 영어 및 다국어 검색에 대해 컨텍스트 길이 1024, 코드에 대해 8192로 평가되었다는 점을 양해 바랍니다.)

참고로 311M 모델을 384 차원으로 축소한 경우(97M 모델의 native 출력과 동일 차원) 세 벤치마크 모두에서 여전히 97M 모델보다 우수합니다. 384 차원 임베딩이 필요하고 311M 모델의 인코딩 비용을 감당할 수 있다면 Matryoshka 축소가 더 강력한 선택입니다.

```
from
 sentence_transformers 
import
 SentenceTransformer

model = SentenceTransformer(
"ibm-granite/granite-embedding-311m-multilingual-r2"
)


# Full 768-dimensional embeddings

full = model.encode([
"example text"
])

print
(full.shape)  
# (1, 768)



# Truncated to 384 dimensions

small = model.encode([
"example text"
], truncate_dim=
384
)

print
(small.shape)  
# (1, 384)
```

97M 모델은 Matryoshka를 지원하지 않습니다 — 384 차원은 이미 컴팩트합니다.

### Cross-lingual Retrieval

MTEB Retrieval 내 교차 언어 태스크의 평균 성능. Belebele은 122개 언어에 걸친 교차 언어 문단 매칭을 측정하고, MLQA는 7개 언어에 걸친 교차 언어 질의응답 검색을 측정합니다.

311M R2 모델은 R1 선행 모델에 비해 Belebele에서 +4.3, MLQA에서 +4.1의 향상을 보여, 더 큰 규모에서의 교차 언어 전달 능력이 개선되었음을 나타냅니다.

97M R2 모델은 Belebele에서 다소 낮은 점수(52.9 vs 55.1, -2.2)를 기록하지만 MLQA에서는 R1 선행 모델과 비슷한 수준을 보입니다(60.5). Belebele 차이는 가지치기 및 어휘 축소 과정에서 발생하는 트레이드오프의 결과입니다 — R2 모델의 학습은 더 넓은 18개 언어의 MTEB 다국어 검색 세트와 긴 문서 검색에 더 가중치를 둔 반면, 더 적은 어휘(180K 대 250K 토큰) 및 감소된 층 수(12 대 22)는 좁은 교차 언어 전이 태스크에 영향을 미칩니다. 다수의 언어 쌍에 걸친 교차 언어 전이가 주요 사용 사례라면 풀 사이즈의 311M 모델이 더 나은 선택입니다.

## Deployment Options

두 모델은 프로덕션 사용을 위한 여러 배포 경로를 제공합니다. 코어 라이브러리는 다음과 같이 설치합니다:

```
pip install sentence-transformers
```

대다수 사용자에게 권장되는 Sentence Transformers:

```
from
 sentence_transformers 
import
 SentenceTransformer, util

model = SentenceTransformer(
"ibm-granite/granite-embedding-97m-multilingual-r2"
)

queries = [
    
"What is the tallest mountain in Japan?"
,          
# 영어

    
"Wer hat das Lied Achy Breaky Heart geschrieben?"
, 
# 독일어

    
"ドイツの首都はどこですか？"
,                            
# 일본어

]

passages = [
    
"富士山은 静岡県と 山梨県에 걸친活火山으로、標高3776.12 mで日本最高峰の独立峰である。"
,  
# 일본어

    
"Achy Breaky Heart is a country song written by Don Von Tress."
,                        
# 영어

    
"Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland."
,                
# 독일어

]

q_emb = model.encode(queries)
p_emb = model.encode(passages)

print
(util.cos_sim(q_emb, p_emb))

# 각 쿼리는 대응하는 문장과 가장 높은 점수를 얻습니다 — 언어 간에 걸쳐
```

LangChain (`pip install langchain-huggingface`):

```
from
 langchain_huggingface 
import
 HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=
"ibm-granite/granite-embedding-97m-multilingual-r2"

)

docs = embeddings.embed_documents([
    
"富士山は日本最高峰の独立峰です。"
,
    
"Mount Fuji is Japan's highest peak."
,
])
query = embeddings.embed_query(
"What is Japan's tallest mountain?"
)

# LangChain이 Embeddings 객체를 받는 곳이라면 어디서나 드롭인 대체
```

LlamaIndex (`pip install llama-index-embeddings-huggingface`):

```
from
 llama_index.embeddings.huggingface 
import
 HuggingFaceEmbedding

from
 llama_index.core 
import
 Settings

embed_model = HuggingFaceEmbedding(
    model_name=
"ibm-granite/granite-embedding-97m-multilingual-r2"

)
Settings.embed_model = embed_model  
# 인덱스나 파이프라인에 전역적으로 적용
```

```
from
 haystack.components.embedders 
import
 (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

from
 haystack.components.retrievers.in_memory 
import
 InMemoryEmbeddingRetriever

from
 haystack.dataclasses 
import
 Document

from
 haystack.document_stores.in_memory 
import
 InMemoryDocumentStore

doc_embedder = SentenceTransformersDocumentEmbedder(
    model=
"ibm-granite/granite-embedding-97m-multilingual-r2"

)
query_embedder = SentenceTransformersTextEmbedder(
    model=
"ibm-granite/granite-embedding-97m-multilingual-r2"

)
doc_embedder.warm_up()
query_embedder.warm_up()


# Embed and index documents

document_store = InMemoryDocumentStore()
result_docs = doc_embedder.run(documents=[
    Document(content=
"富士山は日本最高峰の独立峰です。"
),
    Document(content=
"Mount Fuji is Japan's highest peak."
),
    Document(content=
"Achy Breaky Heart is a country song written by Don Von Tress."
),
    Document(content=
"Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland."
),
])
document_store.write_documents(result_docs[
"documents"
])


# Embed query and retrieve

result_query = query_embedder.run(text=
"What is Japan's tallest mountain?"
)
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
results = retriever.run(query_embedding=result_query[
"embedding"
], top_k=
2
)

for
 doc 
in
 results[
"documents"
]:
    
print
(
f"
{doc.score:
.3
f}
  
{doc.content}
"
)

# 0.961  Mount Fuji is Japan's highest peak.


# 0.913  富士山は日本最高峰の独立峰です。
```

```
from
 pymilvus 
import
 MilvusClient

from
 sentence_transformers 
import
 SentenceTransformer

model = SentenceTransformer(
"ibm-granite/granite-embedding-97m-multilingual-r2"
)


# Use "./milvus.db" for local persistence or a server URI for production

client = MilvusClient(
":memory:"
)
client.create_collection(collection_name=
"multilingual_docs"
, dimension=
384
)

docs = [
    
"富士山は日本最高峰の独立峰です。"
,
    
"Mount Fuji is Japan's highest peak."
,
    
"Achy Breaky Heart is a country song written by Don Von Tress."
,
    
"Berlin ist die Hauptstadt und ein Land der Bundesrepublik Deutschland."
,
]
embeddings = model.encode(docs).tolist()
client.insert(
    collection_name=
"multilingual_docs"
,
    data=[{
"id"
: i, 
"vector"
: emb, 
"text"
: doc} 
for
 i, (emb, doc) 
in
 
enumerate
(
zip
(embeddings, docs))],
)

query_emb = model.encode([
"What is Japan's tallest mountain?"
]).tolist()
results = client.search(
    collection_name=
"multilingual_docs"
,
    data=query_emb,
    limit=
2
,
    output_fields=[
"text"
],
)

for
 hit 
in
 results[
0
]:
    
print
(
f"
{hit[
'distance'
]:
.3
f}
  
{hit[
'entity'
][
'text'
]}
"
)

# 0.961  Mount Fuji is Japan's highest peak.


# 0.913  富士山は日本最高峰の独立峰です。
```

두 모델은 또한 최적화된 CPU/가속기 추론을 위한 미리 변환된 ONNX 및 OpenVINO 가중치를 함께 제공하고, [vLLM](https://docs.vllm.ai/)의 임베딩 엔드포인트(`vllm serve ... --task embed`)로 동작하며, [llama.cpp](https://github.com/ggerganov/llama.cpp)를 사용하여 [Ollama](https://ollama.com/)로 GGUF로 변환할 수 있습니다. 전체 배포 예시는 모델 카드를 확인하세요.

## For Framework Integrators

임베딩 프레임워크, 벡터 스토어, 또는 RAG 파이프라인 라이브러리를 유지 관리하고 있으며 기본값으로 이 모델을 평가 중인 경우, 알아둘 점은 다음과 같습니다:

- 라이선스: Apache 2.0, MS-MARCO 학습 데이터 사용 안 함

- 드롭인 동작: 과제별 지시문 접두사가 필요 없으며 API 수준에서 `all-MiniLM-L6-v2`처럼 동작합니다. `.encode()`를 호출하는 기존 코드는 그대로 작동합니다.

- 차원 수: 384차원 출력(97M) 및 768차원 출력(311M), 대부분의 기존 기본값과 일치합니다. 인덱스 마이그레이션 필요 없음.

- 모델 크기: 97M 모델의 가중치가 195MB(safetensors)로, 가장 일반적인 다국어 기본값인 `paraphrase-multilingual-MiniLM-L12-v2`(471MB)의 절반 이하입니다. 양자화된 ONNX 가중치는 98MB에 불과하며, 200개 이상 언어를 다룹니다.

- CPU 친화적: 최적화된 CPU 추론용 ONNX 및 OpenVINO 가중치를 포함합니다. 시작 가이드를 위한 GPU 의존성은 없습니다.

- 기본적으로 다국어 지원: 현재 기본이 영어 전용인 경우에도 이 한 줄 교환으로 커뮤니티의 모든 사용자에게 200+개 언어 지원을 제공합니다 — 코드 수정 없이 가능합니다.

- 안정적인 식별자: Hugging Face의 `ibm-granite/granite-embedding-97m-multilingual-r2`로, IBM이 Granite 모델군 아래에서 관리합니다.

프로젝트에서 기본값으로 이 모델들을 채택하는 논의는 [ibm-granite/granite-embedding-models](https://github.com/ibm-granite/granite-embedding-models) 이슈를 열어 진행해 주세요 — 통합, 테스트, 라이선스나 배포에 관한 문의에 기꺼이 도움 드립니다.

## Which Model Should You Use?

이 두 다국어 모델은 Granite Embedding R2 계열의 일부로, 영어에 특화된 두 모델도 함께 포함하고 있습니다: [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) (149M 매개변수) 및 [granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) (47M 매개변수). 데이터가 주로 영어인 경우, 영어 모델은 200+ 언어를 포괄하는 비용을 들이지 않고도 영어 벤치마크에서 더 높은 검색 품질을 제공합니다.

## Try The Models

두 모델은 지금 바로 Hugging Face의 [IBM Granite Embedding 컬렉션](https://huggingface.co/collections/ibm-granite/granite-embedding-models)에서 확인할 수 있습니다:

- [granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2)

- [granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2)

또한 곧 Granite Embedding 데모(곧 공개 예정) on Hugging Face Spaces를 통해 작은 모델을 CPU에서 대화식으로 체험하거나 Google Colab에서 전체 예제 노트북을 실행해 볼 수 있습니다:

전체 학습 방법론, 언어별 평가, 가지치기 연구의 자세한 기술 보고서는 여기에 있습니다 [Granite Multilingual Embedding R2 report](https://arxiv.org/abs/2605.13521). 질문이나 피드백, 이슈는 [ibm-granite/granite-embedding-models](https://github.com/ibm-granite/granite-embedding-models)에서 확인해 주세요.

프레임워크 유지 관리자분들: 프로젝트의 기본값으로 이 모델을 채택하고 싶다면 [ibm-granite/granite-embedding-models](https://github.com/ibm-granite/granite-embedding-models) 이슈를 열어 주십시오 — 통합, 테스트 및 배포 관련 라이선스나 기타 문의에 기꺼이 도와드리겠습니다.

한 번 사용해 보시고 임베딩이 마음에 드신다면 Hugging Face의 ❤️ 버튼을 눌러 주세요. 저희 모델도 감정이 있으며, 한 번의 +1이 모델을 밤새 따뜻하게 지켜줍니다.
