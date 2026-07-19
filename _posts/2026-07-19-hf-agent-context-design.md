---
layout: post
title: "Hugging Face Blog Agent의 번역 ECL 파이프라인 설계 전략"
author: hyeonseo
categories: [AI]
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face Blog Agent의 ECL 파이프라인 구축 구축 경험을 담은 글입니다._

---

> Hugging Face Blog Agent의 번역 파이프라인은 ECL(Extract → Contextualize → Load) 구조에서 다음 두 가지 설계 결정을 고려하여 설계하였습니다.
>
> **(1) 실험 기반 컨텍스트 선별**
> 다양한 컨텍스트 조합을 비교하여 번역 품질과 연관성이 높은 컨텍스트 조합을 사용했습니다. 
>
> **(2) 글 특성을 고려한 번역 가이드 컨텍스트 압축**
> 방대한 번역 컨벤션·모범사례 문서를 모두 프롬프트에 넣는 대신, 원문 특성에 맞는 부분만 선별하여 재작성하는 컨텍스트 압축 파이프라인을 구현했습니다. 

# 1. 왜 ETL이 아니라 ECL인가

전통적인 데이터 파이프라인은 ETL(Extract → Transform → Load)로 설계됩니다. 그러나 LLM 기반 번역에서 Transform은 규칙으로 정의되지 않습니다. 번역 품질은 변환 로직이 아니라, 모델이 올바른 판단을 내리도록 구성한 컨텍스트(맥락)이 결정합니다.

이러한 관점 전환은 **ECL(Extract → Contextualize → Load)** 이라는 이름으로 논의되어 왔습니다. [Introducing ETL-C](https://subhadipmitra.com/blog/2024/etlc-context-new-paradigm/)에서 지식 그래프와 RAG 관점의 Contextualize 단계를 제안했고, 2026년 [Data Engineering After AI](https://www.dataengineeringweekly.com/p/data-engineering-after-ai)에서 AI가 변환 코드를 대신 작성하는 시대에 데이터 엔지니어의 역할이 Extract, Contextualize, Link로 재편된다고 정리했습니다. 출발점은 다르지만 두 논의 모두 Transform의 자리를 Context가 대체한다는 공통점이 있으며, 번역 파이프라인에도 해당 방식을 적용해보고자 시도했습니다. 

# 2. ECL 파이프라인 구조

Hugging Face Blog Agent의 번역 파이프라인을 ECL 구조로 정리하면 다음과 같습니다.

| 단계 | 역할 | 실제 구성 |
| --- | --- | --- |
| Extract | 원문 확보와 구조 분해 | RSS fetch, Markdown parse, heading/code/paragraph 블록 분리 |
| Contextualize | 번역 전 맥락 주입 | 제목·heading 경로, glossary 매칭, 블록 역할, 문체 기준, 압축된 가이드 |
| Load | 정형화된 결과물 생성 | JSON 배열 응답 → Markdown 구조 복원 → 한국어 게시물 |

Extract와 Load는 파싱과 정형화된 결과물을 생성하는 절차를 따르기 때문에 한 번 구현하면 결과가 크게 달라지지 않습니다. 
반면 Contextualize는 무엇을 얼마나 넣느냐에 따라 번역 품질과 비용이 함께 변화하는 단계로 설계 과정에서 다음 질문을 고려하게 되었습니다. 

- 컨텍스트 후보(용어집, heading 경로, 블록 구조, 문서 의도, 문체 기준, 번역 메모리 등) 중 무엇을 넣어야 품질 대비 토큰 비용이 가장 좋은가
- 컨텍스트의 원천인 가이드 문서가 계속 길어질 때, 이를 어떻게 프롬프트에 담을 것인가


# 3. 설계 전략 1: 실험 기반 컨텍스트 선별

## 3-1. 실험 설계

기존 컨텍스트 구성을 C1(baseline)으로 두고, 컨텍스트 조합을 더하거나 빼면서 9가지 변형을 구성했습니다.

| 변형 | 이름 | 구성 차이 | C1 대비 토큰 |
| --- | --- | --- | --- |
| C0 | no_context | 번역 지시와 본문만 (컨텍스트 없음) | −465 |
| C1 | ecl_baseline | 제목 + 문서 구조 + 문체 + glossary substring 매칭 | 기준 |
| C2 | hier_heading | + 위계 heading 정보 | +68 |
| C3 | morph_glossary | glossary 매칭을 어형 변형까지 확장 | +32 |
| C4 | doc_intent | + LLM이 요약한 문서 의도 | +1,522 |
| C5 | translation_memory | + 이전 청크에서 확정된 용어 | +159 |
| C6 | block_intent | + 블록 역할 라벨(intro/instruction/explanation) | +254 |
| C7 | all_combined | 전체 결합 (C2+C3+C4+C5+C6) | +2,047 |
| C8 | morphology+block | C3와 C6만 선별 결합 | +286 |

평가 지표는 glossary 준수율, 용어 일관성, 구조 보존율의 정량 지표와 LLM judge 점수(10점 만점)를 사용했고, 비용 측면은 입출력 토큰 수와 지연 시간으로 측정했습니다.

## 3-2. 실험 결과

| C | glossary | 일관성 | 구조 | judge | input tok | output tok | latency(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 0.333 | 0.272 | 1.0 | 8.000 | 1,768 | 1,286 | 24.61 |
| C1 | 1.000 | 0.952 | 1.0 | 9.333 | 2,233 | 1,293 | 26.42 |
| C2 | 1.000 | 0.952 | 1.0 | 9.333 | 2,301 | 1,293 | 23.37 |
| C3 | 1.000 | 0.898 | 1.0 | 8.333 | 2,265 | 1,299 | 28.55 |
| C4 | 1.000 | 0.952 | 1.0 | 9.000 | 3,755 | 1,392 | 35.24 |
| C5 | 1.000 | 0.952 | 1.0 | 8.333 | 2,392 | 1,289 | 28.82 |
| C6 | 1.000 | 0.952 | 1.0 | 6.000 | 2,487 | 1,296 | 26.39 |
| C7 | 1.000 | 0.952 | 1.0 | 8.000 | 4,280 | 1,571 | 35.16 |
| **C8** | **1.000** | **0.952** | **1.0** | **9.667** | 2,519 | 1,334 | 29.92 |

## 3-3. 결과 분석과 설계 결정

- **컨텍스트 유무의 차이** : C0에서는 glossary 준수율 33%, 용어 일관성 27%에 그치지만, C1 수준만 되어도 두 지표 모두 95~100%로 올라갑니다. 
- **풀세트(C7)의 효율 하락** : C7은 C8보다 토큰을 1,761개 더 쓰지만 judge 점수는 낮았습니다. 긴 컨텍스트에서 중간에 위치한 정보가 활용되지 못하는 [lost in the middle](https://arxiv.org/abs/2307.03172) 현상과, 핵심 신호가 부가 정보에 희석되는 문제가 원인으로 추정됩니다.
- **단독 효과와 결합 효과의 괴리** : 블록 역할 라벨(C6)을 단독으로 넣으면 judge가 6.000까지 떨어지지만, glossary 어형 확장(C3)과 결합한 C8은 9.667로 전체 1위를 기록했습니다. 개별 신호의 단독 측정값으로 조합의 효과를 예측할 수 없으므로, 조합 단위의 실험이 필요합니다.

Contextualize 단계는 C8 조합을 채택했습니다. 실험 결과, 어형 변화를 포함한 용어 매칭과 블록별 번역 목적의 조합이 높은 점수를 기록하여 번역 품질에 연관성이 높다고 추정하였습니다. 


# 4. 설계 전략 2: 글 특성을 고려한 번역 가이드 컨텍스트 압축

## 4-1. 문제 정의

번역 Agent가 참조하는 가이드 문서는 일반 번역 컨벤션(`docs/hf_translation_conventions.md`)과, 실제 Hugging Face 한국어 번역 PR·리뷰에서 축적된 모범사례 모음(`docs/hf_ko_translation_best_practice.md`)입니다. 모범사례는 번역과 리뷰가 쌓일수록 문서의 길이가 늘어나게 됩니다.

두 문서를 그대로 프롬프트에 넣으면 다음 문제가 발생합니다.

| 문제 | 내용 |
| --- | --- |
| 비용·지연 증가 | 프롬프트 길이에 비례해 토큰 비용과 latency 상승 |
| 지침 매몰 | 중요한 지침이 긴 문서 사이에 묻힘 |
| 무관한 지침 혼입 | 현재 글과 무관한 규칙이 번역 판단에 개입 |
| 역할 충돌 | glossary와 style guide의 역할 경계가 흐려짐 |

3장의 C7 실험에서 확인한 것과 같은 종류의 문제입니다. 필요한 것은 더 긴 프롬프트가 아니라, 현재 글에 적합한 컨텍스트라고 생각하여 컨텍스트 압축을 고려하게 되었습니다. 

## 4-2. 압축 방식 선택

동일한 문제를 다루는 연구들의 대표적인 갈래는 다음과 같습니다.

| 방식 | 개념 | 대표 연구 |
| --- | --- | --- |
| Filtering / Pruning | 덜 중요한 문장·토큰을 제거 (새 문장 생성 없음) | [Selective Context](https://arxiv.org/abs/2310.06201) |
| Prompt Compression | 프롬프트 전체를 토큰 단위로 축약 | [LLMLingua](https://arxiv.org/abs/2310.05736) |
| Long-context Aware | 핵심 정보를 잘 보이는 위치로 재배치 | [LongLLMLingua](https://arxiv.org/abs/2310.06839) |
| Retrieval Compression | query 관련 부분만 선별해 압축 | [RECOMP](https://arxiv.org/abs/2310.04408) (extractive) |
| Abstractive Capsule | LLM이 긴 내용을 짧은 새 문장으로 재작성 | [RECOMP](https://arxiv.org/abs/2310.04408) (abstractive) |
| Soft Prompt / Latent | 텍스트가 아닌 hidden vector로 압축 | [AutoCompressors](https://arxiv.org/abs/2305.14788), [ICAE](https://arxiv.org/abs/2307.06945) |

공통점은 작업에 덜 중요한 정보를 버리는 손실 압축(lossy compression)이라는 점입니다. 따라서 어떤 방식을 쓸지만큼 **무엇을 압축 대상으로 삼을지**가 중요합니다. 원문과 glossary는 정확성이 우선이므로 압축 대상에서 제외하고, 가이드 문서 두 개로 한정했습니다.

| 압축하지 않는 것 | 압축하는 것 |
| --- | --- |
| 원문 Markdown 본문 | 번역 컨벤션 문서 |
| 코드 블록·inline code placeholder | 모범사례 문서 |
| 링크·이미지 경로·표 구조 | |
| glossary, 출력 JSON 스키마, 복원 규칙 | |

방식 선택에서는 soft prompt 계열이 먼저 제외되었습니다. 모델 내부 표현에 접근해야 하므로 API 기반 workflow에서는 사용할 수 없습니다. 최종적으로 RECOMP의 두 아이디어(retrieval 선택 + abstractive 재작성)에 규칙 기반 pruning을 보조로 결합한 3단계 구성을 채택했습니다.

# 5. 3단계 가이드 압축 파이프라인

전체 흐름은 다음과 같습니다. 번역 진입점에서 압축 기능이 활성화되어 있으면 가이드 문서를 로드하고, 원문을 분석해 profile을 생성하고, 가이드를 chunk로 분할해 점수화·선별한 뒤, 압축 LLM을 한 번 호출해 짧은 capsule을 만들고, 후처리를 거쳐 번역 프롬프트의 [Context]에 삽입합니다. 

## 5-1. 분석: Source Profile 생성

원문에서 SourceProfile을 추출합니다. 제목(frontmatter title, 없으면 첫 heading), heading 최대 12개, 본문 앞부분 발췌(기본 2,500자), 그리고 7가지 feature flag로 구성됩니다.

| Flag | 감지 대상 |
| --- | --- |
| has_code | fenced / inline code |
| has_table | Markdown 표 |
| has_links | 링크 또는 URL |
| has_images | 이미지 |
| has_benchmark_terms | benchmark, mteb, score 등 |
| has_cli_terms | pip, uv, npm, conda, git, --flag 등 |
| has_model_or_dataset_terms | model, dataset, transformers 등 |

단순 regex와 키워드 기반 감지로 embedding이나 별도 분류기 없이 동작합니다. 

## 5-2. 선별: Chunking → Scoring → Selection

가이드 문서는 Markdown heading 단위로 chunk를 나눕니다. 

원문과 겹치는 단어가 많은 chunk일수록 높은 점수를 받고, 점수가 높은 순서대로 추출합니다. 

코드가 많은 글이면 코드 관련 규칙이 뽑히고, 표가 많은 글이면 표 관련 규칙이 뽑히는 방식입니다. 

최종적으로 번역 프롬프트의 [Context]는 다음과 같이 구성됩니다.

```text
[Context]
Document title: Torch MLP Fusion
Headings in document: Why it matters
Style: technical blog, natural Korean

Glossary:
- benchmark -> 벤치마크
- dataset   -> 데이터셋

Compressed translation guide:
- 코드·CLI·model ID는 번역하지 않는다
- 표의 수치와 비교 방향을 보존한다
- 홍보성 표현을 덧붙이지 않는다
```

# 6. 한계

실험 결과를 일반화하기에는 검증 범위가 크지 않습니다. 벤치마크는 문서 1개, 1회 측정 결과이고, 생성과 평가에 같은 계열의 모델을 사용해 judge 점수에 편향 가능성이 있습니다. 긴 글이나 다른 도메인 문서에 대한 검증이 필요하며, Context 조합 실험도 선별 조합으로는 C8(C3+C6)만 측정했기 때문에 다른 선별 조합이 더 나을 가능성을 배제할 수 없습니다. 파이프라인이 매일 실제 번역 PR을 생성하고 있으므로, 운영 데이터를 쌓아 검증을 이어갈 예정입니다.

# 마무리

Hugging Face Blog Agent의 번역 ECL 파이프라인은 두 가지 설계 원칙을 적용했습니다. 
Contextualize 단계에는 실험으로 검증된 최소 컨텍스트만 반영하고, 방대한 가이드 문서는 글의 특성을 고려하여 선별 및 압축합니다. 

이 글이 번역 Agent나 LLM 파이프라인의 컨텍스트 설계와 압축 전략을 고민하는 분들께 참고 자료가 되기를 바랍니다.
