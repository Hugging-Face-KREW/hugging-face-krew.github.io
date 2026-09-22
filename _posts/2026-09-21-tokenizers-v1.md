---
layout: post
title: "tokenizers v1: 인코드, 디코드 및 스케일링 측정"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/tokenizers-v1/thumbnail.png
image: assets/images/blog/posts/2026-09-21-tokenizers-v1/thumbnail.png
authors:
  - user: ArthurZ
  - user: sbrandeis
  - user: mcpotato
  - user: lysandre
slug: "tokenizers-v1"
source_url: "https://huggingface.co/blog/tokenizers-v1"
source_published_date: "2026-09-21"
source_published_at: "2026-09-21T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [tokenizers v1: encode, decode and scaling, measured](https://huggingface.co/blog/tokenizers-v1)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/tokenizers-v1 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# tokenizers v1: 인코드, 디코드 및 스케일링 측정

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-header.static.hf.space" width="100%" height="352" frameborder="0" scrolling="no"></iframe>
</figure>

ML 워크플로에서 토크나이저가 병목이었던 적은 지금까지 많지 않았습니다. 연산 측면에서 토큰화는 파이프라인의 나머지 부분에서 수행되는 무거운 모델링 작업에 비하면 가볍습니다. 하지만 어떤 경우에는 토큰화가 머신 러닝 작업을 가속하거나(또는 느리게 하는) 핵심 요소가 되기도 합니다.

모델이 더 빨라지고 워크로드가 확장되면서 이러한 균형이 바뀌기 시작합니다. 대규모 데이터셋으로 학습하거나, 많은 동시 요청을 서빙하거나, 긴 입력을 반복적으로 처리하면 토크나이저에 충분한 부담이 가해져 모델에 공급할 데이터가 부족해질 수 있습니다.

그래서 tokenizers의 다음 메이저 버전인 v1에서는 성능에 집중하기로 했습니다. 토큰화는 가벼워야 하며 워크플로와 함께 확장되어야 합니다. GPU가 CPU의 토큰화 완료를 기다리며 유휴 상태에 머무는 일은 없어야 합니다.

---

이 글에서는 v1이 v0.23보다 더 빠른 이유를 살펴봅니다. 그 차이는 종종 수십 배에 달합니다.

이 작업은 나머지 생태계 덕분에 전적으로 가능했습니다. 토큰화는 오픈 소스에서 매우 활발하게 개발되는 분야이며, [gigatoken](https://github.com/marcelroed/gigatoken), [tiktoken](https://crates.io/crates/tiktoken-rs), [kitoken](https://crates.io/crates/kitoken), [tokie](https://crates.io/crates/tokie), [fastokens](https://crates.io/crates/fastokens), [wordchipper](https://crates.io/crates/wordchipper), [ai-tokenizer](https://www.npmjs.com/package/ai-tokenizer)과 같은 라이브러리와 그 밖의 많은 프로젝트가 빠른 토크나이저의 가능성을 계속 확장해 왔습니다. 우리는 이러한 작업을 살펴보았고, 아래에서 소개하는 아이디어 중 몇 가지는 다른 프로젝트가 시도해 볼 가치가 있음을 보여주었기 때문에 알게 되었습니다.

이 리팩터링 이전의 tokenizers는 낼 수 있었던 성능에 훨씬 못 미쳤기 때문에, 기여할 가치가 없어 보였을 수도 있습니다. 이번 리팩터링을 통해 tokenizers가 기여할 가치가 있는 라이브러리라는 점을 분명히 보여주고자 합니다.

또한 패치를 기여하고 다양한 하드웨어에서 테스트하여 플랫폼 지원을 넓히는 데 도움을 준 IBM, NVIDIA 및 ExecuTorch 팀에도 감사드립니다.

## 결과 {#section-1}

tokenizers v1 릴리스 후보와 널리 사용되는 다른 대안의 결과를 제시합니다. 단일 스레드 및 멀티스레드 성능, 스레드 수에 따른 스케일링, 모델별 비교, 언어별 비교, 지연 시간, 디코딩 처리량, 메모리 힙, 크레이트 크기를 살펴봅니다.

[tokbench](https://github.com/huggingface/tokbench) 리포지토리에서 이를 실행하며, 원한다면 사용 중인 하드웨어에서 벤치마크를 다시 실행할 수 있는 명령도 제공합니다.

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-results.static.hf.space" width="100%" height="2615" frameborder="0" scrolling="no"></iframe>
</figure>

## V1이란 {#section-2}

v1은 v0.23과 동일한 토큰 ID를 생성합니다. 목표는 출력, API, 어휘 및 병합 순위를 유지하면서 개선할 수 **있는** 모든 것을 개선하는 것이었습니다. 여기에는 지원 범위도 포함됩니다. 라이브러리는 BPE에 특화되는 대신 여러 토크나이저 계열을 포괄하는 범용성을 유지하므로, v1은 v0.23이 로드하던 모든 것을 로드합니다.

토크나이저는 텍스트를 모델이 읽는 정수 목록으로 변환합니다. tokenizers는 이 변환을 네 단계로 수행합니다. 정규화는 원시 텍스트에 소문자화나 유니코드 정규화와 같은 연산을 적용합니다. 사전 토큰화는 텍스트를 프리토큰이라고 하는 더 작은 조각으로 나눕니다. 모델은 각 프리토큰을 토큰으로 변환하고 어휘에서 해당 ID를 매핑합니다. 후처리는 모델이 요구하는 특수 토큰을 추가합니다.

여기서 설명하는 작업 대부분은 모델 단계에서 수행됩니다. 이 글에서 측정한 10개 모델 계열 중 8개는 바이트 페어 인코딩, 즉 BPE를 사용합니다. BPE는 프리토큰의 바이트에서 시작해 순위가 가장 높은 인접 쌍을 더 이상 순위가 매겨진 쌍이 남지 않을 때까지 반복해서 결합합니다. 순위는 토크나이저 학습 시 학습되어 토크나이저와 함께 제공되므로, 동일한 텍스트는 항상 동일한 ID를 생성합니다. 병합은 프리토큰 경계를 넘지 않습니다. 나머지 두 계열은 라이브러리가 지원하는 다른 두 모델 유형인 WordPiece와 Unigram을 사용합니다.

[tokenization pipeline](https://huggingface.co/docs/tokenizers/pipeline) 페이지에서는 네 단계를 설명합니다. [Tokenization algorithms](https://huggingface.co/docs/transformers/tokenizer_summary)에서는 BPE, WordPiece 및 Unigram을 설명합니다.

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-pipeline.static.hf.space" width="100%" height="341" frameborder="0" scrolling="no"></iframe>
</figure>

각 단계가 개선되었습니다. 중요한 변경 사항은 다음과 같습니다.

| 변경 사항 | 기능 |
| --- | --- |
| workspace 분리 | 하나의 크레이트를 workspace로 전환했습니다. `tk-encode`은 필수 런타임이고, `tk-serialize`, `tk-convert`, `tk-train`은 애플리케이션이 필요로 할 때만 연결됩니다. |
| no-alloc 모델 | 병합 작업 집합이 호출자가 소유한 스크래치 버퍼에 저장되며, 루프는 allocator에 전혀 접근하지 않습니다. |
| bitcannon | 분할 패턴을 비트스트림에 대한 불리언 연산으로 변환하여, 정규식 엔진 대신 SIMD 명령으로 분할 지점을 찾습니다. |
| 병합 루프 재작성 | 병합되는 조각을 하나의 사전 할당 버퍼 안에 intrusive 이중 연결 리스트로 구성하므로, 병합 시 데이터를 이동하지 않고 두 인덱스만 갱신합니다. |
| 단어 캐시 | 프리토큰 바이트에서 완성된 ID로 매핑하는 스레드 로컬 메모입니다. 따라서 같은 단어는 한 번만 병합됩니다. |
| 네이티브 병렬성 | 하나의 공유 토크나이저가 여러 스레드에서 동시에 인코딩합니다. 각 스레드는 자체 하위 풀에서 스크래치 버퍼와 단어 캐시를 가져오므로, 스레드가 더 이상 하나의 잠금에서 대기하지 않습니다([#2365](https://github.com/huggingface/tokenizers/pull/2365)). |

### 분할: 정규식 대신 비트스트림

BPE 모델은 정규식을 사용해 입력 텍스트를 프리토큰이라는 더 작고 처리하기 쉬운 청크로 나눕니다. 병합은 프리토큰 내부에서 일어나며 두 프리토큰 사이의 경계를 넘지 않습니다. 따라서 이 분할이 파이프라인의 나머지 부분에서 보게 되는 내용을 결정합니다.

이 정규식은 모델의 고정 매개변수입니다. 토크나이저와 함께 제공되며 런타임에 변경되지 않으므로, 매번 인코딩할 때 범용 정규식 엔진이 이를 해석할 필요가 없습니다. 특정 모델이 실제로 사용하는 패턴에 맞는 동등한 분할 함수를 한 번 직접 작성할 수 있습니다.

그런 다음 직접 작성한 함수는 최신 CPU의 SIMD 명령(단일 명령, 다중 데이터)을 사용할 수 있습니다. SIMD 명령은 한 번에 많은 바이트에 하나의 연산을 적용하며 UTF-8 텍스트에 적합합니다. bitcannon은 입력의 바이트를 병렬 비트 스트림으로 간주하므로, 한 번에 한 문자씩 진행하는 스캔 대신 전체 레지스터에 대한 불리언 연산을 통해 경계를 계산합니다. 레지스터 연산 한 번으로 64바이트를 처리합니다. 동일한 아이디어가 텍스트 처리를 위한 [Parabix](https://www.cs.sfu.ca/~ashriram/papers/2012_HPCA_Parabix.pdf)과 JSON을 위한 [simdjson](https://arxiv.org/abs/1902.08318)에도 적용됩니다.

이는 패턴을 인식할 수 있어야 한다는 전제에 기반합니다. 소수의 문법이 대부분의 바이트 수준 BPE 모델을 포괄하며, 패턴이 그 안에 포함되지 않는 토크나이저는 정규식 경로를 그대로 사용하므로 이러한 속도 향상을 얻지 못합니다. 따라서 앞서의 성능 향상 폭이 이처럼 크게 달라집니다.

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-split.static.hf.space" width="100%" height="304" frameborder="0" scrolling="no"></iframe>
</figure>

### 단어 캐시

실제 텍스트에는 반복되는 단어가 많습니다. BPE는 특정 프리토큰에 대해 항상 동일한 토큰 ID를 생성하므로, v1은 한 번 처리한 결과를 저장할 수 있습니다. 스레드 로컬 캐시는 각 프리토큰의 바이트를 토큰 ID에 매핑하여, 이후에 같은 프리토큰이 등장하면 병합 과정을 건너뛸 수 있게 합니다.

당연히 입력이 커질수록 고유 단어 수는 전체 단어 수보다 느리게 증가할 수 있습니다. 그러면 반복되는 단어가 입력에서 차지하는 비중이 커집니다. 새로운 단어도 계속 등장하므로, 아래 애니메이션에서 간혹 캐시 미스가 발생합니다.

<figure class="image text-center">
  <iframe src="https://lysandre-tokenizers-v1-cache.static.hf.space" width="100%" height="976" frameborder="0" scrolling="no"></iframe>
</figure>

다음 명령으로 공유 접두사 결과를 재현할 수 있습니다.

```bash
tokbench measure prefix-sharing \
  --engine pipeline \
  --engine hf-tokenizers \
  --compare-to pipeline-no-cache \
  --corpus agentic_swe
```


캐싱은 입력에 반복되는 프리토큰이 있을 때 가장 효과적입니다. 반복되는 프리토큰이 적은 입력에서는 많은 캐시 적중을 얻지 못한 채 조회 비용만 발생할 수 있습니다.

### 병합 루프

다음으로 큰 비용은 BPE 병합 루프에서 발생합니다. 루프는 각 프리토큰에 대해 우선순위가 가장 높은 인접 쌍을 반복해서 찾아 병합합니다. 이전 구현은 호출할 때마다 새 메모리를 할당하고, 프리토큰마다 새로운 우선순위 큐를 생성했습니다.

v1은 호출자가 소유한 스크래치 버퍼를 재사용하여 이러한 반복 할당을 제거합니다. 기호를 평면 배열에 저장하고 인접한 기호를 해당 배열의 위치로 연결하므로, 병합 중 갱신 비용이 줄어듭니다. 또한 하나의 모델 호출에서 여러 프리토큰을 배치로 처리합니다.

각 후보 쌍은 하나의 64비트 값으로 패킹되며, 병합 순위는 상위 비트에 저장됩니다. 따라서 두 후보의 비교는 두 정수를 비교하는 것에 불과합니다. 또한 "여기서는 병합하지 않음"이 가능한 가장 큰 값이므로, 루프는 분기 없이 다음 병합을 찾습니다.

## 방법 {#section-3}

벤치마크 설계의 작은 차이도 토크나이저 성능에 큰 차이를 만들 수 있습니다. 엔진 간 비교의 일관성을 유지하기 위해 다음 규칙을 사용했습니다.

| 규칙 | 이유 |
| --- | --- |
| 하나의 타이밍 루프 | 모든 엔진이 동일한 루프를 실행하며, 엔진별 빠른 경로는 없습니다. |
| 로드 제외 | 어휘 로드는 별도로 측정하며 encode 내부에서는 측정하지 않습니다. |
| ID 해시 검증 | 출력 ID에 대한 FNV-1a가 기준선과 정확히 일치해야 합니다. |
| 공통 셀만 사용 | 모든 엔진이 실행하고 검증한 셀에 대해서만 중앙값을 계산합니다. |
| 프로세스별 전체 스윕 | 각 반복은 새 프로세스에서 시작하며 모든 셀을 유지합니다. |
| 물리 코어 고정 | 작업자를 서로 다른 8개의 물리 코어에 고정하며, 같은 코어의 SMT 스레드는 사용하지 않습니다. |
| 독립적인 Jobs | 호스트 간 변동을 측정하기 위해 별도의 Jobs를 사용합니다. |

하나의 문서를 반복해서 인코딩하는 것은 동일한 빌드에서 서로 다른 문서의 스트림을 인코딩하는 것보다 빠를 수 있습니다. 전자의 방식은 문서 전체가 이미 캐시에 표현되어 있는 경우의 성능을 측정합니다. 후자의 방식은 이전에 본 프리토큰을 캐시에 남겨 둔 채 새로운 입력을 처리하는 성능을 측정합니다.

두 조건 모두 서로 다른 워크로드를 측정하지만 때때로 "warm"이라고 표현됩니다. 이 글의 주요 결과는 서로 다른 문서를 사용하며, 전체 코퍼스는 캐시에 들어가기에는 너무 큽니다. 토크나이저 벤치마크는 어떤 워크로드를 사용하는지 밝혀야 합니다. 선택에 따라 결과가 크게 달라질 수 있기 때문입니다.

## 종합 결과 {#section-4}

v1의 encode 경로가 지원하는 10개 모델 계열 전체에서, Apple M4 Max의 단일 스레드 환경에서 v1은 v0.23보다 텍스트를 **3~30배 빠르게** 인코딩합니다. 하한은 t5-base, 상한은 gpt2입니다. 8개 워커에서 선형 확장의 **76%**로 스케일링됩니다. 이 모든 변경에도 불구하고 v1은 릴리스된 라이브러리와 정확히 동일한 토큰 ID를 생성합니다.

전반적인 개선은 여러 변경 사항이 함께 작동한 결과입니다. 정규식 엔진 대신 직접 작성한 분할기를 사용하고, 반복되는 단어를 다시 병합하지 않고 처리하는 캐시를 도입했으며, allocator에 전혀 접근하지 않는 병합 루프를 사용하고, 프리토큰마다 한 번씩 모델을 호출하는 대신 프리토큰 배치마다 한 번 호출합니다. 각각의 변경은 파이프라인의 서로 다른 지점에서 수행되는 작업을 줄입니다.

다음 우선순위는 더 많은 모델 계열을 지원하는 것입니다. `1.0.0` 이전에 추가 모델을 새로운 병합 루프로 옮길 예정입니다. 릴리스 후보가 안정화되면 다음 단계는 transformers 라이브러리와 tokenizers 라이브러리에 의존하는 나머지 생태계에 이러한 개선 사항을 적용하는 것입니다.

이 글은 [tokbench](https://github.com/huggingface/tokbench) 결과를 바탕으로 생성되며 지원 범위가 확대됨에 따라 업데이트됩니다.

## 설치 방법 {#section-5}

v1 릴리스 후보가 crates.io에 공개되어 있습니다. 호출하는 API는 기존과 동일하므로, 변경되는 것은 설치하는 빌드뿐입니다.

일반적인 설치 방법은 다음과 같습니다.

```bash
cargo add tokenizers --pre
```


학습은 기본적으로 활성화된 기능 뒤에 있으며, 이 기능은 C++ 의존성도 함께 가져옵니다. 인코딩만 필요하다면 이 기능을 끄고 학습 구현을 제외할 수 있습니다.

```bash
cargo add tokenizers --pre --no-default-features --features http
```


인코딩 방식은 변경되지 않습니다. 동일한 호출, 동일한 ID입니다.

```rust
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    let tokenizer = Tokenizer::from_pretrained("deepseek-ai/DeepSeek-V4-Flash", None)?;

let encoding = tokenizer.encode("The tokenizer is no longer the bottleneck.", false)?; println!("{:?}", encoding.get_ids()); // [671, 17840, 9160, 344, 1119, 5827, 270, 111127, 16] println!("{:?}", encoding.get_tokens()); // ["The", "Ġtoken", "izer", "Ġis", "Ġno", "Ġlonger", "Ġthe", "Ġbottleneck", "."]

Ok(()) } ```

For a batch, `encode_batch` is what scales across cores. It is the call the scaling view above measures.

```rust
let encodings = tokenizer.encode_batch(documents, false)?;
```


이 글의 모든 그림은 이 크레이트를 기준으로 측정되었습니다. Python 바인딩은 동일한 코드를 래핑하며 `bindings/python`에서 빌드되지만, 호출별 오버헤드를 추가하므로 이러한 측정에는 포함되지 않습니다.

## V1을 향한 진행 상황 {#section-6}

이 글의 벤치마크는 먼저 나열한 릴리스 후보 작업 중 완료된 부분을 다룹니다. 나머지 절에서는 `1.0.0`에 필요한 작업과 그 이후에 탐색할 계획을 보여줍니다.

### 릴리스 후보: 구현 완료

이 작업은 crates.io의 Rust 프리릴리스에 포함되어 있습니다.

```bash
cargo add tokenizers --pre
```


- workspace 분리: 단일 크레이트를 `tk-encode`, `tk-serialize`, `tk-convert`, `tk-train`으로 나누어 애플리케이션이 사용하는 항목만 연결하도록 함
- bitcannon: 인코딩 경로에서 정규식 분할을 GPT-2, cl100k, o200k, Tekken 및 DeepSeek을 지원하는 비트스트림 연산으로 대체함. 이는 처음 제공된 유한 상태 머신을 대체했습니다 [#2201](https://github.com/huggingface/tokenizers/pull/2201) [#2317](https://github.com/huggingface/tokenizers/pull/2317)
- WordCache: 이전에 처리한 프리토큰의 토큰 ID를 재사용함 [#2262](https://github.com/huggingface/tokenizers/pull/2262), `af5a3e3`
- 더 빠른 조회 및 병합 구조: FlatCache, MPHF RankStore, 증분 병합 및 BucketVocabStore를 추가함 [#2190](https://github.com/huggingface/tokenizers/pull/2190) [#2188](https://github.com/huggingface/tokenizers/pull/2188)
- 재사용 가능한 모델 메모리: 임시 모델 상태를 스크래치 버퍼로 옮겨 토큰화 시 호출마다 할당하지 않도록 함 [#2175](https://github.com/huggingface/tokenizers/pull/2175) [#2183](https://github.com/huggingface/tokenizers/pull/2183)
- 파이프라인 후처리: 후처리를 `STAGE_POST` 파이프라인 단계로 노출함 [#2182](https://github.com/huggingface/tokenizers/pull/2182)
- 배치 모델 호출: 한 번의 호출에서 여러 프리토큰 범위를 처리함 [#2304](https://github.com/huggingface/tokenizers/pull/2304)
- 더 빠른 디코딩: 디코딩된 바이트를 재사용 가능한 버퍼에 직접 기록하고, 중간 문자열과 복사를 피하며, 토큰 조회를 가속하고, 버퍼링된 스트리밍을 지원하며, 배치를 병렬로 디코딩함
- `role_to_token` 지원 [#2343](https://github.com/huggingface/tokenizers/pull/2343)
- Node.js 바인딩 [#2281](https://github.com/huggingface/tokenizers/pull/2281)

### 1.0.0

- 하나의 인코딩 구현: 학습 검증 중에 `tk-encode`을 사용하여 학습과 추론이 서로 다른 토큰화 결과를 생성할 수 없도록 함
- 선택적 오프셋 및 마스크: 요청된 경우에만 이 메타데이터를 계산하여 토큰 ID만 사용하는 경로에서는 비활성화함
- normalizer 재작업
- atomnorm을 기반으로 하는 bitnorm 지원 [#2209](https://github.com/huggingface/tokenizers/pull/2209)
- spm 사전 컴파일
- 더 단순한 Python 바인딩: 서브클래싱, 직렬화, 사용자 지정 디코더, 변경 동작 및 free-threaded CPython 지원을 유지하면서 잠금, 래퍼 타입 및 직접 작성한 디스패치 코드를 줄임
- ExecuTorch 및 llama.cpp를 위한 추론 전용 C 및 C++ 바인딩. 이후 JVM, Swift 및 Go 바인딩도 제공할 가능성이 있음

### 1.0.0 이후

- tok-devices: 텍스트와 토큰 ID를 디바이스에 유지하면서 GPU 인코딩 및 배치 디코딩을 탐색함. 디코더는 어휘를 한 번 업로드하고, 출력 위치를 병렬로 계산한 다음, GPU에서 해당 바이트를 수집합니다. 이는 대규모 배치를 대상으로 하는 선택적 구성 요소이며, 추가 프로토타이핑과 측정이 필요합니다.
