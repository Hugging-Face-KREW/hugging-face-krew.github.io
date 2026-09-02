---
layout: post
title: "@huggingface/kernels 소개: 로컬 AI를 위한 200개 이상의 WebGPU 커널"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/webgpu-kernels/thumbnail.png
image: assets/images/blog/posts/2026-09-01-webgpu-kernels/thumbnail.png
authors:
  - user: nico-martin
  - user: Xenova
slug: "webgpu-kernels"
source_url: "https://huggingface.co/blog/webgpu-kernels"
source_published_date: "2026-09-01"
source_published_at: "2026-09-01T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Introducing @huggingface/kernels: 200+ WebGPU Kernels for Local AI](https://huggingface.co/blog/webgpu-kernels)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/webgpu-kernels -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# @huggingface/kernels 소개: 로컬 AI를 위한 200개 이상의 WebGPU 커널

Hugging Face의 WebAI 팀에서 가장 큰 목표 중 하나는 브라우저 추론을 최대한 빠르고 사용자 친화적으로 만드는 것입니다. 이를 달성하려면 여러 층위의 노력이 필요합니다. 모델에는 브라우저에 적합한 표현이 필요하고, 런타임은 효율적인 실행 계획을 구성해야 하며, 스택 최하단의 개별 GPU 연산은 다양한 디바이스와 브라우저 구현을 최대한 활용해야 합니다.

오늘은 이러한 노력의 첫 번째 층인 [`@huggingface/kernels`](https://www.npmjs.com/package/@huggingface/kernels)을 공개합니다. [`@huggingface/kernels`](https://www.npmjs.com/package/@huggingface/kernels)은 Hugging Face Hub에서 최적화된 WebGPU 커널을 로드하고 실행하기 위한 최소한의 라이브러리이며, [huggingface.co/webgpu-kernels](https://huggingface.co/webgpu-kernels)에서 **207개 커널**로 구성된 초기 컬렉션도 함께 제공합니다.

이 컬렉션은 매우 다양한 머신 러닝 아키텍처와 워크로드에서 사용되는 연산을 다룹니다. 더 중요한 점은 각 커널이 완전한 버전 관리 패키지로 게시된다는 것입니다. 인터페이스, 셰이더 템플릿, 정확성 테스트 케이스, 벤치마크 케이스, 사용 지침이 모두 Hub에 함께 포함되어 있습니다.

또한 브라우저에서 GPU 벤치마킹과 테스트를 수행하고, 사용자의 하드웨어에서 커널을 실행해 점수를 매기는 [Fleet](https://webgpu-kernels-fleet.hf.space/)도 출시합니다. Fleet은 자신의 머신에 대한 결과를 제공하는 것을 넘어, 기존의 테스트 랩에서는 다룰 수 없었던 디바이스의 성능 및 정확성 증거를 커뮤니티가 기여할 수 있는 방법을 제공합니다. 사용자가 동의하면 실행할 때마다 비공개 증거가 추가되며, 이를 통해 실패(잘못된 결과, 비정상적으로 느린 경우 등)를 찾고, 커널 변형을 개선하고, 실제 하드웨어 전반에서 더 나은 최적화 결정을 내릴 수 있습니다.

## TL;DR {#section-1}

- **207개의 WebGPU 커널**이 [`webgpu-kernels`](https://huggingface.co/webgpu-kernels) 조직의 개별 저장소로 게시되었습니다. Apache-2.0 라이선스가 적용됩니다.
- **JavaScript 로더** `@huggingface/kernels`는 Hub에서 직접 커널을 다운로드하고 준비한 뒤 실행합니다.
- 모든 커널에 대해 매니페스트, 정확성 테스트, 벤치마크 케이스, WGSL 셰이더 템플릿을 포함한 **명시적 계약과 재현 가능한 증거**를 제공합니다.
- **Fleet**은 실제 GPU 전반에서 정확성과 성능 증거를 크라우드소싱하는 브라우저 기반 벤치마킹 도구로, 커널과 그 변형을 개선하는 데 도움을 줍니다.

## 커널부터 시작하는 이유 {#section-2}

브라우저에서 실행되는 모델은 결국 GPU 연산의 연속으로 변환됩니다. 여기에는 행렬 곱, 정규화, 컨볼루션, 어텐션 기본 연산, 양자화 연산, 데이터 레이아웃 변환 등이 포함됩니다. WebGPU는 이식 가능한 API를 통해 최신 브라우저 전반에서 이러한 연산을 사용할 수 있게 하며, WGSL은 이를 실행하는 셰이더를 위한 공통 언어를 제공합니다.

하지만 이식성이 곧바로 성능을 의미하지는 않습니다. 두 셰이더가 동일한 연산을 구현하고 같은 출력을 생성하더라도, 서로 다른 가속기에서는 완전히 다르게 동작할 수 있습니다. 워크그룹 크기, 메모리 액세스 패턴, 벡터화, 데이터 타입, 퓨전 전략은 모두 성능에 영향을 줄 수 있습니다. 또한 최적의 선택은 입력 형태, 디바이스, 브라우저, 사용 가능한 WebGPU 기능에 따라 달라질 수 있습니다.

이 때문에 커널은 빠른 브라우저 추론을 위한 기반 층을 이룹니다. 상위 수준 런타임의 효율은 런타임이 디스패치하는 연산만큼만 높을 수 있습니다. 이러한 연산을 개별적으로 탐색하고, 테스트하고, 벤치마크하고, 버전 관리할 수 있게 하면 상위 층에는 안정적인 계약을 유지하면서 기반을 독립적으로 개선할 수 있습니다.

## 단순한 셰이더가 아닌 커널 저장소 {#section-3}

컬렉션의 각 커널에는 자체 저장소와 커널 카드가 있습니다. 카드에는 연산의 의미, 입력, 출력, 속성, 지원되는 데이터 타입, 소스 파일, 바로 실행할 수 있는 `@huggingface/kernels` 예제가 문서화되어 있습니다.

예를 들어 [`ai.onnx.Add`](https://huggingface.co/webgpu-kernels/ai.onnx.Add)은 다방향 브로드캐스팅을 지원하는 원소별 덧셈을 구현합니다. 이는 신경망에서 가장 단순한 연산 중 하나로, 잔차 연결부터 바이어스 추가까지 다양한 곳에서 사용됩니다. 카드에는 두 입력, 브로드캐스팅된 출력 형태, 지원되는 데이터 타입, 다양한 형태와 디바이스에 사용할 수 있는 변형이 문서화되어 있습니다.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/webgpu-kernels/ai-onnx-add.png" alt="Files in the ai.onnx.Add WebGPU kernel repository" width="90%">
  <figcaption>The <code>ai.onnx.Add</code> repository packages its manifest, correctness and benchmark cases, and WGSL shader templates together.</figcaption>
</figure>

카드 뒤의 저장소에는 구현을 이해하고 평가하는 데 필요한 아티팩트가 포함되어 있습니다.

- **`manifest.json`**은 연산 계약의 기준이 되는 정보입니다. 입력, 출력, 속성, 타입 제약 조건, 형태 도출 규칙을 정의합니다.
- **`metadata.json`**에는 커널 식별자, 다이제스트, 출처 정보가 기록됩니다.
- **`test.json`**에는 정확성 케이스가 포함되어 있어 구현이 예상된 동작과 일치하는지 확인할 수 있습니다.
- **`bench.json`**에는 커널을 평가하는 데 사용되는 워크로드를 나타내는 벤치마크 및 튜닝 케이스가 포함됩니다.
- **`*.wgsl.jinja`** 파일에는 특정 요청과 디바이스에 맞는 셰이더를 생성하는 데 사용되는 매개변수화된 WGSL 구현이 포함됩니다.

이 구조는 셰이더를 재사용 가능한 소프트웨어 아티팩트로 바꿉니다. WGSL을 읽지 않고도 인터페이스를 확인할 수 있고, 정확성과 성능 케이스가 구현과 함께 제공되며, 게시된 버전을 버전이 지정되지 않은 파일 URL에 의존하지 않고 명시적으로 로드할 수 있습니다. 또한 개발자가 맞춤형 WebGPU 커널을 구축하거나 이러한 연산을 자체 런타임에 통합할 때, 당사의 커널을 참조 구현으로 활용할 수 있습니다.

## Hub에서 커널 로드하기 {#section-4}

npm에서 패키지를 설치합니다.

```bash
npm install @huggingface/kernels@preview
```


> [!NOTE]
> 이러한 커널을 실행하려면 [WebGPU support](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)을 지원하는 브라우저가 필요합니다. WebGPU 사용 가능 여부는 브라우저, 운영 체제, GPU, 드라이버에 따라 달라집니다. JavaScript에서는 `"gpu" in navigator`로 확인할 수 있습니다.

`@huggingface/kernels`은 커널 저장소와 애플리케이션 사이를 연결합니다. Hub 저장소 ID와 계약 버전으로 `getKernel`을 호출한 다음, 반환된 함수를 타입이 지정된 입력 데이터와 텐서 형태로 호출하면 됩니다. 다음은 간단한 바이어스 추가 예제입니다.

```js
import { getKernel } from "@huggingface/kernels";

const add = await getKernel("webgpu-kernels/ai.onnx.Add", { version: 1 });

const { c } = await add({
  a: {
    data: new Float32Array([1, 2, 3, 4, 5, 6]),
    shape: [2, 3],
  },
  b: {
    data: new Float32Array([10, 20, 30]),
    shape: [3],
  },
});
```


두 번째 입력은 첫 번째 차원에 걸쳐 브로드캐스트되며, `[2, 3]` 형태의 출력을 생성합니다. 로더는 매니페스트 계약과 입력으로부터 출력 형태와 논리적 데이터 타입을 도출한 다음 `c`을 자동으로 할당합니다.

6개의 부동소수점 수를 더하는 것은 의도적으로 가능한 한 작은 데모로 구성했습니다. 이 정도 크기에서는 GPU 왕복 비용이 연산 자체보다 훨씬 큽니다. 핵심은 호출 패턴입니다. 최적화된 커널이 실제로 효과를 발휘하는 행렬 곱(`ai.onnx.MatMul`)과 같은 대규모 연산에서도 호출 방식은 정확히 동일하게 유지됩니다. 저장소 ID와 입력만 바뀝니다.

이 기본적인 연산만 보더라도 커널에 변형이 필요한 이유를 알 수 있습니다. 형태가 같은 덧셈에는 직접 벡터화 경로를 사용할 수 있지만, 브로드캐스트된 입력에는 다른 인덱싱 로직이 필요합니다. 게시된 Add 커널에는 동일한 형태, 벡터화된 브로드캐스팅, 스칼라 처리, 일반 브로드캐스팅을 위한 변형이 포함되어 있습니다. 런타임은 애플리케이션을 향한 API를 변경하지 않고 현재 호출과 디바이스에 적합한 구현을 선택할 수 있습니다.

`version: 1` 옵션은 게시된 **커널 계약**의 버전 1을 선택합니다. 이는 ONNX opset, 연산자의 `since_version`, 모델 리비전과는 별개입니다. 이러한 개념을 분리하면 애플리케이션은 안정적인 JavaScript 대상 계약에 의존하면서 커널 구현은 그 이면에서 발전할 수 있습니다.

## 커널은 얼마나 빠른가? {#section-5}

그렇다면 최적화된 커널은 실제로 얼마나 큰 차이를 만들어 낼까요? Apple M4 GPU에서 ONNX Runtime Web `1.30.0-dev.20260826-b1f76d586a`을 사용해 컬렉션을 ORT WebGPU와 직접 비교했습니다. 207개 연산 전체에 걸쳐 1,756개의 테스트 케이스로 시작했으며, 양쪽에서 일치하는 출력과 신뢰할 수 있는 측정 시간을 얻은 809개 케이스를 유지했습니다.

이러한 비교에서 당사의 커널은 **기하 평균 기준 2.57배 빠르고**, **중앙값 기준 1.90배 빨랐으며**, 629승, 176패, 4무를 기록했습니다. 익숙한 네 가지 연산을 좀 더 자세히 살펴보겠습니다.

| 연산 | 비교 케이스 | 당사 WebGPU 커널 | ORT WebGPU | 속도 향상 |
| --- | ---: | ---: | ---: | ---: |
| Add | 5 | 0.064 ms | 0.227 ms | **3.52x** |
| MatMul | 29 | 0.115 ms | 0.131 ms | **1.14x** |
| Softmax | 12 | 0.114 ms | 0.240 ms | **2.11x** |
| LayerNormalization | 6 | 0.061 ms | 0.135 ms | **2.22x** |

개별적인 성능 향상 중에는 훨씬 큰 경우도 있었습니다. 특히 까다로운 이중선형 Einsum 케이스(`i,ij,j`, 크기 4096)는 당사 커널에서 0.136 ms, ORT WebGPU에서 1,396 ms가 걸려 **10,000배 이상 빠른** 결과를 보였습니다. `[256, 4096]`에 대한 행 단위 CumSum은 0.016 ms 대 4.784 ms로 **301배 빨랐습니다**. 이는 모든 경우에 기대할 수 있는 속도 향상이 아니라 특이한 사례이지만, 일반 구현이 느린 경로에 진입할 때 특화된 커널이 얼마나 큰 도움을 줄 수 있는지 보여 줍니다.

커널 로드, 세션 생성, 입력 업로드, 셰이더 컴파일, 출력 읽기와 같은 준비 작업은 제외하고 GPU 자체에서 수행되는 작업의 시간을 측정했습니다. 매우 짧은 워크로드는 본질적으로 측정하기 더 어렵고, 작은 케이스는 GPU 캐시의 이점을 얻을 수 있으므로, 이 수치는 모든 애플리케이션에 대한 보장이 아니라 유용한 비교 자료로 해석하는 것이 좋습니다.

또한 이 결과는 완전한 모델이 아니라 개별 연산에 대한 결과입니다. 정확한 성능은 GPU와 브라우저에 따라 달라지므로, 더 폭넓은 상황을 파악하는 데 Fleet이 매우 중요합니다.

또한 이러한 개선 사항을 업스트림에 반영하여 더 넓은 ONNX Runtime Web 생태계가 혜택을 받을 수 있도록 ONNX Runtime 팀과 협력하고 있습니다.

## 하나의 디바이스에서 Fleet으로 {#section-6}

WebGPU 성능은 GPU, 브라우저, 드라이버에 따라 달라지므로 한 머신의 결과만으로는 전체 상황의 일부만 알 수 있습니다. [Fleet](https://webgpu-kernels-fleet.hf.space/)을 사용하면 누구나 브라우저에서 정확성과 성능 검사를 실행하고 자신의 하드웨어에서 커널이 어떻게 동작하는지 확인할 수 있습니다.

동의하면 각 실행이 비공개로 증거를 제공하며, 이를 통해 디바이스별 실패를 발견하고, 변형을 비교하고, 선택 규칙을 개선할 수 있습니다. 목표는 간단합니다. 실제 환경의 폭넓은 커버리지를 활용해 모든 사용자를 위해 커널을 더 빠르고 안정적으로 만드는 것입니다.

## WebAI를 위한 공유 기반 구축 {#section-7}

초기 207개 커널은 최종 상태가 아니라 시작점입니다. Hub에 커널을 독립적으로 게시하면 모든 셰이더를 각 런타임에 직접 내장하지 않고도 계약을 확인하고, 구현을 비교하고, 정확성 검사를 재현하고, 성능을 개선할 수 있는 공통 장소를 마련할 수 있습니다.

이 컬렉션은 Hub의 더 폭넓은 커널 생태계의 일부이기도 합니다. [Kernels page](https://huggingface.co/kernels?platform=webgpu&sort=trending)에서 WebGPU 커널은 CUDA, ROCm, Metal 및 기타 플랫폼용 커널과 나란히 제공되며, Hub의 다른 아티팩트와 마찬가지로 필터링하고 정렬하고 탐색할 수 있습니다.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/webgpu-kernels/kernels.png" alt="The Hub Kernels page filtered to the WebGPU platform, listing the 207 published kernels" width="90%">
  <figcaption>All 207 WebGPU kernels on the Hub's <a href="https://huggingface.co/kernels?platform=webgpu&sort=trending">Kernels page</a>, filtered by platform.</figcaption>
</figure>

각 구성 요소는 서로를 강화합니다.

1. 커널 저장소는 투명하고 버전이 지정된 연산 계약을 정의합니다.
2. `@huggingface/kernels`은 이러한 연산을 JavaScript에서 간편하게 로드하고 실행할 수 있게 합니다.
3. Fleet은 일반적인 벤치마크 랩보다 훨씬 광범위한 디바이스에서 실제 환경의 증거를 크라우드소싱합니다.
4. 기여된 각 실행은 실패를 드러내고, 튜닝을 이끌고, 변형 선택을 개선하며, 향후 커널 버전을 검증하는 데 도움을 줄 수 있습니다.

이는 브라우저 추론 스택에서 앞으로 나아가기 위한 저수준 기반입니다. 이러한 커널을 상위 수준의 모델 도구와 연결하고, 연산 지원 범위를 계속 확장하며, WebAI 생태계 전반에서 빠른 로컬 추론을 더 쉽게 사용할 수 있도록 노력하겠습니다.

[WebGPU kernel collection](https://huggingface.co/webgpu-kernels)을 살펴보고, [`@huggingface/kernels`](https://www.npmjs.com/package/@huggingface/kernels)을 사용해 보고, [join the Fleet](https://webgpu-kernels-fleet.hf.space/)하여 자신의 디바이스에서 증거를 제공하고 모든 사용자를 위해 커널을 개선하는 데 도움을 주세요.
