---
layout: post
title: AMD Ryzen 및 Radeon을 위한 llama.cpp 벤치마크 대시보드 소개
author: dailybot
categories:
- Translation
- HuggingFace
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/67289c93ff90038d9a6e5675/tbIvyMSISXCm64WYP3atl.png
image: assets/images/blog/posts/2026-11-26-llamacpp-amd-bench/thumbnail.png
authors:
- user: ggml-org
slug: llamacpp-amd-bench
source_url: https://huggingface.co/blog/ggml-org/llamacpp-amd-bench
source_published_date: '2026-11-26'
source_published_at: '2026-11-26T11:37:17+00:00'
locale: ko
translation_status: draft
translator: openai
description: '* TOC {:toc} <!--toc--> _이 글은 Hugging Face 블로그의 [Introducing the llama.cpp
  benchmark dashboard for AMD Ryzen and Radeon](https://huggingface.'
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Introducing the llama.cpp benchmark dashboard for AMD Ryzen and Radeon](https://huggingface.co/blog/ggml-org/llamacpp-amd-bench)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ggml-org/llamacpp-amd-bench -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# AMD Ryzen 및 Radeon을 위한 llama.cpp 벤치마크 대시보드 소개

[llama.cpp × AMD 벤치마크](https://ggml-org-llamacpp-amd-bench.static.hf.space/index.html) 대시보드를 Hugging Face에서 사용할 수 있습니다. 이 대시보드를 사용하면 llama.cpp 빌드, 모델, 양자화에 따른 프롬프트 처리 및 텍스트 생성 성능을 추적할 수 있으므로, AMD Ryzen 및 Radeon 하드웨어에서 로컬 추론이 어떻게 수행되는지 살펴볼 수 있습니다.

모델, 커널, 빌드가 발전함에 따라 로컬 추론 성능도 달라집니다. AMD 개발자라면 누구나 이러한 변화를 쉽게 확인할 수 있도록 하려 합니다. 하드웨어, 모델, 빌드를 선택하고 양자화를 비교하며 시간에 따른 성능을 추적할 수 있습니다. 벤치마크 워크플로와 기반 결과는 공개되어 있으므로, 추론 성능을 어떻게 벤치마크하는지도 직접 확인할 수 있습니다.

## 빠른 시작 {#section-1}

Configuration 사이드바에서 Hardware, Model, llama.cpp build를 선택할 수 있습니다. 현재 선택 가능한 하드웨어는 Radeon 8060S (gfx1151) 및 Radeon AI Pro R9700S (gfx1201)입니다.

- Overview 탭에서는 하나의 하드웨어, 모델, 빌드에 대한 양자화를 비교할 수 있습니다. 두 차트는 PP512(입력 토큰 512개, 생성 없음)와 TG128(생성 토큰 128개)을 보여 주며, 모두 컨텍스트 깊이 0에서 시작합니다.

- Build history 탭에서는 선택한 하드웨어, 모델, 양자화에 대해 기록된 모든 빌드의 PP512 및 TG128을 추적할 수 있습니다.

- Context scaling 탭에서는 선택한 빌드, 하드웨어, 모델, 양자화에 대해 기존 컨텍스트가 커질 때 토큰 생성 처리량이 어떻게 변하는지 확인할 수 있습니다.

## 수치의 의미 {#section-2}

대시보드에는 초당 토큰 수로 측정되는 두 가지 별도 테스트가 표시됩니다.

| 테스트 | 워크로드 | 평가에 도움이 되는 항목 |
| --- | --- | --- |
| PP512 | 입력 토큰 512개 처리 | 모델이 프롬프트를 처리하는 속도 |
| TG128 | 토큰 128개 생성 | 모델이 출력을 생성하는 속도 |

두 테스트 모두 컨텍스트 깊이 0(`n_depth=0`)에서 시작합니다. PP512는 생성 없이 입력 토큰 512개를 처리하는 시간을 측정하고, TG128은 기존 컨텍스트 없이 토큰 128개를 생성하는 시간을 측정합니다. 그래프에 표시되는 값은 평균 처리량(`avg_ts`)입니다. 이는 유용한 참고용 워크로드이며, Context scaling 탭에서는 컨텍스트가 커질 때 토큰 생성 처리량이 어떻게 변하는지 살펴볼 수 있습니다.

## 벤치마크 실행 방법 {#section-3}

[llama.app](https://llama.app/)을 뒷받침하는 빌드 및 설치 프로젝트인 [llama-install.sh](https://github.com/ggml-org/llama-install.sh)를 통해 배포되는 사전 빌드 바이너리를 벤치마크합니다. 공개된 [gfx1151](https://github.com/ggml-org/llama-install.sh/blob/master/.github/workflows/bench-gfx1151.yml) 및 [gfx1201](https://github.com/ggml-org/llama-install.sh/blob/master/.github/workflows/bench-gfx1201.yml) 워크플로는 매주 예약 실행되며, [llama-bench 도구](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)를 사용해 Flash Attention을 켠 경우와 끈 경우 각각 표준 벤치마크를 실행합니다. 구성별로 10회 반복합니다.

모델 구성은 계속 변경되며, 현재는 Inkling 2 Mini, Qwen, Gemma, gpt-oss, LFM 모델이 포함되어 있습니다. 여기에는 dense 아키텍처와 mixture-of-experts 아키텍처가 모두 포함됩니다. 사용 가능한 GGUF 형식에는 `Q4_K_M`, `Q8_0`, 기타 양자화 형식과 `BF16` 및 `F16` 같은 고정밀 형식이 포함됩니다.

## 결과 확인 및 재사용 {#section-4}

이 탐색기는 Hugging Face의 공개 [ggml-org/install.sh-benches-data Storage Bucket](https://huggingface.co/buckets/ggml-org/install.sh-benches-data)에서 직접 데이터를 읽습니다. 차트 제목 옆의 데이터 소스 링크를 클릭하면 선택한 빌드, 하드웨어, 모델에 해당하는 폴더가 열립니다.

표준 결과에는 `<build>/<hardware>/<model>/<quant>.jsonl`를 사용합니다. 워크로드 스윕에는 같은 폴더의 `<quant>-tg.jsonl` 파일 이름을 사용합니다.

이를 사용해 결과를 조사하거나, 직접 차트를 만들거나, 로컬 실행 결과와 비교할 수 있습니다.

## 함께 이야기 나누기 {#section-5}

테스트 인프라를 제공하고 설정을 검증하는 데 도움을 주신 AMD 팀에 감사드립니다! [대시보드](https://ggml-org-llamacpp-amd-bench.static.hf.space/index.html)를 사용해 보고, [데이터](https://huggingface.co/buckets/ggml-org/install.sh-benches-data)를 확인한 다음, 다음에는 어떤 모델과 워크로드를 보고 싶은지 알려 주세요.
