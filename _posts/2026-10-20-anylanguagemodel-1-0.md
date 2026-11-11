---
layout: post
title: "AnyLanguageModel 1.0: Apple 플랫폼에서 로컬 모델을 사용한 1년"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/anylanguagemodel-1-0/banner.png
image: assets/images/blog/posts/2026-10-20-anylanguagemodel-1-0/thumbnail.png
authors:
  - user: mattt
slug: "anylanguagemodel-1-0"
source_url: "https://huggingface.co/blog/anylanguagemodel-1-0"
source_published_date: "2026-10-20"
source_published_at: "2026-10-20T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [AnyLanguageModel 1.0: One Year of Local Models on Apple Platforms](https://huggingface.co/blog/anylanguagemodel-1-0)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/anylanguagemodel-1-0 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# AnyLanguageModel 1.0: Apple 플랫폼에서 로컬 모델을 사용한 1년

지난 11월, 이 블로그에서 [AnyLanguageModel](https://huggingface.co/blog/anylanguagemodel)을 소개했습니다.
Apple의 Foundation Models framework를 위한 drop-in 대체 API로,
앱에서 사용할 모델과 provider를 선택할 수 있게 해 줍니다.
로컬 모델에 관심은 있지만
항상 어디서부터 시작해야 할지 확신하지 못했던 앱 개발자들의 이야기를 들었습니다.
또한 여러 inference provider를 지원해야 하는 복잡성도 피하고 싶어 했습니다.
AnyLanguageModel은 이러한 피드백에서 탄생했습니다.

로컬 추론과 호스팅 추론은 각각 장점이 있으며,
이 새로운 세대의 지능형 앱을 구축하는 데 모두 중요한 역할을 합니다.
공유 API를 사용하면 각 작업에 적합한 모델을 더 쉽게 선택할 수 있습니다.

1년 동안 수십 차례의 릴리스를 거쳐
[AnyLanguageModel 1.0](https://github.com/huggingface/AnyLanguageModel/releases/tag/1.0.0)이 출시되었고,
[swift-huggingface 1.0](https://github.com/huggingface/swift-huggingface/releases/tag/1.0.0) 및
[swift-xet 1.0](https://github.com/huggingface/swift-xet/releases/tag/1.0.0)도 함께 제공됩니다.

무엇이 변경되었고 앞으로 무엇이 진행될지 살펴보겠습니다.

## drop-in 호환 API {#section-1}

AnyLanguageModel의 핵심 아이디어는 여전히 동일합니다.
앱에 적합한 모델과 함께 Foundation Models API를 사용하는 것입니다.
시작하려면 `import` 문을 변경하세요.

```diff
- import FoundationModels
+ import AnyLanguageModel
```


그러면 동일한 세션 코드가 시스템 모델,
llama.cpp, MLX, Core ML을 통한 open-weight 모델,
그리고 아래의 호스팅 provider와 함께 작동합니다.

지난 1년 동안 다음과 같은 주요 기능을 추가했습니다.

- **Tool calling**은 모든 provider에서 작동합니다.
- **Guided generation** 역시 모든 provider에서 작동하며,
  중첩된 `@Generable` 타입과 그 배열도 지원합니다.
- **Image support**를 사용하면 Foundation Models 27의 구문으로
  프롬프트에 이미지를 첨부할 수 있습니다.
  iOS, macOS, visionOS 27에서 시스템 모델을 사용할 때도 마찬가지입니다.

OS 27에서 AnyLanguageModel은 Foundation Models의 `LanguageModel` 프로토콜을
준수하는 모든 모델과도 작동하며,
[Core AI](https://github.com/apple/coreai-models) 모델도 포함됩니다.

AnyLanguageModel과 함께
[swift-huggingface](https://github.com/huggingface/swift-huggingface) 및
[swift-xet](https://github.com/huggingface/swift-xet)도 확장하여
Hugging Face Hub에서 Swift 앱으로 가속된 모델 다운로드를 지원합니다.

## 호스팅 모델도 지원 {#section-2}

올해에는 호스팅 모델에 대한 지원도 확장되었습니다.
이제 AnyLanguageModel은 다음을 지원합니다.

- OpenAI의 Chat Completions 및 Responses API
- 모든 provider에서 호환되는 endpoint를 사용할 수 있는 [Open Responses](https://www.openresponses.org)
- adaptive thinking 및 effort 옵션을 지원하는 Anthropic의 Messages API
- Google의 Gemini API
- Ollama
- OS 27에서 Private Cloud Compute
- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) 및 OpenRouter와 같은 기타 OpenAI 호환 endpoint
  (custom base URL을 통한 연결)

Inference Providers를 사용하면
디바이스에서 실행하기에는 너무 큰 open-weight 모델도
동일한 세션 코드로 사용할 수 있습니다.

```swift
let model = OpenAILanguageModel(
    baseURL: URL(string: "https://router.huggingface.co/v1/")!,
    apiKey: ProcessInfo.processInfo.environment["HF_TOKEN"]!,
    model: "openai/gpt-oss-120b",
    apiVariant: .chatCompletions
)
```


호스팅 provider는 로컬 모델과 동일한 기능을 제공합니다.
Guided generation, streaming을 지원하는 tool calling,
응답의 token usage가 이에 포함됩니다.

이제 reasoning도 transcript의 일부가 됩니다.
provider를 전환하면
새 provider가 reasoning 블록을 재생할 수 없는 경우 해당 블록이 요청에서 자동으로 생략됩니다.

API 키는 문자열 대신 closure에서 가져오므로,
앱이 자체 서버에서 단기 토큰을 가져올 수 있습니다.
[README](https://github.com/huggingface/AnyLanguageModel#api-credentials-and-security)은
사용자가 자신의 키를 가져오도록 할 때와
proxy를 통해 요청을 라우팅할 때를 설명합니다.

AnyLanguageModel은 Linux에서도 실행되므로,
동일한 코드가 서버에서도 작동할 수 있습니다.

## Open-weight 모델의 성능 향상 {#section-3}

로컬 모델을 더 쉽게 사용할 수 있도록 만드는 동안,
모델 자체도 더욱 강력해졌습니다.
따라서 작은 모델부터 시작하는 것도 좋은 선택입니다.
현재 Hugging Face Hub에서 가장 많이 다운로드된 모델 중 일부는 다음과 같습니다.

- Google의 [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf).
  GGUF 형식이며, quantization-aware training을 적용한 `Q4_0`에서 약 3.4 GB입니다.
- Liquid AI의 [LFM2.5 2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF).
  GGUF 형식이며, `Q4_K_M`에서 약 1.7 GB입니다.
- Alibaba의 [Qwen3.8 27B for MLX](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit).
  4비트에서 약 16 GB입니다.

앞의 두 모델은 iPhone에서 실행할 수 있으며,
Qwen3.8 27B는 메모리가 32 GB 이상인 Mac에서 실행됩니다.
MLX 모델을 선택하는 데는 한 줄이면 충분합니다.

```swift
let model = MLXLanguageModel(modelId: "mlx-community/Qwen3.8-27B-4bit")
```


최대 사양의 Mac Studio가 네 대 있다면,
Z.ai의 [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)은 BF16에서 약 1.5 TB를 사용합니다.
그 외의 사용자라면 Inference Providers를 통해 사용할 수 있습니다.

## 1.0으로 업그레이드 {#section-4}

```swift
dependencies: [
    .package(
        url: "https://github.com/huggingface/AnyLanguageModel",
        from: "1.0.0",
        traits: ["MLX"]
    )
]
```


기존 앱을 업그레이드하는 경우 코드를 업데이트하는 데 도움이 되도록
[AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel/releases/tag/1.0.0),
[swift-huggingface](https://github.com/huggingface/swift-huggingface/releases/tag/1.0.0),
[swift-xet](https://github.com/huggingface/swift-xet/releases/tag/1.0.0) 릴리스 노트의
마이그레이션 가이드를 참고하세요.

커뮤니티의 피드백과 기여는 여러 주요 기능을 형성하는 데 중요한 역할을 했습니다.
AnyLanguageModel을 앱 개발자에게 더욱 유용하게 만드는 데 도움을 주신
[everyone who contributed](https://github.com/huggingface/AnyLanguageModel/graphs/contributors)께 감사드립니다.

## 앞으로의 계획 {#section-5}

언어 모델은 제품에 지능을 더하는 강력한 도구임을 입증했습니다.
또한 개발자가 활용할 수 있는 여러 종류의 모델 중 하나이기도 합니다.
이와 동일한 접근 방식을 더 많은 기능으로 확장하게 되어 기대가 큽니다.
디바이스에서 실행되는 모델을 선택할 수 있는 익숙한 Swift API를 제공하는 것입니다.
곧 제공될 기능: 판단하는 모델과 듣는 모델.

## 링크 {#section-6}

- [AnyLanguageModel on GitHub](https://github.com/huggingface/AnyLanguageModel)
- [swift-huggingface on GitHub](https://github.com/huggingface/swift-huggingface)
- [swift-xet on GitHub](https://github.com/huggingface/swift-xet)
- [Introducing AnyLanguageModel](https://huggingface.co/blog/anylanguagemodel)
- [Introducing swift-huggingface](https://huggingface.co/blog/swift-huggingface)
