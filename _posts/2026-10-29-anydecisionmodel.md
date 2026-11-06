---
layout: post
title: "AnySpeechRecognizer와 AnyDecisionModel 소개: Apple 플랫폼을 위한 음성 및 의사 결정 오픈 모델"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/anydecisionmodel/banner.png
image: assets/images/blog/posts/2026-10-29-anydecisionmodel/thumbnail.png
authors:
  - user: mattt
slug: "anydecisionmodel"
source_url: "https://huggingface.co/blog/anydecisionmodel"
source_published_date: "2026-10-29"
source_published_at: "2026-10-29T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Introducing AnySpeechRecognizer and AnyDecisionModel: Open Models for Speech and Decisions on Apple Platforms](https://huggingface.co/blog/anydecisionmodel)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/anydecisionmodel -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# AnySpeechRecognizer와 AnyDecisionModel 소개: Apple 플랫폼을 위한 음성 및 의사 결정 오픈 모델

언어 모델은 많은 일을 할 수 있습니다.
글을 쓰고, 요약하고, 번역하고, 도구를 호출합니다.
하지만 이는 머신러닝이 할 수 있는 일의 일부에 불과합니다.

고객이 지원 전화로 연락했다고 가정해 보겠습니다.
먼저 고객이 무슨 말을 했는지 알고 싶을 것입니다(음성 인식).
그다음에는 고객을 어디로 보낼지 결정해야 합니다(의사 결정).
두 작업을 모두 언어 모델에 맡길 _수도_ 있지만,
각 작업에 맞게 만들어진 모델이 있으며,
이 모델이 더 뛰어나고 빠르며 비용도 적게 듭니다.

지난 [AnyLanguageModel 1.0 recap](https://huggingface.co/blog/anylanguagemodel-1-0)에서는
Swift에서 로컬 및 호스팅 모델을 1년간 살펴봤습니다.
이제 AnyLanguageModel과 swift-huggingface가 1.x 릴리스에 도달했으므로,
같은 접근 방식을 음성 및 의사 결정으로 확장합니다.
익숙한 Swift API와 모델 선택 기능을 제공합니다.

오늘은 두 가지 패키지를 발표합니다.

- [AnySpeechRecognizer](https://github.com/mattt/AnySpeechRecognizer)
  Apple의 새로운 음성-텍스트 API를 구현하며,
  NVIDIA의 Parakeet과 Alibaba의 Qwen3-ASR을 포함한 여러 인식 엔진을 선택할 수 있습니다.
  두 모델은 Core ML에서 로컬로 실행되며, 언제 누가 말하는지 추적할 수 있도록 화자 분리 기능도 추가합니다.
- [AnyDecisionModel](https://github.com/mattt/AnyDecisionModel)
  예/아니요, 객관식, 평점 척도 질문에
  타입이 지정된 결과와 확률로 답합니다.
  텍스트를 생성하는 대신 모델의 확률에서 각 답을 도출하며,
  MLX, Core AI 또는 llama.cpp를 사용해 기기에서 실행되는 모델이나
  Jev 또는 Ollama의 의사 결정 모델을 사용할 수 있습니다.

두 패키지 모두 Hugging Face Hub에서 오픈 모델을 다운로드하고
기기에서 실행할 수 있습니다.

## AnySpeechRecognizer {#section-1}

컴퓨터는 수십 년 동안 음성을 전사해 왔으며,
지난 몇 년 사이에는 그 성능이 매우 좋아졌습니다.
Apple 플랫폼에는 이미 이를 위한 API가 있습니다.
macOS 26과 iOS 26에서는 Apple의 새로운 음성-텍스트 API인 `SpeechAnalyzer`를 도입했습니다.
`SpeechTranscriber` 및 `SpeechDetector`와 같은 모듈을 분석기에 추가하고,
파일이나 오디오 버퍼 스트림을 전달한 다음,
비동기 시퀀스에서 도착하는 결과를 읽습니다.
`AssetInventory`가 모델을 설치하므로,
다운로드가 이루어지는 시점을 앱이 결정할 수 있습니다.

잘 설계된 API이지만,
이를 뒷받침하는 엔진은 Apple의 엔진 하나뿐입니다.
AnySpeechRecognizer는 API를 그대로 유지하면서 엔진을 선택할 수 있게 합니다.

```diff
- import Speech
+ import AnySpeechRecognizer
```


이 패키지는 `SpeechAnalyzer`, `SpeechTranscriber`, `DictationTranscriber`,
`SpeechDetector`, `AssetInventory`를 동일하게 제공하며,
Apple의 Speech 프레임워크를 기본 엔진으로 사용합니다.
코드가 이 공통 API를 사용한다면 import만 변경하면 됩니다.
대신 Parakeet을 사용하려면
transcriber를 생성하는 방법만 변경하면 됩니다.

```swift
let transcriber = SpeechTranscriber(
    model: ParakeetSpeechModel(variant: .v3),
    transcriptionOptions: [],
    reportingOptions: [.volatileResults],
    attributeOptions: [.audioTimeRange]
)
```


에셋 설치, 분석기 실행,
결과 읽기 단계는 그대로 유지됩니다.
[README](https://github.com/mattt/AnySpeechRecognizer#transcribe-a-file)에는
완전한 예제가 포함되어 있으며,
Apple 프레임워크의 [lists the differences](https://github.com/mattt/AnySpeechRecognizer#differences-from-apples-speech-framework)도 제공합니다.

### 모델 만나 보기

**Parakeet v3**는 NVIDIA의 [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)입니다.
다국어 모델로,
유럽의 25개 언어를 인식하고 어떤 언어가 들리는지도 판별하므로
로캘을 전달할 필요가 없습니다.

**Parakeet Ultra**는 Moondream의 [post-trained take on v3](https://huggingface.co/moondream/parakeet-ultra)입니다.
대부분의 측정에서 더 정확했습니다.

**Parakeet Unified**는 NVIDIA의
[English-only Parakeet](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)입니다.

세 Parakeet 모델 모두 패키지 자체의 Core ML 런타임에서 실행되며,
파일 및 라이브 입력, 휘발성 결과, 단어 타이밍,
단어 신뢰도를 지원합니다.

**Qwen3-ASR**은 Alibaba의 음성 모델로,
[0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 및 [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) 크기로 제공됩니다.
30개 언어와 22개 중국어 방언을 인식하며,
Parakeet이 지원하지 않는 만다린, 광둥어, 일본어, 한국어도 포함됩니다.
언어를 자동으로 감지하거나 직접 지정할 수도 있습니다.

```swift
let transcriber = SpeechTranscriber(
    model: QwenSpeechModel(variant: .large, language: Locale.Language(identifier: "yue")),
    transcriptionOptions: [],
    reportingOptions: [],
    attributeOptions: []
)
```


Qwen3-ASR은 최대 40초 길이의 각 세그먼트에 대해 최종 결과 하나를 제공하며,
휘발성 결과, 단어 타이밍 또는 신뢰도는 제공하지 않습니다.
GPU에서 실행되며 Parakeet보다 느리므로,
라이브 입력보다는 파일 전사에 더 적합합니다.

다른 엔진을 추가하려면 해당 구성에 대해 `SpeechRecognitionModel`를,
하나의 모듈 세션에 대해 `SpeechRecognitionExecutor`를 구현합니다.
README에는 시작을 돕기 위해 자리표시자 텍스트를 보고하는 [small provider](https://github.com/mattt/AnySpeechRecognizer#custom-providers)가 포함되어 있습니다.

### Hub의 Core ML 모델

앱에서 이러한 모델을 실행할 수 있도록
Core ML로 변환하고 Hub에 게시했습니다.

| 모델 | Core ML 변환 | 원본 체크포인트 | 라이선스 |
| --- | --- | --- | --- |
| Parakeet TDT 0.6B v3 | [mattt/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/mattt/parakeet-tdt-0.6b-v3-coreml) | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | CC BY 4.0 |
| Parakeet Ultra | [mattt/parakeet-ultra-coreml](https://huggingface.co/mattt/parakeet-ultra-coreml) | [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) | CC BY 4.0 |
| Parakeet Unified EN 0.6B | [mattt/parakeet-unified-en-0.6b-coreml](https://huggingface.co/mattt/parakeet-unified-en-0.6b-coreml) | [nvidia/parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
| Qwen3-ASR 0.6B | [mattt/qwen3-asr-0.6b-coreml](https://huggingface.co/mattt/qwen3-asr-0.6b-coreml) | [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | Apache 2.0 |
| Qwen3-ASR 1.7B | [mattt/qwen3-asr-1.7b-coreml](https://huggingface.co/mattt/qwen3-asr-1.7b-coreml) | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | Apache 2.0 |
| Streaming Sortformer 4spk v2 | [mattt/diar-streaming-sortformer-4spk-v2-coreml](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2-coreml) | [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | CC BY 4.0 |
| Streaming Sortformer 4spk v2.1 | [mattt/diar-streaming-sortformer-4spk-v2.1-coreml](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2.1-coreml) | [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |

각 저장소에는 소스 체크포인트와
모든 파일의 체크섬이 기록되어 있으며,
패키지는 이를 고정된 리비전에 고정합니다.
Parakeet 저장소에는 라이브 입력을 위한 8초 인코더도 추가되어 있으며,
다소 높은 단어 오류율을 감수하는 대신
라이브 분석 시간을 약 3분의 1로 줄입니다.
저장소에는 변환 워크플로도 포함되어 있고,
대부분 FluidInference의 [mobius](https://github.com/FluidInference/mobius) 스크립트를 따르며,
각 변환을 원본 모델과 대조해 확인합니다.

다운로드는 앱이 요청할 때만 이루어집니다.
각 Parakeet은 약 600MB, 라이브 인코더는 약 590MB,
Qwen3-ASR은 0.9~2.4GB, Sortformer는 약 250MB입니다.
파일은 표준 Hugging Face 캐시에 저장되므로,
샌드박스 외부의 Mac 앱은 Python 및 다른 Hugging Face 도구와 이를 공유합니다.

### 화자 분리

대화를 전사할 때는
누가 무엇을 말했는지 알고 싶은 경우가 많습니다.
이를 화자 분리라고 하며,
Apple의 Speech 프레임워크에는 이를 위한 모듈이 없으므로
AnySpeechRecognizer가 하나를 추가합니다.
`SpeakerDiarizer`는 NVIDIA의
[Streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)를
Apple 또는 Parakeet transcriber와 함께 Core ML에서
동일한 타임라인으로 실행합니다.

```swift
let diarizer = SpeakerDiarizer(model: SortformerDiarizationModel())
if let request = try await AssetInventory.assetInstallationRequest(
    supporting: [transcriber, diarizer]
) {
    try await request.downloadAndInstall()
}
let analyzer = SpeechAnalyzer(modules: [transcriber, diarizer])
```


화자 분리기는 서로 겹쳐서 말하는 경우에도
최대 네 명의 화자를 추적합니다.
`SpeakerTimeline`가 결과를 수집하고
각 시간 지정 단어를 화자에게 귀속하므로,
화자별 발화 순서로 전사문을 출력할 수 있습니다.
이는 AnyLanguageModel의 이미지 지원과 마찬가지로,
현재 Apple API가 제공하는 범위를 넘어섭니다.

화자 분리기는 기본적으로 Sortformer v2를 사용합니다.
[v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)를 선택할 수도 있습니다.
NVIDIA가 더 많은 회의 데이터로 학습한 모델이며,
측정 결과 회의 데이터에서도 더 정확했습니다.

### 성능은 어느 정도일까요?

macOS 27.0.1이 실행되는 M4 Max Mac Studio에서
`hear` 명령줄 도구를 사용해 패키지를 측정했습니다.
다음은 3~8분 길이의 LibriSpeech 파일에 대한 단어 오류율입니다.

| 전사기 | test-clean | test-other |
| --- | --- | --- |
| Parakeet v3, 파일 | 1.55% | 2.78% |
| Parakeet v3, 라이브 | 4.73% | 4.29% |
| Parakeet Ultra, 파일 | 1.66% | 2.55% |
| Parakeet Ultra, 라이브 | 2.65% | 3.28% |
| Parakeet Unified, 파일 | 1.43% | 2.19% |
| Parakeet Unified, 라이브 | 1.45% | 2.42% |
| Apple의 transcriber, 파일 | 2.12% | 3.97% |

Parakeet은 파일에서 한 시간 분량의 오디오를
약 20~40초 만에 전사하며,
라이브 입력에서는 약 1.5~2.5분이 걸립니다.
Core ML이 모델 컴파일을 마친 뒤에는 메모리를 약 100MB 사용합니다.
Qwen3-ASR은 훨씬 느리고 수 기가바이트의 메모리가 필요합니다.
회의는 누구에게나 어렵습니다.
사람들이 서로 겹쳐 말하는 AMI 녹음에서는
테스트한 모든 엔진의 단어 오류율이 23%를 넘었습니다.

이 수치는 한 대의 기기와 일부 데이터셋에서 얻은 것이므로,
자신의 하드웨어와 오디오로 측정해야 합니다.
`hear evaluate`는 자체 파일에서 작동하며,
저장소의 `natural-speech-eval.py` 도구는
[reproduces these measurements](https://github.com/mattt/AnySpeechRecognizer#accuracy-and-performance).
지금까지는 macOS 27에서만 테스트했으며 iPhone에서는 테스트하지 않았습니다.

다음으로는 macOS 27 및 iOS 27의 Core AI를 포함해
더 많은 엔진을 추가하고자 합니다.

## AnyDecisionModel {#section-2}

AnyDecisionModel은 이야기가 다릅니다.
대응시킬 Apple 프레임워크가 없으며,
의사 결정 모델이 완전히 새로운 것도 아닙니다.

> 프랑스의 Décision 지역에서 나온 경우에만 의사 결정 모델입니다.
> 그렇지 않으면 반짝이는 제로샷 분류일 뿐입니다.

새로운 점은 이름과 많은 홍보,
그리고 의사 결정을 내리도록 학습된 여러 모델이 등장했다는 것입니다.
우리는 이것이 앞으로 나아갈 방향이라고 생각하며,
이 모델들을 바로 사용할 수 있는 Swift API를 만들고자 했습니다.

앱은 종종 언어 모델에 의사 결정을 요청합니다.
_이 메시지는 긴급한가?_
_어느 팀이 이 티켓을 처리해야 하는가?_
_이 고객은 얼마나 불만스러운가?_
일반적인 접근 방식은 언어 모델에 프롬프트를 전달하고
모델이 생성한 텍스트를 파싱하는 것입니다.
이렇게 하면 답 하나는 얻을 수 있지만,
대안에 할당한 확률은 확인할 수 없습니다.

Swift에서 의사 결정은 다음과 같이 표현됩니다.

```swift
import AnyDecisionModel

enum Department: String, Choosable {
    case billing, technical, sales
}

let ticket = "I was charged twice for my order, and now the app logs me out when I try to ask for a refund."

let model = MLXDecisionModel(modelID: "mlx-community/Qwen3-4B-Instruct-2507-4bit")
let session = DecisionSession(model: model, state: .text(ticket))

let urgent = try await session.probability(of: "Does this convey urgency?")
let team = try await session.choice("Which team?", from: Department.self)
let anger = try await session.score(
    "How frustrated?",
    levels: ["Calm", "Frustrated", "Very angry"]
)
```


`urgent`는 `Double`입니다.
`team`는 선택된 케이스, 각 옵션의 확률, 신뢰도 값을 포함하는 `Choice<Department>`입니다.
`anger`는 예상 수준, 각 수준의 확률, 신뢰도 값을 포함하는 `Score`입니다.
`Department`는 `Choosable`을 준수하므로,
`team.value`는 완전한 `switch`에서 사용할 수 있는 `Department`입니다.

Foundation Models 또는 AnyLanguageModel을 사용해 본 적이 있다면,
모델과 세션 패턴이 익숙하게 느껴질 것입니다.
`DecisionSession`는 텍스트 또는 JSON일 수 있는 단일 상태를 보유하며,
모든 질문은 이 상태에 관한 것입니다.
질문은 독립적이므로
한 질문의 답이 이후 질문의 입력이 되지 않습니다.
여러 질문을 한 번에 하려면
`decide(_:)`에 배치로 전달합니다.

각 질문에 대해 MLX 백엔드는
허용된 각 답에 대해 하나의 토큰 레이블을 포함하는 프롬프트를 구성한 다음,
한 번의 순전파에서 각 레이블의 확률을 읽습니다.

### 직접 사용해 보기: Emoji Magnet

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/anydecisionmodel/emoji-magnet.mp4" title="&quot;things you wear&quot;부터 &quot;junk food&quot;까지 일곱 문구에 맞춰 이모지를 분류하는 Emoji Magnet"></video>

Emoji Magnet은 343개의 이모지 더미 위에 텍스트 필드가 있는 macOS 앱입니다.
"things you wear"와 같은 문구를 입력하면
해당하는 이모지가 더미에서 떠오릅니다.
이를 "things you wear in winter"로 구체화하면
더 이상 해당하지 않는 이모지는 다시 내려갑니다.

앱에는 내장 키워드 규칙이 없습니다.
모델이 추론 시점에 모든 것을 결정합니다.
입력을 잠시 멈추면 모델에 각 이모지에 대해 예/아니요 질문 하나를 요청합니다.
예를 들면 다음과 같습니다.

```
Statement: a heavy winter coat fits the description "things you wear". Is this statement true?
```


세션의 상태에는 문구가 저장됩니다.
위 데모에서는 호스팅된 Jev 백엔드를 사용하며,
343개의 질문에 모두 한 번의 요청으로 답하므로
각 문구에 대해 이모지가 약 0.5초 만에 정렬됩니다.
`TYPESAFE_AI_API_KEY`가 없으면 앱은 MLX 백엔드를 사용해 Mac에서 Qwen3 4B를 실행합니다.
이 백엔드는 프리픽스 캐싱을 사용해 상태를 한 번 평가한 다음,
순전파 한 번마다 16개의 질문에 답합니다.
M4 Max에서는 모델이 로드된 후 16개 그룹을 처리하는 데 약 0.45초가 걸립니다.
343개 이모지 전체를 처리하는 데는 약 10초가 걸리며,
각 그룹의 결과는 도착하는 즉시 표시됩니다.

Emoji Magnet은 [AnyDecisionModel-Examples](https://github.com/mattt/AnyDecisionModel-Examples) 저장소에 있습니다.

### 의사 결정 헤드와 Clef

다음 토큰 확률을 읽는 방식은 범용 언어 모델에서 작동하지만,
이러한 모델은 의사 결정을 내리도록 학습되지 않았습니다.
Cloudflare의 [Clef](https://huggingface.co/Cloudflare/clef) 모델은
더 새로운 접근 방식의 한 예입니다.
학습된 의사 결정 헤드를 가진 모델이
상태와 타입이 지정된 질문 스키마를 의사 결정으로 변환하고,
한 번의 순전파에서 질문 전체의 허용된 모든 옵션에 대한 확률을 반환합니다.
[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash)는 9B 변형입니다.

AnyDecisionModel은 MLX를 사용해 Mac에서 Clef-Flash를 실행합니다.

```swift
let model = MLXHeadDecisionModel(
    modelID: "mlx-community/clef-flash-4bit",
    head: .jointSchema
)
```


동일한 세션과 질문을 사용하지만,
`MLXHeadDecisionModel`는 다음 토큰 확률 대신 Clef의 학습된 의사 결정 헤드에서 각 의사 결정을 읽습니다.
[4-bit model](https://huggingface.co/mlx-community/clef-flash-4bit)는 6.2GB 다운로드이며,
모델 카드에서는 최소 16GB 메모리를 갖춘 Mac을 권장합니다.
Clef는 이미지와 동영상을 읽을 수 있지만,
이 백엔드는 텍스트 및 JSON 상태만 허용합니다.

### 확률 보정 및 확인

언어 모델의 원시 확률은 과도하게 확신하는 경우가 많습니다.
로컬 백엔드는 `Calibration`를 허용하며,
이는 모든 답에 온도를 적용하고 예/아니요 답에는 편향을 적용합니다.
별도의 데이터셋을 사용해 보정을 맞추고 결과를 평가하세요.

언어 모델은 목록에서의 위치 때문에 특정 옵션을 선호할 수도 있습니다.
MLX, Core AI, llama.cpp의 다음 토큰 백엔드에서는
`rotationDebiasing`를 켜서
선택지의 순열마다 선택 질문을 한 번씩 요청하고 확률을 평균낼 수 있습니다.
순열마다 프롬프트 평가 한 번의 비용이 발생합니다.

각 로컬 선택 및 점수에는 확률의 엔트로피를 기반으로 한
신뢰도 값도 포함됩니다.
이는 확률이 얼마나 집중되어 있는지를 측정할 뿐,
답이 맞는지는 측정하지 않습니다.
따라서 임계값을 선택하기 전에 자신의 데이터에서 오류율을 확인하세요.
다음 토큰 백엔드의 진단 정보는
모델이 허용된 답에 할당한 확률의 양을 보고합니다.
값이 낮다는 것은 모델이 확률의 대부분을 다른 곳에 할당했다는 의미입니다.

### 백엔드

AnyDecisionModel에는 여섯 가지 백엔드가 있습니다.

- **MLX**는 Apple silicon에서 소형 언어 모델을 실행하며
  [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)를 사용해
  Hub에서 모델을 다운로드합니다.
- **MLX decision heads**는 Clef와 같이 자체 학습된 의사 결정 헤드를 가진 모델을 실행합니다.
- **Core AI**는 macOS 27 및 iOS 27에서 Core AI 모델을 사용해 MLX와 동일한 작업을 수행합니다.
- **llama.cpp**는 [llama.swift](https://github.com/mattt/llama.swift)를 통해
  로컬 GGUF 모델에서 의사 결정을 읽습니다.
- **Jev**는 [System One API](https://docs.typesafe.ai/api/system-one)를 통해 TypeSafe의 Jev를 호출하거나,
  [Kev](https://github.com/jaredpalmer/kev)와 같이 이를 구현하는 서버를 호출합니다.
- **Ollama**는 로컬 Ollama 서버 또는 Ollama의 클라우드에서
  System One 의사 결정 모델을 호출합니다.
  Ollama 0.35.0 이상이 필요하며, 클라우드 모델에는 0.40.5 이상이 필요합니다.

다음 토큰 백엔드는 답을 "yes"와 "no", A부터 Z, 또는 0부터 9로 레이블링하므로,
선택지는 최대 26개, 점수는 최대 10개 수준까지 가능합니다.
의사 결정 헤드에는 이러한 제한이 없습니다.

AnyLanguageModel과 마찬가지로 이 패키지는 Swift 패키지 traits를 사용합니다.
trait가 없으면 서드파티 의존성이 없습니다.
핵심 타입과 Jev 및 Ollama 백엔드를 사용할 수 있으며,
이 백엔드는 Linux에서도 빌드됩니다.
로컬 백엔드에는 `MLX`, `CoreAI` 또는 `Llama` trait를 활성화하세요.

## 시작하기 {#section-3}

두 패키지 모두 Swift 6.2 이상과 macOS 26 또는 iOS 26 이상이 필요합니다.
Parakeet 및 Sortformer 런타임과 MLX 백엔드는 Apple silicon이 필요합니다.
AnyDecisionModel의 MLX 빌드 작업에는 Xcode의 Swift 6.4 이상과 Metal 툴체인이 필요하며,
Core AI에는 Xcode 27과 OS 27 런타임이 필요합니다.

`Package.swift`에 패키지를 추가합니다.

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnySpeechRecognizer", from: "0.1.0"),
    .package(
        url: "https://github.com/mattt/AnyDecisionModel",
        from: "0.6.0",
        traits: ["MLX"]
    ),
]
```


현재 AnyDecisionModel의 `CoreAI` trait는 AnySpeechRecognizer와 함께 사용할 수 없습니다.
coreai-kit 0.7.3은 swift-huggingface 1.0 미만을 요구하는 반면,
AnySpeechRecognizer는 1.x를 요구하기 때문입니다.

각 README에는 요구 사항과 백엔드에 관한 더 많은 예제와 세부 정보가 있습니다.
AnySpeechRecognizer에는 명령줄 도구 `hear`도 포함되어 있으며,
이 도구는 로컬 파일을 전사하고 화자를 표시하며 결과를 평가합니다.
모든 `hear` 명령에는 `--backend apple`, `--backend parakeet` 또는 `--backend qwen`가 필요합니다.
분석 전에 동일한 백엔드와 모델 옵션으로 `hear prepare`를 실행해
필요한 에셋을 설치하세요.

앱에서 사용해 보고 무엇이 잘되고 무엇이 잘되지 않는지 알려 주세요.
두 저장소 모두에서 이슈와 pull request를 환영합니다.

## 링크 {#section-4}

- [AnySpeechRecognizer on GitHub](https://github.com/mattt/AnySpeechRecognizer)
- [AnyDecisionModel on GitHub](https://github.com/mattt/AnyDecisionModel)
- [AnyDecisionModel-Examples on GitHub](https://github.com/mattt/AnyDecisionModel-Examples)
- [AnyLanguageModel on GitHub](https://github.com/huggingface/AnyLanguageModel)
- [swift-huggingface on GitHub](https://github.com/huggingface/swift-huggingface)

_Parakeet, Sortformer, NeMo를 제공한 NVIDIA,
Parakeet Ultra를 제공한 Moondream,
Parakeet 런타임과 Core ML 변환이 따르는 규칙을 문서화한 FluidInference의 FluidAudio 및 mobius 프로젝트,
Clef를 제공한 Cloudflare,
그리고 두 패키지가 그 패턴을 차용한 AnyLanguageModel에 기여한 모든 분께 감사드립니다._
