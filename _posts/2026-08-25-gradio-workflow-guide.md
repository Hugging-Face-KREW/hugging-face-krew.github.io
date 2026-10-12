---
layout: post
title: "gr.Workflow로 무엇이든 빌드하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/gradio-workflow-guide/thumbnail.png
image: assets/images/blog/posts/2026-08-25-gradio-workflow-guide/thumbnail.png
authors:
  - user: ysharma
  - user: abidlabs
slug: "gradio-workflow-guide"
source_url: "https://huggingface.co/blog/gradio-workflow-guide"
source_published_date: "2026-08-25"
source_published_at: "2026-08-25T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Wire It, Run It, Deploy It: AI Workflows in Gradio](https://huggingface.co/blog/gradio-workflow-guide)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/gradio-workflow-guide -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# gr.Workflow로 무엇이든 빌드하기

흥미로운 AI 앱은 대부분 파이프라인입니다. 이미지를 생성한 다음, 원한다면 배경을 잘라내거나 새로운 이미지로 편집합니다. 스크립트를 작성한 다음 음성을 생성하거나, 스크립트는 그대로 둔 채 음성만 바꿉니다. 일반적으로 이러한 단계를 Python으로 연결하고, 무언가 이상해지는 순간 어떤 단계에서 이상한 값이 생성됐는지 찾기 위해 다시 출력 디버깅으로 돌아갑니다.

**`gr.Workflow`**는 Gradio에 바로 내장되어 있으며, 파이프라인을 *인터페이스*로 만듭니다. 단계를 타입이 지정된 노드의 그래프로 설명하면 Gradio가 드래그 앤 드롭 캔버스를 제공하고, 모든 노드를 실행할 수 있으며 모든 중간 결과를 확인할 수 있습니다. 동일한 그래프는 REST API이자 Hugging Face Spaces에 한 번의 명령으로 배포할 수 있는 대상이기도 합니다.

이 개념을 이해하는 가장 좋은 방법은 몇 가지 워크플로가 실제로 동작하는 모습을 보는 것입니다. 아래의 모든 앱은 열어서 실행하고 복제할 수 있는 라이브 Hugging Face Space입니다.

## 이미지 편집 {#section-1}

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/Image edit workflow - sample2.mp4"></video>

이미지를 업로드하고 편집 내용을 입력하면("눈 내리는 겨울 풍경으로 바꿔 줘", "선글라스를 추가해 줘", "차를 빨간색으로 바꿔 줘" 등) 편집된 사진을 받을 수 있습니다. 전체 앱은 Hugging Face Inference Providers에서 [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)를 호출하는 단일 노드입니다.

👉 **[Try the Image Editor Pipeline](https://huggingface.co/spaces/ysharma/gr-workflow-image-editor)**

## 실제 모델을 연결해 미디어 스튜디오 만들기 {#section-2}

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/App 04.mp4"></video>

하나의 그래프에 세 개의 파이프라인이 있습니다. 프롬프트로 시작해 [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-schnell)로 이미지를 생성한 다음, 이를 [background-removal Gradio Space](https://huggingface.co/spaces/not-lain/background-removal)에 전달해 스티커로 변환합니다. 하나의 주제는 [text-to-speech Gradio Space](https://huggingface.co/spaces/mrfakename/MeloTTS)를 통해 보이스오버가 되고, 같은 주제는 [LLM](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 호출을 통해 눈길을 끄는 에피소드 제목이 됩니다.

하나의 캔버스에서 Hugging Face [Inference Providers](https://huggingface.co/docs/inference-providers/en/index)를 통한 두 번의 모델 호출과 Gradio Spaces에 대한 두 번의 호출이 이루어집니다.

이것은 워크플로이므로 세 출력 각각에도 고유한 REST 엔드포인트가 생성됩니다: `/sticker`, `/voiceover`, `/episode_title`. UI를 열지 않고도 코드에서 어느 엔드포인트든 직접 호출할 수 있습니다. 실행 가능한 예시는 아래의 [Call it from code](#call-it-from-code)을 참고하세요.

👉 **[Try the AI Media Studio](https://huggingface.co/spaces/ysharma/gr-workflow-04-ai-media-studio)**

## 이미지를 병렬로 팬아웃 생성하기 {#section-3}

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/fan-out-generation-sample1.mp4"></video>

하나의 아이디어를 입력하면 생성된 여러 작품이 한 번에 만들어집니다. FLUX로 생성한 기본 이미지, 해당 이미지를 AI로 재해석한 두 가지 버전(부드러운 수채화 버전과 네온 사이버펑크 버전), 그리고 LLM이 작성한 갤러리 제목이 생성됩니다.

각 이미지는 Inference Providers를 사용하는 모델 노드가 프롬프트에서 직접 생성하고, 제목은 LLM을 호출하는 `fn` 노드에서 생성됩니다. 이것이 바로 팬아웃 패턴입니다. 하나의 아이디어가 여러 연산자에 동시에 전달되어 모두 병렬로 생성 작업을 수행할 수 있습니다.

👉 **[Try the Generative Art Lab](https://huggingface.co/spaces/ysharma/gr-workflow-01-generative-art-lab)**

## Hugging Face 데이터셋 프로파일링하기 {#section-4}

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/data-detective-sample1.mp4"></video>

`stanfordnlp/imdb` 또는 `mteb/tweet_sentiment_extraction`와 같은 Hugging Face 데이터셋 ID를 입력하면, 하나의 입력이 네 개의 연산자 노드로 팬아웃되어 [Datasets Server](https://huggingface.co/docs/dataset-viewer) API를 사용해 데이터셋을 실시간으로 분석합니다.

개요 카드, 처음 몇 개 행의 미리보기, 열별 통계, 분포 차트를 얻을 수 있으며, 이 모든 결과는 서로 독립적으로 병렬 계산됩니다. 이것이 바로 워크플로의 힘입니다!

👉 **[Try Data Detective](https://huggingface.co/spaces/ysharma/gr-workflow-03-data-detective)**

## 직접 GPU 모델 실행하기 {#section-5}

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/zerogpu-animate-sample1.mp4"></video>

지금까지 살펴본 모든 노드는 Hugging Face에 연결됩니다. 하지만 `fn` 노드는 단순한 Python이므로, Space 내부의 GPU에서 모델을 실행할 수도 있습니다.

바인딩된 함수에 `@spaces.GPU`를 데코레이터로 적용하면, 노드가 실행될 때 [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)가 해당 호출을 위해 GPU를 할당하고 모델을 실행한 후 GPU를 반환합니다. 항상 Inference Providers나 기존 Gradio Spaces에 의존해야 하는 것은 아닙니다.

Diffusers를 통해 로드한 [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video-0.9.7-distilled)로 정지 이미지를 애니메이션으로 변환하는 데모를 확인해 보세요. 전체 작업이 하나의 노드에서 실행됩니다. `gr.Workflow`는 GPU 설정에 대해 아무것도 알 필요가 없습니다. 단순히 바인딩된 함수를 호출합니다.

👉 **[Try the ZeroGPU Animator](https://huggingface.co/spaces/ysharma/gr-workflow-zerogpu-animate)**

## 작동 방식 한눈에 보기 {#section-6}

모든 워크플로는 세 종류의 노드로 구성된 그래프입니다. **references**(입력), **operators**(작업을 수행하는 단계), **subjects**(출력)입니다. 연산자는 직접 작성한 Python 함수일 수도 있고, Hugging Face Inference Providers의 모델, 다른 Gradio Space, Hub 데이터셋의 행일 수도 있습니다. 타입이 지정된 포트 사이를 드래그해 연결하고 Run을 누르면 각 결과가 해당 위치에 나타나는 모습을 확인할 수 있습니다.

## 코드에서 호출하기 {#section-7}

빌드한 모든 워크플로는 추가 작업 없이 API이기도 합니다. 각 출력은 레이블을 이름으로 사용하는 REST 엔드포인트가 되며, Gradio client를 사용해 Python에서 호출할 수 있습니다. 다음은 여러 엔드포인트를 제공하는 데모 Space를 대상으로 한, 토큰이 필요 없는 라이브 예제이며 원문 그대로 실행할 수 있습니다.

```python
from gradio_client import Client

client = Client("ysharma/gr-workflow-multi-endpoint-API")

print(client.predict("hello there friend", api_name="/word_count"))  # -> 3
print(client.predict(20, api_name="/fahrenheit"))                    # -> 68.0
```


모델이나 Space를 호출하는 엔드포인트는 Hugging Face token으로 실행되므로, client를 생성할 때 토큰을 전달해야 합니다.

```python
from gradio_client import Client, handle_file

client = Client("ysharma/gr-workflow-image-editor", token="hf_...")

edited = client.predict(
    handle_file("dog.jpg"),
    "turn it into a snowy winter scene",
    api_name="/edited_image",
)
```


일반 HTTP를 선호하시나요? 모든 엔드포인트는 `curl`을 통해서도 접근할 수 있습니다.

```bash
curl -s https://ysharma-gr-workflow-multi-endpoint-API.hf.space/gradio_api/call/word_count \
  -H "Content-Type: application/json" -d '{"data": ["hello there friend"]}'
```


## 직접 빌드하기 {#section-8}

시작하는 가장 빠른 방법은 위의 데모 중 하나를 열고 **Duplicate**를 클릭한 다음 연결을 다시 구성하는 것입니다. Python에서는 다음처럼 간단합니다.

```python
import gradio as gr

def your_function(text: str) -> str:
  pass

gr.Workflow(bind=[your_function]).launch()
```


전체 튜토리얼, 연산자 종류, JSON 스키마, 재사용 가능한 패턴은 Gradio docs의 공식 [gr.Workflow guide](https://gradio.app/guides/workflows)을 참고하세요.

`gr.Workflow`을 사용하면 [AUTOMATIC1111](https://github.com/automatic1111/stable-diffusion-webui)처럼 복잡한 작업도 빌드할 수 있습니다. 다음 게시물에서는 이를 단계별로 빌드하는 과정을 다룰 예정이니 기대해 주세요. 먼저 살짝 엿보면 다음과 같습니다 😉👇

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow-guide/workflow1111-sample2.mp4"></video>
