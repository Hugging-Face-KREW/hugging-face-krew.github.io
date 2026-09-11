---
layout: post
title: "Gradio Workflow로 AUTOMATIC1111 재구축하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/gradio-workflow1111/thumbnail.png
image: assets/images/blog/posts/2026-09-10-gradio-workflow-1111/thumbnail.png
authors:
  - user: ysharma
  - user: abidlabs
slug: "gradio-workflow-1111"
source_url: "https://huggingface.co/blog/gradio-workflow-1111"
source_published_date: "2026-09-10"
source_published_at: "2026-09-10T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Rebuilding AUTOMATIC1111 with Gradio Workflow](https://huggingface.co/blog/gradio-workflow-1111)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/gradio-workflow-1111 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Gradio Workflow로 AUTOMATIC1111 재구축하기

지난 [last post](https://huggingface.co/blog/gradio-workflow-guide)에서 다섯 개의 작은 `gr.Workflow` 그래프를 만들고, AUTOMATIC1111의 [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)처럼 복잡한 것을 구축하려면 무엇이 필요한지 살펴보았습니다. 이번 글에서는 하나의 워크플로 캔버스에서 AUTOMATIC1111의 기능 대부분을 재구축한 **Workflow1111**을 소개합니다.

Workflow1111은 **73개의 노드**로 구축된 **11개의 미디어 파이프라인** 그래프입니다. 텍스트-이미지, 고해상도 보정, 이미지-이미지, 프롬프트 매트릭스 그리드, VLM interrogate, 감지-인페인트 마스크, ControlNet 스타일 annotator, 배경 제거, PNG Info 저장, 이미지-동영상에 SOTA 모델을 한데 모았습니다.

Hugging Face 계정으로 로그인하거나 access token을 제공하면 이러한 파이프라인을 실행할 수 있습니다. 로그인하면 모델 호출에 자신의 quota가 사용됩니다.

👉 **[Try Workflow1111](https://huggingface.co/spaces/ysharma/Workflow1111)**하거나, Space를 복제해 자신의 사용 사례에 맞게 다시 연결해 보세요.

이제 캔버스를 살펴보겠습니다.

## 캔버스에는 무엇이 있나 {#section-1}

모든 미디어 파이프라인은 지난 글과 [official guide](https://gradio.app/guides/workflows#operator-kinds)에서 다룬 동일한 네 가지 operator 종류로 구축됩니다. 캔버스의 각 노드는 하나의 operator를 감싸며, operator의 입력과 출력은 edge를 연결하는 port가 됩니다. 네 가지 operator 종류를 간단히 정리하면 다음과 같습니다. `fn`은 Python 함수이고, `model`는 `InferenceClient`를 통해 호출되는 모델이며, `space`는 또 다른 Gradio Space이고, `dataset`는 Hub 데이터셋의 한 행입니다.

파이프라인을 하나씩 살펴보겠습니다.

### 텍스트-이미지

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/txt2img.mp4"></video>

이것이 핵심 파이프라인입니다. A1111의 txt2img 탭에서 볼 수 있는 컨트롤인 negative prompt, steps, CFG, seed, width와 height, 그리고 체크포인트를 선택하는 `model_id` 필드를 제공합니다. 프롬프트는 먼저 prompt-builder `fn` 노드를 거치며, 여기서 선택한 스타일 preset을 추가하고 텍스트를 정리합니다. 그런 다음 Inference Providers를 통해 체크포인트를 호출하는 `model` 노드로 전달됩니다. 마지막으로 post-process `fn` 노드가 생성 파라미터를 PNG의 metadata에 기록합니다. 이후 PNG Info 파이프라인이 이 정보를 다시 읽습니다.

### 고해상도 보정

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/hires-fix.mp4"></video>

Automatic1111의 hi-resolution fix는 먼저 txt2img 출력을 upscale한 다음 두 번째 denoising pass를 실행합니다. 여기서는 대신 두 노드를 거치는 우회 경로를 사용합니다. 텍스트-이미지 결과가 refine 지시문("enhance fine detail and micro-texture, keep the composition identical")과 함께 [FLUX.1-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) `model` 노드로 들어가고, 더 선명하고 크게 변환되어 나옵니다.

### 이미지-이미지

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2img.mp4"></video>

동일한 Kontext 노드가 이미지-이미지 탭의 역할도 합니다. 이미지를 업로드하고 원하는 변경 사항을 설명하면 편집된 이미지를 반환합니다.

### LLM에게 프롬프트 작성시키기

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/prompt-magic.mp4"></video>

"A lighthouse in a storm."처럼 대략적인 프롬프트로 시작합니다. 이 파이프라인은 프롬프트를 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) `model` 노드로 보내고, 작은 `fn` 노드가 응답을 최대 40개의 깔끔한 tag 목록으로 변환합니다. 예를 들면 다음과 같습니다. "stormy sea, wet rocks, dramatic composition, low angle shot, volumetric lighting, ominous tone." 이 출력에 어떤 diffusion model 노드든 연결해 이미지를 렌더링할 수 있습니다.

ComfyUI와 달리 custom node는 필요하지 않습니다. Gradio workflow에서는 LLM과 diffusion model이 모두 같은 캔버스 위의 일반적인 `model` operator입니다.

### 이미지를 다시 프롬프트로 읽어들이기

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/interrogate.mp4"></video>

AUTOMATIC1111의 Interrogate 버튼과 비슷하지만, CLIP 대신 VLM이 interrogate를 수행합니다. [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)가 야시장 사진을 살펴보고 그 사진을 만들어 냈을 법한 프롬프트를 작성합니다. [ViT](https://huggingface.co/google/vit-base-patch16-224) classifier 노드는 같은 이미지를 읽고 다음과 같은 label을 반환합니다. restaurant 51.9%, tobacco shop 15.6%, toyshop 9.1%.

두 노드는 같은 이미지 입력을 사용하므로 `gr.Workflow`가 이들을 병렬로 실행합니다. 따라서 하나를 실행하는 데 걸리는 시간 정도만으로 두 결과를 모두 얻을 수 있습니다.

### 감지 결과를 인페인트 마스크로 변환하기

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/detect-and-mask.mp4"></video>

AUTOMATIC1111에서는 인페인트 마스크를 직접 그려야 합니다. 이 파이프라인은 대신 detector를 사용해 마스크를 생성합니다. [DETR](https://huggingface.co/facebook/detr-resnet-50)가 거리 사진에서 여섯 개의 객체(사람 세 명, 개 한 마리, 자전거 한 대, 자동차 한 대)를 찾고, 이후 workflow가 두 갈래로 나뉩니다. 한쪽은 원본 이미지에 감지된 box를 그리고, 다른 한쪽은 이를 downstream의 inpaint 파이프라인에 넣을 수 있는 mask로 변환합니다.

그리기와 mask 생성은 모두 Pillow와 NumPy를 사용해 로컬에서 수행됩니다. machine 밖으로 나가는 것은 detection 호출뿐입니다.

### 프롬프트 매트릭스

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/prompt-matrix.mp4"></video>

AUTOMATIC1111의 prompt matrix와 비슷한 기능입니다. `fn` 노드가 기본 프롬프트 "a lone oak tree"에 네 가지 suffix(at sunrise, in a thunderstorm, under the Milky Way, in autumn fog)를 결합하고, 각 변형을 별도의 text-to-image 노드로 보냅니다. 마지막 노드는 네 결과를 하나의 contact sheet로 이어 붙입니다.

`gr.Workflow`에는 loop operator가 없으므로 네 개의 text-to-image 노드가 캔버스에 나란히 배치됩니다. 이들은 동일한 dependency depth에 있으므로 병렬로 실행되며, 네 이미지의 생성이 동시에 시작됩니다.

### Upscale 및 배경 제거

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/extras-upscale.mp4"></video>

Automatic1111의 Extras 탭과 비슷한 기능입니다. 두 개의 upscaler 노드가 있으며 서로 다른 경로를 사용합니다. 첫 번째는 `fn` 노드에서 수행하는 로컬 [Lanczos](https://en.wikipedia.org/wiki/Lanczos_resampling) resample로, network call이 필요 없고 Pillow가 resize할 수 있는 속도로 완료됩니다. 두 번째는 [AuraSR ×4](https://huggingface.co/spaces/gokaygokay/AuraSR-v2)이며, 캔버스의 첫 번째 `space` 노드입니다. Hub에서 [Space](https://huggingface.co/spaces/gokaygokay/AuraSR-v2)를 호출하고 그 결과를 다른 노드 출력과 동일하게 처리합니다.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/extras-background.mp4"></video>

배경 제거도 같은 방식으로 작동합니다. [BRIA RMBG-2.0](https://huggingface.co/spaces/briaai/BRIA-RMBG-2.0)는 또 다른 `space` 노드이므로 전체 모델은 자체 Space에 있고, 이 캔버스는 그 Space를 호출하기만 합니다.

### Annotator

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/annotators.mp4"></video>

Canny, line art, sketch, luma-depth, posterize는 일반적으로 Automatic1111의 ControlNet extension에서 사용하는 preprocessor입니다. 여기서는 각각이 순수 NumPy로 작성된 `fn` 노드이며, 뒤에서 실행되는 model은 없습니다. 미리 로드된 건물 facade 예시 사진에서 각 annotator는 CPU로 약 0.5초가 걸립니다.

앱에는 36개의 operator 노드가 있으며, 그중 32개가 `fn` 노드이고, 이 중 22개는 network call 없이 전적으로 in-process로 실행됩니다. 연결이 끊겨도 캔버스의 대략 3분의 2는 계속 작동합니다. 이들은 일반적인 Python 함수이므로 캔버스, server, GPU 없이도 직접 테스트할 수 있습니다.

### PNG Info

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/png-info.mp4"></video>

AUTOMATIC1111은 PNG의 `parameters` text chunk에 생성 세부 정보를 저장하고, PNG Info 탭이 이를 다시 읽습니다. Workflow1111도 동일하게 작동합니다. text-to-image 파이프라인의 post-process 노드가 metadata를 기록하고, 이 파이프라인이 이를 다시 읽어 prompt, negative prompt, steps, CFG, seed, image size, model을 가져옵니다.

### 이미지-동영상

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2video.mp4"></video>

PNG Info가 읽는 image 노드는 [Wan 2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) 노드에도 연결되어 이미지를 애니메이션으로 만듭니다. 데모 예시에서는 잠들어 있던 여우가 깨어나 움직이기 시작합니다. 두 번째 업로드 상자가 필요하지 않은 이유는 하나의 reference 노드가 필요한 만큼 많은 downstream 파이프라인에 입력을 제공할 수 있기 때문입니다. 따라서 한 번 업로드한 이미지의 metadata를 읽고 같은 캔버스에서 애니메이션도 만들 수 있습니다.

## 자신의 GPU에서 모델 실행하기 {#section-2}

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/gradio-workflow1111/img2video-zerogpu.mp4"></video>

지금까지 모든 model call은 Inference Providers나 Space를 통해 다른 사람의 hardware에서 실행되었습니다. 그래서 자신의 GPU가 없어도 Workflow1111 같은 것을 구축하고 실행할 수 있습니다.

`fn` 노드는 그저 Python이므로, model을 로컬에서 로드해 자신의 GPU에서 실행할 수도 있습니다. [FastVideo/fastvideo-fasth3-preview](https://huggingface.co/spaces/FastVideo/fastvideo-fasth3-preview)은 정확히 그렇게 하는 `gr.Workflow` 앱입니다. 이 앱은 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)의 4단계 distillation인 [FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)를 실행하고, ZeroGPU에서 soundtrack이 포함된 video를 생성합니다.

전체 앱은 하나의 bound function으로 요약됩니다.

```python
@spaces.GPU(duration=get_duration, size=GPU_SIZE)
def _generate(prompt_embeds, text_token_tags, height, width, num_frames, seed):
    ...

gr.Workflow(bind={"generate": generate, "status": status}).launch()
```


[ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu)는 function이 필요할 때 GPU를 할당하고 호출이 끝나면 해제합니다. `gr.Workflow`은 이 모든 것을 알 필요가 없습니다. 그저 `fn` 노드를 호출하면 됩니다.

이 방식은 Spaces에만 해당하지도 않습니다. `bind=`를 local checkpoint를 로드하는 function에 연결하고 자신의 machine에서 `.launch()`을 실행하면, Workflow1111 캔버스가 자신의 GPU를 구동할 수 있습니다.

## 모든 출력은 API입니다 {#section-3}

캔버스의 모든 output 노드는 직접 route를 작성하지 않아도 REST endpoint가 됩니다. Workflow1111은 아홉 개의 endpoint를 제공합니다. `/image`, `/edited_image`, `/generated_prompt`, `/recovered_prompt`, `/detected_objects`, `/x_y_grid`, `/upscaled_local`, `/annotator_map`, `/png_info`입니다.

```python
from gradio_client import Client

client = Client("ysharma/Workflow1111", oauth_token="hf_...")

image, params, hires = client.predict(
    "a red fox in a snowy pine forest",  # Prompt
    "",                                  # Negative prompt
    "Cinematic",                         # Style preset
    "enhance fine detail",               # Hires refine instruction
    api_name="/image",
)
```


동일한 endpoint는 [MCP](https://modelcontextprotocol.io) tool이기도 합니다. `mcp_server=True`([guide](https://www.gradio.app/guides/building-mcp-server-with-gradio))로 실행하면 모든 output 노드가 AI assistant가 호출할 수 있는 tool로 표시됩니다. Claude Code, Cursor 또는 MCP client를 server URL에 연결하세요.

```json
{
  "mcpServers": {
    "workflow1111": {
      "url": "https://ysharma-workflow1111.hf.space/gradio_api/mcp/",
      "headers": { "X-HF-Token": "hf_..." }
    }
  }
}
```


이제 agent는 별도의 glue code 없이 더 큰 task의 단계로 이미지를 생성하고, prompt를 다시 읽거나, detection을 실행할 수 있습니다. 각 caller는 `X-HF-Token` header에 자신의 token을 보내므로 Space에는 자체 token이 저장되지 않습니다.

## ComfyUI와 비교하면 {#section-4}

AUTOMATIC1111이 기능 목록을 제공했지만, 실제로 Gradio Workflow와 비교되는 도구는 ComfyUI입니다. 둘 다 node graph이기 때문입니다. 사람들이 구축하고 배포하려는 작업 중 상당수는 `gr.Workflow`로 동일하게 처리할 수 있습니다.

* **node가 자신이 소유하지 않은 hardware일 수 있습니다.** [Inference Providers](https://huggingface.co/docs/inference-providers/index)를 통해 실행하거나, Hub의 어떤 Space나 API든 호출하거나, 데이터셋에서 가져올 수 있습니다. Workflow1111이 자체 GPU 없이 실행되는 이유입니다.
* **모든 output이 typed REST endpoint가 됩니다.** endpoint는 graph에서 생성됩니다.
* **방문자는 자신의 identity로 workflow를 실행할 수 있습니다.** [OAuth](https://huggingface.co/docs/hub/spaces-oauth)을 켜고 public URL을 공유하면 누구나 로그인해 아무것도 설치하지 않고 앱을 사용할 수 있습니다.
* **같은 캔버스에서 model과 modality를 조합할 수 있습니다.** diffusion model, LLM, VLM, detector, video model을 모두 하나의 workflow에 포함할 수 있습니다.
* **custom 기능이 필요하신가요? function을 작성하세요.** custom node는 Python function이므로 Python으로 할 수 있는 모든 작업을 수행할 수 있습니다.

그 결과 사람들은 browser에서 열고, 로그인하고, 즉시 사용하며, code에서 호출할 수 있는 multi-model pipeline을 얻게 됩니다.

## 직접 구축하기 {#section-5}

Workflow1111에는 73개의 node가 있지만, 시작은 다음과 같이 간단했습니다.

```python
import gradio as gr

def your_function(text: str) -> str:
    pass

gr.Workflow(bind=[your_function]).launch()
```


`bind=`는 function을 node로 바꾸고, `edges=`는 node를 연결하며, `.launch()`는 browser에서 canvas를 열어 계속 편집할 수 있게 합니다. 준비가 되면 `gradio deploy`가 전체를 Space에 올립니다. [gr.Workflow guide](https://gradio.app/guides/workflows)에는 JSON schema와 모든 operator type을 포함한 전체 세부 정보가 있습니다.

이미 작동하는 것에서 시작하고 싶다면 [Workflow1111](https://huggingface.co/spaces/ysharma/Workflow1111)을 열고 **Duplicate**를 누른 다음, 변경할 11개 pipeline 중 하나를 선택하세요. node를 삭제하고, model을 바꾸고, flow를 다시 연결할 수 있습니다. 더 작은 규모로 시작하고 싶다면 [previous post](https://huggingface.co/blog/gradio-workflow-guide)에 각각 약 1분 만에 실행할 수 있는 다섯 개의 workflow가 있습니다.

무엇을 만들든 X에 게시하고 [@gradio](https://x.com/Gradio)을 tag해 주세요. 여러분의 workflow를 널리 알리는 데 기꺼이 도움을 드리겠습니다.
