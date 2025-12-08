---
layout: post
title: "Python으로 구현하는 MCP 서버: Gradio를 활용한 AI 쇼핑 어시스턴트"
author: sohyun
categories: [MCP, Gradio]
image: assets/images/blog/posts/2025-09-14-Implementing-MCP-Servers-in-Python/thumbnail.png
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Implementing MCP Servers in Python: An AI Shopping Assistant with Gradio](https://huggingface.co/blog/gradio-vton-mcp)를 한국어로 번역한 글입니다._

---


# Python으로 구현하는 MCP 서버: Gradio를 활용한 AI 쇼핑 어시스턴트

Python 개발자 여러분, LLM에 특별한 능력을 부여하고 싶으신가요? 그렇다면 Gradio가 가장 빠른 방법입니다! Gradio의 MCP(Model Context Protocol) 연동을 이용하면 LLM을 Hugging Face [Hub](https://hf.co)에 호스팅된 수천 개의 AI 모델과 Space에 직접 연결할 수 있습니다. LLM의 일반적인 추론 능력과 Hugging Face의 모델들의 특화된 능력을 결합한다면, LLM은 단순히 텍스트 질문에 답하는 것을 넘어 일상생활의 문제를 해결해줄 것 입니다!

Gradio가 제공하는 다음과 같은 기능 덕분에 Python 개발자들이 강력한 MCP 서버를 매우 쉽게 구현할 수 있습니다:
* **Python 함수를 LLM 도구로 자동 변환:** Gradio 앱의 각 API 엔드포인트는 해당하는 이름, 설명, 입력 스키마를 가진 MCP 도구로 자동 변환됩니다. 함수의 docstring은 도구와 전달인자의 설명을 생성하는 데 사용됩니다.
* **실시간 진행 상황 알림:** Gradio는 MCP 클라이언트에 진행 상황 알림을 스트리밍하기 때문에, 직접 구현하지 않고도 실시간으로 상태를 모니터링할 수 있습니다.
* 공개 URL 지원 및 다양한 유형의 파일 처리를 포함한 **자동 파일 업로드**

이런 상황을 상상해봅시다. 쇼핑은 시간이 너무 많이 걸려서 싫고, 직접 옷을 입어보는 것도 귀찮습니다. 이때 LLM이 쇼핑을 대신해준다면 어떨까요? 이 포스트에서는 온라인 의류 매장을 탐색하고, 특정 옷을 찾고, 가상 피팅 모델을 사용해 여러분이 그 옷을 입을 때 어떨지 보여주는 LLM 기반 AI 어시스턴트를 만들어보겠습니다. 아래 데모를 확인해보세요:

<video src="https://github.com/user-attachments/assets/e5bc58b9-ca97-418f-b78b-ce38d4bb527e" controls alt="AI Shopping Assistant Demo using Gradio python sdk and MCP"></video>

## 목표: 나만의 개인 AI 스타일리스트

AI 쇼핑 어시스턴트를 구현하기 위해 세 가지 핵심 구성 요소를 결합할 것입니다:

1. [IDM-VTON](https://huggingface.co/yisol/IDM-VTON) Diffusion 모델: 이 AI 모델은 가상 피팅 기능을 담당합니다. 원본 사진을 편집하여 사진 속 사람이 다른 옷을 입고 있는 것처럼 보이게 할 수 있습니다. 우리는 [여기](https://huggingface.co/spaces/yisol/IDM-VTON)에서 접근 가능한 IDM-VTON의 Hugging Face Space를 사용할 것입니다.

2. Gradio: Gradio는 AI 기반 웹 애플리케이션을 쉽게 구축할 수 있게 해주는 오픈소스 Python 라이브러리로, 우리 프로젝트에서 중요한 부분은 MCP 서버를 생성할 수 있다는 점입니다. Gradio는 LLM이 IDM-VTON 모델과 다른 도구들을 호출할 수 있도록 다리 역할을 합니다.

3. Visual Studio Code의 AI Chat 기능: 임의의 MCP 서버를 추가할 수 있는 VS Code의 내장 AI 채팅을 사용하여 AI 쇼핑 어시스턴트와 대화할 것입니다. 이는 명령을 내리고 가상 피팅 결과를 보기 위한 사용자 친화적인 인터페이스를 제공합니다.

## Gradio MCP 서버 구축하기
AI 쇼핑 어시스턴트의 핵심은 Gradio MCP 서버입니다. 이 서버는 한 가지 주요 도구를 제공합니다:

1. `vton_generation`: 이 함수는 사람 모델 이미지와 의류 이미지를 입력으로 받아 IDM-VTON 모델을 사용하여 그 사람이 해당 옷을 입고 있는 이미지를 생성합니다.


다음은 Gradio MCP 서버를 위한 Python 코드입니다:

```python
from gradio_client import Client, handle_file
import gradio as gr
import re


client = Client("freddyaboulton/IDM-VTON",
                hf_token="<Your-token>")


def vton_generation(human_model_img: str, garment: str):
    """IDM-VTON 모델을 사용하여 옷을 입고 있는 사람의 이미지를 생성합니다."""
    """
    Args:
        human_model_img: 옷을 입어볼 사람 모델
        garment: 입을 의상
    """
    output = client.predict(
        dict={"background": handle_file(human_model_img), "layers":[], "composite":None},
        garm_img=handle_file(garment),
        garment_des="",
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )

    return output[0]

vton_mcp = gr.Interface(
    vton_generation,
    inputs=[
        gr.Image(type="filepath", label="Human Model Image URL"),
        gr.Image(type="filepath", label="Garment Image URL or File")
    ],
    outputs=gr.Image(type="filepath", label="Generated Image")
)

if __name__ == "__main__":
    vton_mcp.launch(mcp_server=True)
```

`launch()` 메서드에서 `mcp_server=True`로 지정해주면, Gradio가 자동으로 Python 함수를 LLM이 이해하고 활용할 수 있는 MCP 도구로 변환합니다. 함수의 docstring은 도구와 매개변수의 설명을 생성하는 데 사용됩니다.

> [!TIP]
> 원래 IDM-VTON space는 자동 MCP 기능이 있기 전인 Gradio 4.x로 구현되었습니다. 따라서 이 데모에서는 Gradio API 클라이언트를 통해 원래 space를 호출하는 Gradio 인터페이스를 구축할 예정입니다.

마지막으로 이 스크립트를 python으로 실행합니다.

## VS Code 설정하기

Gradio MCP 서버를 VS Code의 AI 채팅에 연결하려면 `mcp.json` 파일을 편집해야 합니다. 파일 내부 설정값을 통해 AI 채팅에 MCP 서버 위치와 상호작용 방법을 알려줍니다.

명령 패널에 `MCP`를 입력하고 `MCP: Open User Configuration`을 선택하여 설정 파일을 찾을 수 있습니다. 파일을 열고 다음 서버들이 있는지 확인하세요:

```json
{
  "servers": {
  "vton": {
    "url": "http://127.0.0.1:7860/gradio_api/mcp/"
  },
  "playwright": {
    "command": "npx",
    "args": [
      "-y",
      "@playwright/mcp@latest"
    ]
   }
  }
}
```

playwright MCP 서버는 AI 어시스턴트가 웹을 탐색할 수 있게 해줍니다.

> [!TIP]
> `vton` 서버의 URL이 이전 섹션에서 콘솔에 출력된 URL과 일치하는지 확인해보세요. playwright MCP 서버를 실행하려면 node가 [설치](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)되어 있어야 합니다.

## 실제로 활용해보기

이제 AI 쇼핑 어시스턴트와 대화할 수 있게 되었습니다. VS Code에서 새 채팅을 열어서 어시스턴트에게 다음과 같이 요청해보세요. 
"유니클로 웹사이트에서 파란색 티셔츠를 찾아보고, [your-image-url]에 있는 내 사진을 사용해서 그 중 세 개를 내가 입었을 때 어떻게 보이는지 보여줘."

예시는 위의 비디오를 참조하세요!

## 결론

Gradio, MCP, 그리고 IDM-VTON과 같은 강력한 AI 모델의 조합으로, 똑똑하고 유용한 AI 어시스턴트를 만들 수 있다는 가능성이 열렸습니다. 이 블로그 포스트에 설명된 대로 따라가보면 여러분이 가장 관심 있는 문제를 해결하기 위한 자신만의 어시스턴트를 구축할 수 있습니다!