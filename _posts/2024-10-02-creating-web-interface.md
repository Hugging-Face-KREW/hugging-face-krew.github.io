---
layout: post
title: "Gradio 기반 NL2SQL 챗봇 만들기 - Part 2: 웹 인터페이스 만들기"
author: chaewon
categories: [NLP, Web]
image: assets/images/blog/posts/2024-09-19-creating-gradio-based-NL2SQL-chatbot/gradio_free_1160.png
---
* TOC
{:toc}
<!--toc-->
안녕하세요🤗 오늘은 Hugging Face에서 개발한 Gradio 라이브러리를 사용하여 머신러닝 모델을 위한 웹 인터페이스를 쉽고 빠르게 만드는 방법을 알아보겠습니다. Gradio를 사용하면 복잡한 웹 개발 지식 없이도 데모나 인터페이스를 빠르게 제작할 수 있습니다.

## <a href="#introduction">1. Gradio 소개</a><a id="introduction"></a>

Gradio는 머신러닝 모델이나 Python 함수를 위한 웹 기반 GUI를 몇 줄의 코드로 만들 수 있게 해주는 강력한 도구입니다. 주요 특징은 다음과 같아요.

- 다양한 입력/출력 컴포넌트 지원 (텍스트, 이미지, 오디오 등)
- 간단한 Python API
- 로컬 호스팅 및 클라우드 배포 옵션
- 다양한 머신러닝 프레임워크와의 호환성

## <a href="#basic-usage">2. 기본 사용법: Hello World 예제</a><a id="basic-usage"></a>

먼저, 가장 기본적인 Gradio 애플리케이션을 만들어 보겠습니다. 이 예제에서는 사용자의 이름을 입력받아 인사말을 반환하는 간단한 함수를 만들고, 이를 웹 인터페이스로 구현해볼 것입니다.

```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

if __name__ == "__main__":
    demo.launch()
```

이 코드를 실행하면, 로컬 서버가 시작되고 웹 브라우저에서 인터페이스를 볼 수 있습니다. 텍스트 상자에 이름을 입력하고 'Submit' 버튼을 클릭하면, 인사말이 출력됩니다.

## <a href="#interface-class">3. Gradio Interface 클래스 상세 설명</a><a id="interface-class"></a>

Gradio의 핵심은 `Interface` 클래스입니다. 이 클래스를 사용하여 함수와 입력/출력 컴포넌트를 연결하고 웹 인터페이스를 생성합니다.

### 주요 매개변수

- `fn`: 인터페이스로 감싸고자 하는 함수
- `inputs`: 입력 컴포넌트 (예: "textbox", "image" 등)
- `outputs`: 출력 컴포넌트
- `examples`: 예제 입력값 리스트
- `title`: 인터페이스 제목
- `description`: 인터페이스 설명

## <a href="#advanced-example">4. 고급 예제: 이미지 분류기</a><a id="advanced-example"></a>

이번에는 조금 더 복잡한 예제로, 이미지 분류기를 위한 인터페이스를 만들어 보겠습니다.

```python
import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

# 사전 훈련된 ResNet50 모델 로드
model = models.resnet50(pretrained=True)
model.eval()

# 이미지 전처리를 위한 변환 정의
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ImageNet 클래스 레이블 로드
with open('imagenet_classes.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

def image_classifier(img):
    # 이미지 전처리
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    # 추론
    with torch.no_grad():
        output = model(input_batch)

    # 결과 처리
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # 결과를 사전 형식으로 변환
    results = {}
    for i in range(top5_prob.size(0)):
        results[labels[top5_catid[i]]] = float(top5_prob[i])

    return results

demo = gr.Interface(
    fn=image_classifier,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="이미지 분류기",
    description="이미지를 업로드하면 상위 5개의 예측 클래스와 확률을 보여줍니다."
)

if __name__ == "__main__":
    demo.launch()
```

이 예제에서는 이미지를 입력으로 받아 상위 5개의 예측 클래스와 확률 출력하는 인터페이스를 만들었습니다.

## <a href="#launching">5. 인터페이스 실행하기</a><a id="launching"></a>

`launch()` 메서드를 사용하여 인터페이스를 실행할 수 있습니다. 다양한 옵션을 제공하여 실행 방식을 커스터마이즈할 수 있습니다.

```python
demo.launch(
    share=True,  # 공유 가능한 링크 생성
    auth=("username", "password"),  # 기본 인증 설정
    server_port=7860  # 서버 포트 지정
)
```

## <a href="#queue-system">6. 큐 시스템 활용하기</a><a id="queue-system"></a>

Gradio는 큐 시스템을 제공하여 동시 요청을 관리하고 사용자 경험을 향상시킬 수 있습니다.

```python
demo = gr.Interface(image_classifier, gr.Image(), gr.Label())
demo.queue(max_size=50)  # 최대 50개의 요청을 큐에 저장
demo.launch()
```

이렇게 하면 동시에 들어오는 요청을 효과적으로 관리하고, 사용자에게 대기 시간을 알려줄 수 있습니다.

## <a href="#conclusion">결론</a><a id="conclusion"></a>

Gradio를 사용하면 복잡한 머신러닝 모델도 쉽게 웹 인터페이스로 만들 수 있습니다. 이 튜토리얼에서는 기본적인 사용법부터 고급 기능까지 살펴보았습니다. Gradio의 다양한 컴포넌트와 옵션을 활용하여 여러분의 프로젝트에 맞는 인터페이스를 만들어보세요!

다음 포스팅에서는 이러한 Gradio의 기능을 활용하여 Text-to-SQL 챗봇을 구현해볼 예정입니다. 이 챗봇은 다음과 같은 특징을 가질 예정이에요.

- 자연어 질문을 SQL 쿼리로 변환하는 NLP 모델 통합
- 데이터베이스 스키마 정보를 컨텍스트로 활용
- 생성된 SQL 쿼리와 실행 결과를 동시에 표시하는 다중 출력 인터페이스
- 대화 기록을 유지하는 채팅 인터페이스 구현

Gradio의 다양한 컴포넌트와 옵션을 활용하면, 이러한 복잡한 기능도 비교적 적은 코드로 구현할 수 있습니다. 여러분의 프로젝트에 맞는 고유한 인터페이스를 만들어보세요!

## <a href="#references">참고 자료</a><a id="references"></a>

- [Gradio 공식 문서](https://www.gradio.app/docs/)
- [Gradio GitHub 저장소](https://github.com/gradio-app/gradio)