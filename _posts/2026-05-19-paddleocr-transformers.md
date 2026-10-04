---
layout: post
title: "PaddleOCR 3.5: Transformers 백엔드를 활용한 OCR 및 문서 파싱 작업 실행"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-05-19-paddleocr-transformers/thumbnail.jpeg
slug: "paddleocr-transformers"
source_url: "https://huggingface.co/blog/PaddlePaddle/paddleocr-transformers"
source_published_date: "2026-05-18"
source_published_at: "2026-05-18T15:12:46+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [PaddleOCR 3.5: Running OCR and Document Parsing Tasks with a Transformers Backend](https://huggingface.co/blog/PaddlePaddle/paddleocr-transformers)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/PaddlePaddle/paddleocr-transformers -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# PaddleOCR 3.5: Transformers 백엔드를 활용한 OCR 및 문서 파싱 작업 실행

PaddleOCR 3.5은 OCR 및 문서 파싱 작업을 Hugging Face 생태계에 더 가깝게 가져옵니다. 이 릴리스로, 지원되는 PaddleOCR 모델은 다음과 같이 추론 백엔드로 Hugging Face Transformers를 설정해 실행할 수 있습니다:

```
engine=
"transformers"
```

PaddleOCR은 PP-OCRv5와 같은 OCR 모델 시리즈와 PaddleOCR-VL 1.5와 같은 문서 파싱 모델 시리즈를 계속 제공하는 한편, Transformers는 이들을 실행하기 위한 지원 백엔드 중 하나가 됩니다.

Hugging Face Spaces에서 라이브 데모를 시도해 보세요:
[https://huggingface.co/spaces/PaddlePaddle/paddleocr-3.5-transformers-demo](https://huggingface.co/spaces/PaddlePaddle/paddleocr-3.5-transformers-demo)

## What changed?

PaddleOCR 3.5는 더 유연한 추론 엔진 인터페이스를 도입합니다. 개발자는 `engine` 파라미터를 통해 백엔드를 선택하고, `engine_config`를 통해 백엔드별 옵션을 전달할 수 있습니다.

실제로 이는 다음을 의미합니다:

- 이 작업들의 파이프라인은 PaddleOCR가 관리하므로, 개발자가 내부 구성 요소를 일일이 호출할 필요가 없습니다.

- Transformers가 실행 가능한 추론 백엔드 중 하나가 되어, 지원되는 PaddleOCR 모델을 실행할 수 있습니다.

- 개발자는 `engine_config`를 통해 `dtype`, 디바이스 배치, 어텐션 구현 등 백엔드 관련 옵션을 구성할 수 있습니다.

스택을 이해하는 간단한 방법:

이 릴리스는 주로 추론 백엔드 계층에 관한 것입니다. PaddleOCR은 OCR 및 문서 파싱 기능을 계속 제공하고, Transformers는 Hugging Face 중심 환경에 자연스럽게 맞는 또 다른 백엔드 옵션을 제공합니다. 더 큰 Document AI 워크플로우는 여전히 개발자와 애플리케이션 빌더의 손에 있습니다.

## 왜 이것이 중요합니까

RAG, Document AI, 및 문서 에이전트 애플리케이션의 경우, 어려운 부분은 종종 LLM 이전에 시작됩니다.

개발자는 먼저 PDF, 스캔 문서, 스크린샷, 표, 차트, 수식, 그리고 복잡한 페이지 레이아웃을 신뢰할 수 있는 구조화 데이터로 변환해야 합니다. 이 수집 단계가 약하면, 다운스트림 LLM 워크플로우가 핵심 정보를 놓치거나 잘못된 컨텍스트를 검색하거나 신뢰할 수 없는 답변을 생성할 수 있습니다.

PaddleOCR은 PP-OCRv5와 같은 OCR 시리즈 모델과 PaddleOCR-VL-1.5와 같은 문서 파싱 시리즈 모델을 제공함으로써 이 문서 수집 문제를 해결하는 데 도움을 줍니다.

PaddleOCR 3.5를 통해 이 기능들은 이제 Transformer 중심의 스택과 더 쉽게 연결될 수 있습니다. 지원되는 PaddleOCR 모델은 Transformers 백엔드로 실행될 수 있고, PaddleOCR은 뒤에서 OCR 또는 문서 파싱 파이프라인을 계속 관리합니다.

개발자에게 이는 통합 마찰이 줄고, 문서에서 다운스트림 RAG, 에이전트, 검색, 분석, 자동화 워크플로우로 이어지는 경로가 더 자연스러워짐을 의미합니다.

## Quick start

PaddleOCR 3.5, PaddleX, Transformers 및 하드웨어에 맞는 호환 PyTorch 빌드를 설치합니다.

예를 들어 CUDA 12.6 환경에서:

```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install 
"paddleocr==3.5.0"
 
"paddlex==3.5.2"
 
"transformers>=5.4.0"
```

CPU, ROCm, 또는 기타 환경의 경우 대상 하드웨어에 맞는 PyTorch 빌드를 설치합니다.

다음과 같이 명령줄에서 실행합니다:

```
paddleocr ocr \
  -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png \
  --device gpu:0 \
  --engine transformers
```

또는 Python API를 사용합니다:

```
from
 paddleocr 
import
 PaddleOCR

pipeline = PaddleOCR(
    device=
"gpu:0"
,
    engine=
"transformers"
,
    use_doc_orientation_classify=
False
,
    use_doc_unwarping=
False
,
    use_textline_orientation=
False
,
    engine_config={
        
"dtype"
: 
"float32"
,
    },
)

results = pipeline.predict(
    
"https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"

)


for
 result 
in
 results:
    
print
(result)
```

Hugging Face Space는 broad compatibility를 위해 `float32`를 사용합니다. 자체 하드웨어의 경우, 백엔드별 옵션을 `engine_config`를 통해 조정할 수 있습니다:

```
engine_config = {
    
"dtype"
: 
"bfloat16"
,
    
"device_type"
: 
"gpu"
,
    
"device_id"
: 
0
,
    
"attn_implementation"
: 
"sdpa"
,
}
```

최적의 구성은 모델, 하드웨어, 배포 환경에 따라 다릅니다.

## 언제 Transformers 백엔드를 사용해야 하나요?

PaddleOCR의 OCR 및 문서 파싱 기능이 Hugging Face 중심 스택에 더 자연스럽게 맞도록 하고 싶을 때 Transformers 백엔드를 사용합니다.

이는 특히 RAG, Document AI, 검색, 분석, 또는 에이전트 애플리케이션을 구축 중이며 모델 로딩, 실험, 배포, 또는 모델 아카이브 관리에 이미 PyTorch / Transformers 인프라를 의존하고 있을 때 유용합니다.

Transformers 백엔드는 다음과 같이 원할 때 적합합니다:

- Transformers를 이미 사용하는 팀에게 더 친숙한 개발 경험,

- 지원되는 PaddleOCR 모델의 Hub 호환 모델 검색 및 배포,

- 기존 PyTorch / Transformers 서비스와의 통합 용이성.

OCR 또는 문서 파싱 처리량을 최대화하는 것이 우선일 때는 PaddleOCR의 기본 `paddle_static` 백엔드가 일반적으로 권장되는 선택입니다.

이 릴리스는 한 백엔드를 다른 백엔드로 교체하는 것에 관한 것이 아니라, 개발자에게 더 많은 유연성을 제공하는 데 관한 것입니다: OCR 및 문서 파싱 기능은 PaddleOCR로 제공하고, 인퍼런스 백엔드는 스택에 가장 잘 맞는 것을 선택하면 됩니다.

## 지금 바로 사용해 보세요

PaddleOCR 3.5 Transformer 데모를 Hugging Face Spaces에서 체험해 보세요:

[https://huggingface.co/spaces/PaddlePaddle/paddleocr-3.5-transformers-demo](https://huggingface.co/spaces/PaddlePaddle/paddleocr-3.5-transformers-demo)

Hub에서 PaddleOCR 모델을 둘러보세요:

[https://huggingface.co/PaddlePaddle/models](https://huggingface.co/PaddlePaddle/models)

PaddleOCR 3.5는 OCR 및 문서 파싱 기능을 Transformers 중심 워크플로우에 더 가까이 가져오면서, 개발자에게 더 큰 Document AI 애플리케이션을 구축할 자유를 제공합니다.

## 리소스

- PaddleOCR 문서: [https://www.paddleocr.ai/](https://www.paddleocr.ai/)

- PaddleOCR GitHub: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

- Hugging Face의 PaddlePaddle 조직: [https://huggingface.co/PaddlePaddle](https://huggingface.co/PaddlePaddle)

- Spaces의 PaddleOCR 3.5 Transformers 데모: [https://huggingface.co/spaces/PaddlePaddle/paddleocr-3.5-transformers-demo](https://huggingface.co/spaces/PaddlePaddle/paddleocr-3.5-transformers-demo)

## Acknowledgements

PaddleOCR 3.5 Transformers 통합을 지원해 주신 Hugging Face 엔지니어들에게 진심으로 감사합니다.

관련 PR의 검토 및 병합을 포함한 끝에서 끝까지의 참여를 해주신 Anton Vlasjuk 에게 특별한 감사를 드립니다.

또한 가치 있는 PR 리뷰와 피드백을 주신 Raushan Turganbay 및 Yoni Gozlan께도 감사드립니다.

그들의 안내가 Hugging Face 커뮤니티의 통합 품질, 문서화, 개발자 경험을 개선하는 데 도움이 되었습니다.
