---
layout: post
title: "n8n으로 번역 MCP Server 워크플로우 구축하기: 실전 트러블슈팅과 실행 방식 비교" 
author: minju
categories: [Robotics, AI, Community]
image: assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/meme.jpg
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 문서 번역 프로젝트의 MCP Server를 활용한 n8n 워크플로우 구축 경험을 담은 글입니다._

---

## 들어가며

MCP(Model Context Protocol) Server를 활용한 번역 워크플로우를 구축하면서, Claude Desktop Client로 직접 실행하는 방식과 n8n을 통한 워크플로우 자동화 방식을 모두 경험해보았습니다. 이 글에서는 n8n 워크플로우 구축 과정에서 마주친 실질적인 문제들과 해결 방법, 그리고 두 실행 방식의 장단점을 정리해보겠습니다.

## n8n 번역 MCP Server 워크플로우 소개

이번에 구축한 워크플로우는 Gradio 기반의 번역 MCP Server를 n8n과 연동하여, 텍스트를 자동으로 번역하고 PR을 등록하는 시스템입니다. 

**주요 구성 요소:**
- **Gradio MCP Server**: 번역 기능을 MCP Tool로 제공
- **n8n Workflow**: AI Agent를 통해 번역 작업을 오케스트레이션
- **GitHub Integration**: 번역 결과를 자동으로 문서화하고 PR 업로드

**기본 워크플로우:**
1. 번역할 텍스트 검색
2. n8n AI Agent가 MCP Server의 번역 Tool 호출
3. 번역 결과를 GitHub PR로 자동 저장

## n8n 워크플로우 구축 결과

초안으로 구축한 n8n 워크플로우를 먼저 보여드리겠습니다.

### 번역 워크플로우
AI Agent와 AWS Bedrock Chat Model을 사용하여 Claude 모델을 연결하였습니다. 이 AI Agent는 번역 문서 검색 MCP Server와 번역 MCP Server가 Tool로 연결되어 있습니다. 따라서 번역할 문서를 검색하고 번역 결과물을 Notion 페이지로 저장하는 워크플로우입니다. 

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/translate.png" 
         width="500"
         alt="번역 워크플로우 구조"/>
    <p><em>번역 문서 검색 및 Notion 저장 워크플로우</em></p>
</div>

### 번역 워크플로우 결과물
초기에 Notion을 사용했던 이유는, Notion에서 번역 결과물이 저장된 이후, 사람이 편집하기 쉬운 UI를 가지고 있기 때문이었습니다. 이후에 Notion Tool을 사용하지 않게 되었는데, 트러블슈팅 섹션에서 이유를 말씀드리겠습니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/translation_result.png" 
         width="500"
         alt="Notion에 저장된 번역 결과"/>
    <p><em>Notion 페이지에 저장된 번역 결과물</em></p>
</div>

### GitHub PR 워크플로우
AI Agent와 Notion Tool 및 PR 업로드 MCP Server가 연결되어 있습니다. 앞 단계인 번역 워크플로우가 실행된 이후, Notion 데이터베이스에서 번역 결과물을 받아와서 PR로 업로드를 자동화해줍니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/pr_upload.png" 
         width="500"
         alt="GitHub PR 업로드 워크플로우"/>
    <p><em>번역 결과를 GitHub PR로 업로드하는 워크플로우</em></p>
</div>

### PR 결과물
PR 워크플로우로 생성한 PR 결과물입니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/pr_result.png" 
         width="500"
         alt="생성된 GitHub PR"/>
    <p><em>자동으로 생성된 번역 PR</em></p>
</div>

### PR Review 워크플로우
PR URL을 AI Agent에게 알려주면, PR Review 코멘트를 남겨주는 MCP Server를 실행합니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/review.png" 
         width="500"
         alt="PR 리뷰 워크플로우"/>
    <p><em>자동 PR 리뷰 워크플로우</em></p>
</div>

## 실행 방식 비교: AI Chat Client vs n8n

워크플로우 구축 과정에서 마주친 여러 문제들을 이야기하기 전에, 먼저 n8n을 사용하게 된 배경을 얘기해보려고 합니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/meme.jpg" 
         width="500"
         alt="AI 자동화에 대한 기대와 현실을 보여주는 밈"/>
    <p><em>AI 자동화에 대한 기대와 현실</em></p>
</div>

인터넷에서 밈처럼 돌아다니는 이 사진을 본 적이 있으실까요? 저는 MCP Server를 개발하면서 이 사진에 공감하게 되었습니다.

번역 MCP Server를 구현하고 나서, Hugging Face Chat, Claude Desktop, Gemini CLI 등과 같은 AI Chat Client에 직접 MCP Server를 연결하고 테스팅을 해보았습니다. 이 과정에서 사람이 직접 Client와 소통하고 참여해야만 이 MCP Server들이 사용된다고 느껴졌습니다. 사람의 손을 타지 않고 이러한 워크플로우가 돌아간다면 MCP Server의 사용성이 더 높아질 거라 기대되었습니다. 

따라서 사람 대신에 워크플로우를 돌릴 수 있게 MCP Server Orchestration Tool을 찾아다니다가, 직접 MCP Client를 개발할 필요가 없는 노코드형 툴인 n8n을 선택하게 되었습니다. 

### AI Chat Client 직접 실행 (Hugging Face Chat, Claude Desktop, Gemini CLI)

**장점:**

1. **빠른 프로토타이핑**: MCP Server 설정 파일만 수정하면 즉시 테스트 가능
2. **자연스러운 대화형 인터페이스**: 복잡한 요청도 자연어로 표현
3. **컨텍스트 유지**: 대화 흐름 속에서 이전 결과를 자연스럽게 참조
4. **높은 번역 품질**: Claude의 Full Context를 활용한 일관된 번역
5. **디버깅 용이**: 실시간으로 Claude의 사고 과정 확인 가능

**단점:**

1. **수동 작업 필요**: 매번 사용자가 명령을 입력해야 함
2. **자동화 불가**: 반복 작업이나 스케줄링 불가능
3. **확장성 제한**: 대량 처리나 병렬 처리 어려움

**적합한 사용 사례:**
- 일회성 번역 작업
- 복잡한 맥락이 필요한 번역
- 빠른 실험과 테스트
- 대화형 번역 검토 및 수정

### n8n 워크플로우 실행

**장점:**

1. **완전 자동화**: 트리거만 설정하면 자동 실행
2. **확장성**: 대량의 문서를 병렬로 처리 가능
3. **통합성**: Notion, Slack, GitHub 등 다양한 서비스와 연동
4. **재사용성**: 한 번 구축한 워크플로우를 반복 사용
5. **스케줄링**: 정해진 시간에 자동 실행

**단점:**

1. **초기 구축 비용**: 워크플로우 설계와 구현에 시간 소요
2. **제한된 유연성**: 정해진 로직 외의 요청 처리 어려움
3. **복잡한 디버깅**: 문제 발생 시 여러 노드를 확인해야 함
4. **Max Iteration 제약**: 복잡한 작업은 분할 필요
5. **Agent 지능 한계**: 예상치 못한 상황에 유연하게 대응하도록 사람이 도와줄 수 없음

**적합한 사용 사례:**
- 정기적인 대량 번역 작업
- 다단계 자동화 프로세스
- 번역 결과의 자동 배포 및 알림

### 하이브리드 접근법

실제로는 두 방식을 상황에 맞게 조합하여 사용하였습니다. 개발 초창기에는 MCP Server의 테스트를 위해 AI Chat Client들을 사용하였고, 이후 n8n으로 자동화를 도입했습니다.

**1단계: Desktop Client로 프로토타입**
- 새로운 번역 요구사항 테스트
- 최적의 프롬프트와 파라미터 발견
- 번역 품질 확인

**2단계: n8n으로 자동화**
- 검증된 프로세스를 워크플로우로 구현
- 반복 작업 자동화
- 모니터링 및 개선

**3단계: 피드백 루프**
- n8n 실행 결과 분석
- 문제 발생 시 Desktop Client로 재검증
- 워크플로우 개선 및 업데이트

이제 이러한 배경을 이해한 상태에서, Gradio MCP Server를 활용한 n8n 워크플로우 구축 과정에서 실제로 마주친 문제들을 살펴보겠습니다.

## 트러블슈팅: 실전에서 마주친 문제들

### 1. Gradio MCP Server는 선택적 노출이 불가능하다

**문제 상황:**
필요하지 않은 Tool이 등록되면 Agent에게 혼란을 줄 수 있기 때문에, 처음에는 `app.py`에 정의된 여러 함수 중 특정 함수만 MCP Tool로 노출하려고 했습니다. 하지만 Gradio는 그런 방식으로 동작하지 않았습니다.

**원인:**
Gradio 공식 문서를 확인해보니, `demo.launch(mcp_server=True)` 옵션을 활성화하면 **앱에 정의된 모든 인터페이스 함수가 자동으로 MCP Tool로 변환**됩니다. 함수 단위로 선택적으로 제외하는 설정은 공식적으로 제공되지 않습니다.

**해결 방법:**
- MCP Server로 노출하고 싶은 함수만 별도의 `app.py` 파일에 분리
- 또는 노출하고 싶지 않은 함수는 내부 헬퍼 함수로 구조 변경
- 필요한 경우 여러 개의 Gradio 앱을 만들어 각각 다른 포트로 실행

```python
# 잘못된 접근 (특정 함수만 노출 불가)
# app.py
def translate(text): ...
def summarize(text): ...  # 이것도 함께 노출됨

demo.launch(mcp_server=True)

# 올바른 접근 (필요한 함수만 별도 파일로 분리)
# app.py
def translate(text): ...
demo.launch(mcp_server=True)  # translate만 노출됨
```

### 2. n8n Tool Call ID로 인한 Schema Checking 실패

**문제 상황:**
n8n에서 MCP Tool을 호출할 때 스키마 검증 에러가 발생했습니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/toolcallid.png" 
         width="500"
         alt="toolCallId 에러 화면"/>
    <p><em>toolCallId 파라미터로 인한 Schema 검증 실패</em></p>
</div>

**원인:**
[Gradio Python Client](https://github.com/gradio-app/gradio/blob/main/client/python/README.md)는 입력 파라미터의 타입과 구조를 엄격하게 검증합니다. 문제는 n8n에서 AI Agent가 Tool을 호출할 때마다 고유한 `toolCallId`가 자동으로 생성되어 파라미터에 주입된다는 것이었습니다. Gradio는 정의되지 않은 이 파라미터를 거부하면서 Schema 검증이 실패했습니다. 이는 현재 n8n의 알려진 이슈([n8n-io/n8n#21716](https://github.com/n8n-io/n8n/issues/21716))와 동일했습니다.

**해결 방법:**
- 모든 Tool 정의에 `toolCallId: str = None` 파라미터를 추가하여 n8n이 주입하는 값을 수용
- Gradio 앱 정의 시 명확한 타입 힌트 사용
- 테스트 단계에서 Gradio의 `/gradio_api/mcp/schema` 엔드포인트로 정확한 스키마 확인

```python
# n8n의 toolCallId를 수용하도록 수정
def translate(
    text: str,
    source_lang: str = "en",
    target_lang: str = "ko",
    toolCallId: str = None  # n8n에서 주입되는 파라미터
) -> str:
    # toolCallId는 내부적으로 사용하지 않음
    ...
```

### 3. n8n에서 Tool Description 필수 입력

**문제 상황:**
MCP Server를 n8n에 연결했을 때, Tool들이 제대로 인식되지 않는 현상이 발생했습니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/description.png" 
         width="500"
         alt="toolCallId 에러 화면"/>
    <p><em>toolCallId 파라미터로 인한 Schema 검증 실패</em></p>
</div>

**원인:**
n8n의 AI Agent는 각 Tool의 **Description**을 매우 중요하게 취급합니다. Gradio에서 함수만 정의하고 설명을 생략하면, 에러가 나는 경우가 있습니다.

**해결 방법:**
- Gradio 인터페이스 생성 시 `description` 파라미터를 명확하게 작성
- 각 파라미터에도 설명 추가
- Tool이 무엇을 하는지, 언제 사용해야 하는지 구체적으로 명시

```python
import gradio as gr

# 설명이 없는 경우 (❌)
def translate(text: str, target_lang: str = "ko") -> str:
    ...

demo = gr.Interface(fn=translate, inputs=["text", "text"], outputs="text")

# 설명이 있는 경우 (✅)
def translate(text: str, target_lang: str = "ko") -> str:
    """주어진 텍스트를 목표 언어로 번역합니다.
    
    Args:
        text: 번역할 텍스트.
        target_lang: 어떤 언어로 번역할지.
    """
    ...

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="text", info="번역할 텍스트를 입력하세요"),
        gr.Textbox(label="target_lang", info="목표 언어 코드 (예: ko, en, ja)")
    ],
    outputs="text",
    title="번역 도구",
    description="텍스트를 지정한 언어로 번역하는 Tool입니다. 기술 문서나 일반 텍스트 번역에 사용하세요."
)
```

**추가 팁:**
- Description은 AI Agent가 Tool 선택의 기준으로 사용하므로, 구체적일수록 좋습니다.
- "이 Tool은 ~할 때 사용하세요"와 같이 사용 시나리오를 포함하면 더욱 효과적입니다.

### 4. Gradio의 MCP Tools Success Rate 대시보드

**발견:**
Gradio Space에서 MCP Server를 사용하다 보니, 각 Tool의 **성공률(Success Rate)**을 실시간으로 보여주는 대시보드가 있다는 것을 발견했습니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/dashboard.png" 
         width="500"
         alt="Gradio Success Rate 대시보드"/>
    <p><em>각 Tool의 성공률을 보여주는 Gradio 대시보드</em></p>
</div>

**활용 방법:**
- 어떤 Tool이 자주 실패하는지 한눈에 파악 가능
- 워크플로우 개선 시 우선순위 결정에 활용
- 프로덕션 환경에서 모니터링 지표로 사용

이 기능 덕분에 번역 Tool의 성공률을 모니터링하고, 에러 패턴을 추적하여 각 MCP Server를 개선할 수 있었습니다.

### 5. Notion API: Rich Text의 2000자 제한

**문제 상황:**
긴 번역 결과를 Notion 페이지로 저장하려고 할 때 에러가 발생했습니다.

**원인:**
[Notion API 문서](https://developers.notion.com/reference/request-limits#size-limits)를 확인해보니, Rich Text 블록은 **최대 2000자까지만 허용**됩니다. 저희 프로젝트의 경우 번역 결과물이 2000자를 넘는 경우가 자주 발생했기 때문에 다른 방식이 필요했습니다.

**해결 방법:**
- 현서님께서 Notion Tool 대신 GitHub MCP Server로 대체하는 방식을 찾아주셨습니다
- GitHub로 파일을 바로 저장하고, 번역 결과물 또한 GitHub에서 읽어오도록 변경
- Notion UI의 편리함을 포기해야 했지만, Notion API의 사이즈 제한 문제로 꼭 필요한 대체였습니다

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/github.png" 
         width="500"
         alt="GitHub 기반으로 변경된 워크플로우"/>
    <p><em>Notion 대신 GitHub를 사용하도록 변경된 워크플로우</em></p>
</div>

**참고사항:**
만약 Notion을 사용해야 한다면, 2000자가 넘지 않는 콘텐츠 저장을 대상으로 하는지 미리 확인하시기 바랍니다.

### 6. 더 똑똑한 AI Agent의 필요성

**문제 인식:**
2000자 제한 문제를 해결하면서 근본적인 의문이 들었습니다. "AI Agent가 Tool 사용 중 제약을 만나면 스스로 대응할 수는 없을까?"

**현재 상황:**
- 2000자 제한을 넘으면 에러 발생
- 개발자가 미리 분할 로직을 구현해야 함
- Agent는 단순히 주어진 Tool을 실행만 함

**이상적인 시나리오:**
똑똑한 AI Agent라면 이렇게 동작해야 하지 않을까요?
1. 번역 결과가 2000자를 넘는다는 것을 인지
2. 자동으로 텍스트를 분할
3. Notion Tool을 여러 번 호출하여 순차적으로 저장
4. 전체 작업 완료 후 성공 보고

**시사점:**
현재의 n8n 워크플로우 구조는 정해진 시퀀스를 실행하는 데 최적화되어 있습니다. 하지만 앞으로는 **도구의 제약을 이해하고 동적으로 대응할 수 있는 더 지능적인 Agent**가 필요해 보입니다. 이는 단순한 워크플로우 엔진을 넘어서는 차원의 문제입니다.

### 7. 유연한 재시도 어려움 및 결과물 품질 문제

**문제 상황:**
n8n으로 워크플로우를 실행하면 AI Chat Client로 직접 실행할 때보다 번역 과정에서 사람의 개입이 제한적입니다. 사람은 MCP를 직접 실행하면서 중간에 개입하여 결과물의 품질을 높이기 위해 고쳐달라고 요청하거나 재시도를 요청할 수 있습니다. 

반면에, n8n 워크플로우는 Chat History처럼 중간 결과물이 메모리에 저장되어 있지 않다보니 워크플로우를 중간부터 재시도하지 못하고, 처음부터 다시 실행해야 합니다.

**해결 시도:**
- 시스템 프롬프트에 "완전한 번역 제공" 명시
- 프롬프트 최적화로 어느 정도 개선

완벽한 해결은 아니지만, 프롬프트를 잘 작성하면 품질을 어느 정도 유지할 수 있습니다.

### 8. Max Iteration의 함정

**문제 상황:**
복잡한 번역 작업을 수행할 때 워크플로우가 중간에 멈추는 현상이 발생했습니다.

<div align="center">
    <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/iteration.png" 
         width="500"
         alt="Max Iteration 초과 에러"/>
    <p><em>Max Iteration 초과로 인한 워크플로우 중단</em></p>
</div>

**원인:**
n8n AI Agent는 무한 루프 방지를 위해 **Max Iteration** 설정이 있습니다. Tool 호출 횟수가 이 값을 초과하면 강제로 종료됩니다.

**해결 방법:**
- 작업을 적절한 크기로 나누어 각 단계별로 Tool 체인 구성

    ```
    Before: 
    [Input] → [AI Agent: 20 tool calls] → [Output]
             ❌ Max Iteration 초과

    After:
    [Input] → [AI Agent: Step 1] → [AI Agent: Step 2] → [Output]
             ✅ 각 5 tool calls      ✅ 각 5 tool calls
    ```

- Max Iteration 값을 작업 복잡도에 맞게 조정 

    <div align="center">
        <img src="../assets/images/blog/posts/2026-01-05-hf-translation-mcp-n8n/max.png" 
             width="500"
             alt="Max Iteration 설정"/>
        <p><em>n8n AI Agent의 Max Iteration 설정</em></p>
    </div>

- 긴 작업은 여러 워크플로우로 분리하고 순차 실행

## 마치며

n8n으로 MCP Server 워크플로우를 구축하는 과정은 예상보다 많은 시행착오가 필요했습니다. Gradio의 전체 노출 특성, Schema 검증, Notion의 2000자 제한 등 실제로 부딪혀봐야 알 수 있는 제약들이 많았습니다.

하지만 이러한 과정을 통해 얻은 가장 큰 교훈은 **자동화의 가치와 한계를 동시에 이해하게 되었다**는 것입니다. n8n은 반복 작업을 효율화하는 데 탁월하지만, Claude Desktop Client의 유연성과 품질을 완전히 대체할 수는 없습니다. 

앞으로는 단순한 워크플로우 엔진을 넘어, **도구의 제약을 이해하고 동적으로 대응할 수 있는 더 지능적인 Agent**가 등장하기를 기대합니다. 2000자 제한을 만나면 스스로 분할해서 처리하고, Max Iteration에 도달하면 작업을 재구성하는 그런 Agent 말입니다.

지금은 개발자가 모든 예외 상황을 예측하고 처리해야 하지만, 언젠가는 AI가 도구의 특성을 학습하고 최적의 사용법을 스스로 찾아가는 날이 올 것입니다. 그때까지는 Desktop Client와 n8n을 적재적소에 활용하는 지혜가 필요해 보입니다.
