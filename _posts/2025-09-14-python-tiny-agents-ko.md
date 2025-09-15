---
layout: post
title: "파이썬 Tiny Agents: 약 70줄의 코드로 MCP 기반 에이전트 구현하기"
author: minju
categories: [Agent]
image: assets/images/blog/posts/2015-09-14-python-tiny-agents/thumbnail.png
---
* TOC
{:toc}
<!--toc-->

_이 글은 Hugging Face 블로그의 [Tiny Agents in Python: an MCP-powered agent in ~70 lines of code](https://huggingface.co/blog/python-tiny-agents)를 한국어로 번역한 글입니다._

---
# 파이썬 Tiny Agents: 약 70줄의 코드로 MCP 기반 에이전트 구현하기

> [!TIP]
> NEW: tiny-agents가 이제 [AGENTS.md](https://agents.md/) 표준을 지원합니다. 🥳

[Tiny Agents in JS](https://huggingface.co/blog/tiny-agents)에서 영감을 받아, 이 아이디어를 파이썬 🐍으로 개발하고 [`huggingface_hub`](https://github.com/huggingface/huggingface_hub/) 클라이언트 SDK를 확장하여 MCP 클라이언트로써 MCP 서버에서 도구를 가져와 추론 중에 LLM에 전달할 수 있도록 했습니다. 

MCP ([Model Context Protocol](https://modelcontextprotocol.io/))는 대규모 언어 모델(LLM)이 외부 도구 및 API와 상호 작용하는 방식을 표준화하는 개방형 프로토콜입니다. 본질적으로 각 도구에 대한 개별적인 통합을 개발할 필요가 없어졌으며, 이를 통해 LLM에 새로운 기능을 더 쉽게 연결할 수 있습니다.

이 블로그 게시물에서는 강력한 도구 기능을 활용할 수 있도록 MCP 서버에 연결된 파이썬의 Tiny Agent를 시작하는 방법을 보여줍니다. 자신만의 에이전트를 얼마나 쉽게 구축하고 바로 개발을 시작할 수 있는지 직접 확인해 보세요!

> [!TIP]
> _스포일러_ : 에이전트는 본질적으로 MCP 클라이언트 위에 구축된 while 루프입니다!

## 데모 실행 방법

이 섹션에서는 기존 Tiny Agents를 사용하는 방법을 안내합니다. 에이전트를 실행하기 위한 설정 및 명령을 다루겠습니다.

먼저, 필요한 모든 구성 요소를 얻으려면 `mcp` 추가 기능과 함께 `huggingface_hub`의 최신 버전을 설치해야 합니다.

```bash
pip install "huggingface_hub[mcp]>=0.32.0"
```

이제 CLI를 사용하여 에이전트를 실행해 봅시다!

가장 멋진 점은 Hugging Face Hub [tiny-agents](https://huggingface.co/datasets/tiny-agents/tiny-agents) 데이터셋에서 직접 에이전트를 로드하거나, 자신만의 로컬 에이전트 설정에 경로를 지정할 수 있다는 것입니다!

```bash
> tiny-agents run --help
                                                                                                                                                                                     
 Usage: tiny-agents run [OPTIONS] [PATH] COMMAND [ARGS]...                                                                                                                           
                                                                                                                                                                                     
 Run the Agent in the CLI                                                                                                                                                            
                                                                                                                                                                                     
                                                                                                                                                                                     
╭─ Arguments ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   path      [PATH]  Path to a local folder containing an agent.json file or a built-in agent stored in the 'tiny-agents/tiny-agents' Hugging Face dataset                         │
│                     (https://huggingface.co/datasets/tiny-agents/tiny-agents)                                                                                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


```

특정 에이전트 설정에 경로를 제공하지 않으면, Tiny Agent는 기본적으로 다음 두 MCP 서버에 연결됩니다.

- 당신의 데스크톱에 접근 권한을 갖는 "표준" [파일 시스템 서버](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem),
- 샌드박스 환경의 Chromium 브라우저를 사용하는 방법을 아는 [Playwright MCP](https://github.com/microsoft/playwright-mcp) 서버.


다음 예시는 Nebius 추론 공급자를 통해 [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) 모델을 사용하도록 구성된 웹 탐색 에이전트를 보여줍니다. 이 에이전트에는 웹 브라우저를 사용할 수 있게 해주는 Playwright MCP 서버가 함께 제공됩니다! 에이전트 설정은 Hugging Face 데이터셋의 [tiny-agents/tiny-agents](https://huggingface.co/datasets/tiny-agents/tiny-agents/tree/main/celinah/web-browser)에 있는 경로를 지정하여 불립니다.

<video controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/python-tiny-agents/web_browser_agent.mp4" type="video/mp4">
</video>

에이전트를 실행하면, 연결된 MCP 서버에서 발견한 도구 목록을 불러오는 것을 볼 수 있습니다. 이제 여러분의 프롬프트에 응답할 준비가 되었습니다!

이 데모에서 사용된 프롬프트:

> Brave Search에서 HF 추론 공급자를 웹 검색하고 첫 번째 결과를 연 다음 Hugging Face에서 지원되는 추론 공급자 목록을 알려주세요.

Gradio Spaces를 MCP 서버로 사용할 수도 있습니다! 다음 예시는 Nebius 추론 공급자를 통해 [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) 모델을 사용하고, MCP 서버로 `FLUX.1 [schnell]` 이미지 생성 HF Space에 연결합니다. 에이전트는 Hugging Face Hub의 [tiny-agents/tiny-agents](https://huggingface.co/datasets/tiny-agents/tiny-agents/tree/main/julien-c/flux-schnell-generator) 데이터셋에 있는 구성에서 로드됩니다.

<video controls autoplay loop>
  <source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/python-tiny-agents/image-generation.mp4" type="video/mp4">
</video>

이 데모에서 사용된 프롬프트:

> 달 표면에서 알에서 부화하는 작은 우주 비행사의 1024x1024 이미지를 생성하세요.

이제 기존 Tiny Agents를 실행하는 방법을 살펴보았으니, 다음 섹션에서는 Tiny Agents가 작동하는 방식과 자신만의 에이전트를 구축하는 방법에 대해 더 자세히 설명합니다.

### 에이전트 설정

각 에이전트의 동작(기본 모델, 추론 공급자, 연결할 MCP 서버, 초기 시스템 프롬프트)은 `agent.json` 파일에 정의됩니다. 더 자세한 시스템 프롬프트를 위해 동일한 디렉토리에 사용자 지정 `PROMPT.md`를 제공할 수도 있습니다. 다음은 예시입니다.

`agent.json`
`model` 및 `provider` 필드는 에이전트가 사용하는 LLM 및 추론 공급자를 지정합니다.
`servers` 배열은 에이전트가 연결할 MCP 서버를 정의합니다.
이 예시에서는 "stdio" MCP 서버가 구성되어 있습니다. 이 유형의 서버는 로컬 프로세스로 실행됩니다. 에이전트는 지정된 `command` 및 `args`를 사용하여 서버를 시작한 다음 stdin/stdout을 통해 통신하여 사용 가능한 도구를 검색하고 실행합니다.
```json
{
	"model": "Qwen/Qwen2.5-72B-Instruct",
	"provider": "nebius",
	"servers": [
		{
			"type": "stdio",
			"command": "npx",
			"args": ["@playwright/mcp@latest"]
		}
	]
}

```
`PROMPT.md`

```
You are an agent - please keep going until the user’s query is completely resolved [...]


```


> [!TIP]
> Hugging Face 추론 공급자에 대한 자세한 내용은 [여기](https://huggingface.co/docs/inference-providers/index)에서 확인할 수 있습니다.

### LLM은 도구를 사용할 수 있습니다.

최신 LLM은 함수 호출(또는 도구 사용)을 위해 구축되어 사용자가 특정 사용 사례 및 실제 작업에 맞춰진 애플리케이션을 쉽게 구축할 수 있도록 합니다.

함수는 스키마에 의해 정의되며, 이는 LLM에 함수가 무엇을 하는지, 어떤 입력 인수를 예상하는지 알려줍니다. LLM은 도구를 사용할 시기를 결정하고, 에이전트는 도구 실행을 조율하고 결과를 다시 전달합니다.

```python
tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Paris, France"
                        }
                    },
                    "required": ["location"],
                },
            }
        }
]
```

`InferenceClient`는 [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/function-calling?api-mode=chat)와 동일한 도구 호출 인터페이스를 구현하며, 이는 추론 공급자 및 커뮤니티의 확립된 표준입니다.

## 파이썬 MCP 클라이언트 구축

`MCPClient`는 도구 사용 기능의 핵심입니다. 이제 `huggingface_hub`의 일부이며 `AsyncInferenceClient`를 사용하여 LLM과 통신합니다.

> [!TIP]
> 전체 `MCPClient` 코드는 [여기](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/mcp_client.py)에서 찾을 수 있습니다. 실제 코드를 따라가고 싶다면 참고하세요 🤓

`MCPClient`의 주요 책임:

- 하나 이상의 MCP 서버에 대한 비동기 연결 관리.
- 이러한 서버에서 도구 검색.
- LLM을 위한 도구 형식 지정.
- 올바른 MCP 서버를 통해 도구 호출 실행.

MCP 서버에 연결하는 방법은 다음과 같습니다(`add_mcp_server` 메서드):

```python
# `MCPClient.add_mcp_server`의 111-219줄
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/mcp_client.py#L111:L219
class MCPClient:
    ...
    async def add_mcp_server(self, type: ServerType, **params: Any):
        # 'type'은 "stdio", "sse", 또는 "http"일 수 있습니다.
        # 'params'는 서버 유형에 따라 다릅니다. 예:
        # "stdio"의 경우: {"command": "my_tool_server_cmd", "args": ["--port", "1234"]}
        # "http"의 경우: {"url": "http://my.tool.server/mcp"}

        # 1. 유형(stdio, sse, http)에 따라 연결 설정
        #    (mcp.client.stdio_client, sse_client, 또는 streamablehttp_client 사용)
        read, write = await self.exit_stack.enter_async_context(...)

        # 2. MCP ClientSession 생성
        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream=read, write_stream=write, ...)
        )
        await session.initialize()

        # 3. 서버에서 도구 목록 가져오기
        response = await session.list_tools()
        for tool in response.tools:
            # 이 도구에 대한 세션 저장
            self.sessions[tool.name] = session 
            # 사용 가능한 도구 목록에 도구 추가 및 LLM을 위한 형식 지정
            self.available_tools.append({ 
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            })

```
로컬 도구(예: 파일 시스템 액세스)를 위한 `stdio` 서버와 원격 도구를 위한 `http` 서버를 지원합니다! 또한 원격 도구의 이전 표준인 `sse`와도 호환됩니다.

## 도구 사용: 스트리밍 및 처리

`MCPClient`의 `process_single_turn_with_tools` 메서드는 LLM 상호 작용이 발생하는 곳입니다. 대화 기록과 사용 가능한 도구를 `AsyncInferenceClient.chat.completions.create(..., stream=True)`를 통해 LLM에 보냅니다.

### 1. 도구 준비 및 LLM 호출

먼저, 이 메서드는 현재 차례에 LLM이 알아야 할 모든 도구를 결정합니다. 여기에는 MCP 서버의 도구와 에이전트 제어를 위한 특별한 "루프 종료" 도구가 포함됩니다. 그런 다음 LLM에 스트리밍 호출을 합니다.

```python
# `MCPClient.process_single_turn_with_tools`의 241-251줄
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/mcp_client.py#L241:L251

    # 옵션에 따라 도구 목록 준비
    tools = self.available_tools
    if exit_loop_tools is not None:
        tools = [*exit_loop_tools, *self.available_tools]

    # LLM에 스트리밍 요청 생성
    response = await self.client.chat.completions.create(
        messages=messages,
        tools=tools,
        tool_choice="auto",  # LLM이 도구가 필요한지 결정
        stream=True,  
    )

```

LLM으로부터 청크가 도착하면, 메서드는 이를 반복합니다. 각 청크는 즉시 반환되며, 그런 다음 완전한 텍스트 응답과 모든 도구 호출을 재구성합니다.

```python
# `MCPClient.process_single_turn_with_tools`의 258-290줄 
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/mcp_client.py#L258:L290
# 스트림에서 읽기
async for chunk in response:
      # 각 청크를 호출자에게 반환
      yield chunk
      # LLM의 텍스트 응답 및 도구 호출 부분 집계
      …
```

### 2. 도구 실행

스트림이 완료되면, LLM이 도구 호출을 요청한 경우(`final_tool_calls`에 완전히 재구성됨), 메서드는 각 호출을 처리합니다.

```python
# `MCPClient.process_single_turn_with_tools`의 293-313줄 
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/mcp_client.py#L293:L313
for tool_call in final_tool_calls.values():
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments or "{}")

    # 도구 결과를 저장할 메시지 준비
    tool_message = {"role": "tool", "tool_call_id": tool_call.id, "content": "", "name": function_name}

    # a. 이것이 특별한 "루프 종료" 도구인가요?
    if exit_loop_tools and function_name in [t.function.name for t in exit_loop_tools]:
        # 그렇다면 메시지를 반환하고 이 턴의 처리를 종료합니다.
        messages.append(ChatCompletionInputMessage.parse_obj_as_instance(tool_message))
        yield ChatCompletionInputMessage.parse_obj_as_instance(tool_message)
        return # 에이전트의 메인 루프가 이 신호를 처리합니다.

    # b. 일반 도구입니다: MCP 세션을 찾아 실행합니다.
    session = self.sessions.get(function_name) # self.sessions는 도구 이름을 MCP 연결에 매핑합니다.
    if session is not None:
        result = await session.call_tool(function_name, function_args)
        tool_message["content"] = format_result(result) # format_result는 도구 출력을 처리합니다.
    else:
        tool_message["content"] = f"Error: No session found for tool: {function_name}"
        tool_message["content"] = error_msg

    # 도구 결과를 기록에 추가하고 반환합니다.
    ...

```

먼저 호출된 도구가 루프를 종료하는지(`exit_loop_tool`) 확인합니다. 그렇지 않으면 해당 도구에 대한 올바른 MCP 세션을 찾아 `session.call_tool()`을 호출합니다. 결과(또는 오류 응답)는 형식화되어 대화 기록에 추가되고 반환되어 에이전트가 도구의 출력을 알 수 있도록 합니다.

## 우리의 Tiny Python Agent: 거의 루프일 뿐입니다!

`MCPClient`가 도구 상호 작용에 대한 모든 작업을 수행하므로 `Agent` 클래스는 놀랍도록 간단해집니다. `MCPClient`를 상속하고 대화 관리 로직을 추가합니다.

> [!TIP]
> Agent 클래스는 작고 대화 루프에 중점을 둡니다. 코드는 [여기](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/agent.py)에서 찾을 수 있습니다.

### 1. 에이전트 초기화

에이전트가 생성될 때, 에이전트 구성(모델, 공급자, 사용할 MCP 서버, 시스템 프롬프트)을 가져와 시스템 프롬프트로 대화 기록을 초기화합니다. `load_tools()` 메서드는 서버 구성(agent.json에 정의됨)을 반복하고 각 구성에 대해 `add_mcp_server`(부모 `MCPClient`에서)를 호출하여 에이전트의 도구 상자를 채웁니다.

```python
# `Agent`의 12-54줄 
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/agent.py#L12:L54
class Agent(MCPClient):
    def __init__(
        self,
        *,
        model: str,
        servers: Iterable[Dict], # MCP 서버 구성
        provider: Optional[PROVIDER_OR_POLICY_T] = None,
        api_key: Optional[str] = None,
        prompt: Optional[str] = None, # 시스템 프롬프트
    ):
        # 모델, 공급자 등으로 기본 MCPClient 초기화
        super().__init__(model=model, provider=provider, api_key=api_key)
        # 로드할 서버 구성 저장
        self._servers_cfg = list(servers)
        # 시스템 메시지로 대화 시작
        self.messages: List[Union[Dict, ChatCompletionInputMessage]] = [
            {"role": "system", "content": prompt or DEFAULT_SYSTEM_PROMPT}
        ]

    async def load_tools(self) -> None:
        # 구성된 모든 MCP 서버에 연결하고 도구 등록
        for cfg in self._servers_cfg:
            await self.add_mcp_server(**cfg)

```

### 2. 에이전트의 핵심: 루프

`Agent.run()` 메서드는 단일 사용자 입력을 처리하는 비동기 제너레이터입니다. 에이전트의 현재 작업이 완료될 시기를 결정하면서 대화 턴을 관리합니다.

```python
# `Agent.run()`의 56-99줄
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_mcp/agent.py#L56:L99
async def run(self, user_input: str, *, abort_event: Optional[asyncio.Event] = None, ...) -> AsyncGenerator[...]:
    ...
    while True: # user_input 처리를 위한 메인 루프
        ...

        # LLM 및 도구와 한 단계 상호 작용하기 위해 MCPClient에 위임합니다.
        # 이는 LLM 텍스트, 도구 호출 정보 및 도구 결과를 스트리밍합니다.
        async for item in self.process_single_turn_with_tools(
            self.messages,
            ...
        ):
            yield item 

        ... 
        
        # 종료 조건
        # 1. "exit" 도구가 호출되었습니까?
        if last.get("role") == "tool" and last.get("name") in {t.function.name for t in EXIT_LOOP_TOOLS}:
                return

        # 2. 최대 턴에 도달했거나 LLM이 최종 텍스트 답변을 제공했습니까?
        if last.get("role") != "tool" and num_turns > MAX_NUM_TURNS:
                return
        if last.get("role") != "tool" and next_turn_should_call_tools:
            return
        
        next_turn_should_call_tools = (last_message.get("role") != "tool")
```

`run()` 루프 내부:
- 먼저 사용자 프롬프트를 대화에 추가합니다.
- 그런 다음 `MCPClient.process_single_turn_with_tools(...)`를 호출하여 LLM의 응답을 얻고 추론의 한 단계에 대한 도구 실행을 처리합니다.
- 각 항목은 즉시 반환되어 호출자에게 실시간 스트리밍을 가능하게 합니다.
- 각 단계 후에 종료 조건을 확인합니다. 특별한 "루프 종료" 도구가 사용되었는지, 최대 턴 제한에 도달했는지, 또는 LLM이 현재 요청에 대한 최종 텍스트 응답을 제공하는지 여부입니다.

## 다음 단계

MCP 클라이언트와 Tiny Agent를 탐색하고 확장할 수 있는 멋진 방법이 많이 있습니다 🔥
시작하는 데 도움이 되는 몇 가지 아이디어는 다음과 같습니다.

- 다양한 LLM 모델 및 추론 공급자가 에이전트 성능에 미치는 영향을 벤치마킹합니다. 각 공급자가 다르게 최적화할 수 있으므로 도구 호출 성능이 다를 수 있습니다. 지원되는 공급자 목록은 [여기](https://huggingface.co/docs/inference-providers/index#partners)에서 찾을 수 있습니다.
- [llama.cpp](https://github.com/ggerganov/llama.cpp) 또는 [LM Studio](https://lmstudio.ai/)와 같은 로컬 LLM 추론 서버로 Tiny Agent를 실행합니다.
- .. 물론 기여하세요! Hugging Face Hub의 [tiny-agents/tiny-agents](https://huggingface.co/datasets/tiny-agents/tiny-agents) 데이터셋에 자신만의 고유한 Tiny Agent를 공유하고 PR을 엽니다.

풀 리퀘스트 및 기여를 환영합니다! 다시 한 번, 여기 있는 모든 것은 [오픈 소스](https://github.com/huggingface/huggingface_hub)입니다! 💎❤️
