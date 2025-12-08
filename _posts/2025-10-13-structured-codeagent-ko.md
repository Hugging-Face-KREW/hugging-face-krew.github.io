---
layout: post
title: "CodeAgents + Structure: 액션 실행을 위한 더 좋은 방법"
author: Jimin
categories: [Agent]
image: assets/images/blog/posts/2025-10-13-structured-codeagent/thumbnail.png
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [CodeAgents + Structure: A Better Way to Execute Actions](https://huggingface.co/blog/structured-codeagent)를 한국어로 번역한 글입니다._

---


# CodeAgents + Structure: 액션 실행을 위한 더 좋은 방법

오늘 우리는 AI 에이전트 설계에서 두 가지 강력한 패러다임을 연결하는 연구를 소개합니다. 하나는 코드 기반 액션의 표현력이고, 다른 하나는 구조화된 생성의 신뢰성입니다. 연구 결과, **CodeAgents**에게 사고(thoughts)와 코드를 모두 구조화된 JSON 형식으로 생성하도록 하면, 여러 벤치마크에서 기존 방식보다 성능이 크게 향상됨을 확인할 수 있습니다.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/structured-codeagent/accuracy.png)
그림 1: 세 가지 접근 방식의 정확도 비교 — Structured CodeAgent(파란색), CodeAgent(주황색), ToolCallingAgent(회색) — <ins>[**SmolBench (GAIA, MATH, SimpleQA, and Frames**](https://huggingface.co/datasets/akseljoonas/smolbench)</ins>)에서 오차 막대는 95% 신뢰구간을 나타냄

## 🤔 에이전트 액션의 진화

AI 에이전트는 API 호출, 데이터 처리, 복잡한 문제 해결과 같은 실제 세계 액션을 취할 수 있어야 합니다. 에이전트가 이러한 액션을 표현하는 방식은 여러 패러다임을 거쳐 발전했습니다.

### **기존의 JSON Agent (ToolCallingAgent)**: 구조화된 JSON을 생성해 도구를 호출하는 에이전트

```json
{"tool": "get_weather", "arguments": {"city": "Paris"}}
```

이 에이전트들은 미리 정의된 도구 목록에서 필요한 도구를 선택하고, JSON 형식으로 호출을 생성하는 방식으로 동작합니다. 이러한 도구 호출 방식은 OpenAI의 <ins>[**function calling API**](https://openai.com/index/function-calling-and-other-api-updates/)</ins>를 통해 널리 알려졌으며, 이후 가장 일반적으로 사용하는 도구 호출 방식이 되었습니다.

이 방식은 신뢰할 수 있지만, 다음과 같은 한계가 있습니다:

* **제한된 액션 범위**: 에이전트가 수행할 수 있는 액션은 미리 정의된 도구에만 한정되어 있어 기능이 제한됨
* **부족한 조합성**: 여러 출처의 정보를 조합해야 하는 작업의 경우, JSON 기반 에이전트는 각 도구 호출 사이에 중간 상태를 유지할 수 없어 어려움을 겪음. 일부 모델은 병렬 도구 호출을 지원하나, 복잡한 시나리오(이전 결과에 따른 이후 액션 결정, 여러 결과를 비교/처리해야 하는 경우)는 다루기 어려움
* **경직된 구조**: 제공된 도구가 실제 필요한 작업과 정확히 일치하지 않는 경우, 유연하게 대응하기 어려움

### **Code Agents**: 고유한 코딩 능력을 활용해 실행 가능한 Python 코드를 직접 작성하는 에이전트

```python
# 한 번의 모델 호출만으로, 3개 도시의 평균 기온을 구할 수 있음.
temperature_sum = 0
for city in ["Paris", "Tokyo", "New York"]:
    temp = get_weather(city)
    temperature_sum += temp
    
print(f"Average temperature: {temperature_sum / 3:.1f}°C")
```

이러한 변화는 처음에 <ins>[“**Executable Code Actions Elicit Better LLM Agents**”](https://arxiv.org/abs/2402.01030)</ins> 논문에서 CodeAct라는 이름으로 제안되었으며, 기존의 단순 도구 호출 방식에 더해 임의의 실행 가능한 Python 코드를 작성할 수 있는 유연성을 AI 에이전트에 부여했습니다.

여기서 핵심 아이디어는, **도구 호출이 코드 내부에서 직접 이루어진다는 점**입니다. 이를 통해 변수와 상태 관리가 훨씬 더 안정적이고 신뢰할 수 있게 됩니다. 에이전트는 루프, 함수, 조건문 안에서 도구를 호출할 수 있으며, 이는 본질적으로 각 액션마다 동적으로 변화하는 도구 실행 그래프를 생성할 수 있습니다!

<ins>[**CodeAgent**](https://github.com/huggingface/smolagents/blob/6a12ebdf210207eec22d5940157f522463fc1c59/src/smolagents/agents.py#L1344)</ins> 사용의 장점:

* **똑똑한 도구 사용**: 상황에 따라 어떤 도구를 사용할지 스스로 결정할 수 있음
* **제한없는 유연성**: 목표 달성을 위해 Python의 모든 기능을 활용할 수 있음
* **사고 검증 가능**: 에이전트가 가설을 세우고 검증(test) 할 수 있어 액션에 더 큰 유연성을 확보 가능

하지만 마크다운에서 코드를 파싱하는 과정은 오류가 발생하기 쉽습니다. 그렇다면 한 가지 제안을 해볼 수 있습니다: 코드 액션을 생성할 때 구조화된 생성을 활용해보는 것은 어떨까요?

## ➡️ Code Agent에 구조화된 출력 추가하기

구조화된 출력을 사용하면, LLM이 사고 과정과 코드를 명확하게 JSON 형식으로 생성하도록 유도할 수 있습니다.

```json
// "code" 블록은 실행 가능한 Python으로 파싱됨.
{
  "thoughts": "I want to find the average temperature across 3 cities.",
  "code": "temperature_sum = 0\nfor city in [\"Paris\", \"Tokyo\", \"New York\"]:\n    temp = get_weather(city)\n    temperature_sum += temp\n\nprint(f\"Average temperature: {temperature_sum / 3:.1f}°C\")"
}
```

기존 방식과의 주요 차이점은 출력 형식이 강제된다는 점입니다. 기존에는 단순히 프롬프트를 통해 사고 과정과 코드 순서로 출력을 유도했다면, <ins>[**structured outputs**](https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance)</ins>을 사용해 정해진 구조를 반드시 따르도록 강제할 수 있습니다.

이러한 접근 방식은 코드 실행의 유연성과 구조화된 생성의 신뢰성을 동시에 활용할 수 있어, 두 방식의 장점을 모두 얻을 수 있습니다.

* **명시적 추론**: `thoughts` 필드는 에이전트가 액션을 수행하기 전에 반드시 추론하도록 유도
* **신뢰할 수 있는 파싱**: JSON 구조를 사용하여 마크다운 파싱 오류를 제거
* **완전한 코드 표현력**: `code` 필드를 통해 CodeAgent가 가진 모든 코드 유연성을 그대로 활용 가능
* **명확한 분리**: 계획(planning)과 실행(execution)을 명확하게 구분

## 🧪 벤치마크 결과

우리는 GAIA, MATH, SimpleQA, Frames등 여러 벤치마크를 대상으로 세 가지 패러다임을 비교했습니다. 결과는 명확합니다. **`코드 액션 + 구조화된 생성`의 조합은 성능이 우수한 모델에서 일관되게 성능을 향상시킵니다.**

대부분의 고성능 모델에서 구조화된 접근법은 기존의 CodeAgent 방식보다 평균 2~7%p 더 높은 성능을 보입니다.

* **OpenAI 모델**: 구조화된 접근법 적용 시, 가장 큰 성능 향상을 보였으며, 특히 추론이 중요한 작업에서 큰 향상이 나타남
* **Claude 모델**: 전반적으로 구조화의 이점을 보였으며, 특히 Claude 3.7 Sonnet에서 큰 성능 향상이 나타남
* **Qwen 모델**: 구조화를 통해 전반적으로 성능이 향상되었으나, 작은 모델(Qwen3-4B)에서는 “structure tax (다음 세션에서 설명)”가 발생

## 💡 구조화가 (일반적으로) 도움이 되는 이유

### 파싱 문제는 실제로 존재한다

<ins>[**smolagents의 CodeAgent 구현**](https://github.com/huggingface/smolagents/blob/6a12ebdf210207eec22d5940157f522463fc1c59/src/smolagents/agents.py#L1344)</ins>은 LLM 출력에서 Python 코드를 추출하는데, 다음과 같은 상황에서 실패할 수 있습니다:

* 마크다운 코드 블록이 불완전하거나 잘못된 형식으로 작성된 경우
* 하나의 응답에 여러 코드 블록이 포함된 경우

구조화된 생성은 안정적인 JSON 파싱으로 이러한 문제를 제거합니다.

구조화된 생성의 효과를 검증하기 위해, 우리는 벤치마크 전반에서 15,724개의 에이전트 실행 로그(agent trace)를 분석했습니다. 그 결과는 매우 인상적이었습니다:

* 첫 번째 호출에서 파싱 오류가 발생한 경우: **2.4%**
* 첫 호출에 **파싱 오류가 있는** 경우 성공률: **42.3%**
* 첫 호출에 **파싱 오류가 없는** 경우 성공률: **51.3%**

**첫 호출에서 파싱 오류가 없는 에이전트는, 오류가 있는 경우보다 성공률이 21.3% 더 높았습니다**

이는 단순한 편의성의 문제가 아닙니다. 파싱 오류는 연쇄적으로 실패를 일으켜 전체 에이전트 성능에 큰 영향을 줍니다. 첫 번째 액션을 제대로 실행하지 못하면, 에이전트가 이후 회복하기 어려워 최적의 문제 해결 경로를 벗어나게 됩니다.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/structured-codeagent/parsing_error.png)
그림 2: 첫 호출 응답에서 파싱 오류가 발생하면 성공률이 21.3% 감소하고, 평균 스텝 수는 3.18에서 4.63으로 증가합니다.

#### 추기: 강제된 추론 과정

구조화된 생성과 명시적으로 `thoughts` 필드를 사용하면, 에이전트가 액션을 수행하기 전에 반드시 추론하도록 강제할 수 있습니다. 이는 다음과 같은 효과를 가져옵니다:

* **더 나은 계획 수립**: 에이전트가 문제 해결을 더 체계적으로 사고
* **신뢰성 향상**: 명시적인 추론 과정을 통해 논리적 오류를 조기에 발견

### Structure Tax: 구조화 비용

실험 결과, 구조화된 생성을 활용할 때는 명확한 성능 임계점이 존재합니다. 모델은 지시문에 대한 충분한 이해와, JSON 형식에 대한 사전 학습 경험을 갖추고 있어야 구조화 접근법의 이점을 얻을 수 있습니다. 구조화된 접근법은 다음과 같은 모델에 가장 효과적입니다:

* 충분히 크고, 잘 학습된 모델
* 지시문 따르기(instruction-following)가 뛰어난 모델
* 구조화된 생성에 특화되어 파인튜닝된 모델

#### 구조화가 실패하는 경우: 실제 예시

작은 모델(`mistralai/Mistral-7B-Instruct-v0.3`)이 구조화된 코드를 생성하려 할 때, 인지 부하가 커집니다:

```json
{
  "thought": "I need to find the height...",
  "code": "web_search(query=\"Eiffel Tower height\")\", "
}
```

모델이 생성한 Python 코드는 `web_search(query="Eiffel Tower height")",`와 같이 문법적으로 잘못되어 SyntaxError와 코드 실행 실패로 이어집니다.

이것이 “structure tax”의 예입니다. 작은 모델은 JSON 형식, Python 문법, 실제 문제 해결 로직을 동시에 처리하지 못해 단순한 마크다운 기반 코드 생성보다 성능이 떨어질 수 있습니다.

## 🚀 Structured CodeAgents는 언제 사용해야 될까?

#### ✅ Structured CodeAgents를 사용하면 좋은 경우:

* 강력한 모델(32B+ 파라미터 또는 프런티어 모델)을 사용할 때
* 복잡한 추론과 코드 실행이 필요한 작업일 때
* 에이전트 출력에 대해 신뢰할 수 있는 파싱이 필요할 때

#### ⚠️ 다음과 같은 경우에는 다른 접근법을 고려:

* 구조화된 생성에 어려움을 겪는 작은 모델을 사용할 때
* 단순하고 미리 정의된 워크플로우로 충분할 때

### smolagents에서 사용 방법:

매우 간단합니다! `use_structured_outputs_internally:` 옵션만 활성화하면 됩니다.

```python
from smolagents import CodeAgent, InferenceClientModel, GoogleSearchTool

Configure agent for structured generation
agent = CodeAgent(
    tools=[GoogleSearchTool(provider="serper")],
    model=InferenceClientModel("Qwen/Qwen3-235B-A22B", provider='nebius'),
    use_structured_outputs_internally=True # 구조화된 출력 활성화
)

result = agent.run("Calculate the time for a cheetah to run across the Golden Gate Bridge")
```

이에 대해, LLM은 다음과 같이 답변을 생성할 수 있습니다:

```json
{
  "thoughts": "I need to find the length of the Golden Gate Bridge and the top speed of a cheetah, then calculate the time.",
  "code": "bridge_info = web_search('Golden Gate Bridge length meters')\ncheetah_speed = web_search('Cheetah top speed') ..."
}
```

그 후 "code" 부분은 기존 CodeAgent와 동일하게 실행됩니다. 단, 이제 파싱 신뢰도는 100%입니다!

### 구현 팁

1. **명확한 프롬프트 작성**: 원하는 JSON 구조를 프롬프트에 명확히 명시
2. **모델 선택**: 구조화된 생성 능력이 강한 모델을 선택
3. **적절한 provider 선택**: OpenAI나 Anthropic 같은 일부 API provider는 구조화된 생성을 기본적으로 지원함. Hugging Face Inference Provider를 사용하는 경우, 구조화된 생성 지원 여부는 provider마다 다름. 구조화된 생성을 지원하는 목록은 다음을 참고: <ins>[**Structured generation support for Models in smolagents‣**](https://www.notion.so/huggingface2/Structured-generation-support-for-Models-in-smolagents-1f51384ebcac8074a051e6dd03d1fe1d)</ins>

### 더 큰 그림 - 다음 단계는?

이 연구는 우리가 에이전트 아키텍처를 더 정교하게 이해하는 방향으로 나아가고 있음을 보여줍니다. 이것은 단순히 “에이전트가 무엇을 할 수 있는가?”가를 넘어, “에이전트가 그것을 어떻게 생각하고 수행해야 하는가?”에 대한 문제입니다.

추론 과정을 더 명시적으로 표현하는 것은 모델이 문제 해결 경로를 더 잘 유지하도록 돕거나, 혹은 단순히 파싱이 쉬워서일 수도 있습니다. 어느 쪽이든 좋습니다.

하지만 이것은 시작에 불과합니다. 앞으로 탐구할 질문은 많습니다:

* 또 어떤 구조적 개선이 성능 향상에 도움이 될까?
* 특히 작은 모델에서 이를 더 잘 작동하게 만들 방법은 무엇일까?
* 이것이 AI 추론 본질에 대해 어떤 통찰을 주는가?

지금 smolagents를 사용 중이거나 자체 CodeAgent 시스템을 만들고 있다면, 구조화된 출력을 시도해보는 것을 추천합니다. 파싱 오류가 사라지고, 성능이 눈에 띄게 개선될 수도 있습니다!
