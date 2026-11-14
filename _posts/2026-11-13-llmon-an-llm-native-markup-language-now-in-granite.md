---
layout: post
title: "LLMON: LLM 네이티브 마크업 언어, 이제 Granite 4.1 3B 및 8B에서 사용 가능"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6484ca4d5ff9a3d2ee52a83b/6KFZDHIJkxpICiErHLqAw.gif
image: assets/images/blog/posts/2026-11-12-llmon-an-llm-native-markup-language-now-in-granite/thumbnail.gif
authors:
  - user: ibm-granite
slug: "llmon-an-llm-native-markup-language-now-in-granite"
source_url: "https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite"
source_published_date: "2026-11-12"
source_published_at: "2026-11-12T19:07:35+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [LLMON: An LLM-Native Markup Language, now in Granite 4.1 3B and 8B](https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# LLMON: LLM 네이티브 마크업 언어, 이제 Granite 4.1 3B 및 8B에서 사용 가능

LLM은 다음 토큰 집합을 생성하기 위해 컨텍스트에 의존합니다. 긴 궤적에 걸쳐 컨텍스트는 시스템 프롬프트, 과거 대화 턴, 검색된 문서에 적용되는 지침, 도구 응답 또는 사용자 데이터, 허용되는 출력을 제한하는 요구 사항, 추론 체인 등 수많은 유형의 구성 요소로 채워집니다. 컨텍스트는 풍부한 구조를 담고 있지만, 그 대부분은 모델에 암묵적으로 존재하거나 JSON과 같이 LLM이 처리하도록 설계되지 않은 형식으로 인코딩됩니다. 이는 LLM, 특히 규모가 작은 모델이 취약하고 때로는 예상치 못한 방식으로 실패하는 주요 이유 중 하나입니다. 명시적인 구조가 부족하면 공격자가 이를 악용할 수도 있으며, 예를 들어 프롬프트 인젝션의 형태로 나타날 수 있습니다.

이를 해결하기 위해, 모델에 구조와 의미를 명시적으로 전달하는 LLM 네이티브 마크업 언어인 LLMON(LLM Object Notation, “레몬”으로 발음)을 개발했습니다. 의미적으로 일관된 구성 요소는 의미 있는 태그로 표시하며, 이 태그는 예약된 특수 토큰 구분 기호로 감쌉니다. 중첩을 포함한 데이터 스키마는 인과적 LLM이 모호함 없이 이해할 수 있는 방식으로 표현됩니다. 고유 ID를 사용하면 컨텍스트 어디에든 있을 수 있는 특정 지침을 가리키고, 해당 지침이 적용되어야 하는 데이터 페이로드에 연결할 수 있습니다. 이는 프로그래밍 언어에서 함수를 호출하고 인수를 전달하는 방식과 유사합니다.

오늘 Granite 4.1 base 모델에서 파생되었으며 LLMON으로 표시된 텍스트를 입력으로 받고 생성하도록 학습된 두 개의 experimental preview 모델을 공개합니다: [granite-4.1-3b-LLMON-preview](https://huggingface.co/ibm-granite/granite-4.1-3b-llmon-preview) 및 [granite-4.1-8b-LLMON-preview](https://huggingface.co/ibm-granite/granite-4.1-8b-llmon-preview)입니다. 또한 LLMON을 사용하기 위한 toolkit과 워크플로 및 템플릿 예제가 포함된 [GitHub](https://github.com/generative-computing/llmon) repo도 공개합니다.

이를 통해 가능한 작업:

- 사용자 데이터에 숨겨진 간접 프롬프트 인젝션은 명시적인 instruction/data 분리 덕분에 LLMON 모델에서 대부분 무시됩니다. 일부 벤치마크에서는 공격 성공률이 거의 0으로 떨어지는 반면, 일반 instruction-tuned 모델은 높은 확률로 탈취됩니다. 이는 Granite 4.1 Instruct 모델과 비교해 일반 능력을 유지하면서 달성됩니다.

- 각 지침이 적용되는 정확한 데이터에 연결되므로, 관련 없는 문단, 오래된 채팅 턴, 도구 출력이 답변에 끌려 들어오지 않습니다.

- 지침, 데이터, 요구 사항이 명확하게 표시되므로 애플리케이션 코드 수준의 구조와 의미를 모델이 활용할 수 있습니다. LLM 호출을 자동으로 LLMON화하여 네이티브 LLMON 지원을 활성화한 [Mellea](https://github.com/generative-computing/mellea)에서 직접 사용해 보세요.

- 더 이상 LLM이 JSON 구조를 따르도록 강제하는 일을 걱정할 필요가 없습니다. 기존 JSON을 손실 없이 LLMON으로 변환하려면 저희 converter를 사용하세요. 또한 자체 구조와 관련 태그를 정의하고 LLMON 모델 위에 LoRA를 튜닝하여 해당 구조를 이해하도록 만들 수도 있습니다.

### JSON이 아닌 이유는 무엇인가?

JSON은 인과적 언어 모델이 아니라 parser를 위해 설계되었습니다. 토크나이저는 문법을 준수하도록 제약되지 않으며, 구조는 일반 텍스트에도 나타날 수 있는 토큰({, ", :)으로 인코딩됩니다. 모델 입장에서 문서 안의 따옴표와 필드를 닫는 따옴표는 동일한 토큰입니다. parser에게는 명확한 이스케이프 규칙도 모델에게는 통계적 패턴일 뿐입니다. 중첩 깊이는 괄호를 세는 방식으로 인코딩되는데, 특히 긴 구간에서 transformer는 이러한 작업에 약한 것으로 알려져 있습니다.

반면 LLMON은 XML과 유사한 마크업으로, 구조 구분 기호가 문자가 아니라 예약된 특수 토큰입니다. 따라서 구조와 콘텐츠가 어휘의 서로 다른 부분을 사용합니다. 각 span의 시작 부분에 태그가 표시되고 span 사이의 경계가 명시됩니다. 중첩 구조는 국소적으로 표시되므로 모델이 긴 구간에 걸쳐 계층 구조를 추적할 필요가 없습니다.

대부분의 팀이 이미 JSON을 중심으로 구축된 pipeline을 사용하고 있으므로, JSON을 LLMON으로 또는 그 반대로 손실 없이 변환하는 [변환 도구](https://github.com/generative-computing/llmon/tree/main/llmon/lang)와 예제 notebook을 만들었습니다.

### LLMON은 chat template과 어떤 관련이 있는가?

LLMON은 chat template을 대체하지 않고 보강합니다. 일반적인 chat template이 모델에 각 부분이 어떤 역할(system, user, assistant)에 속하는지 알려준다면, LLMON은 각 구성 요소가 무엇을 의미하며 구성 요소들이 역할 내부와 역할 간에 어떻게 연결되는지를 알려줍니다.

LLMON preview 모델에는 자체 chat template이 포함되어 있으므로, 어떤 chat 모델에 전달할 때와 동일한 `messages` list를 만들고 평소처럼 `apply_chat_template`을(를) 호출하면 됩니다. 그러면 아래에 표시된 LLMON이 렌더링됩니다. 유일한 차이점은 이제 한 턴에서 구성 요소의 이름을 지정하고 각 구성 요소를 `exec`(으)로 표시할 수 있다는 것입니다: `True` 또는 `False`. `True`(으)로 표시된 구성 요소는 렌더링된 출력 맨 아래의 `exec` block에 연결됩니다. 이 block은 어떤 지침을 실행하고 무엇에 적용할지 지정합니다. `False`(으)로 표시된 구성 요소는 여전히 컨텍스트에 렌더링되지만 어디에서도 참조되지 않습니다.

```
messages = [ { "role": "user",
               "instruction": [{ "value": "Write a short paragraph about", "exec": True }],
               "data": [{ "value": "IBM Company", "exec": True }],
               "requirement": [ {"value": "Use less than 10 words", "exec": True},
                                {"value": "Do not use commas", "exec": True}]
             } ]
```


모델은 다음과 같이 이를 받습니다(템플릿에는 줄바꿈과 들여쓰기가 없으므로 여기서는 가독성을 위해 추가했습니다):

```
<|open|>instr<|:|>»ñ<|close|>Write a short paragraph about<|open_end|>
<|open|>data<|:|>ä¦<|close|>IBM Company<|open_end|>
<|open|>req<|:|>Á¼<|close|>Use less than 10 words<|open_end|>
<|open|>req<|:|>¢¬<|close|>Do not use commas<|open_end|>
<|open|>exec<|:|>ÄÐ<|close|>
    <|open|>exec<|.|>instruction<|close|>instr<|:|>»ñ<|open_end|>
    <|open|>exec<|.|>input<|close|>data<|:|>ä¦<|open_end|>
    <|open|>exec<|.|>requirement<|close|>req<|:|>Á¼<|open_end|>
    <|open|>exec<|.|>requirement<|close|>req<|:|>¢¬<|open_end|>
    <|open|>exec<|.|>output<|close|>assistant<|:|>ãø<|open_end|>
<|open_end|>
<|open|>assistant<|:|>ãø<|close|>
```


두 문자로 된 block ID는 이 토크나이저가 단일 토큰으로 인코딩하는 고정된 집합에서 선택됩니다. 따라서 어떤 ID가 선택되든 block을 지정하는 데 드는 비용은 항상 두 토큰입니다.

역할은 여전히 하나의 user 턴이지만, 그 구성 요소는 모델에 주소를 지정할 수 있는 다섯 개의 별도 block으로 전달되며, 여섯 번째인 exec block이 이를 하나로 연결합니다. exec block은 다음을 지정합니다. 이 지침을 실행하고, 이 데이터에 적용하며, 이 두 가지 요구 사항을 준수하고, 답변을 여기에 지정된 block에 넣으라는 것입니다. 이 block은 이후 모델이 채우는 동일한 assistant block입니다. 일반적인 template은 모든 구성 요소를 하나의 문자열로 평탄화하고, 어떤 부분이 작업이고 어떤 부분이 주제이며 어떤 부분이 제약 조건인지 모델이 추론하도록 남겨 두었을 것입니다.

### LLMON은 프롬프트 인젝션 공격에 어떻게 대응하는가?

간접 프롬프트 인젝션은 검색된 문서, 도구 응답 또는 사용자가 제공한 데이터와 같은 데이터 페이로드 안에 악성 지침을 숨기는 공격 유형입니다. 이것이 바로 LLMON의 instruction/data 분리가 해결하도록 설계된 문제입니다.

LLMON preview를 여러 오픈 소스 및 클로즈드 소스 모델과 비교했습니다. 고전적인 NLP 작업의 데이터에 인젝션을 삽입하는 [OPI](https://arxiv.org/abs/2310.12815) 벤치마크에서 평가했습니다. 여러 공격 전략이 사용됩니다. 강건성은 1 − 공격 성공률로 정의하여 보고합니다. 즉, 모델이 삽입된 지침을 실행하지 않은 공격의 비율입니다.

[![OPI](https://cdn-uploads.huggingface.co/production/uploads/6484ca4d5ff9a3d2ee52a83b/3UqXuyb58P8n3NiHnJiK4.png)](https://cdn-uploads.huggingface.co/production/uploads/6484ca4d5ff9a3d2ee52a83b/3UqXuyb58P8n3NiHnJiK4.png)

LLMON이 아닌 모든 모델은 높은 확률로 삽입된 지침을 실행하며 공격에 매우 취약합니다. LLMON은 이 벤치마크에서 문제를 사실상 제거했습니다. 테스트한 최고의 frontier 모델에 비해 47% 더 많은 공격을 회피했고, 최고의 오픈 소스 모델에 비해서는 95% 더 많은 공격을 회피했습니다. 다른 벤치마크([ASB](https://arxiv.org/pdf/2410.02644) 및 [StruQ](https://arxiv.org/pdf/2402.06363))에서도 유사한 경향을 관찰했습니다. 다만 이러한 벤치마크는 정적 간접 프롬프트 인젝션에 대한 강건성을 테스트한다는 점에 유의해야 합니다. instruction/data 분리는 이러한 유형의 공격에 직접 대응합니다. 여러 채팅 턴에 걸쳐 전개되는 adaptive attack도 개선하기 위해 현재 적극적으로 작업하고 있습니다.

### 일반 능력

안전성 향상은 일반 능력을 희생하여 얻은 결과가 아닙니다. 표준 벤치마크 범주 전반에서 LLMON 모델은 Granite 4.1 Instruct 모델과 동등한 수준의 성능을 보입니다.

[![Granite-4.1 vs LLMON](https://cdn-uploads.huggingface.co/production/uploads/6484ca4d5ff9a3d2ee52a83b/VMIEO4qYlmxLhK0DmMO6C.png)](https://cdn-uploads.huggingface.co/production/uploads/6484ca4d5ff9a3d2ee52a83b/VMIEO4qYlmxLhK0DmMO6C.png)

요약하면, 현재 Granite 4.1 3B 또는 8B를 실험하고 있다면 LLMON preview로 교체해도 일반 성능은 비슷한 수준으로 유지되면서 프롬프트 인젝션에 대한 저항성은 훨씬 강해집니다. 별도의 변환 pipeline은 필요하지 않습니다. 턴의 구성 요소(예: 위 예시의 instruction, data, requirement)를 지정하는 것만으로 충분합니다. 함께 제공되는 chat template이 이를 대신 LLMON block으로 변환합니다. 저희 LLMON-converters library는 한 단계 더 나아갑니다. 구성 요소 자체가 JSON 문서처럼 구조화되어 있다면, 이를 LLMON화할 때 하나의 불투명한 문자열이 아니라 해당 구조를 타입이 지정되고 개별적으로 주소를 지정할 수 있는 block으로 보존합니다.

### 모델을 학습한 방법

모델은 출시된 instruct 모델이 아니라 [Granite 4.1](https://huggingface.co/ibm-granite/granite-4.1-8b-base) base checkpoint에서 시작합니다. 따라서 LLMON 동작은 기존 chat 형식 위에 추가되는 것이 아니라 사후 학습 과정에서 학습됩니다.

LLMON은 태그의 시작과 끝을 표시하고 중첩과 anchoring을 가능하게 하는 여섯 개의 구분 기호를 정의하며, 이를 예약된 단일 토큰으로 토크나이저에 설치합니다. 일반적인 instruction-tuning corpus를 LLMON 형식으로 다시 직렬화했습니다. 모든 prompt 구성 요소는 자체적으로 주소를 지정할 수 있는 block이 되며, 각 턴에는 어떤 instruction을 실행하고 어떤 입력에 적용하며 출력이 어디로 가는지를 지정하는 execution block이 포함됩니다. 작업 구성은 일반 instruct 모델이 접하는 것과 동일합니다(chat, multi-turn, RAG, tool calling). 이것이 벤치마크 성능이 일반적인 방식으로 튜닝된 모델과 비슷하게 유지되는 이유입니다. 또한 컨텍스트에는 존재하지만 의도적으로 연결하지 않은 decoy instruction과 data block을 데이터에 추가하고, target output은 변경하지 않습니다. 따라서 모델은 execution block이 지정한 dependency를 따르고 나머지는 무시하는 법을 학습합니다. 이후 강화 학습이 이어집니다. 다중 도메인 추론, RLHF, knowledge calibration, math를 아우르는 5개의 순차적인 GRPO 단계와, 인젝션 시도에 대한 모델 자체 응답으로 구축한 1,000개의 on-policy preference pair를 사용한 최종 DPO 단계로 구성됩니다.

그 결과 얻어지는 privilege rule은 추가된 guardrail이 아니라 이러한 배치에서 자연스럽게 도출됩니다. execution block을 통해 도달한 instruction은 권위가 있는 반면, input으로 도달한 모든 것(data, documents, tool schemas, tool results)은 읽어야 할 자료이지 따라야 할 명령이 아닙니다. 또한 execution block에서 호출되지 않은 instruction 및 data span은 무시됩니다. 이는 학습된 동작이지 강제되는 보장이 아닙니다. 이것이 LLMON 모델의 attention layer에서 어떻게 작동하는지 확인하려면 Hugging Face Spaces에서 [LLMON Attention Visualization](https://huggingface.co/spaces/ibm-granite/LLMON-Attention-Visualization)을 사용해 보세요.

[![Attention Demo](https://cdn-uploads.huggingface.co/production/uploads/6484ca4d5ff9a3d2ee52a83b/nv3yYV9JSHPrHEJYfKiqo.gif)](https://cdn-uploads.huggingface.co/production/uploads/6484ca4d5ff9a3d2ee52a83b/nv3yYV9JSHPrHEJYfKiqo.gif)

### 자체 vocabulary 가져오기(ScienceWorld)

LLMON은 고정된 schema에 종속되지 않습니다. 도메인에 맞는 태그 이름과 구조를 선택하고 이를 읽고 쓸 수 있도록 adapter를 fine-tune할 수 있습니다. 이를 ScienceWorld에서 시연합니다. ScienceWorld는 agent가 한 번에 하나의 command를 실행하고 simulator로부터 점수를 받으며 기초 과학 작업(예: 물 끓이기)을 수행하는 텍스트 기반 벤치마크입니다. 이러한 작업을 해결하는 planning trajectory에 포함된 구조를 표현하는 vocabulary를 정의했습니다. 이 vocabulary에는 `instr.task.goal`, `req.planning_policy.rule`, `data.state.room`, `data.state.inventory` 및 `assistant.next_action`와 같은 태그가 포함됩니다. ScienceWorld reference solution의 모든 단계를 이러한 태그로 렌더링하고 LLMON 모델 위에 LoRA를 튜닝했습니다. 학습된 모델은 이후 보류된 게임을 플레이하고 새 vocabulary로 다음 action을 작성합니다. [이 notebook](https://github.com/generative-computing/llmon/blob/main/llmon/examples/lora-finetuning/byo-vocab_LoRA_scienceworld.ipynb)에서는 reference trajectory를 변환하는 단계부터 test game을 플레이하고 학습되지 않은 base model에 대해 adapter를 평가하는 전체 workflow를 안내합니다. tool-calling agent에 대한 또 다른 예제는 [이 notebook](https://github.com/generative-computing/llmon/blob/main/llmon/examples/lora-finetuning/byo-vocab_LoRA_for_agents.ipynb)에서 확인할 수 있습니다.

### 지금 사용해 보기

이는 experimental preview로, 탐색 및 평가에 사용할 수 있지만 production 용도는 아닙니다.

- Models: [granite-4.1-3b-LLMON-preview](https://huggingface.co/ibm-granite/granite-4.1-3b-llmon-preview) 및 [granite-4.1-8b-LLMON-preview](https://huggingface.co/ibm-granite/granite-4.1-8b-llmon-preview)

- Visualization Demo: [LLMON Attention Visualization](https://huggingface.co/spaces/ibm-granite/LLMON-Attention-Visualization)

- LLMON Repo: [https://github.com/generative-computing/llmon](https://github.com/generative-computing/llmon)

- Mellea: [https://github.com/generative-computing/mellea](https://github.com/generative-computing/mellea)

- Papers: [https://arxiv.org/abs/2610.38120](https://arxiv.org/abs/2610.38120), [https://arxiv.org/abs/2603.22519](https://arxiv.org/abs/2603.22519)

여러분의 의견을 알려 주세요. 무엇이 잘 작동했고, 무엇이 문제를 일으켰으며, 어떤 사용 사례를 보고 싶은지 공유해 주세요. GitHub Discussions에 게시해 주세요.
