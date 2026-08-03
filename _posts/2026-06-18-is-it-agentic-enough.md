---
layout: post
title: "충분히 에이전트적인가요? 자체 도구로 오픈 모델 벤치마킹하기"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-06-18-is-it-agentic-enough/thumbnail.png
authors:
  - user: lysandre
  - user: SaylorTwift
  - user: pcuenq
slug: "is-it-agentic-enough"
source_url: "https://huggingface.co/blog/is-it-agentic-enough"
source_published_date: "2026-06-18"
source_published_at: "2026-06-18T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Is it agentic enough? Benchmarking open models on your own tooling](https://huggingface.co/blog/is-it-agentic-enough)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/is-it-agentic-enough -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 충분히 에이전트적인가요? 자체 도구로 오픈 모델 벤치마킹하기

![Benchmarking transformers revisions across different metrics](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_6.png)

<p align="center">
  <em>Benchmarking transformers revisions across different metrics</em>
</p>
<br/>

> 이것은 인간이 만든, 에이전트 중심의 블로그 글입니다.

코딩 에이전트가 점점 우리 대신 우리의 소프트웨어를 다루고 있다: 작업을 설명하면 에이전트가 라이브러리를 선택하고,
호출들을 작성하고 실행하며 자신의 실수를 디버그한다. 라이브러리가 길을 막을 때는 이를 기꺼이 우회하고
처음부터 로직을 다시 작성한다. 이것은 라이브러리 개발에 새로운 개념을 도입한다: 코드는 단지 정확하고 빠를 뿐만 아니라
에이전트가 이를 효과적으로 이끌 수 있도록 설계되어야 한다. 어색한 API나 오래된 문서는 개발자인 우리를 짜증나게 하지만,
이제는 에이전트가 더 길고 비용이 큰 경로로 이끌기도 합니다.

대부분의 벤치마크는 최종 답변만 봅니다. 우리는 그 전체 과정을 원했습니다: 에이전트가 정답을 얻었는지뿐 아니라 그것에 이르는 데 얼마나 많은 작업이 필요했고, 그것이 모델, 라이브러리 개정, 과제에 따라 어떻게 달라지는지 말입니다. 우리는 바로 그 목적을 `transformers`를 사례 연구로 정확히 측정했습니다.

여기서는 정답이 어떻게 발견되었는지에 초점을 맞춘 도구 특화 벤치마크를 도입하고, 하나의 해처스 중 하나의 간단한 구현을 제공합니다. 이는 [pi](https://www.npmjs.com/package/@mariozechner/pi-coding-agent) 코딩 에이전트가 구동하는 오픈 모델에서 완전히 실행되며, 모델 × 개정 × 과제의 전체 범위를 [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)를 가로질러 확장하여 모든 실행이 동일한 하드웨어를 보게 합니다.

<br/>
<p align="center">
    But, <b><i>how do you optimize software for agents?</i></b>
</p>
<br/>

다음 두 가지 소프트웨어 원칙을 강하게 믿습니다:

- 테스트되지 않았다면 작동하지 않습니다
- 문서화되지 않았다면 존재하지 않는 것입니다

에이전트적 최적화 도구의 영역에서도 이것은 여전히 동일하며, 이번에는 두 가지가 서로 직접 연결되어 있습니다.

에이전트가 사용할 도구가 존재하기를 원한다면, 그것은 발견 가능해야 한다. API는 명확해야 하고 문서는 광범위해야 한다. 유용한 파일과 예제에 에이전트가 빠르게 접근할 수 있도록 구조화되어야 한다. 도구가 에이전트에게 작동하기를 원한다면, 에이전트적 사용을 위해 테스트해야 한다.

## 에이전트적 사용을 위한 소프트웨어 테스트 {#section-1}

본 블로그 포스트 전반에 걸쳐 `transformers`를 예로 사용할 것이다: 이를 *사용하는* 에이전트가 ML 작업을 해결하도록 하는 것이며, 코드를 기여하지는 않지만, 해처스는 명령줄에서 작동할 수 있는 어떤 도구와도 작동하도록 설계되었다.

`transformers`에 대한 우리의 직관은 몇 가지 변경으로 사용성을 극적으로 단순화할 수 있다는 것이었습니다: CLI, 스킬, 그리고 독립적이며 작업별 예제들. 이는 최근 [`hf` CLI, redesigned to be agent-optimized](https://huggingface.co/blog/hf-cli-for-agents)에 적용된 것과 같은 조리법이며, 그곳에서 에이전트는 토큰을 1.3–1.8배(최대 6배) 더 적게 사용했습니다. 우리는 그러한 승리가 일반화될 수 있는지, 그리고 트랜스포머에도 유용할 수 있는지 알고 싶었습니다.

직관은 강력한 도구이지만, `transformers` 같은 널리 사용되는 코드베이스에 수천 줄의 코드를 추가하는 PR을 열기 전에 더 많은 증거가 필요했습니다. 우리가 측정하려고 한 것은 성공이 어떤 모습인지였습니다.

### 모든 성공이 동등하지는 않다

두 에이전트가 감정 분석 작업에서 올바른 레이블을 생성할 수 있지만, 하나는 다음과 같습니다:

- 40줄짜리 파이썬 스크립트를 작성하고, `transformers`를 임포트하고, 모양 에러를 디버그한 뒤,
  두 번 재실행하고, 마침내 정답을 출력한다;

반면에

- `transformers classify --model ... --text "..."`를 입력하고 한 번의 호출로 끝난다.

두 경로 모두 `POSITIVE (0.9999)`에 도달했고, 이 특정 작업에서 에이전트가 실제로 취한 두 경로는 다음과 같습니다:

```diff
# Task: classify the sentiment of "I absolutely loved the movie, it was fantastic!"

- # one agent: pipe a script into python and parse the output
- python - <<'PY'
- from transformers import AutoTokenizer, AutoModelForSequenceClassification
- import torch
- import torch.nn.functional as F
-
- model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
- tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
- inputs = tokenizer("I absolutely loved the movie, it was fantastic!", return_tensors="pt")
- with torch.no_grad():
-     logits = model(**inputs).logits
- probs = F.softmax(logits, dim=1)
- idx = torch.argmax(probs, dim=1).item()
- print(model.config.id2label[idx], probs[0][idx].item())
- PY

+ # the other agent: one command
+ transformers classify \
+   --model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
+   --text "I absolutely loved the movie, it was fantastic!"
```


두 방법 모두 같은 결과에 도달합니다. 그러나 **비용**, **지연 시간**, **토큰 사용**, 및 **실패** 측면에서 아주 다른 프로필을 보입니다.

평가가 최종 문자열만 확인한다면, 이러한 점들뿐 아니라 라이브러리에 적용한 변화(CLI 개선, 더 나은 오류 메시지, 스킬)가 에이전트에게 실제로 도움이 되었는지 역시 놓치게 됩니다.

이 해처스를 통해 우리의 목표는 에이전트가 주어진 과제를 수행하는 데 필요한 작업량을 평가하고, 라이브러리의 변경이 성능을 개선하는지 여부를 확인하는 것입니다.

### 평가를 어떻게 실행합니까?

여기에서 에이전트를 평가하는 방식에 대해 간단히 설명합니다.

우리는 모든 과제를 세 가지 변형(또는 '계층')으로 실행합니다; `transformers`에 에이전트가 접근하는 세 가지 서로 다른 방식:

```text
bare     pip install transformers, and nothing else
clone    the full transformers source, checked out in the working directory
skill    a packaged Skill: the CLI's docs + task examples, loaded in context
```


이 두 가지는 중첩되지 않습니다: `skill`은 `clone`를 포함하지 않습니다(선별된 문서만 제공하고 소스 트리는 포함하지 않음), 또한 서로를 엄밀히 포함하지도 않습니다. 각각은 에이전트에게 다른 유형의 도움을 제공합니다. 보시다시피 어떤 경우에는 `clone`에서 `skill`보다 더 잘 작동할 수 있습니다.

추가 선택 사항:

- 현재로서는 결정론적 작업에만 집중합니다. 이들은 정확한 매치를 제공하므로 실험하기에 아주 좋은 기반을 제공합니다. 모델-심판으로 삼는 방식과 다른 스킴은 다른 작업에 대한 명백한 차례의 다음 단계입니다.
- 모든 실행은 자체적인 허깅페이스 작업: (모델 × 개정 × 과제)당 하나씩이므로 전체 스윕이 동일한 하드웨어에서 병렬로 실행되어 규모에 맞는 비교를 공정하게 유지합니다.
- 결과와 추적은 허깅페이스 버킷에 저장됩니다: 빠르고 버전 관리가 필요 없으며, 매우 높은 쓰기 동시성을 처리합니다.

### 어떤 모델을 벤치마크 대상으로 삼을까요?

에이전트를 구동하는 모든 모델이 같지 않으며, 그 차이가 모델을 실행할 때 무엇을 살펴봐야 하는지 영향을 줍니다.

*대형 오픈 모델*

한쪽 끝에는 가장 크고 능력도 가장 뛰어난 오픈 모델들이 있습니다. 비교적 흔한 작업에서 이들은 결국 올바른 답을 얻어야 합니다. 이 경우 작업 완료율은 100%에 거의 도달하고 도구에 대해 더 이상 많은 정보를 제공하지 않습니다; 더 관련성 높은 벤치마크는 에이전트가 그 자리에 이르기까지 들인 노력입니다: 몇 차례의 대화, 토큰 수, 걸린 시간, 그리고 그들이 깨끗한 경로를 밟았는지 여부, 아니면 더 이상 사용되지 않는 API를 사용했는지 여부입니다.

*로컬*

로컬 모델은 크기가 크게 다르고 그 능력도 다릅니다. __"매칭 %"__ 같은 지표는 더 큰 상대 모델에 비해 더 관련성이 있으며, 이는 귀하의 특정 도구에서 모델의 크기/능력이 결과에 어떤 영향을 주는지 확인할 수 있기 때문입니다.

이 해처스는 에이전트 간의 상호작용에 대한 저장소 개선 방법에 대해 라이브러리 유지관리자에게 지침을 제공하는 데 그치지 않고, 사용자가 관심을 가지는 작업에서 서로 다른 에이전트와 모델이 어떻게 수행하는지 평가하는 데에도 도움을 줍니다.

해처스는 각 실행을 여러 축에서 점수를 매기므로, 모델 분류별로 실제로 무엇이 중요한지 물어볼 수 있습니다:

- **일치 %**: 최종 답변에 기대 결과가 포함되어 있었는지(과제별, 대소문자 구분 없는 서브스트링/정규식/정확 일치, 보고서에 모두 명시됨);
- **중간 시간** 및 **중간 토큰**(새로 생성된 토큰 / 캐시된 토큰 / 생성된 토큰);
- **에러 발생 비율**: 아무것도 출력하지 않는 실행을 표시하는 가드를 포함하여(출력 토큰 0, 도구 호출 없음, 답이 없음) 조용한 실패가 "0"으로 속이지 않도록 합니다;
- **마커 채택**: 도구 정의된 동작 마커; 아래에 이것이 무엇인지에 대한 설명이 있습니다.

모두가 직접 검토할 수 있는 보고서에 담깁니다:

<p align="center">
<iframe
	src="https://transformers-community-is-transformers-agentic.static.hf.space"
	frameborder="0"
	width="100%"
	height="900"
	style="max-width:100%"
></iframe>
<br>
  <em>The live report: Overview, Coverage, and Results, all client-side.</em>
</p>

또한 모든 실행의 네이티브 에이전트 추적을 캡처하기 때문에 숫자는 시작에 불과합니다: 에이전트가 명령마다 실제로 무엇을 했는지 정확히 읽을 수 있습니다. 이 추적은 Hub의 [agent-traces viewer](https://huggingface.co/docs/hub/agent-traces)를 통해 공유할 수 있습니다:

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_11.png" alt="A run rendered in the Hub's agent-traces viewer: MiniMax-M2.7 on the answer-question task" width="85%"><br>
  <em>A run rendered in the Hub's agent-traces viewer: MiniMax-M2.7 on the answer-question task.</em><br>
  <a href="https://huggingface.co/buckets/lysandre/transformers-agentic-use/tree/traces/22404f7951/pi/MiniMaxAI--MiniMax-M2.7/bare__answer-question__run1.jsonl"><b>Open this trace on the Hub ↗</b></a>
</p>

결과를 보기 전에 설정에 대한 간단한 요약입니다. 각 실행은 네 가지를 다릅니다: 에이전트를 구동하는 **모델**, 실행 대상인 **`transformers` 개정판**, **과제**, 그리고 **계층**( `bare` / `clone` / `skill` ). 다루었던 바와 같이, 두 가지 다른 모델 범주에 대해 서로 다른 지표를 살펴봅니다.

### 대형 오픈 모델: 모델은 유지하고 개정판은 바꾼다

대형 오픈 모델은 일반적으로 올바른 결과에 도달하므로, 실제로 측정하는 것은 그것을 달성하는 데 필요한 노력입니다. 열 번의 대화가 걸렸나요, 아니면 한 번이었나요? 구식 문서를 신뢰하여 폐기된 API 경로를 따랐나요? 예측하지 못한 오류를 만났나요?

자연스러운 실험은 하나의 강력한 모델을 고정하고 도구의 개정을 바꾸는 것입니다: 우리가 테스트하는 `transformers`의 연속된 Git 버전들로, 공개 태그들인 `v5.8.0` 및 `v5.9.0`에서 CLI와 스킬을 도입하는 특정 커밋까지. 에이전트에 가해지는 부하가 증가하는지 감소하는지 관찰하고자 합니다. `transformers`에서 해처스를 사용하여 전용 CLI와 스킬을 추가하는 것이 에이전트의 작업 부담을 실제로 줄이는지 확인했습니다.

세 가지 대형 모델에서 모든 작업에 소요된 평균 시간은 스킬 커밋이 과제를 수행하는 데 소비하는 시간이 더 짧아진다는 것을 시사합니다:

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_13.png" alt="Median time per revision, by tier" width="85%"><br>
  <em>Median time per revision, by tier: the skill commit (green dot) is the fastest.</em>
</p>

반면에 저장소를 복제한 실험에서는 CLI와 예제를 도입한 커밋으로 인해 토큰 소비가 크게 증가하는 것을 볼 수 있습니다. 이는 곧 확인하겠습니다.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_12.png" alt="Median new tokens per revision, by tier" width="85%"><br>
  <em>Median new tokens per revision, by tier: the clone variant jumps once the CLI lands in the repo.</em>
</p>

복제 변형의 추적을 읽어보면 이유를 설명합니다. 이 커밋은 명령을 추가하지만, 동시에 CLI의 구현과 `cli/agentic/*.py` 사용 예제 모음을 저장소에 직접 포함합니다.

`clone` 변형의 경우 에이전트 앞에 트랜스포머의 전체 체크아웃이 있고, 약 3분의 1 정도의 실행은 새로운 표면(`/cli/` 트리와 예제 스크립트)를 읽어 인터페이스를 배우고 호출하기 전에 이를 학습합니다. 이로 인해 입력의 중앙값이 약 4k에서 6.4k 토큰으로 증가합니다.

두 차트는 한 가지 트레이드오프의 양면입니다: 커밋은 대형 모델에 더 적은 시간을 주되(그들이 CLI를 찾고 디버깅 파이썬 대신), 더 많은 토큰을 필요로 합니다(그들이 CLI를 가르친 코드를 읽기 때문). PR을 병합하기 전에 알아두면 좋은 트레이드오프입니다.

다만 CLI에 유리하게 작용하는 한 가지 경고가 있는데, 아직 벤치마크되지 않았습니다: 읽는 비용은 후속 실행과 함께 상쇄됩니다. 우리의 설정은 일회성 실험에 맞춰져 있습니다. 각 실행은 CLI를 처음부터 다시 발견하는 새로운 에이전트이며, 따라서 매번 발견 비용을 부담합니다. 실제 사용에서는 에이전트가 인터페이스를 학습한 뒤 같은 세션에서 계속해서 과제를 해결하고 그 비용을 다수의 요청에 걸쳐 상쇄합니다. 여기에서 측정한 토큰 증가분은 일상의 사용자에게 보이는 것보다 최악의 경우에 더 가깝습니다.

### 소형 모델: 개정을 유지하고 모델을 바꾼다

오픈 모델은 이곳에서 가장 중요한 변수인 크기, 구성, 양자화, 추론 제공자, 학습 등 모델마다 다를 수 있는 요소를 아주 세밀하게 제어할 수 있게 해줍니다. 또한 도구 표면이 가장 큰 차이를 만드는 곳이기도 합니다: `bare` 환경에서 ' `transformers`를 사용해 X를 수행하라'고 요청받은 작은 모델은 몇 차례의 릴리스 전에 변경된 API를 추측해 불필요한 도구 호출을 할 수 있고, 잘못된 답을 낼 수 있습니다.

따라서 이 실험은 위와 정반대입니다: 개정판을 유지하고 모델을 스윕합니다. 이는 토큰 수와 시간뿐만 아니라 도구 호출을 신뢰성 있게 다룰 수 없는 모델까지 어떤 모델이 실제로 작업을 처리하는지 확인하는 데 도움이 됩니다. 우리의 직관은 모델이 작아질수록 도구 사용과 작업이 더 어려워진다는 것이며, 이를 정확히 시험하기 위해 다양한 크기의 모델들에 대해 해처스를 실행했습니다.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_14.png" alt="Match % across models, by tier" width="85%"><br>
  <em>Match % across models, by tier: the skill tier lifts the larger models but drops the smaller ones.</em>
</p>

이는 또한 투입된 토큰 수와 상관관계가 있는 것으로 보입니다.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_15.png" alt="Median new tokens across models, by tier" width="85%"><br>
  <em>Median new tokens across models, by tier.</em>
</p>

> 한 가지 공정한 비교에 대한 주의: 커버리지가 불균형할 때(빠르게 끝난 작업만 있는 모델은 빠르게 보임) 단순히 작업들 간의 평균을 내는 것은 오해를 불러일으킵니다. 보고서는 **"공유된 작업만"** 토글(모델 및/또는 개정판 간)과 **커버리지** 히트맵을 제공하므로, 같은 조건으로 비교하고 어떤 작업 × 개정판 × 모델 셀들이 실제로 실행되었는지 정확히 볼 수 있습니다.

## 도구 미세 조정: 마커와 결과 {#section-2}

두 가지가 여기에 함께 합쳐집니다: 에이전트가 성공했는지 여부를 넘어 그것이 무엇을 했는지, 그리고 그것을 어떻게 했는지 보는 방법; 또한 해처스에서 처음으로 뽑아낸 결과들에 대해서도.

### 마커란 무엇인가?

매칭 %, 토큰 수, 그리고 시간은 실행의 비용을 알려주지만, 내부에서 벌어지는 일을 많이 알려주지는 않습니다.

이것이 우리가 마커의 개념을 도입한 이유입니다. 마커는 프로필(해처스가 특정 라이브러리를 구성하고 구동하는 방법을 가르쳐 주는 도구별 플러그인)에 의해 실행과 매칭되는 이름 있는 패턴입니다.

에이전트가 실행한 셸 명령, 작성한 코드, 읽은 파일, 또는 최종 답변에 대해 관심 있는 동작에 대한 한 줄 레이블입니다. 한 실행은 여러 마커를 작동시킬 수도 있고 어떠한 마커도 없을 수도 있습니다; 보고서는 각 마커가 모델별, 개정판별로 얼마나 자주 발동했는지 보여줍니다.

`transformers`에 대해 몇 가지를 선언하지만, 가장 관련성이 높은 두 가지만 보겠습니다:

- **`cli`**: 에이전트가 `transformers` 명령줄 도구를 호출했고(Python을 작성하지 않음) 예: `transformers classify …`.
- **`pipeline`**: 고수준의 `pipeline(...)` Python API에 의지했습니다.

이것들이 변화가 실제로 에이전트의 행동을 바꿨는지 보는 지표들입니다. 흥미롭게도 여기서는 모델이 클수록 새로운 맥락을 더 많이 활용하고 기억을 덜 사용하며, 따라서 새로 도입된 CLI를 활용합니다.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_16.png" alt="CLI adoption by tier across models" width="85%"><br>
  <em>CLI adoption by tier across models: only the skill tier reaches for it, and more so as models grow.</em>
</p>

CLI 도입은 새롭습니다: CLI는 단일 커밋에 도입되었고 어떤 모델의 학습 데이터에도 포함되지 않았으며, 문서는 간략하게만 남아 있습니다. 효과는 명확합니다: CLI의 문서를 포함하는 스킬 변형이 실제로 그것을 활용하는 경향이 가장 뚜렷하며, 그 비율은 55.3%입니다.

### CLI + 스킬 커밋이 도움이 되고 있나요?

모델 크기별 커밋을 비교하면, CLI + 스킬이 더 큰 모델에 도움을 줍니다: `skill` 계층에서 Kimi와 다른 대형 에이전트들이 CLI를 이용하고 더 적은 수의 턴으로 완료합니다. (`clone`에서는 앞서 본 대로 새 CLI 코드를 읽느라 입력 토큰을 더 많이 소비하므로, 이 승리는 토큰 수가 아니라 시간과 턴에서 나타납니다.)

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_5.png" alt="Kimi-K2.6, GLM-5.1, and MiniMax-M2.7 across revisions" width="85%"><br>
  <em>Kimi-K2.6, GLM-5.1, and MiniMax-M2.7 across revisions</em>
</p>

그러나 일부 소형 모델 설정에서는 성능에 해를 입히는 것으로 보입니다. 그럴듯한 설명 중 하나는 작은 모델들이 기억된 API 패턴에 의존하고, 훈련 데이터에서 본 `pipeline(...)` 스니펫을 재현하기 때문입니다. 새 개념은 그러한 모델들이 잘못 이해할 수 있는 더 큰 표면을 제공합니다. 해처스에서 이를 직접 확인할 수 있습니다: 매칭 %이 낮아지고 재시도가 늘어나고, `cli` 마커는 거의 발동하지 않습니다. 특히 Qwen3-4B 모델에서 두드러지는데, 스킬은 매칭률을 거의 바꾸지 않으나 비용 분포에 상당한 영향을 미칩니다.

거의 모든 것은 `clone` 계층에서 발생합니다. 체크아웃에는 이제 CLI의 구현과 `cli/agentic/*.py` 예제가 포함되어 있으며, 4B 에이전트는 이를 대량으로 읽습니다: 중간 신규 토큰 수가 약 2.4k에서 23k로 증가하고, 시간과 출력도 급증하지만 정확도에는 이득이 없습니다.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_7.png" alt="Qwen3-4B cost distributions across revisions: elapsed, new tokens, repeat tokens, out tokens" width="85%"><br>
  <em>Qwen3-4B across revisions. The CLI + Skill commit fans the cost distribution wide open, on the <code>clone</code> tier the agent reads the newly-shipped CLI source in bulk (~10× the new tokens), for no gain in match %. (<code>repeat tokens</code> stays flat: this setup uses no prompt caching.)</em>
</p>

다만 간혹 스킬이 정확도를 완전히 깨뜨리는 경우도 있습니다. 트레이스를 읽어보면 예를 들어 Qwen3-14B의 경우: 스킬 추가로 전체 매칭률이 67%(기본)에서 43%로 떨어지며, 가장 간단한 작업에서는 그 붕괴가 매우 뚜렷하게 나타납니다: `classify-sentiment`은 `clone` 변형에서 100%에서 스킬을 사용하면 0%로 떨어집니다.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_17.png" alt="Qwen3-14B classify-sentiment match % by tier across revisions" width="85%"><br>
  <em>Qwen3-14B on <code>classify-sentiment</code>, by tier: <code>clone</code> (blue) holds at 100% across revisions, but the Skill variant (green) collapses to 0% at the CLI + Skill revision.</em>
</p>

트레이스를 보면, 모델이 CLI를 *직접 호출할 수 있는 도구*로 오해합니다(에이전트-해처스 도구와 같은, 예를 들어 웹 검색). 스킬은 실행 가능한 도구가 아닙니다: 에이전트의 맥락에 로드된 문서이며, `transformers` CLI는 오직 쉘에서 실행되도록 의도되었고( `bash`를 통해), 따라서 이 방법은 작동하지 않습니다.

Qwen3-14B는 스킬을 읽고, 56회의 스킬 실행 중 39회에서 `transformers(command="classify", ...)` 도구 호출을 하거나(등록되지 않은 도구), `read`/`bash`/`edit`/`write` 도구들 중 그와 같은 것을 찾지 못해, 모델을 실행할 수 없다고 여기고 포기합니다. 어쨌든 1줄짜리 `pipeline(...)`가 `clone` 체크아웃에서 100%를 기록했지만, 이 경우에는 작업을 불가능하다고 선언합니다.

<p align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/is-it-agentic-enough/img_10.png" alt="Qwen3-14B gives up on classify-sentiment under the Skill variant" width="85%"><br>
  <em>Qwen3-14B on classify-sentiment (Skill variant): it reasons that read/bash/edit/write can't run a model, and gives up.</em>
</p>

이런 현상은 우리가 해처스를 통해 포착하려 했던 바로 그것입니다: 대형 모델의 속도를 높여주는 동일한 변화가 소형 모델의 작업을 망가뜨리는 결과를 낳았고, 처음에는 우리에게 다소 직관에 어긋나는 것이었으며, 그대로 배포했을 수도 있었습니다. 유지관리자에게의 시사점: 에이전트-대상 API는 모델 크기에 따라 평가되어야 한다. 새로운 기능은 강력한 모델의 작업을 줄이고 작은 모델에는 모호성을 더할 수 있다. 또한 해결책에 대한 힌트를 제공합니다: 강한 모델에 대해 스킬을 수동으로 작성한 뒤 이를 검증하기보다는, 약한 모델에 대해 미리 하나를 생성하고 검증하는 방식이 앞당길 수 있습니다.

바로 이것이 [Upskill](https://huggingface.co/blog/upskill)가 하는 일입니다: 강력한 모델의 해결책을 작은 모델에 실제로 도움이 될 때만 스킬로 바꿉니다.

## 직접 해보기 {#section-3}

해처스는 단 하나의 CLI인 `agent-eval`입니다. 이를 설치하고, 한 묶음을 실행하고, 허깅페이스 작업에서 모델 × 개정에 걸쳐 확산시키고, 보고서를 허깅페이스 스페이스에 게시하십시오.

> [경고]
> **신뢰할 수 있는 로컬 사용 전용.** 이 해처스는 권한 우회를 통해 코드 에이전트를 실행하고, 당신이 가리키는 어떤 개정에서든 코드를 실행하며, 추적에는 프롬프트, 출력 및 로컬 경로가 포함될 수 있습니다. [SECURITY.md](https://github.com/huggingface/is-it-agentic-enough/blob/main/SECURITY.md)를 먼저 확인하십시오, 당신이 작성하지 않은 코드에 이를 지시하거나 결과를 공유하기 전에.

전체이고 최신의 설정 및 사용 지침은 [README](https://github.com/huggingface/is-it-agentic-enough)에 있습니다.

## 마무리 {#section-4}

최종 답을 확인하는 것은 에이전트가 귀하의 라이브러리를 사용할 수 있는지 여부를 알려주지만, 그것이 비용이 얼마나 되는지(대화 턴 수, 토큰, 오류, 그리고 그 경로)까지는 알려주지 않습니다. 이 해처스는 사용자가 선택한 개정판과 모델 전반에 걸쳐 이를 측정합니다.

`transformers`에서는 CLI + 스킬이 대형 오픈 모델에는 도움이 되고 소형 모델에는 해를 끼친다는 점을 확인했습니다. 합치기 전에 알아두면 좋습니다!

프로필 기반이며, 적응 가능하게 설계되었습니다: 자신의 라이브러리에 가리키고, 몇 가지 작업과 그 기대 답을 정의하면 같은 보고서를 얻습니다. 코드와 작업은 [repo](https://github.com/huggingface/is-it-agentic-enough)에 있으며, 추적은 허브(Hub)에 있습니다. 프로젝트에 사용하신다면 알려 주세요!

## 감사의 말씀 {#section-5}

이 해처스는 전적으로 [pi](https://www.npmjs.com/package/@mariozechner/pi-coding-agent)에 의존합니다. 마리오 제크너의 코딩-에이전트 CLI: 이는 모든 오픈 모델 실행을 구동하며, 모델을 서비스하려면 오직 `HF_TOKEN`가 필요할 뿐입니다. 이것이 바로 오픈 모델 스윕을 실제로 가능하게 만든 요인입니다.

우리가 훑어본 모델 뒤에 있는 모델 제작자들과 추론 제공자들에게 감사드립니다. 전반적으로 그들은 `bare` 기준선이 제시하는 것보다 훨씬 뛰어난 성능을 보였습니다.
