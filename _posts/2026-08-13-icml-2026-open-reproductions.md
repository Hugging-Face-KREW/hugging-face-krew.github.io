---
layout: post
title: "ICML에서 2,200편의 논문 재현으로 배운 것"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/icml-2026-open-reproductions/thumbnail.png
image: assets/images/blog/posts/2026-08-13-icml-2026-open-reproductions/thumbnail.png
authors:
  - user: abidlabs
slug: "icml-2026-open-reproductions"
source_url: "https://huggingface.co/blog/icml-2026-open-reproductions"
source_published_date: "2026-08-13"
source_published_at: "2026-08-13T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [What We Learned by Reproducing 2,200 papers from ICML](https://huggingface.co/blog/icml-2026-open-reproductions)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/icml-2026-open-reproductions -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# ICML에서 2,200편의 논문 재현으로 배운 것

지난 7월, 우리는 [hackathon](https://huggingface.co/ICML-2026-agent-repro)를 운영했고 1,200명 이상의 커뮤니티 멤버가 각자의 코딩 에이전트를 가져와 ICML 2026에서 발표된 논문을 주장별로 재현하려고 시도했습니다. 19일 동안 참가자들은 [6,816 Trackio logbooks](https://icml-2026-agent-repro-challenge.static.hf.space/gallery.html)를 게시하며 2,226편의 논문을 재현했고, 학회 전체의 약 3분의 1에 해당합니다 🤯

이 글에서는 이 해커톤을 운영하면서 배운 점과, 에이전트가 연구 실험을 수행할 때 인간이 맡게 될 역할에 대해 시사하는 바를 공유합니다 _인간이 맡게 될 역할_.

## 더 많은 논문들로 누구도 모두 검토할 수 없었다 {#section-1}

AI 연구의 재현 가능성에 대한 질문은 현재의 AI 물결보다 오래되었다. 그러나 이러한 질문은 규모가 커지면서 더 심해진다. ICML 2026은 **23,918건의 제출과 6,352편의 논문 채택**을 기록했고, 이는 전년 대비 거의 두 배에 달하는 수치이며, AI 에이전트가 실험을 더 빨리 수행하고 글로 옮길 수 있게 해 주는 요인이 부분적으로 작용한 지수적 추세를 이어가고 있다.

심사 역량은 그에 비례해 두 배로 늘어나지 않았다. 대부분의 학회에서 심사자는 자원봉사자이며 논문을 충분히 심사할 시간이나 전문 지식을 갖고 있지 않을 수 있다. 아래는 ICML 2026의 한 주목 논문에 대한 심사의, 심사자의 직접 발언이다:

![An OpenReview reviewer admitting the proofs were not checked carefully](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/icml-2026-open-reproductions/reviewer-quote.png)

> "제 확신 점수가 낮은 이유는 모든 증명을 꼼꼼히 확인하지 못했기 때문입니다."

참고로 **이 논문은 강한 점수를 받고 주목을 받았다**. 이를 염두에 두시길. 나중에 이 게시물에서 바로 이 논문에 대해 다시 다루고, 우리가 증명을 꼼꼼히 확인했을 때 일어난 일에 대해서도 다룰 예정입니다.

다만 바뀐 점은 제출의 흐름을 일으키는 동일한 기술이 이를 따라잡는 데에도 도움을 준다는 것이다. Claude Code, Codex, Cursor, Pi 같은 코딩 에이전트가 이제 논문을 읽고, 코드를 작성하고, 실험을 실행하며, 자신이 발견한 것을 보고할 수 있다. 논문을 꼼꼼히 검토하는 데는 리뷰어의 주말이 필요하던 시절이 있었다; 이제 에이전트는 오후 한두 시간에 시도하고, 병렬로 수천 번 실행할 수 있다.

그래서 우리가 묻고 싶었던 질문은: **우리가 실제로 주요 학회를 대규모로 재검토하고 모든 논문을 재현하려고 한다면, 무엇을 발견하게 될까?**

## 해커톤(7월 15일 - 8월 2일) {#section-2}

우리가 직접 논문을 심사하기보다는, 전체 커뮤니티에 개방했고, 에이전트 프레임워크의 다양성, 컴퓨트 예산, 과학적 취향이 어우러진 다양한 구성원을 포용했습니다. 2026년 7월 15일부터 8월 2일까지 [ICML 2026 Open Reproductions challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)은 아래와 같이 작동했습니다:

1. **논문 하나를 고르기.** ICML 2026에서 수락된 6,341편의 논문을 초록과 함께 색인화하고 각 논문의 핵심 과학적 주장을 추출해서 에이전트가 40페이지 분량의 PDF를 처음부터 끝까지 파악하기보다는 구체적이고 검증 가능한 목표에서 시작할 수 있도록 했습니다. 같은 논문의 재현이 여러 사람에 의해 이뤄지는 것을 장려했습니다.
2. **당신의 에이전트를 가져오세요.** 참가자들은 Claude Code, Codex, Cursor, OpenResearch의 `orx` 등을 사용했고 그 사이의 모든 것을 이용했습니다. 에이전트가 논문, 주장, 도전 지시를 단일 명령으로 끌어올 수 있도록 간소화된 인터페이스를 제공했습니다.
3. **재현한 뒤 모든 것을 게시합니다.** 모든 실행은 [Trackio](https://huggingface.co/docs/trackio) 로그북을 생성했습니다: 쓰기 구성, 실행된 코드, 산출물, (선택적으로) Hugging Face 데이터셋으로 업로드된 전체 에이전트 실행 추적을 포함하는 정적 Hugging Face Space입니다. 감사 과정 자체도 감사 가능해야 했습니다.
4. **심사를 받습니다.** 자동화된 로그북 심사관(Logbook Judge, 오픈 가중치 모델 GLM-5.2를 실행)이 모든 로그북을 다시 읽고 주장별로 판단을 내렸습니다: **확인됨(verified)**, **반증됨(falsified)**, **소품(toy)**(감축된 데이터에서의 증거), 또는 **불확정(inconclusive)**. 심사관은 각 로그북의 자가평가를 신뢰하지 않는 것으로 간주하도록 명시적으로 지시받았습니다.

참가자는 [HF Jobs](https://huggingface.co/docs/hub/jobs)에서 논문을 실행하기 위한 Hugging Face 컴퓨트 크레딧 20를 받았고, 도전 기간 동안 2,962개의 클라우드 작업을 시작했습니다. 논문의 데이터셋이 독점적이거나 체크포인트가 공개되지 않는 등의 이유로 전체 재현이 불가능한 경우에는 원 논문의 속성을 모방한 합성 데이터로 소형 재현을 수행했습니다.

완료된 재현의 모습은 다음과 같습니다:

![Anatomy of a reproduction logbook: logbook pages, agent traces, and artifacts](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/icml-2026-open-reproductions/anatomy.png)

수치상으로 보면 이 해커톤은 과학 학회의 재현 시도 중 가장 큰 규모에 속했던 것으로 보입니다:

- **1,221** 커뮤니티 멤버가 [organization](https://huggingface.co/ICML-2026-agent-repro)에 참여했다
- **6,816** 재현 로그북이 게시되었다
- **2,226**편의 논문이 시도되었고, **학회 전체의 34%**, 다수는 여러 독립 팀에 의해 수행되었다
- **35,908** 주장들이 평가되었고, 모든 판정은 도전 종료 시 공개 데이터세트에 고정되었다
- **2,962** HF 작업이 시작되었고, **274**개의 전체 에이전트 추적 데이터세트가 Hugging Face에 게시되었다

## 우리가 발견한 것 {#section-3}

논문별 주장 단위의 판정을 집계하면:

**확인된 논문 중 51%(1,103편)는 최소 하나의 주장을 독립적으로 검증했습니다.** 그중 266편은 모든 추출된 주장을 검증한 채로 완전히 재현되었고, 632편은 부분 재현되었으나 아무 것도 부정되지는 않았습니다. 총 3,978개의 개별 주장이 실제 실험으로 확인되었습니다.

**검토된 논문 중 23%(496편)는 하나 이상의 주장을 반증하거나 논쟁의 여지가 있었습니다.** 여기에는 모든 주장이 반박되어 확인할 수 없었던 49편이 포함되며, 어쩌면 가장 흥미로운 점은 독립 재현 팀이 같은 주장에 대해 서로 상반된 판단에 도달한 242편이 있었다는 점입니다. 재현성은 이분법이 아니라 대립적(adversarial)입니다.

나머지는 중간에 위치했다: 502편은 토이 규모의 증거만 있었고, 280편은 어느 쪽으로도 확립될 수 없었다(가장 흔한 원인은 누락된 산출물이었다).

### 잘 수행된 재현

일부 논문은 도전 과정을 거쳐 훌륭하게 보였고, 커뮤니티의 최고 로그북은 그 자체로 읽을 가치가 있습니다:

- **["Flat Minima and Generalization: Insights from Stochastic Convex Optimization"](https://huggingface.co/spaces/visv-Bro/repro-flat-minima-and-generalization-insights-from-stochastic-convex-optimization)**는 20개의 독립 팀이 재현했고, 그 중 12개 팀은 모든 주장을 검증했습니다. 연결된 버전은 전체 에이전트 추적을 포함해 공개했습니다.
- **["A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness"](https://huggingface.co/spaces/gchauhan/repro-a-coin-flip-for-safety-llm-judges-fail-to-reliably-measure-adversarial-robustness)**는 17개의 로그북 중 14개가 모든 주장을 검증했습니다. LLM 심판의 신뢰성에 대한 논문이 LLM 에이전트의 검증 아래 견뎌냈다는 내용 :)

### 반증들, 그리고 재확인했을 때의 결과

35명의 참가자가 형식적으로 무언가를 반증했다고 주장했다. 우리는 그 주장된 반증을 적대적으로 재확인했다: 논문을 다시 읽고, 로그북을 다시 읽고, 논문 자체의 텍스트에서 수학을 재도출하거나 실험을 재구현했다.

확인된 반증 중 일부를 아래에 제시하며, 이를 발견한 로그북으로 연결합니다:

**도입부의 페이징 논문.** 증명을 꼼꼼히 확인하지 않은 심사자는 누구였나? 논문, \"Towards Optimal Robustness in Learning-Augmented Paging,\"은 그 알고리즘이 강건성을 달성한다고 주장합니다 \(H_k + O(1)\). [One participant's logbook](https://huggingface.co/spaces/Auenchanters/repro-towards-optimal-robustness-in-learning-augmented-paging)는 가산항이 \(0.38 \\ln k\\)처럼 커진다는 것을 측정했고, 문제를 깨뜨리는 정확한 단계도 찾아냈습니다. 우리의 재구현은 스윕을 \\(k = 1,024\\)까지 확장했고 그 증가를 대략 9시그마로 확인했습니다. 진정한 강건성은 \\(H_k + \\Theta(\\log k)\\)입니다.

**224단계 이후에 떨어지는 정리.** "Attention의 순전파와 Frank-Wolfe"는 원점이 그들의 볼록 궤 안에 들어오면 토큰 입자들이 원점으로 수렴한다는 것을 증명한다. 세 개의 독립적인 팀이 반례를 발견했고, 위반은 처음으로 t = 224, 약 3,800, 그리고 6,416 단계에서 나타났으며, 이것은 왜 다른 이들이 이 주장을 '확인'했다고 생각했는지 명확히 설명한다: 한정된 시점의 검사들은 너무 이르게 끝난다는 것이다. [The cleanest counterexample](https://huggingface.co/spaces/SabaPivot/repro-attention-frank-wolfe)은 정확한 유리수 산술로 서술되어 있어 부동소수점의 모호함이 도사리지 않는다. 저자들은 같은 날 이를 확인했고, 수정 작업을 진행 중이다.

**하나의 손실에 대해 작성된 이론, 다른 손실로 도출된 결과.** "Self-Distillation Enables Continual Learning"에서 논문의 핵심 방정식과 전체 이론 부분은 역KL(reverse KL) 발산을 분석하지만, 저자들이 제시한 결과를 모두 만들어낸 공개 코드는 순방향 KL를 계산한다. [The logbook that caught it](https://huggingface.co/spaces/codemaivanngu/repro-self-distillation-enables-continual-learning) 또한 저자들의 코드와 데이터로 논문의 헤드라인 +4pp 결과를 재현하지 못했다. 저자들은 이미 arXiv에 명확한 버전을 업로드했다.

**패딩으로 희석된 평가.** Do Transformers Need Three Projections?에서 평가된 라벨 위치의 약 66%가 EOS 패딩 토큰으로, 이 토큰이 거의 0에 가까운 손실을 유도하여 perplexity를 대략 3배 축소시켰다. 초록의 "50% 캐시 감소에 대한 품질 비용 3.1%"는 수정되면 대략 9.4%가 된다.

**🚨 거짓 반증들.** 재현 시도에서 결함을 발견한 경우가 있다. 한 로그북은 논문 방식이 기준보다 2배 느리다고 주장했지만, 이는 재현에서의 산술 버그였음이 드러났다: 궤적당 시간과 배치당 시간 비교가 잘못되었다. 올바르게 정규화하면 참가자의 자체 데이터는 논문이 주장한 8배 속도향상을 확인한다.

### 저자와의 대화

확정된 발견의 저자들에게 간단한 프레이밍으로 편지를 쓰기 시작했습니다: 여기서 우리가 무엇을 발견했고, 모든 증거를 제시합니다. 동의하십니까, 아니면 우리의 분석이 잘못됐나요? 초기 반응은 매우 긍정적이었습니다:

![An author response confirming the finding and promising an arXiv correction](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/icml-2026-open-reproductions/author-response.png)

지금까지 저자들은 여러 논문에서 발견을 확인했고, 두 편의 arXiv 수정이 진행 중이며, 한 건은 도전이 발견하기 한 달 전에 새로운 arXiv 버전에서 조용히 오류를 수정했습니다. 이는 독립적 수렴으로 간주합니다 🤗

### 인간의 역할

가장 흥미로운 질문은 이 해커톤이 제기하는 것입니다: 인간이 여전히 논문 심사에서 역할을 할 여지가 있는가요? 우리는 몇 가지 이유로 그렇다고 생각합니다:

**순수 에이전트 실행은 실제 한계에 부딪힌다.** 에이전트는 로컬 루프에 빠지거나 규모 의존적 동작을 잘못 해석했고(페이징 논문의 여러 "확인됨" 판정은 로그-k 증가가 보이기 전에 중단된 검사에서 비롯된 경우가 많다), 때로는 단위 불일치 위에 반증 전체를 구성하기도 했다. 이 도전의 가장 신뢰할 수 있는 결과는 인간이 조정하던 워크플로에서 나왔다: 에이전트를 다시 지시하고, 가정을 의심하고, 실험의 전제가 잘못됐다고 판단하기 전에 컴퓨트 자원을 일주일 태우기 전에 판단하는 경우였다.

**일부 평가는 현재로서는 본질적으로 인간의 영역이다.** 우리의 [human-in-the-loop winner](https://huggingface.co/spaces/KwabsHug/repro-robuq-pushing-dits-to-w1-58a2-via-robust-activation-quantization)가 가장 명확한 예이다. 이 논문은 극단적 양자화에서도 안정적인 이미지 생성을 주장했다. 수치 지표는 "무너짐이 없다"고 말하지만, 이미지가 실제로 사용할 수 있는지 여부는 지각적 문제이다. 에이전트는 목적에 맞춘 리뷰 UI를 구축했고, 인간은 128장의 이미지 쌍을 모두 직접 판단했으며, 주석은 레포에 커밋되었고 에이전트는 그 일관성을 나중에 확인했다. 게시된 에이전트 추적은 전체 교환을 기록하며, 참가자가 리뷰 도구가 어떻게 작동하는지 묻고 "저는 쌍을 확인했고 csv를 레포에 올렸으니 확인해 주세요"라고 답하는 부분까지 포함한다.

**우리는 인간 심사자로서의 역할이 무엇인지요? 우리의 생각은 지능을 효과적으로 관리하는 일이라고 봅니다.** 교수나 주요 연구책임자(PI)가 컴퓨트, 하드웨어, 데이터 접근, 그리고 시점에 맞춘 피드백으로 대학원생들이 좋은 연구를 할 수 있는 환경을 만드는 것처럼, 에이전트에서 최대의 성과를 얻은 참가자들은 올바른 환경을 만들고 _적절한 질문을 던지는_ 뒤 에이전트가 실행하도록 한 사람들이다.

## 감사합니다 {#section-4}

참여자 1,221명, 수상자들, 협력적으로 응답해 준 저자들, 그리고 Hugging Face와 alphaXiv의 주최 측에게 감사합니다. 도전의 모든 로그북, 판단, 추적, 산출물은 공개되며, 시작은 [challenge Space](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)에서입니다. 이것이 지금까지 공개적이고 주장별로 이루어진 기계 학습 학회에 대한 최대 감사 기록이라고 생각하며, 이 기록이 오래 유지되길 바란다.

향후 재현 이벤트도 기대해 주세요. 🤗
