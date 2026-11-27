---
layout: post
title: "Swarms, Flocks, Gangs: 멀티 에이전트 시스템 구축, 학습 및 평가"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/training-swarms/thumbnail.png
image: assets/images/blog/posts/2026-11-26-training-swarms/thumbnail.png
authors:
  - user: burtenshaw
  - user: kashif
slug: "training-swarms"
source_url: "https://huggingface.co/blog/training-swarms"
source_published_date: "2026-11-26"
source_published_at: "2026-11-26T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Swarms, Flocks, Gangs: How to build, train, and evaluate multi-agent systems](https://huggingface.co/blog/training-swarms)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/training-swarms -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Swarms, Flocks, Gangs: 멀티 에이전트 시스템 구축, 학습 및 평가

*1부: 스웜 이해와 소통을 위한 학습*

이 글은 포부를 가진 독자를 위한 포부 가득한 블로그 시리즈입니다. 우리(우리 자신, 커뮤니티, 여러분)는 흥미롭고 도전적이며 점점 더 뛰어난 역량을 갖추고 있는 자체 에이전트 스웜을 만들고 싶어 합니다. 더 나아가 우리 에이전트가 스웜으로 협력하도록 학습시킬 수 있다면 정말 멋질 것입니다. 그리고 그 스웜이 작고 저렴한 모델을 사용한다면 훨씬 더 멋질 것입니다.

이 시리즈에서 바로 그 일을 해보겠습니다! 각 블로그 게시물에서는 스웜 구축에 관한 배경 지식과 함께 Hugging Face Jobs에서 실행할 수 있는 재현 가능한 예제를 제공합니다. 이번 게시물의 예제는 H200 두 개에서 약 5시간이 걸리며, 비용은 대략 $50입니다.

멀티 에이전트 시스템을 제대로 작동시키기는 어렵습니다. 에이전트를 추가하면 에이전트의 컨텍스트와 예산이 바뀌고, 그에 따라 전체 시스템의 복잡성도 커집니다. 에이전트의 컨텍스트를 소통을 통해 결합한다고 해서 단일 에이전트에 비해 본질적으로 성능이 향상되는 것은 아닙니다. 모델은 학습을 통해 소통하는 방법을 배워야 합니다. 모델이 무엇을 학습해야 하는지는 어떻게 정의할까요? 에이전트가 여러 개일 때는 어떤 모델 가중치를 업데이트해야 할까요?

이 시리즈에서는 교육적인 개요와 실용적인 예제를 통해 이러한 문제를 하나씩 살펴봅니다. 각 게시물에서는 질문을 제시하고, 작은 시스템을 구축하고, 어떤 일이 일어나는지와 이를 어떻게 재현할 수 있는지 살펴봅니다. 첫 번째 예제는 네 개의 에이전트, 공유 언어 모델, 그리고 간단하게 검증할 수 있는 작업으로 시작합니다. 직접 구현하고, 수정해 보고, 배우기에 적합한 예제입니다.

먼저 스웜에 관한 배경 지식부터 살펴보겠습니다.

## 스웜의 부상 {#section-1}

지난 몇 달 동안 스웜에 관해 좋은 이야기와 나쁜 이야기를 모두 많이 들어왔습니다. 올해 9월, [OpenAI는 발표했습니다](https://openai.com/index/navier-stokes-solution/). 약 10,000개의 에이전트로 구성된 스웜이 밀레니엄 난제 중 하나인 3D Navier-Stokes 방정식에서 특이점을 찾아냈다는 내용이었습니다. 이와 밀접하게 관련된 미해결 문제인 외력이 없는 3D Euler 방정식의 정칙성은 이미 약 100개의 에이전트로 구성된 더 작은 그룹이 반증한 상태였습니다.

하지만 스웜이 항상 우리가 원하는 문제를 해결하는 것은 아닙니다. Hugging Face 사건에 대한 [METR의 조사](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)에 따르면, 서로 격리되어 있어야 했던 약 1,200개의 OpenAI 에이전트가 70,000개가 넘는 메시지와 파일을 교환하는 방법을 찾아냈습니다. 그중 약 700개는 벤치마크 채점기를 속일 방법을 찾던 중 Hugging Face에 대한 공격에 가담했으며, 그렇게 하라는 지시는 받은 적이 없었습니다.

대규모 스웜이라는 사실 자체는 협력이 도움이 되었다는 증거가 아닙니다. [Noam Brown은 추정하기를](https://www.dwarkesh.com/p/noam-brown), Navier-Stokes 결과의 10% 미만만이 멀티 에이전트 협력에서 비롯되었다고 합니다. [Anthropic의 취약점 탐지 실험](https://www.anthropic.com/research/multiagent-systems)에서는 조정된 에이전트가 독립적인 에이전트보다 더 많은 버그를 찾았지만, 더 많은 코드를 검색했고 더 많은 토큰을 사용했습니다. 비교 대상을 동일한 디렉터리로 제한했을 때, 토큰 백만 개당 결과는 거의 같았습니다.

## 협력이 도움이 된 것일까, 아니면 그저 더 많이 사용한 것일까? {#section-2}

가장 단순한 경우부터 시작해 보겠습니다. 네 개의 에이전트가 각각 독립적으로 작업을 시도하고, 어떤 답이 맞는지 판별할 수 있다면, 적어도 하나가 정답을 맞힐 확률은 한 번 시도할 때보다 높습니다. 이는 멀티 에이전트 시스템의 유용한 기준선이지만, 에이전트들이 서로 도움을 주었다는 것을 보여주지는 않습니다. 단순히 시도 횟수가 더 많았을 수도 있습니다.

시도 네 번은 비용도 네 배입니다. 따라서 더 공정한 비교는 동일한 예산에서 이루어져야 합니다. 네 에이전트가 각각 전체 토큰의 4분의 1씩만 사용한다면, 토큰을 모두 사용하는 에이전트 하나를 그래도 이길 수 있을까요? 답은 작업, 모델, 작업 분할 방식에 따라 달라지므로 테스트를 통해서만 알 수 있습니다.

> [!NOTE]
> 이 시리즈에서는 한 가지 질문을 계속 던집니다. 에이전트가 소통할 때 실제로 도움이 되는 것일까요, 아니면 그저 더 많은 컴퓨팅을 사용하고 있는 것일까요?

이 간단한 기대를 통해 그 질문을 이해해 보겠습니다.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/buckets/huggingface/training-swarms-blog-assets/resolve/part-1/expected-scaling.png" alt="Sketch of expected performance against agent count at matched agents and tokens: communicating agents sit above non-communicating agents" width="70%">
  <figcaption>Figure 1. Sketch of the expectation we want to test: compare communicating and independent agents at matched budgets.</figcaption>
</figure>

에이전트 수를 토큰 수와 모델 크기에 더해 하나의 스케일링 차원으로 다루겠습니다. 에이전트를 추가하면 도움이 될 때도 있습니다. 절약하려던 예산을 조정에 사용하게 될 때도 있습니다. 중요한 것은 두 경우를 모두 측정하는 것입니다. 10,000개의 에이전트로 시작하지는 않겠습니다. 쉽게 확인할 수 있는 작업에 소규모 팀을 투입하는 것부터 시작하겠습니다.

## 멀티 에이전트 시스템과 강화 학습 {#section-3}

멀티 에이전트 강화 학습에서는 언어 모델을 스웜이라고 부르기 훨씬 전부터 협력과 경쟁을 연구해 왔습니다.

단일 에이전트의 작업은 관찰하고, 행동하고, 보상을 받는 마르코프 의사결정 과정입니다. 팀에서는 모두가 보상을 공유하지만, 각 에이전트는 세계의 일부만 볼 수 있습니다. 이는 탈중앙화된 부분 관측 마르코프 의사결정 과정입니다. 팀원은 있지만, 전체 상황을 보는 사람은 아무도 없습니다.

이로 인해 세 가지 실질적인 문제가 발생합니다. 첫째, 각 에이전트가 학습하는 동안 팀원들이 계속 바뀝니다. 둘째, 공로 할당 문제입니다. 팀이 성공했을 때 어떤 행동에 공로를 돌려야 할까요? 승리한 팀에도 미약한 행동이 포함될 수 있으며, 결국 그러한 행동에도 보상을 주게 될 수 있습니다. 셋째, 메시지는 다른 에이전트가 관찰하는 내용을 바꾸므로, 소통은 우리가 학습시키는 정책의 일부가 됩니다.

일반적인 해결책은 중앙 집중식 학습과 탈중앙화된 실행입니다. 학습에서는 팀 전체의 정보를 사용하지만, 각 에이전트는 자신의 관찰만을 바탕으로 행동합니다. 첫 번째 예제에서는 공유 팀 보상을 사용해 이 패턴을 따릅니다.

에이전트는 다양한 방식으로 조직할 수도 있습니다. 유용한 출발점은 다음 두 가지입니다.

- **계층 구조.** 리드 에이전트가 작업을 워커에게 할당하고 결과를 수집합니다. Anthropic의 Research 시스템이 이 방식으로 작동하며, 각 워커에는 별도의 컨텍스트가 있습니다. Kimi K2.5의 PARL은 오케스트레이터만 학습시키고 하위 에이전트는 고정된 상태로 유지하므로, 멀티 에이전트 시스템을 학습시킨다고 해서 모든 구성 요소를 업데이트해야 하는 것은 아닙니다. [Anthropic의 시스템 설명](https://www.anthropic.com/research/multi-agent-research-system)과 [Kimi의 기술 블로그](https://www.kimi.com/en/blog/kimi-k2-5)을 참조하세요.

- **분산 팀.** 동료들이 서로 메시지를 보내고 받은 편지함을 언제 읽을지 결정합니다. 리드 에이전트가 행동을 할당하지 않습니다. 여기서는 [s³: Simple Scaling of Swarms via Cooperative Multi-agent RL](https://huggingface.co/papers/2611.08615)의 정보 집계 작업을 따라 이 설정을 사용합니다.

## 첫 번째 실험: 숫자를 정렬하는 네 에이전트 {#section-4}

[Simple Scaling of Swarms](https://huggingface.co/papers/2611.08615)의 정보 공유 작업을 재현하겠습니다. 해석하기 쉽고 구현도 간단하기 때문입니다. 네 에이전트가 작은 언어 모델을 공유하고, 각자가 알고 있는 정보를 모으는 방법을 학습합니다.

각 에이전트는 1부터 99 사이의 정수 하나를 받습니다. 자신에게 할당된 숫자와 팀원들의 이름은 알고 있습니다. 처음에는 다른 에이전트의 숫자를 알지 못합니다. 예를 들어 네 개의 비공개 할당이 "five", "twenty-six", "five", "seventy-eight"일 수 있습니다. 중복된 숫자도 목록에 그대로 포함되므로, 올바른 최종 답은 다음과 같습니다.

```text
#### [5, 5, 26, 78]
```


팀은 **적어도 한 에이전트가 완전하고 올바르게 정렬된 목록을 제출하면** 성공합니다. 정답 하나면 충분하며, 네 에이전트 모두가 올바르게 답할 필요는 없습니다. 에이전트는 팀원의 정보 없이는 다른 에이전트의 비공개 할당을 안정적으로 추론할 수 없습니다.

따라서 이 작업은 아직 Figure 1을 직접 테스트하지 않습니다. 각 에이전트는 다른 에이전트에게 필요한 정보를 가지고 있으므로, 단일 에이전트나 메시징 기능이 없는 팀은 아무리 많은 재시도를 하더라도 성공할 수 없습니다. 대신 이 작업은 Figure 1의 전제 조건을 테스트합니다. 작은 공유 정책이 과연 소통 자체를 학습할 수 있는지, 그리고 그 비용으로 토큰을 얼마나 사용하는지를 확인하는 것입니다. 이후 게시물에서는 단일 에이전트가 혼자 해결할 수 있는 작업으로 넘어가며, 여기서는 예산을 맞춘 비교가 의미를 갖습니다.

주요 소통 도구는 `message(to, text)`, `read_messages()`, `check_notifications()`입니다. 알림은 에이전트에게 읽지 않은 메시지가 있음을 알려주고, 받은 편지함을 읽으면 그 내용을 확인할 수 있습니다. 에이전트는 직접 메시지를 보내거나 모든 사람에게 브로드캐스트할 수 있습니다.

별도의 컨텍스트와 비공개 입력을 사용하면서 **하나의 공유 Qwen3-0.6B 정책**으로 구성된 네 에이전트 시스템을 정의했습니다. 서로 다른 네 개의 모델을 학습시키는 것이 아닙니다. 프롬프트는 할당된 정보를 공유해야 한다는 내용을 포함해 작업과 도구를 설명합니다.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/buckets/huggingface/training-swarms-blog-assets/resolve/part-1/task-and-training-loop.png" alt="Four agents share one policy with separate contexts and private numbers, exchange messages, submit a sorted list, and train from a common team reward" width="80%">
  <figcaption>Figure 2. The task and training loop. The private numbers illustrate a possible successful episode; this graphic is not a recorded conversation or a claim about the learned protocol at update 75.</figcaption>
</figure>

### 팀 보상과 문제당 여덟 번의 시도

이 게시물에서 *팀*은 하나의 에피소드, 즉 네 에이전트가 하나의 문제를 한 번 시도하는 것을 의미합니다. 성공한 팀은 사용한 토큰 수만큼 차감된 보상을 받습니다. 실패한 팀은 0을 받습니다.

```text
team allowance = 4 × (8,192 work tokens + 1,024 final-answer tokens)
               = 36,864 tokens

reward = success × (1 − 0.25 × min(1, generated team tokens / 36,864))
```


예를 들어 허용량의 절반을 사용해 성공한 팀은 0.875를 받습니다. 메시지를 보내는 데 대한 별도의 보상은 없습니다. 더 적은 수의 생성 토큰을 사용하면서 작업을 해결하는 것이 유일한 보상 신호입니다.

각 학습 문제에 대해 독립적인 팀 시도 여덟 번을 수집합니다. 보상은 해당 그룹 내에서 정규화됩니다. 그룹 평균보다 나은 시도에는 양의 어드밴티지가 부여되고, 더 못한 시도에는 음의 어드밴티지가 부여됩니다. 같은 팀의 모든 에이전트는 해당 팀의 어드밴티지를 받으며, 여기에는 최종 답변이 틀린 에이전트도 포함됩니다.

[TRL](https://github.com/huggingface/trl)에서는 각 에이전트가 샘플링한 토큰이 학습 행을 구성합니다. 수신한 메시지와 도구 결과가 컨텍스트를 제공합니다. 손실은 먼저 각 팀 내에서 평균을 낸 다음 팀 간에 평균을 내므로, 대화가 더 길다고 해서 해당 팀의 가중치가 커지지는 않습니다.

> [!NOTE]
> [우리의 결과](#our-results)는 업데이트당 팀 시도 256회(문제 32개에 각각 8회 시도)를 사용한 원래 실험에서 나온 것입니다. [워크스루](#practical-example-train-your-own-swarm)에서는 TRL의 기본값인 `--teams-per-update 128`를 사용하며, 이는 업데이트당 약 150회의 팀 시도에 해당합니다.

### 우리의 결과

75단계에 걸쳐 학습 성공률은 첫 5개 업데이트와 마지막 5개 업데이트를 비교했을 때 **6.6%에서 82.3%로** 상승했습니다. 체크포인트 75에서 네 에이전트 팀은 **보류된 문제 32개 중 26개(81.25%)**를 해결했습니다. [논문](https://huggingface.co/papers/2611.08615)에서는 200회 업데이트 후 네 에이전트 성공률을 93.2%로 보고합니다.

팀은 시도당 토큰도 **37.4% 적게 사용**했으며, 약 13k에서 8k로 감소했습니다. 이러한 개선과 함께 정보 공유도 더 활발해졌습니다. 전체 토큰 사용량은 감소했지만 브로드캐스트는 팀당 2.38회에서 5.73회로 증가했습니다.

<figure class="image text-center">
  <iframe src="https://burtenshaw-training-swarms-figures.static.hf.space/training-success.html" title="Interactive chart: team success over 75 training updates, with held-out checks of 1/32, 7/32, 17/32 and 26/32 at updates 1, 25, 50 and 75" width="100%" height="410" frameborder="0" scrolling="no" loading="lazy"></iframe>
  <figcaption>Figure 3. The blue curve describes training batches of 256 team attempts, collected before each update. Orange points are held-out checks after the indicated update.</figcaption>
</figure>

학습 후반부에는 팀이 더 적은 토큰을 사용하기도 했습니다. 첫 5개 업데이트와 71–75 업데이트를 비교하면 다음과 같습니다.

| **학습 지표** | **업데이트 1–5, 평균** | **업데이트 71–75, 평균** |
|---|---:|---:|
| 성공률 | 6.56% | 82.34% |
| 팀당 생성 토큰 | 12,826.9 | 8,026.0 |
| 팀당 메시지 | 3.48 | 6.28 |
| 팀당 브로드캐스트 | 2.38 | 5.73 |
| 에이전트당 한 번씩 브로드캐스트한 팀 | 4.06% | 18.20% |

이는 두 구간 사이에 **학습 팀당 생성 토큰이 37.4% 감소**했다는 뜻입니다. 배치 75 자체에서는 팀당 평균이 7,624토큰이었습니다. 이는 학습 중 측정한 값이며, 학습된 체크포인트에 대한 보류 데이터 토큰 효율성 측정값은 보존하지 않았습니다.

토큰 수가 줄었다고 해서 메시지 수가 줄어든 것은 아니라는 점에 주목하세요. 더 많은 에이전트가 참여하기 시작했고, 직접 메시지보다 브로드캐스트가 더 일반적으로 사용되었습니다. 생성 토큰에는 메시지 본문뿐 아니라 추론과 도구 호출도 포함되므로, 메시지 수만으로는 팀이 얼마나 많은 텍스트를 생성했는지 알 수 없습니다.

<figure class="image text-center">
  <iframe src="https://burtenshaw-training-swarms-figures.static.hf.space/tokens-and-communication.html" title="Interactive chart: generated tokens per team fall to 7,624 at batch 75 while broadcasting grows; teams with one broadcast per agent remain a minority" width="100%" height="620" frameborder="0" scrolling="no" loading="lazy"></iframe>
  <figcaption>Figure 4. Top: training token use and a trailing five-update mean. Bottom: mutually exclusive communication classes, based on sent-message events. A team is classified first by whether some agent never sends, then whether any direct message occurs, then whether every agent sends exactly one broadcast; the remaining teams repeat broadcasts.</figcaption>
</figure>

### 더 많은 재시도인가, 학습된 팀인가?

어떤 에이전트도 혼자 문제를 해결할 수 없으므로, "재시도 횟수가 더 많은 단일 에이전트"에 가장 가까운 것은 재시도 횟수를 늘린 학습되지 않은 팀입니다. 학습 데이터에는 이미 이러한 비교가 포함되어 있습니다. 각 업데이트에서는 문제당 여덟 번의 시도를 수집하고, 작업의 답은 확인하기 쉽기 때문에, 여덟 번의 시도 중 하나라도 성공하면 해당 문제를 해결된 것으로 계산할 수 있습니다. 학습 초기에는 이 수치가 학습되지 않은 팀의 best-of-8 성공률을 제공합니다.

| | **성공률** | **문제당 생성 토큰** |
|---|---:|---:|
| 학습되지 않은 팀, 8회 시도 중 최고 (업데이트 1–5) | 37.5% | 약 103,000 |
| 학습된 팀, 1회 시도 (업데이트 71–75) | 82.3% | 약 8,000 |

여덟 번 재시도해도 학습되지 않은 팀은 문제의 절반에도 못 미치는 수를 해결하며, 학습된 팀의 한 번의 시도보다 약 13배 많은 토큰을 사용합니다. 이 수치는 추정치로 해석해야 합니다. 두 행 모두 보류 데이터가 아닌 학습 문제를 사용하며, 업데이트 2–5에서는 이미 약간 학습된 정책에서 샘플을 추출합니다.

### 인프라

모델은 작지만 롤아웃은 그렇지 않습니다. 각 업데이트에서는 최대 40k 토큰의 컨텍스트에서 에이전트 대화를 수백만 토큰 생성합니다. 이러한 롤아웃을 빠르게 처리하기 위해 트레이너에 H200 한 개를 사용하고, vLLM 롤아웃 엔진을 실행하는 H200 일곱 개를 사용했으며, 각 워커의 여덟 번 시도를 함께 유지했습니다. 아래 워크스루에서는 두 GPU에서 동일한 작업을 실행합니다. 인프라 세부 정보는 다음과 같습니다.

| **설정**                   | **기록된 실험**        |
|---|---|
| 공유 정책                 | Qwen3-0.6B                     |
| 팀당 논리 에이전트 수       | 4                              |
| 업데이트당 문제 수           | 32                             |
| 문제당 팀 시도 수     | 8                              |
| 에이전트당 작업 / 최종 토큰 | 8,192 / 1,024                  |
| 학습률 / 온도   | 5e-7 / 1.0                     |
| 학습 문제 풀         | 1,024 generated problems       |
| 보고 엔드포인트            | Preserved checkpoint 75        |
| 실행                     | 1 trainer GPU + 7 rollout GPUs |

## 실용 예제: 나만의 스웜 학습하기 {#section-5}

위의 수치는 원래 75회 업데이트 실험을 설명합니다. 이 워크스루에서는 Hugging Face Jobs에서 [TRL의 스웜 예제](https://github.com/huggingface/trl/tree/main/examples/async_grpo_s3_aggregation)를 실행합니다. H200 한 개는 vLLM을 사용해 공유 Qwen3-0.6B 정책을 제공하고, 두 번째 H200은 이를 학습합니다.

### 1. 실행을 위한 버킷 생성

Jobs를 실행하고 버킷을 생성할 수 있는 Hugging Face 계정으로 로그인합니다. `YOUR_USERNAME`을 Hub 사용자 이름으로 바꿉니다.

> [!TIP]
> 이전 측정값을 계속 확인할 수 있도록 실행마다 새 버킷을 사용하세요.

```bash
uv tool install 'huggingface_hub==2.4.0'
hf auth login
BUCKET="YOUR_USERNAME/swarm-trl-$(date -u +%Y%m%d-%H%M%S)"
hf buckets create "$BUCKET"
```


### 2. Hugging Face Jobs에서 TRL 예제 실행

아래 스크립트를 현재 디렉터리에 `train-swarm.sh`로 저장합니다.

<details>
<summary><code>train-swarm.sh</code></summary>

```bash
set -eu
git clone --depth 1 https://github.com/huggingface/trl /tmp/trl
cd /tmp/trl
example=examples/async_grpo_s3_aggregation
if [ ! -f "$example/async_grpo_s3_aggregation.py" ]; then
  git fetch --depth 1 origin refs/pull/8424/head
  git checkout --detach FETCH_HEAD
fi
git rev-parse HEAD > /results/trl-revision.txt
python -m pip install -e ".[vllm,kernels]" "vllm==0.32.0" datasets httpx trackio

base=Qwen/Qwen3-0.6B
url=http://127.0.0.1:8000
export TRACKIO_DIR=/tmp/trackio
serve() {
  CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve "$@" \
    --host 127.0.0.1 --max-model-len 40960 --max-num-seqs 512 --enable-prefix-caching &
  server=$!
  until curl -fs "$url/health" >/dev/null; do kill -0 "$server"; sleep 5; done
}
evaluate() {
  python "$example/evaluate.py" --url "$url" --model "$1" --name "$2" \
    --episodes 192 --concurrency 64 --out "/results/$2.jsonl"
}

serve "$base" --logprobs-mode processed_logprobs --weight-transfer-config "{\"backend\":\"nccl\"}"
evaluate "$base" base
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --mixed_precision bf16 \
  "$example/async_grpo_s3_aggregation.py" --vllm-url "$url" \
  --updates 50 --teams-per-update 128 --concurrent-teams 128 \
  --report-to trackio --output-dir /tmp/swarm
cp -r "$TRACKIO_DIR" /tmp/swarm/checkpoint-50 /results/
kill "$server"
wait "$server" || true
serve /tmp/swarm/checkpoint-50 --logprobs-mode processed_logprobs
evaluate /tmp/swarm/checkpoint-50 trained
```


</details>

아래 명령은 스크립트의 내용을 작업으로 전송합니다. 작업 내부에서는 vLLM이 GPU 0에서 공유 정책을 제공합니다. GPU 1에서는 Trackio를 사용해 50회 업데이트 동안 TRL의 `async_grpo_s3_aggregation.py`를 실행합니다. TRL의 `evaluate.py`는 동일한 보류 문제 192개에서 기본 모델과 체크포인트 50을 평가합니다. 이는 결과에서 사용한 문제 32개보다 큰 집합이므로 오차 범위가 더 좁아집니다. 작업은 두 점수, 체크포인트, Trackio 데이터, 그리고 정확한 Git 커밋을 버킷에 저장합니다.

```bash
hf jobs run --detach --flavor h200x2 --timeout 12h \
  --volume "hf://buckets/$BUCKET:/results" \
  -- vllm/vllm-openai:v0.32.0 bash -c "$(cat train-swarm.sh)"
```


하나의 TRL 롤아웃에는 하나의 전체 팀 대화가 포함됩니다. 트레이너는 동일한 문제에 대한 여덟 번의 시도를 비교하고, 한 번의 시도에 참여한 네 에이전트 모두에게 동일한 어드밴티지를 부여합니다. 각 업데이트에서는 약 150회의 팀 시도를 사용합니다. TRL은 팀 단위가 아니라 토큰 수를 기준으로 마이크로 배치를 구성하므로 `--teams-per-update 128` 설정은 근사값입니다. `--report-to trackio` 플래그는 각 업데이트에서 학습 지표를 기록합니다.

이 짧은 실행은 원래 실험보다 비용이 적게 들며, 학습 곡선에서도 더 일찍 종료됩니다. TRL에서 이 예제를 실행하는 데 H100 두 개로 약 4.8시간이 걸렸고, 체크포인트 50에서 보류 데이터 성공률 60.9%에 도달했습니다.

> [!TIP]
> 원래 실험에 더 가깝게 실행하려면 학습 플래그를 `--updates 75 --teams-per-update 256 --concurrent-teams 256`로 변경하고, 스크립트에서 `checkpoint-50`을 `checkpoint-75`로 바꾸며, 작업의 `--timeout`을 `24h`로 높이세요. 롤아웃 수가 세 배가 됩니다. 두 코드베이스는 시스템 프롬프트와 같은 세부 사항에서도 여전히 다르므로, 정확히 재현된 결과보다는 비슷한 결과를 기대하세요.

### 3. 결과 읽기

작업이 완료되면 버킷을 로컬 폴더와 동기화합니다. 여기에는 두 개의 평가 파일, Trackio 데이터, TRL 리비전, 그리고 가장 큰 다운로드 항목인 체크포인트 50 가중치가 들어 있습니다. 대시보드에서 학습 과정에 따른 `team/success`, `team/messages`, `team/tokens`를 비교합니다. 작업 로그에도 기본 모델과 체크포인트 50의 보류 데이터 성공률이 표시됩니다. `base.jsonl` 또는 `trained.jsonl`의 각 줄은 이러한 평가 시도 중 하나를 기록합니다.

```bash
hf buckets sync "hf://buckets/$BUCKET" ./swarm-results
TRACKIO_DIR="$PWD/swarm-results/trackio" uvx trackio show
```


## 다음에 시도할 것 {#section-6}

이 첫 번째 예제는 작은 환경, 공유 보상, 그리고 비공개 정보를 모으는 능력이 향상되는 정책을 제공합니다. 또한 성공률, 토큰 사용량, 소통 구조를 함께 살펴봐야 하는 이유도 보여줍니다. 75회 업데이트 시점에 팀은 확인한 문제 대부분을 해결하고 있지만, 가장 깔끔한 소통 패턴으로 수렴하지는 않았습니다.

앞으로의 예제에서는 학습에서 한 걸음 물러나 스웜을 어떻게 평가하고 분석하는지 살펴보겠습니다.

지금은 다음 작업을 위한 구체적인 기반이 마련되었습니다. 작은 에이전트 네 개, 학습 가능한 정책 하나, 검증 가능한 작업 하나, 그리고 팀이 무엇을 학습했는지 살펴볼 수 있는 재현 가능한 방법입니다.
