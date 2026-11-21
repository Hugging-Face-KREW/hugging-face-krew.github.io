---
layout: post
title: "harness adapter 소개: 선호하는 모델을 harness에 맞춰 성능과 비용 개선하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/harness-adapters-at-scale/thumbnail.png
image: assets/images/blog/posts/2026-11-20-harness-adapters-at-scale/thumbnail.png
authors:
  - user: ybelkada
  - user: burtenshaw
  - user: SaylorTwift
  - user: lysandre
  - user: aminediroHF
  - user: AdithyaSK
  - user: evalstate
  - user: sergiopaniego
  - user: joelniklaus
  - user: lewtun
slug: "harness-adapters-at-scale"
source_url: "https://huggingface.co/blog/harness-adapters-at-scale"
source_published_date: "2026-11-20"
source_published_at: "2026-11-20T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Introducing harness adapters: adapt your favorite model to your harness to improve performance and cost](https://huggingface.co/blog/harness-adapters-at-scale)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/harness-adapters-at-scale -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# harness adapter 소개: 선호하는 모델을 harness에 맞춰 성능과 비용 개선하기

사용자가 선호하는 agent harness에서 사용자 지정 task distribution에 대한 RL을 결합하면, 해당 harness와 task distribution에 맞춰 더 정확하고 효율적인 모델을 얻을 수 있습니다.

모델은 모든 harness와 사용자가 수행하는 모든 task에 완전히 최적화되어 있지 않습니다. 대신 일반적인 성능을 목표로 합니다. 따라서 모델과 harness를 결합하면 실제로 수행하는 작업에서 성능과 효율성을 충분히 활용하지 못하게 됩니다. RL을 사용하면 사용자 지정 harness adapter를 학습하여, 특정 harness와 task 조합에서 모델의 성능을 최대한 끌어낼 수 있습니다.

Qwen3.8-27B를 예로 들어 보겠습니다. pi harness에서 Terminal-Bench 2.1 점수는 39%이며 task당 평균 2.9K 토큰을 사용합니다. 이 가중치를 고정하고 사용자 지정 RL 학습 LoRA adapter를 추가하면 평균 토큰 사용량 1.5K로 55% pass@1을 달성할 수 있습니다. 이는 대부분 모델이 추론을 덜 하고, tool 호출을 줄이며, 첫 시도에 더 잘 수행하기 때문입니다.

이전 작업([multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl), [harness adapters](https://huggingface.co/blog/harness-hub/harness-adapters))에서는 자체 task distribution에서 자체 harness 내부에 RL을 적용하면 더 정확하고 효율적인 모델을 얻을 수 있음을 보였습니다. 해당 실험은 데이터 분석 및 코드 구축 task를 대상으로 했습니다. 여기서는 이 결과가 27B 모델에서도, 더 어려운 실제 소프트웨어 엔지니어링 task에서도, 그리고 Pi, OpenCode, Hermes Agent라는 세 harness 전반에서도 유지되는지 살펴보며 한 단계 더 나아가고자 했습니다. 무엇보다도 누구나 자신의 task, harness, 모델 조합에서 재현할 수 있도록 완전히 재사용 가능한 레시피를 공유하고자 했습니다.

## TL;DR {#section-1}

- agent harness(pi, Hermes Agent, OpenCode)별로 Qwen3.8-27B용 LoRA adapter를 하나씩 학습했으며, harness 자체에서 RL을 수행했습니다.
- Datasets, 학습 및 agent rollout은 모두 TRL의 `AsyncGRPOTrainer`, OpenEnv, harbor를 사용하는 Jobs를 통해 Hub에서 수행됩니다.
- 각 harness adapter는 두 benchmark 모두에서 자체 harness를 사용하는 base model을 성능과 효율성 측면에서 모두 능가합니다.
- 세 LoRA를 하나의 super adapter로 병합했으며, 이 adapter는 pi, Hermes Agent, OpenCode에서 변경 없이 실행됩니다.
- 모든 artifact(adapter, GGUF, RL task, safeagents plugin)는 이 [Hugging Face collection](https://huggingface.co/collections/harness-hub/qwen38-27b-harness-adapters)에서 확인할 수 있습니다.

## 목차 {#section-2}

* [agent harness 내부에서 학습하기](#section-3)
* [Hugging Face에서 사용자 지정 RL task 생성하기](#section-4)
* [짧은 학습 실행으로 filter 테스트하기](#section-5)
* [더 효율적인 GRPO 학습을 위한 `AsyncGRPOTrainer`](#section-6)
* [학습 설정](#section-7)
* [Super-adapter: 모든 adapter를 하나로 병합하기](#section-8)
* [최종 결과](#section-9)
* [미해결 질문: 모델은 실제로 무엇을 학습했는가?](#section-10)
* [맺음말과 향후 방향](#section-11)

## agent harness 내부에서 학습하기 {#section-3}

최종 학습에 사용할 harness 내부에서 모델을 학습하기 위해 [OpenEnv의 Harbor integration](https://huggingface.co/docs/openenv/environments/harbor)과 [TRL의 OpenEnv](https://huggingface.co/docs/trl/openenv) 가이드를 활용했습니다. 모든 RL task에 대해 target harness가 내부에 사전 설치된 즉시 사용 가능한 docker image를 구축했습니다. 최종 docker image에는 Pi `1.5.2`, OpenCode `2.2.7`, Hermes Agent `v0.22.6`를 사용합니다. 앞서 언급한 blogpost의 아래 다이어그램은 전체 protocol을 잘 설명합니다.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/loop.html" width="100%" height="520" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

대부분의 RL framework는 agent loop를 재구현한 것을 학습합니다. 즉 agent는 framework 자체의 interface에 맞춰 작성됩니다. 예를 들면 [verl의 AgentLoop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), [SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL) environment, [rLLM](https://github.com/rllm-org/rllm)의 agent 및 environment class, [slime](https://github.com/THUDM/slime)의 custom rollout function, 또는 자체 Python agent 내부의 [ART](https://github.com/OpenPipe/ART) client가 있습니다. 여기서는 harness가 배포되는 그대로 실행되며, trainer는 harness와 vLLM 사이의 proxy가 기록한, harness가 실제로 생성한 token을 사용해 학습합니다. 최근 작업도 같은 방향으로 나아가고 있습니다. [Agent Lightning v1.0](https://huggingface.co/papers/2608.17528)은 proxy를 통해 배포 시점의 harness를 loop 안에 유지하며("harnessed agentic RL"), [SkyRL의 Harbor integration](https://novasky-ai.notion.site/skyrl-harbor)은 Harbor trial을 대상으로 학습하고 해당 chat history를 trainer용 token으로 변환합니다.

## Hugging Face에서 사용자 지정 RL task 생성하기 {#section-4}

이 레시피를 위해 Hub, OpenEnv, [Repo2RLEnv](https://github.com/huggingface/Repo2RLEnv)를 사용해 agentic coding용 사용자 지정 dataset을 만들었습니다. 기존 RL environment에서 task를 가져오고, Repo2RLEnv로 task를 생성하며, OpenEnv로 task를 검증하고, sampling하는 과정을 하나의 data pipeline으로 결합했습니다. 이를 통해 Qwen3.8-27B를 agentic coding에 맞게 개선할 수 있는, 검증되고 관련성 높은 최고 품질의 task 집합을 얻었습니다. 이 절에서는 이를 구축한 방법을 설명하므로, 여러분의 domain에서도 직접 시도해 볼 수 있습니다.

RL 학습에는 agent가 시도할 수 있는 task, 각 시도를 평가하는 verifier, 그리고 verifier가 올바르다는 근거가 필요합니다. 잘못된 답을 받아들이는 verifier는 잘못된 행동을 학습시킵니다. 이미 해결 가능한 task는 모델에 아무것도 가르치지 못합니다.

이 절에서는 세 가지 source에서 task set을 구축한 방법을 설명합니다. 두 source는 Hugging Face Hub에 있고, 세 번째는 Repo2RLEnv의 pipeline인 Tasksmith로 생성했습니다. 모든 task는 `openenv validate`를 사용한 eligibility screen과 Qwen3.8-27B base model을 대상으로 한 동일한 difficulty check를 통과합니다.

<p align="center">
    <img src="https://huggingface.co/buckets/huggingface/harness-adapters-blog-assets/resolve/images/task-pipeline.png" alt="Task pipeline: Tasksmith, SWE-smith and SWE-Gym sources pass through an eligibility screen, Harbor packaging, openenv validate on Hugging Face Jobs, and a difficulty check with Qwen3.8-27B" width="80%">
</p>

### 세 가지 source

각 task는 Terminal-Bench 2.1에서도 사용하는 Harbor format을 따릅니다. Hub에서 Harbor environment를 광범위하게 검색한 결과 약 59,000개의 task가 나왔지만, 중복 task, evaluation set, 학습에 사용하지 말라고 표시된 task가 포함되어 있었습니다. 이를 학습용으로 구축된 두 Hub source로만 줄이고, 세 번째 set은 직접 생성했습니다.

- SWE-smith([Python](https://huggingface.co/datasets/SWE-bench/SWE-smith-py), [Go](https://huggingface.co/datasets/SWE-bench/SWE-smith-go))는 실제 Python 및 Go repository에 bug를 주입합니다. LLM rewrite를 사용하거나 operator를 뒤집는 등의 mutation을 적용합니다. 각 bug를 발견하는 test가 task의 test가 됩니다. 데이터 규모를 늘려 줍니다.
- TaskTrove의 [TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove)에서 Harbor task로 패키징된 SWE-Gym은 실제 GitHub issue와 병합된 수정 사항을 task로 변환합니다. 실제 환경의 bug를 추가합니다.
- [Tasksmith](https://huggingface.github.io/Repo2RLEnv/pipelines/tasksmith/)는 병합된 pull request에서 task를 생성합니다. LLM agent가 instruction과 test를 작성하고, Tasksmith가 결과를 확인합니다. reference와 유효한 대안은 통과해야 하며, 수정되지 않은 code와 알려진 오답 solution은 실패해야 합니다.

target의 공백을 채우기 위해 Tasksmith를 사용했습니다. target benchmark(Terminal-Bench 2.1, SWE-bench Pro의 hard subset, DeepSWE)가 테스트하는 skill을 매핑한 뒤, HTTP streaming, filesystem operation, serialization과 같은 skill을 실행하는 pull request를 선택했습니다. 병합된 pull request 30개에서 Tasksmith는 10개의 Python repository에 대해 검사를 통과한 task 24개를 생성했으며, 두 번째 실행에서는 그중 17개가 통과했습니다. 이 검사는 우리의 검사를 대체하지 않으므로, 해당 task도 나머지 task와 동일한 pipeline을 거칩니다.

### `openenv validate`를 사용한 검증

먼저 metadata screen을 실행합니다. permissive license가 적용된 training split의 task만 남깁니다. 중복 task, 학습에 사용하지 말라고 표시된 task, target benchmark에 포함된 253개 task, 그리고 해당 benchmark가 사용하는 repository의 모든 task를 제거합니다. 그런 다음 모든 task를 새 trial에서 `openenv validate`의 semantic check로 실행했습니다.

- reference solution은 최대 reward를 획득합니다(`semantic.oracle_max`).
- 아무것도 하지 않는 경우의 score는 명확히 더 낮습니다(`semantic.floor_gap`).
- 어떤 canary도 아무것도 하지 않는 경우보다 높은 score를 내지 않습니다(`semantic.canary_floor`).
- reference를 replay하면 동일한 reward를 얻습니다(`runtime.episode_determinism`).

각 trial은 별도의 Hugging Face Job으로 실행했기 때문에 수십 개의 task를 동시에 검증할 수 있었습니다. `openenv validate`는 모든 Harbor task에 대해 동일한 옵션을 제공합니다.

```bash
openenv validate path/to/task --sandbox hf-sandbox
```


Hub 후보 1,127개 중 858개가 통과했습니다. 오답 제어를 통해 결함을 발견했으며, 여기에는 불완전한 구현을 받아들인 Tasksmith task와 알려진 오답을 받아들인 Terminal-Bench 2.1 verifier(regex-log)가 포함됩니다. 대부분의 실패는 image가 시작되지 않는 것과 같은 infrastructure 문제였습니다. 실패한 task는 실패한 검사를 기록한 채 rejected split에 남겨 둡니다.

### base model을 사용한 난이도 확인

유효한 task라도 아무것도 가르치지 못할 수 있습니다. GRPO(group relative policy optimization)는 동일한 task에 대한 다른 시도와 비교해 각 시도를 평가하므로, 모든 시도가 통과하거나 모두 실패하는 task는 학습 signal을 제공하지 않습니다. 학습 전에 Qwen3.8-27B가 각 task를 4번 시도했으며, repository당 6개 task로 층화 추출한 6개 repository의 sample에 대해서는 8번 시도했습니다.

점수가 산출된 task 442개 중 Qwen3.8-27B는 151개를 매번 해결했고 120개는 한 번도 해결하지 못했습니다. 때때로만 해결한 171개 task가 qualified set을 구성합니다. 아래 figure는 score가 양쪽 끝에 몰리는 양상과 source별 차이를 보여 줍니다. 모델은 대부분의 Go task를 매번 해결하고, SWE-smith Python task의 3분의 1 이상에서는 모든 시도에 실패하며, SWE-Gym task의 73%에서는 중간 영역에 해당하는 결과를 냅니다.

<p align="center">
    <img src="https://huggingface.co/buckets/huggingface/harness-adapters-blog-assets/resolve/images/difficulty-check.png" alt="Qwen3.8-27B results on 442 scored tasks, split by attempts solved and by source" width="90%">
</p>

*점수가 산출된 task 442개에 대한 Qwen3.8-27B 결과. 녹색은 qualified task, 즉 모델이 일부 시도에서는 해결하지만 모든 시도에서 해결하지는 못하는 task를 나타냅니다.*

이 예에서는 1,127개 task를 filter하고 검증하여 최종 dataset에 직접 포함한 278개로 줄인 과정을 보여 줍니다. 그런 다음 pipeline은 남은 rejected task 522개를 Tasksmith로 수정하고 복잡하게 만든 뒤 검증을 반복했습니다.

## 짧은 학습 실행으로 filter 테스트하기 {#section-5}

pipeline은 두 가지 filter를 적용합니다. validation은 손상된 task를 제거하고, difficulty check는 base model이 때때로만 해결하는 task를 남깁니다. 이 절에서는 이러한 filter가 모델이 학습하는 내용을 바꾸는지 테스트합니다. pipeline의 서로 다른 단계에 해당하는 task로 최대 50 step까지 네 번의 짧은 실행을 학습하고, 이를 Terminal-Bench 2.1에서 base model과 비교했습니다.

각 실행은 TRL의 `AsyncGRPOTrainer`를 사용해 32개 task에서 LoRA와 GRPO로 Qwen3.8-27B를 학습합니다. 네 실행 모두 동일한 agent harness(OpenCode, 최대 48회의 model call), 동일한 설정, 하나의 8×H200 Hugging Face Job을 사용하며, reward는 verifier의 pass 또는 fail입니다. task set은 적용된 filter에 따라 다릅니다.

- **Unvalidated**: 통과한 task 24개와 validation에 실패한 task 8개.
- **Validated**: 난이도와 관계없이 선택한 통과 task 32개.
- **Unqualified**: base model이 항상 또는 전혀 해결하지 못하는 통과 task 32개.
- **Qualified**: base model이 일부 시도에서 해결하는 통과 task 32개.

아래 figure는 task가 모델 학습에 어떤 영향을 미쳤는지 보여 줍니다. 표에는 각 실행의 step과 평가에 사용하는 checkpoint가 나와 있습니다. chart는 validation 상태가 서로 다른 task subset에서 학습 reward가 어떻게 변하는지 추적합니다.

<iframe src="https://harness-hub-tb21-lift-status.static.hf.space" title="TB2.1 lift experiment status" width="100%" height="760" style="border:0;max-width:880px" loading="lazy"></iframe>

### dataset

`harness-hub/agentic-coding-envs`에는 35개 repository의 task 871개가 포함되어 있습니다. Hub task는 repository당 최대 40개로 제한하여 단일 project가 결과를 지배하지 않도록 했습니다. 아래 표는 각 source가 기여하는 내용을 보여 줍니다. dataset에는 두 가지 view가 있습니다. 모든 accepted task와 GRPO 학습을 위한 qualified subset입니다. rejected attempt는 별도의 config에 남겨 둡니다.

| Source | Task type | Tasks | Qualified |
|---|---|---:|---:|
| SWE-smith | Python 및 Go에 주입한 bug | 717 | 144 |
| SWE-Gym (TaskTrove) | 실제 GitHub issue | 130 | 27 |
| Tasksmith | 병합된 pull request에서 생성 | 24 | 22 |
| **Total** | | **871** | **171** |

## 더 효율적인 GRPO 학습을 위한 `AsyncGRPOTrainer` {#section-6}

TRL의 안정 버전인 `AsyncGRPOTrainer`를 활용했습니다. 이 trainer의 주요 장점은 모든 rollout이 완료될 때까지 기다린 후 다음 학습 optimization step으로 넘어가지 않는다는 점입니다.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/asyncgrpo-sync-async.html" width="100%" height="390" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

trainer와 그 구성 요소에 대한 자세한 내용은 [AsyncGRPO blogpost](https://huggingface.co/spaces/trl-lib/async-grpo-stable)를 참조하세요.

## 학습 설정 {#section-7}

최종 harness adapter를 얻기 위해 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 위에 TRL의 `AsyncGRPOTrainer`를 사용해 LoRA adapter를 학습했습니다. adapter는 `all-linear` flag를 사용해 모델에 연결했습니다. 학습 효율을 극대화하기 위해 rollout 단계에서는 thinking mode를 끄고, 더 폭넓게 비교하기 위해 최종 adapter를 thinking 사용 및 미사용 상태에서 모두 평가했습니다.

최종 reward function은 verifier의 correctness에 작은 tool-efficiency bonus를 더한 것입니다(0.1, 0.1이 가장 효과적임을 보인 [harness adapters의 이전 iteration에서 수행한 초기 실험](https://huggingface.co/blog/harness-hub/harness-adapters)에서 도출).

Termgrade environment에서 50 step 동안 작은 LR sweep(1e-5, 5e-5, 1e-4)을 수행했습니다. 1e-5는 학습 correctness를 거의 변화시키지 않았고(0.52에서 0.50), 1e-4는 개선 대신 rollout을 붕괴시켰으며(turns per rollout이 18에서 6으로 감소하고 entropy는 상승), 5e-5만 명확한 개선을 보였습니다(step 40에서 0.49에서 0.60으로). 이후 더 긴 5e-5 실행에서 불안정성 징후(rollout length 붕괴, entropy 상승)가 나타나 최종 실행에는 2e-5를 사용했습니다.

### 학습 infrastructure

각 harness adapter는 NVIDIA 2xB200 GPU를 사용해 학습했으며, 한 GPU는 vLLM server에, 다른 한 GPU는 학습 process에 할당했습니다. 먼저 하나의 optimization step 안에서 학습할 수 있는 RL task 수를 극대화하기 위해 stress testing을 수행했습니다. 이를 위해 현재 설정에서 `AsyncGRPOTrainer`의 `--token-budget` argument를 한계까지 ‘maxed out’했습니다. 아래 chart는 각 harness에서 step당 group 수의 변화를 보여 줍니다(파란색: Pi, 녹색: OpenCode, 주황색: Hermes).

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/groups.html" width="100%" height="500" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

학습 중 모델이 점점 더 좋아지면서 각 training step에 더 많은 group을 담을 수 있게 되고, 그 결과 각 optimization step에서 모델이 받는 signal이 다양해집니다.

### 최종 mixture

최종 data mixture는 [harness-adapters RL environments](https://huggingface.co/datasets/harness-hub/harness-adapters)의 task 750개와 [termgrade](https://huggingface.co/datasets/ai-and/termgrade-environments)의 task 300개로 구성됩니다. 학습 중 sampled task의 다양성을 극대화하기 위해 data sampling strategy를 약간 사용자 지정했으며, 아래에서 설명합니다.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/sampling.html" width="100%" height="430" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

학습 전에 task를 family별로 interleave하는 고정 순서로 배치합니다(생성한 RL environment에서는 repository가 family입니다. Termgrade에서는 각 task를 자체 family로 간주하므로 이름순으로 정렬했습니다). 따라서 mixture의 distribution을 따르면서 각 training step에 가능한 많은 repository가 섞입니다.

### loop 안의 safeagents

학습 중 모델의 잘못된 행동을 차단하기 위해 [safeagents](https://github.com/huggingface/safeagents) tool도 연결했습니다. 최종 plugin(RL task와 test path에 특화된 regex 및 pattern rule로 구성)은 [harness-hub/qwen-safeagents-plugin](https://huggingface.co/harness-hub/qwen-safeagents-plugin)에 호스팅되어 있으며, 누구든 학습 설정을 재현하고 싶다면 최신 버전의 safeagents를 사용해 바로 이용할 수 있습니다.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/protocol.html" width="100%" height="430" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

학습 중 safeagents가 차단하는 tool 수는 꾸준히 감소했으며, 이는 각 optimization step 후 모델이 ‘점점 더 안전해진다’는 것을 보여 줍니다. 아래는 각 harness adapter에 대해 training step별 1000회 tool call당 safeagents의 blocked call을 나타낸 diagram입니다.

<p align="center">
    <img src="https://huggingface.co/buckets/huggingface/harness-adapters-blog-assets/resolve/images/safeagents-blocked.png" alt="safeagents blocked calls per 1,000 tool calls by training step for the pi, hermes and opencode adapters" width="100%">
</p>

## Super-adapter: 모든 adapter를 하나로 병합하기 {#section-8}

모델을 병합하면 각 모델이 수행할 수 있던 작업의 대부분을 유지하면서 capability를 결합하는 경우가 많습니다. 세 harness adapter를 학습한 뒤 두 가지 질문을 던졌습니다. 병합으로 accuracy가 떨어지는가? 그리고 하나의 "super-adapter"가 모든 harness에서 정확하고 효율적일 수 있는가?

먼저 세 adapter의 weight update를 layer별로 비교했습니다. 어느 두 adapter 사이의 cosine similarity도 약 0.01–0.02였습니다. 각 adapter가 학습 중 모델을 거의 직교하는 방향으로 밀어간 것입니다. update가 거의 간섭하지 않으므로, 이는 병합에 가장 유리한 경우입니다.

rank dimension을 따라 세 LoRA adapter를 concatenate했습니다(PEFT의 `combination_type="cat"`). 병합된 adapter의 rank는 96(3 × 32)이며 정확히 w · (ΔW_pi + ΔW_hermes + ΔW_opencode)를 적용합니다. 병합에 대한 이전 연구([task arithmetic](https://huggingface.co/papers/2212.04089), [TIES](https://huggingface.co/papers/2306.01708), [KnOTS](https://huggingface.co/papers/2410.19735))에서는 각 update를 1보다 작게 scaling하는 것이 가장 효과적이라고 보고하므로, step-120 adapter를 사용해 세 weight를 89개 Terminal-Bench 2.1 task 전체에서 각 harness별로 비교했습니다.

- ⅓(평균): 각 harness가 자체 update의 3분의 1만 유지.
- ½: 각 update를 절반의 강도로 적용.
- 1(단순 합): 모든 update를 그대로 적용.

| Weight | pi | hermes | opencode | 세 harness 전체 | Task당 tool call |
|---|---:|---:|---:|---:|---:|
| base model | 39.1% | 46.0% | 44.8% | 43.3% | 49.2 |
| ⅓(평균) | 49.4% | 51.2% | 47.1% | 49.2% | 35.6 |
| **½** | 48.8% | 53.5% | 55.2% | **52.5%** | 25.9 |
| 1(합) | 40.2% | 62.8% | 47.1% | 50.0% | 13.4 |

단순 합은 update를 과도하게 적용합니다. agent가 매우 짧아져 task당 tool call이 base model의 49회에서 13회로 줄고, Hermes는 62.8%로 최고점을 기록하는 반면 pi는 base model 수준으로 돌아갑니다. 평균은 각 harness 자체의 update를 희석합니다. ½이 가장 균형이 좋으므로, 공개된 checkpoint(pi step 160, hermes step 160, opencode step 150)를 weight ½로 사용해 최종 super-adapter를 구축했습니다.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/super-results.html" width="100%" height="560" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

### Adapter merging evaluation 결과

각 harness에서 모든 실행이 평가한 task(SWE-bench Pro 50–51개, Terminal-Bench 2.1 82–85개)를 대상으로 task당 한 번 시도한 결과입니다(괄호 안은 base model 대비 efficiency gain).

| Harness | Benchmark | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Qwen3.8-27B-\<harness\>-adapted | [Qwen3.8-27B-super-adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter) | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), thinking | [Qwen3.8-27B-super-adapter](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter), thinking |
|---|---|---:|---:|---:|---:|---:|
| pi | SWE-bench Pro (hard 51) | 40.0% | 64.0% (−60%) | 46.0% (−73%) | 74.0% | 74.0% (−48%) |
| hermes | SWE-bench Pro (hard 51) | 29.4% | 43.1% (−56%) | 47.1% (−60%) | 45.1% | 60.8% (−36%) |
| opencode | SWE-bench Pro (hard 51) | 35.3% | 51.0% (−72%) | 45.1% (−63%) | 47.1% | 66.7% (−31%) |
| pi | Terminal-Bench 2.1 | 40.0% | 56.5% (−56%) | 48.2% (−56%) | 52.9% | 57.6% (−25%) |
| hermes | Terminal-Bench 2.1 | 44.7% | 58.8% (−70%) | 55.3% (−53%) | 58.8% | 63.5% (−19%) |
| opencode | Terminal-Bench 2.1 | 47.6% | 52.4% (−53%) | 52.4% (−68%) | 54.9% | 58.5% (−22%) |

"Thinking"은 chat template의 기본 effort(xhigh)에 해당하는 thinking mode입니다. specialist는 thinking을 끈 상태로 학습하고 serving했습니다.

thinking을 끈 상태에서 super-adapter는 6개의 harness–benchmark pair 모두에서 base model보다 향상됩니다(mean pass@1 39.5% → 49.0%; 새롭게 해결한 task 71개, 잃은 task 34개, p < 0.001).

thinking을 사용해 전혀 학습하지 않았음에도, thinking mode에서 평가한 모델 중 가장 강력했습니다. 동일한 effort의 base model thinking과 비교하면 6개 pair 중 5개에서 승리하고 나머지 하나에서는 동률입니다(mean 55.5% → 63.5%; 새롭게 해결한 task 67개, 잃은 task 37개, p = 0.004). 가장 큰 향상은 SWE-bench Pro에서 나타났습니다. hermes에서 +16 points, opencode에서 +20입니다. thinking만으로 이미 74%에 도달하는 SWE-bench Pro의 pi에서는 동률입니다.

thinking을 끈 상태에서는 specialist harness adapter를 사용하세요. 각 harness의 자체 adapter는 6개 pair 중 4개에서 앞서며, pi에서 가장 큰 차이를 보입니다(SWE-bench Pro에서 64.0% 대 46.0%, Terminal-Bench에서 56.5% 대 48.2%). 하나의 모델을 세 harness로 나누면 pi가 얻은 gain을 가장 많이 잃습니다. 하나의 harness만 사용하고 thinking을 끈 상태로 유지한다면 해당 harness의 specialist를 사용하세요.

super adapter는 효율성도 훨씬 높습니다. thinking을 끈 상태에서 base model과 super-adapter가 모두 해결한 task에서는 task당 tool call이 53–73% 줄고 token 생성량이 44–60% 감소하며, 130개 task 중 111개에서 tool call이 더 적습니다. thinking을 켜면 SWE-bench Pro에서 tool call이 31–48%, token이 24–43% 줄어듭니다. Terminal-Bench에서는 tool call이 19–25% 감소하고 token 수는 거의 동일하게 유지됩니다.

½ weight는 step-120 adapter를 사용한 Terminal-Bench 2.1에서 선택했으므로 Terminal-Bench 수치는 약간 낙관적입니다. SWE-bench Pro는 선택에 전혀 사용되지 않았습니다. 전체 paired 결과는 [super-adapter의 model card](https://huggingface.co/harness-hub/Qwen3.8-27B-super-adapter)에 있습니다.

## 최종 결과 {#section-9}

이 절에서는 학습 중 발생한 결과와, 자체 harness 안팎의 held-out benchmark에서 adapter가 얻은 score를 차례로 보고합니다.

### 학습 결과

학습은 누적 총 598 step 동안 진행되었으며, 총 78,114개 sample을 학습했습니다. 이는 정확히 167M개의 학습 token에 해당합니다.

학습 중 모델이 점점 더 적은 step으로 task를 해결하는 것을 보여 주는 명확한 signal이 나타났으며, saturation의 뚜렷한 징후는 없었습니다. 때때로 명확한 상승 추세가 보였지만, training correctness는 전반적으로 평평하게 유지되는 것으로 보입니다. 이는 학습 중 모델이 새로운 task를 만나지만, trajectory가 짧아짐에 따라 한 step의 token budget에 담을 수 있는 group 수가 두 배 이상 증가하기 때문으로 설명할 수 있습니다. accuracy gain은 아래 절의 held-out benchmark에서 나타납니다.

한편 safeagents는 여러 의심스러운 시도를 차단했습니다. 세 번의 training run에서 총 1.3M회의 tool call을 검사했고, 총 647회의 tool call을 차단했습니다.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/reward.html" width="100%" height="500" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/toolcalls.html" width="100%" height="500" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

### Evaluation 결과

이 절에서는 evaluation setup, harness 내부 결과, harness 간 transfer, 그리고 out-of-distribution check를 다룹니다.

#### Evaluation setup

각 adapter를 학습한 harness 내부에서 두 agentic coding benchmark를 사용해 평가합니다. [Terminal-Bench 2.1](https://huggingface.co/datasets/harborframework/terminal-bench-2.1)의 task 89개 전체(task당 1시간 및 100 agent step)와 SWE-bench Pro의 가장 어려운 task 51개(task당 2시간 및 100 agent step)를 사용합니다. task당 최대 token 수는 131072로 제한합니다. agent 설정은 조정하지 않고 모든 harness에서 기본 setup을 사용합니다. 예를 들어 Pi는 이전 reasoning turn을 유지하지 않는 반면 OpenCode와 Hermes는 유지하며, 이는 context가 빠르게 넘치는 SWE-bench Pro 결과에 주로 영향을 줍니다. 따라서 서로 다른 harness의 결과를 비교할 수는 없습니다(Pi를 사용한다고 해서 반드시 Hermes나 OpenCode보다 나은 결과가 나오는 것은 아닙니다). 그러나 동일한 harness 내부의 결과를 비교하고 결론을 내리는 것은 타당합니다.

모든 rollout은 학습과 동일한 Harbor 및 OpenEnv stack을 통해 HF sandbox에서 실행하며, temperature 0.8, thinking off 상태로 task당 한 번 시도합니다. 각 LoRA checkpoint는 base weight에 병합한 뒤 단일 GPU에서 vLLM으로 단독 serving하며, base model 평가와 동일한 concurrency를 사용합니다.

모든 evaluation benchmark에 대해 pass@1과 함께 전체 task 및 해결된 task에 대한 tool call 수와 total turn을 보고합니다.

투명성을 위해 최종 evaluation의 모든 model trajectory를 이 public bucket에 업로드했습니다: [harness-hub/harness-adapters-trajectories](https://huggingface.co/buckets/harness-hub/harness-adapters-trajectories).

#### 최종 evaluation 결과

세 specialist adapter 모두 두 benchmark에서 자체 harness를 사용할 때 base model을 능가했습니다. SWE-bench Pro(hard 51)에서 pi는 39.2%에서 64.7%로 상승했습니다(+25.5 points: 새롭게 해결한 task 15개, 잃은 task 2개, p = 0.002). hermes는 29.4%에서 43.1%로, opencode는 35.3%에서 51.0%로 상승했습니다. Terminal-Bench 2.1 전체에서는 pi가 +16.3 points(16 대 2, p = 0.001), hermes가 +14.1 points(19 대 7, p = 0.03) 향상되었습니다. Opencode는 전체적으로 +6.0, checkpoint 선택에 사용되지 않은 66개 task에서는 +13.6(12 대 3, p = 0.035) 향상되었습니다.

adapter는 task도 훨씬 효율적으로 해결합니다. adapter와 base model이 모두 해결한 task에서 task당 tool call이 SWE-bench Pro에서는 56–72%, Terminal-Bench 2.1에서는 53–70% 감소했습니다. 단일 super-adapter도 세 harness 모두에서 base model을 능가했습니다(SWE-bench Pro에서 +6~+18 points, Terminal-Bench 2.1에서 +5~+10 points). tool call도 동일하게 크게 줄었습니다(53–73%). thinking을 끈 상태에서는 각 specialist의 gain을 덜 유지하며 pi에서 가장 큰 손실이 나타났지만, thinking을 켜면 6개 harness–benchmark pair 중 5개에서 동일한 effort의 base model thinking을 능가합니다([super-adapter section](#super-adapter-merging-all-adapters-into-one) 참조). 각 model card에서 전체 paired 결과를 확인할 수 있습니다.

<iframe src="https://harness-hub-harness-adapters-figures.static.hf.space/results.html" width="100%" height="470" frameborder="0" sandbox="allow-scripts" referrerpolicy="same-origin"></iframe>

#### 효율성과 성능 능력이 harness 간에 transfer되는가?

이 질문에 답하기 위해 pi harness adapter를 Hermes Agent와 OpenCode에서도 평가했습니다.

OpenCode에서는 Terminal-Bench 2.1에서 OpenCode 자체 specialist와 동일한 수준을 보였습니다(44.8% → 54.0%, 자체 specialist는 46.4% → 52.4%). Hermes에서는 specialist gain의 약 3분의 2를 유지했습니다(46.3% → 56.1%, 자체 specialist는 44.7% → 58.8%). 두 경우 모두 tool call이 약 45% 동일하게 감소했습니다. 따라서 새롭게 학습된 capability가 harness 간에 transfer되는 것으로 보입니다.

#### distribution 밖의 task와 setup에서 평가하기

adapter를 학습한 harness 외부의 out-of-distribution task에서 평가하면 어떻게 될까요?

다섯 모델 모두를 300개의 [AutomationBench task](https://github.com/zapier/AutomationBench)에서 실행했습니다. 이 task는 business-automation workflow 평가로 구성되며, Pi, Hermes, OpenCode 대신 AutomationBench 자체 agent loop를 사용합니다. 이를 통해 두 가지 질문을 교차 확인할 수 있습니다. 모델이 SWE 관련 task 외부의 문제를 해결할 수 있는가? 모델을 학습한 harness 외부에 배치하면 어떻게 되는가?

결과에 따르면 adapter는 base model과 비슷한 수준을 유지합니다. Pi와 OpenCode는 약간 앞섰고(23.0% 대비 pass@1 27.0% 및 25.3%), merged adapter는 비슷한 수준(23.7%)이었으며, Hermes는 몇 points 뒤처졌습니다(17.7%). Hermes를 제외하면 차이는 noise 범위 안에 있습니다. 따라서 harness 내부의 RL은 이처럼 매우 다른 설정으로 명확하게 transfer되지도, 모델을 망가뜨리지도 않았습니다.

| Model | pass@1(모든 assertion 충족) | base 대비 | Partial credit(assertion 충족 비율) | base 대비 |
|---|---:|---:|---:|---:|
| base | 23.0% | | 62.2% | |
| pi 160 | 27.0% | +4.0 | 66.2% | +4.0 |
| opencode 150 | 25.3% | +2.3 | 65.2% | +3.0 |
| merged | 23.7% | +0.7 | 60.6% | −1.6 |
| hermes 160 | 17.7% | −5.3 | 56.6% | −5.6 |

## 미해결 질문: 모델은 실제로 무엇을 학습했는가? {#section-10}

사용자와 community의 feedback에 감사드립니다. 초기 testing과 community의 결과는 전반적으로 매우 긍정적으로 보였으며, 앞으로도 이 방향을 계속 이어가고자 합니다. 또한 [harness adapter가 Swift-1.5와 같은 Qwen3.8 derivative에서 매우 잘 작동한다는 사용자 보고](https://huggingface.co/harness-hub/Qwen3.8-27B-opencode-adapted-GGUF/discussions/2)와 같은 놀라운 결과도 확인했습니다.

이 질문은 release 후 고민하게 된 미해결 질문입니다. evaluation benchmark에서 생성된 trajectory를 면밀히 살펴보니, non-thinking mode의 모델은 tool call을 제안하기 전에 짧은 text를 생성하는 경향이 있었습니다. 반면 base model은 사전에 거의 아무 text도 생성하지 않고 바로 tool call을 생성합니다.

이 현상은 일종의 ‘reasoning leakage’로 볼 수 있습니다. 모델이 때때로 reasoning trace처럼 보이는 text(예: ..wait? ..let me see 등)를 생성하여 non-thinking version에서 thinking version으로 ‘탈출’하는 법을 학습하는 것입니다. [최근 논문](https://huggingface.co/papers/2610.14270)도 RL 단계에서 LLM이 보이는 행동을 잘 설명합니다.

이것이 모델을 무효화할까요? 그렇지는 않다고 생각합니다. 최종 결과는 여전히 유효하기 때문입니다. 정확히 동일한 조건(고정된 token budget)에서 우리의 adapter는 base non-thinking model보다 훨씬 뛰어난 성능을 보이며 thinking version의 성능과 맞먹는 동시에 token 효율성은 훨씬 높습니다. 이는 community의 긍정적인 feedback을 통해서도 다시 확인되었습니다.

그러나 이는 적절한 후속 작업의 필요성을 제기합니다. 후속 작업에서는 pure thinking model에 동일한 pipeline을 다시 적용하고, hybrid thinking / non-thinking 학습도 탐색하여 hybrid thinking model에서 RL 학습이 보이는 행동을 더 잘 이해하고자 합니다. 확실한 결과를 얻는 대로 다음 결과(및 harness adapter)를 community와 공유하겠습니다.

## 맺음말과 향후 방향 {#section-11}

이번 작업을 통해 target harness 내부에서 모델에 RL 학습을 적용하면, 동일한 조건에서 평가할 때 더 효율적이고 따라서 더 정확한 최종 모델을 얻을 수 있음을 보였습니다. 이러한 관찰은 Hugging Face team의 여러 이전 작업, 예를 들어 [multi-harness-RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)이나 [harness adapters의 초기 iteration](https://huggingface.co/blog/harness-hub/harness-adapters)과 맥락을 같이하며 이를 이어갑니다.

앞으로는 이 레시피를 더 많은 모델에 적용하는 동시에 hybrid thinking / non-thinking model의 학습 strategy를 더 면밀히 연구할 예정입니다.

또한 모든 내용을 재현할 수 있도록 전체 recipe와 pipeline을 community에 공개적으로 공유합니다. 모든 사람이 자신의 codebase 또는 관련 자료에서 도출한 자체 RL task를 만들고, 동일한 학습 recipe와 아이디어를 선호하는 모델과 harness에 적용해 더 효율적인 agentic model을 구축하기를 바랍니다. 이 blogpost가 앞으로 builder들이 자신만의 harness adapter를 구축하고 공유하는 출발점이 되기를 기대합니다.
