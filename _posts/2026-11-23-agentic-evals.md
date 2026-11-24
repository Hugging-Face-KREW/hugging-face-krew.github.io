---
layout: post
title: "에이전트 평가를 구축하고 유지 관리하는 방법"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/agentic-evals/thumbnail.png
image: assets/images/blog/posts/2026-11-23-agentic-evals/thumbnail.png
authors:
  - user: evijit
  - user: burtenshaw
  - user: irenesolaiman
slug: "agentic-evals"
source_url: "https://huggingface.co/blog/agentic-evals"
source_published_date: "2026-11-23"
source_published_at: "2026-11-23T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [How to build and maintain an agentic evaluation](https://huggingface.co/blog/agentic-evals)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/agentic-evals -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 에이전트 평가를 구축하고 유지 관리하는 방법

## 개요 {#section-1}

AI 평가는 사람들이 어떤 시스템을 만들고 도입하며 신뢰할지를 형성하는 데 기여합니다. 평가는 시스템에 태스크를 제시하고 답변이나 행동을 확인한 뒤 점수를 산출합니다. 따라서 유용한 평가를 구축하고 유지 관리하려면 도메인 전문성을 반영하는 태스크를 정의하고, 이를 신뢰성 있게 채점하며, 결과가 제시된 주장을 뒷받침하는지 확인하는 실질적인 작업이 필요합니다. 방법, 코드, 데이터가 비공개로 남아 있으면 다른 사람들이 이를 검토하거나 기반으로 발전시키기 더 어려워집니다.

오픈 소스는 더 강력하고, 더 자주 업데이트되며, 서로 비교 가능한 평가와 이를 둘러싼 인프라를 구축하는 수단을 제공합니다. 기존 태스크, 환경, 도구는 구축자에게 출발점을 제공하므로, 구축자는 도메인 전문성과 검증에 역량을 집중할 수 있습니다. 무엇이 잘못되었는지를 포함해 평가가 어떻게 구축되었는지 공유하면 다른 사람들이 오류를 찾고, 개선하고, 재사용할 수 있습니다.

Hugging Face Hub는 AI 평가를 위한 공간입니다. 약 3500개의 eval leaderboard를 호스팅하고 있으며, 지난 1년 동안 허브의 eval dataset이 약 [16배 증가했습니다](https://huggingface.co/spaces/huggingface/hub-stats). 현재 약 17000개의 dataset이 있습니다. 최근 몇 달 동안 평가와 관련된 agentic 추상화 지원도 추가했습니다. 예를 들어 에이전트의 작업을 쉽게 검토하고 공유할 수 있도록 저장하는 [Traces](https://huggingface.co/docs/hub/en/agent-traces), leaderboard와 함께 모델 평가에 사용된 dataset을 보여주는 [Benchmarks](https://huggingface.co/datasets?benchmark=benchmark%3Aofficial), agentic RL 또는 평가 환경으로 사용되는 dataset을 보여주는 [Environments](https://huggingface.co/blog/rl-environments)가 있습니다.

이 가이드에서는 네 가지를 다룹니다:

- [**평가의 세 시대**](#1-introduction-how-evaluation-design-has-changed) (1절). 평가는 답변 확인에서 에이전트가 수행한 작업 확인으로, 다시 에이전트 팀이 만들어 낸 결과 확인으로 발전했습니다. 각 단계마다 점수를 바꿀 수 있는 소프트웨어가 추가됩니다.
- [**에이전트 평가의 단계**](#2-building-an-eval-what-to-reuse-adapt-and-validate) (2절). 주장에서 보고된 점수에 이르기까지 하나의 태스크를 여덟 단계로 따라갑니다. 명령어를 포함한 전체 과정은 [Appendix A](#appendix-a-practical-example-agentic-deployment-in-a-physics-lab)에 있습니다.
- [**벤치마크가 잘못되었을 때**](#3-bitter-lessons-when-a-benchmark-is-wrong) (3절). 태스크와 grader의 오류는 흔합니다. 오류가 어떻게 발견되고 수정되는지 살펴봅니다.
- [**무엇을 전략적으로 공개할지와 그 이유**](#4-open-sourcing-an-evaluation-strategically) (4절). 구성 요소별로 무엇을 공유하고 무엇을 보호할 수 있는지 표로 정리합니다.

> [!NOTE]
> 연구 커뮤니티에서 AI 평가 용어는 [일관성이 없기로 악명 높습니다](https://ai.scal.ing/p/evals-intro-1). 문서 끝의 [용어집](#glossary)에서는 사용하는 각 용어를 정의하고, 접하게 될 수 있는 다른 명칭도 설명합니다.

## 1. 서론: 평가 설계는 어떻게 변해 왔는가 {#section-2}

언어 모델의 성능이 향상되고 더 복잡한 작업을 수행할 수 있게 되면서 AI 평가 설계는 세 가지 형태로 발전했습니다. 바로 질의응답, 에이전트 평가, 팀 또는 swarm입니다. [MMLU](https://huggingface.co/papers/2009.03300)(2020)와 [GPQA](https://huggingface.co/papers/2311.12022)(2023) 같은 질의응답 벤치마크는 모델이 생성한 답변을 통해 지식과 추론을 측정합니다. 모델이 계획을 세우고 도구를 사용할 수 있게 되면서 [SWE-bench](https://huggingface.co/papers/2310.06770)와 [WebArena](https://huggingface.co/papers/2307.13854) 같은 에이전트 벤치마크는 소프트웨어 문제 해결과 웹사이트 작업 완료까지 범위를 넓혔습니다. 이러한 태스크를 평가하려면 시스템이 작동하는 환경과 시스템이 가하는 변화를 고려하는 평가 설정이 필요합니다.

최근에는 다중 에이전트 시스템 연구가 또 다른 차원을 추가하기 시작했습니다. 에이전트가 어떻게 협력하고 함께 무엇을 달성하는지를 평가하는 것입니다. 여기에는 [분산형 swarm](https://huggingface.co/papers/2505.04364), [오케스트레이터가 관리하는 팀](https://aclanthology.org/2026.findings-emnlp.662/), [과학 작업에서 협력하는 전문 에이전트](https://arxiv.org/abs/2610.07763)가 포함됩니다. Figure 1은 이 세 평가 시대의 대략적인 연대표를 보여줍니다.

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig1_eval_types.html" title="Three types of language model evaluations" width="100%" height="664" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 1. 언어 모델 평가의 세 가지 유형. 질의응답 벤치마크는 응답을 정답과 비교합니다. 에이전트 벤치마크는 에이전트가 환경에서 무엇을 변경했는지 확인합니다. 팀 및 swarm 벤치마크는 에이전트 그룹이 만들어 낸 결과를 채점합니다.*

이 글에서는 두 가지 이유로 에이전트 평가에 집중합니다. 첫째, 에이전트 배포에 대한 사용자와 기업의 관심이 높아지면서 [모델 출시 발표](https://claude.ai/artifact/22cBPJt6Aos6bYpjmB1BHw#distribution)에서 점점 가장 두드러지는 유형이 되고 있습니다. 둘째, 평가 설계가 모델 성능에 어떤 영향을 미치는지 이해하는 유용한 출발점이 됩니다. 팀 및 swarm 평가는 에이전트 간 상호작용으로 인해 더 복잡해지므로 이 가이드의 범위에서는 제외합니다. 에이전트가 도구를 사용해 태스크를 완료하면 주변 소프트웨어의 각 부분이 점수에 영향을 줄 수 있습니다. 예를 들어 [Terminal-Bench 감사](https://icml.cc/virtual/2026/82793)에서 연구자들은 모델을 고정한 채 agent harness를 변경했습니다. 정확도는 최대 29.9 percentage points까지 달라졌습니다. 이러한 차이를 이해하기 위해 에이전트 평가의 구조를 살펴보고, 에이전트가 받는 지침부터 통과 여부를 결정하는 검사까지 하나의 태스크를 따라갑니다.

### 에이전트 평가의 구조

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig2_anatomy.html" title="Components of an agentic evaluation" width="100%" height="488" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 2. 에이전트 평가를 구성하는 여러 요소.*

에이전트 평가에는 여러 부분이 있습니다. 태스크는 에이전트에 지시를 제공합니다. Agent harness는 모델과 도구를 감싸고, 고정된 환경 안에서 작업할 수 있게 합니다. 하나의 rollout 동안 에이전트는 처음부터 끝까지 태스크를 시도합니다. 여러 명령을 실행하고 출력을 확인한 뒤 멈추기 전에 다시 시도할 수도 있습니다. Evaluation harness는 각 태스크에 대해 여러 rollout을 실행합니다. Rollout이 끝나면 verifier가 환경의 최종 상태를 채점합니다. 각 rollout은 trace도 남기며, trace와 점수는 함께 최종적으로 공개되는 증거의 일부가 됩니다. 각 부분은 독립적으로 공개하고, 검사하고, 재사용할 수 있습니다.

### 벤치마크 공개 결정이 중요한 이유

독립적인 평가에 대한 수요가 커지고 있습니다. 정책 입안자, 기업, 연구자는 검증할 수 있는 평가를 원하지만 현재는 대부분 그렇게 할 수 없습니다. [모델-벤치마크 점수의 98.2%가 단 한 당사자에 의해 보고됩니다](https://huggingface.co/papers/2606.09809). 또한 연구자들은 [사내 평가만으로는 충분하지 않다](https://huggingface.co/papers/2503.16861)고 주장해 왔습니다.

동시에 오염을 줄이기 위해 더 많은 벤치마크가 테스트 항목을 비공개로 유지하고 있습니다. 즉, 질문이나 해답이 학습 데이터에 들어가 점수가 사전 노출을 반영할 수 있는 문제를 줄이는 것입니다. Artificial Analysis는 2026년 10월 발표한 [Intelligence Index v5의 60%](https://x.com/ArtificialAnlys/status/2115265133467635877)가 비공개 테스트 세트 또는 보류된 답변을 사용하는 평가에 기반한다고 말합니다. Google DeepMind는 최근 [이중 블라인드 평가](https://deepmind.google/blog/piloting-the-worlds-first-double-blind-ai-evaluations/)를 시범 운영했습니다. 여기서는 보안 enclave가 외부 평가자의 테스트 프롬프트를 모델 제공자에게서, 모델 가중치를 평가자에게서 숨깁니다. 이는 오염을 줄일 수 있지만, 프라이버시만으로 품질이 보장되지는 않으며 지원 자료를 공개하지 않으면 전문가 검토가 더 어려워질 수 있습니다. 우리는 구축자가 인프라, 채점 방법, 태스크 설계에 관한 증거를 여전히 공유할 수 있고 또 공유해야 한다고 봅니다. 검토가 공개에 적합하지 않은 경우의 통제된 검토를 포함해, 다른 사람들이 측정에 의문을 제기하고 오류를 수정할 방법을 제공해야 검증이 가능합니다.

## 2. 평가 구축: 무엇을 재사용·수정·검증할 것인가 {#section-3}

이제 구조를 살펴보았으므로 실제 사용 사례를 위한 에이전트 평가 하나를 구축해 보겠습니다. 물리학 실험실의 측정값을 분석하는 에이전트입니다. 처음부터 시작하는 대신 오픈 태스크를 수정해 사용하므로, 대부분의 노력은 도메인 전문성과 검증에 투입됩니다.

여덟 단계를 진행합니다. [주장 정의](#step-1-define-the-claim), [태스크 찾기](#step-2-find-a-starting-task), [변경 없이 실행](#step-3-run-the-task-unchanged), [수정](#step-4-adapt-the-task), [평가 확인](#step-5-check-the-evaluation), [trace 읽기](#step-6-run-it-and-read-the-traces), [점수 보고](#step-7-report-the-score), [모델 비교](#step-8-compare-models)입니다. 그런 다음 무엇을 재사용했는지 정리합니다. 전체 과정은 [Appendix A](#appendix-a-practical-example-agentic-deployment-in-a-physics-lab)에 있습니다. Figure 3은 예시에서 각 단계와 그 결과를 보여줍니다.

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig3_eight_steps.html" title="Eight steps to build an agentic evaluation" width="100%" height="837" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 3. 에이전트 평가 구축의 여덟 단계. 왼쪽 패널은 각 단계를 차례로 강조하고, 오른쪽 패널은 예시에서 해당 단계가 포함한 내용을 보여줍니다. 노란색 단계는 평가 자체를 다루고, 주황색 단계는 모델을 실행하고 결과를 읽습니다.*

### 방법

**주장.** Raman spectrum의 peak를 fit하는 에이전트를 원하는 물리학 실험실을 생각해 봅시다. Raman spectrum은 물질의 특성을 보여주는 peak가 있는 측정값이며, 여기서는 graphene을 대상으로 합니다. 에이전트의 작업은 일반적인 코딩입니다. 데이터 파일을 읽고, 각 peak에 curve를 fit하고, 결과를 저장합니다(Appendix A의 Figure 5). 이 글에서 주장은 의도적으로 좁게 설정했습니다. 지정된 peak를 fit하고, 요청된 D-to-G ratio(한 peak의 intensity를 다른 peak와 비교한 값으로 물질의 결함을 나타낼 수 있으며, defect ratio라고도 부름)를 계산하고, 정확한 결과를 일관되게 저장하는 것입니다. 전체 실험실 보고서를 검토하려면 추가적인 기술과 증거가 필요합니다.

**시작 태스크와 baseline.** [Harbor](https://docs.harborframework.com/) 태스크 형식과 Terminal-Bench의 [raman-fitting](https://hub.harborframework.com/tasks/terminal-bench/raman-fitting) 태스크를 재사용했습니다. Harbor는 오픈 소스 evaluation harness이고, Terminal-Bench는 에이전트가 terminal에서 완료하는 태스크의 오픈 벤치마크입니다. 먼저 태스크의 reference solution을 실행해 통과하는 것을 확인한 뒤, terminal agent harness인 [Terminus-2](https://docs.harborframework.com/agents/pre-integrated-agents)와 OpenAI의 open-weight model인 [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)로 원래 태스크를 세 번 실행했습니다. Harbor는 HF Job 안에서 실행했고, 각 rollout은 자체 HF Sandbox를 사용했으며, Inference Providers가 모델 응답을 제공했습니다. runner와 태스크 형식을 재사용함으로써 물리학, 채점, 검토에 집중할 수 있었습니다. 새 작업이 필요하지 않았던 것은 agent harness뿐이었습니다. 나머지 모든 구성 요소는 확인하거나 새로 작업해야 했으며, 이를 [Appendix A의 끝](#how-we-reused-existing-open-components)에서 부분별로 정리합니다.

**수정.** 알려진 parameter와 다양한 noise 및 defect level을 가진 8개의 synthetic spectrum으로 [raman-lab](https://huggingface.co/datasets/burtenshaw/raman-lab)을 구축했습니다. 수정된 버전은 두 개가 아닌 세 개의 peak를 fit하고 defect ratio를 보고합니다. 에이전트의 도움으로 spectrum을 생성하고, 이를 수작업으로 검토했으며, 새 지침·reference fit·output check를 작성했습니다. 새 지침에는 원래 지침에 암묵적으로만 있던 단위와 parameter 정의를 명시했습니다. 이 8개 태스크는 evaluation을 구축하는 동안 사용하는 태스크인 development set입니다. 더 넓은 범위는 실제 측정값의 변화를 따라야 합니다.

**확인.** 추가 모델 실행 전에 평가 자체를 테스트했습니다. reference solution, 아무 작업도 하지 않는 에이전트, 필수 작업을 건너뛰는 shortcut을 실행했습니다. reference solution은 모든 태스크를 통과해야 하고, 나머지 둘은 실패해야 합니다. shortcut은 데이터에서 각 peak의 높이를 바로 읽고 폭을 추정했습니다. spectrum을 fit하지 않았는데도 8개 태스크 모두에서 예상 defect ratio를 반환했으므로, fit 자체를 확인하는 검사를 추가했습니다.

**Shortcut.** 평가가 부정행위에 얼마나 강건한지 확인하기 위해 shortcut도 테스트했습니다. 우리의 shortcut은 실제 fitting을 건너뛰는 짧은 script입니다. 데이터에서 각 peak의 가장 높은 지점을 바로 읽고, peak 폭을 추정한 뒤, D 높이를 G 높이로 나눕니다. 그럼에도 8개 태스크 모두에서 예상 defect ratio를 반환했으므로 ratio만 확인하는 grader라면 만점을 주었을 것입니다. 각 fitted peak에 대한 검사를 추가했고, 최종 verifier는 이를 거부합니다.

### 결과

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig4_results.html" title="Results from building and testing the raman-lab evaluation" width="100%" height="1256" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 4. 평가를 구축하며 여러 측면을 측정해 얻은 내용. (A) shortcut으로 grader를 테스트한 결과 ratio만 확인하는 검사는 속일 수 있었습니다. (B) gpt-oss-120b의 trace를 읽은 결과 대부분의 실패가 너무 일찍 멈춘 데서 비롯되었습니다. (C) 다른 모델을 실행한 결과 태스크 자체는 해결할 수 있었습니다.* [*Appendix A*](#appendix-a-practical-example-agentic-deployment-in-a-physics-lab) *에는 Figure 7(gpt-oss-120b의 시도가 종료된 방식), Figure 8(전체 6개 모델), Figure 9(각 모델이 완료되었다고 말한 시점)에 상세 결과가 있습니다.*

**Baseline.** 원래 태스크에서 gpt-oss-120b는 3개 rollout 중 0개를 통과했습니다. 두 번은 관련 출력을 확인하기 전에 완료를 선언했고, 세 번째에는 데이터 파일을 읽지 못했습니다. 이 파일은 reference solution이 로드 전에 변환하는 decimal comma를 사용합니다. 점수만으로는 이러한 실패와 잘못된 fit을 구분할 수 없었지만 trace로는 가능했습니다.

**Trace와 점수.** 수정한 8개 태스크에 대해 태스크당 3개의 rollout을 실행했습니다. 수정된 설정은 24개 rollout 중 11개를 통과했습니다. 19개에서는 모델이 첫 응답에서 `mark_task_complete`를 호출했고, 그중 11개는 results file을 생성하지 않았습니다. 여러 시도는 NumPy 또는 SciPy가 없어 script가 실패하는 것으로 시작했는데, 태스크는 에이전트가 이를 설치할 것을 요구합니다. 따라서 대부분의 실패는 너무 일찍 멈추거나 package가 없어서 발생했고, 13개의 실패 중 fit check에 실패한 것은 2개뿐이었습니다(Appendix A의 Figure 7). pass@3은 7/8이었습니다. 7개 태스크가 적어도 한 번 성공했습니다. pass^3은 1/8이었습니다. 한 태스크만 매번 성공했습니다. 사람이 출력을 검토한다면 재시도가 허용될 수 있습니다. 무인 처리는 일관성이 필요하므로 두 번째 측정값이 더 중요합니다. 평균 pass rate는 0.46이었고, rollout을 태스크별로 cluster했을 때 standard error는 약 0.11이었습니다. 반복 실행에서는 24개 중 10개가 통과했습니다. development task가 8개뿐이므로 어느 결과든 새로운 spectrum으로 일반화할 때는 신중해야 합니다.

**모델 비교.** 같은 8개 태스크에서 6개 모델을 비교했습니다(Figure 4). 여러 provider가 서로 다른 quantization과 context limit으로 같은 open-weight model을 제공할 수 있으므로 모델마다 하나의 provider를 고정했습니다. DeepInfra를 provider로 고정했을 때 gpt-oss-120b는 24개 rollout 중 13개를 통과했습니다. 이는 default provider를 사용한 이전 실행의 11개보다 많습니다. 첫 응답에서 완료를 선언한 모델은 gpt-oss-120b뿐이었으며, 24개 rollout 중 22개에서 그렇게 했습니다. 나머지 5개 모델은 어떤 rollout에서도 그렇게 하지 않았습니다(Appendix A의 Figure 9). 비교 결과 development set의 한계도 드러납니다. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)는 모두 24개 raman-lab rollout을 통과했으므로 이 태스크로는 이들을 순위화할 수 없습니다. 더 어려운 태스크가 필요합니다.

## 3. 쓰라린 교훈: 벤치마크가 잘못되었을 때 {#section-4}

우리의 shortcut은 spectrum을 fit하지 않고도 예상 defect ratio를 반환했습니다. ratio만 확인하는 verifier라면 8개 태스크 모두에 만점을 주었을 것입니다. 우리는 불완전하다는 것을 알고 있는 solution으로 grader를 깨뜨려 보다가 이를 발견했고, fitted spectrum에 대한 검사를 추가했습니다([Appendix A, Step 5](#step-5-check-the-evaluation)).

shortcut은 작업 과정을 베끼지 않고 최종 숫자만 베낀 학생처럼, 잘못된 이유로 정답을 냈습니다. 우리에게 주어진 교훈은 출력이 무엇을 입증하는지 테스트해야 한다는 것입니다. 더 빠르지만 유효한 방법은 통과해야 하고, 필수 기술을 건너뛰는 방법은 통과해서는 안 됩니다. 도메인 전문가는 verifier가 다른 올바른 fit은 받아들이면서도 태스크의 핵심을 놓친 작업은 받아들이지 않도록 tolerance를 설정하는 데 도움을 줄 수 있습니다.

이러한 오류는 흔하며 점수를 상당히 바꿉니다. [에이전트 벤치마크 검토](https://huggingface.co/papers/2507.02825)에서는 태스크 설정이나 reward 설계의 결함이 측정된 성능을 상대적으로 최대 100%까지 바꿀 수 있다고 밝혔습니다.

### 벤치마크를 수정하려면 공개 검토가 필수다

공개 검토는 벤치마크 구축자가 놓친 지침과 테스트의 오류를 드러낼 수 있습니다. Terminal-Bench 2.1은 사용자 보고 후 [89개 태스크 중 28개를 수정했습니다](https://www.tbench.ai/news/terminal-bench-2-1). 여기에는 지침은 PostgreSQL을 요청하면서 테스트는 Spark SQL을 예상한 태스크도 포함되며, 수정 과정에서도 [에이전트 순위는 안정적으로 유지되었습니다](https://icml.cc/virtual/2026/82793).

증거에 접근할 수 있는지도 외부 감사에 필요한 작업량을 결정합니다. CritPt는 reference solution을 비공개로 유지했으므로 [이를 감사한 physicist](https://arxiv.org/abs/2609.13009)들은 검토한 56개 challenge 중 21개에서 오류를 찾기 전에 문제를 직접 해결해야 했습니다. 이후 저자들은 [수정 버전](https://artificialanalysis.ai/evaluations/critpt-v2)을 공개했습니다. OpenAI의 [SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) 감사는 누구나 읽을 수 있는 태스크와 테스트를 바탕으로 조사한 138개 문제 중 59.4%에서 중대한 문제를 발견했습니다. 구축자에게 외부 검토는 검증의 일부입니다. 검토자가 태스크에 이의를 제기하는 데 필요한 자료를 공개하고, 문제를 보고할 방법을 제공하며, 무엇이 변경되었는지 기록해야 합니다.

SWE-bench Pro에서도 비슷한 결함이 나타났습니다. 2026년 7월 OpenAI는 [SWE-Bench Pro 태스크의 약 30%가 손상되었다고 추정했고](https://openai.com/index/separating-signal-from-noise-coding-evaluations/), 도입 권고를 철회했습니다. 벤치마크 검증은 공개 이후에도 계속되어야 합니다.

### 코드뿐 아니라 주장도 확인하라

일관된 점수도 여전히 잘못된 것을 측정할 수 있습니다. 29명의 전문가가 445개 LLM 벤치마크를 검토한 [Measuring what Matters](https://huggingface.co/papers/2511.04703)는 construct validity, 즉 태스크와 metric이 의도한 능력에 관한 주장을 뒷받침하는지 여부를 약화시키는 태스크와 metric의 반복적인 패턴을 발견했습니다.

예를 들어 synthetic spectrum을 fit한다고 해서 에이전트가 실험실 보고서를 검토할 수 있다는 뜻은 아닙니다. 잘못된 측정을 인식하거나 도움을 요청하는 능력도 필요할 수 있습니다. 모든 태스크가 사용할 수 있는 spectrum을 제공한다면, fit이 부적절한 경우를 에이전트가 아는지는 보여줄 수 없습니다.

[Harder Than the Task](https://arxiv.org/abs/2611.03494)는 이러한 우려를 권한, 분산된 지식, 의존성이 성공을 좌우할 수 있는 조직 작업으로 확장합니다. 저자들은 주요 벤치마크 중 이러한 차원을 다룬다고 주장하는 것이 4분의 1 미만이라고 밝혔습니다. 에이전트는 다른 팀의 calibration data나 측정을 교체할 권한이 필요할 수 있는데, 처음부터 모든 것을 제공하면 이러한 어려움이 사라집니다. 그것이 주장에 중요하다면 평가에 포함해야 합니다. 중요하지 않다면 아직 테스트하지 않았다고 말하면 충분합니다.

각 오류에 대해서는 실패한 가정, 발견 방법, 변경 사항을 짧게 기록할 가치가 있습니다. 기록에는 영향을 받은 버전과 이전 점수를 계속 비교할 수 있는지도 명시해야 합니다.

### 오염과 포화

앞서 설명한 SWE-bench Verified 감사에서는 오염도 발견했습니다. 테스트한 모든 frontier model이 일부 태스크에서 reference patch 또는 문제의 세부 내용을 그대로 재현할 수 있었습니다. SWE-bench의 공개 repository와 광범위한 사용으로 인해 태스크가 학습 데이터에 노출되었습니다. 잘못된 테스트를 검토할 수 있게 한 개방성이 평가 전에 항목을 접할 가능성도 높인 것입니다.

하지만 학습 중 오염을 신중하게 통제하더라도 테스트 세트를 숨기는 것만으로 포화를 막을 수는 없습니다. [60개 언어 모델 벤치마크 연구](https://huggingface.co/papers/2602.16763)는 공개 벤치마크와 비공개 벤치마크에서 비슷한 포화를 발견했으며, 벤치마크가 널리 채택되면 테스트 데이터를 숨겨도 포화를 막을 수 없다고 결론 내렸습니다. 즉, 좋은 벤치마크도 개발자가 이를 통해 배우고 기반으로 발전시킬 수 없는 상태에서 포화로 인해 사용되지 않게 됩니다. 포화율을 낮추는 것과 관련이 있었던 요인은 전문가의 선별과 더 큰 테스트 세트였습니다.

따라서 실무적인 공개 문제가 남습니다. 활성 테스트 항목을 보호하면서 검토를 위해 무엇을 공개할 수 있을까요?

## 4. 전략적으로 평가를 오픈 소싱하기 {#section-5}

오염 위험과 포화 사이의 균형을 맞추려면 어떤 콘텐츠를 공개할지 전략적으로 접근해야 합니다. 개방성을 전부 아니면 전무인 결정으로 다루기보다, 벤치마크 제작자는 구성 요소별로 공개 여부를 결정할 수 있습니다. 이를 통해 활성 테스트 태스크는 데이터 유출로부터 보호하면서도 외부 연구자가 평가를 감사하고 기반으로 발전시키는 데 필요한 기본 인프라, 채점 방법, reference example을 공유할 수 있습니다.

기존 벤치마크는 다양한 공개 수준을 보여줍니다. [ARC-AGI-2](https://arcprize.org/arc-agi/2)는 training set과 120개 태스크로 구성된 evaluation set을 공개하고, 각각 120개 태스크로 된 semi-private set과 private set은 보류합니다. [AILuminate](https://mlcommons.org/ailuminate/faq/)는 공식 prompt를 비공개로 유지하고 방법과 taxonomy를 공개하며, 승인된 감사자와 신뢰할 수 있는 연구자가 통제된 방식으로 검토할 수 있도록 기본 증거를 보관합니다. Terminal-Bench는 태스크를 공개하고 파일에 [canary string](https://huggingface.co/datasets/harborframework/terminal-bench-2.1/blob/main/tasks/raman-fitting/solution/solve.sh)를 표시해 training corpus에서 제외해 달라고 요청합니다. 공개 검토자는 보호된 항목을 확인할 수 없으므로, 비공개로 남기는 항목에는 자체적인 검토 경로가 필요합니다.

Table 1은 평가를 오픈 소싱할 때의 핵심 고려 사항을 정리합니다. 무엇을 공개할 수 있는지, Hugging Face Hub에서 어디에 둘지, 무엇을 보호할 수 있는지를 보여줍니다. Figure 2의 각 구성 요소마다 한 행이 있으므로, 곧 공개하려는 부분으로 바로 이동할 수 있습니다.

| 부분 | 공개할 수 있는 것 | Hugging Face Hub에서 공유하는 방법 | 공개 검토를 위한 대안을 두고 보호할 수 있는 것 |
| --- | --- | --- | --- |
| 태스크 | 사양, 범주, 구축 방법, 공개 development split | source task, 변경 사항, license, 한계를 기록한 [dataset card](https://huggingface.co/docs/hub/datasets-cards)와 함께 dataset repository에 게시 | 별도의 [gated dataset](https://huggingface.co/docs/hub/datasets-gated)에 활성 테스트 항목과 답변을 보관하고 각 access request를 수동 승인합니다. 허용되는 경우 만료된 항목을 공개합니다. |
| 환경 | Container 정의, build file, 고정 dependency, synthetic fixture | Dataset repository에서 dataset card에 [environment type](https://huggingface.co/datasets?other=rl-environment&sort=trending)로 선언 | 비공개 데이터, credential, license가 있는 asset, 위험한 세부 정보. 공개 대체물이 어떻게 다른지 문서화합니다. |
| Agent harness 및 도구 | Agent-loop code, tool interface, adapter, permission, budget | 사용한 configuration과 함께 고정 버전을 run record에 보관하고 harness source 링크를 제공합니다. 각 `.eval_results` 항목의 notes field에 고정된 harness version을 제공해 점수 옆에 표시합니다. | Secret, 비공개 service, 민감한 tool 구현. 안전한 대체물과 승인된 검사를 제공합니다. |
| Rollout 및 trace | 공개 태스크 trace, 검토된 발췌, 분석 code | Dataset 또는 storage bucket. Harbor의 [trace exporter](https://github.com/harbor-framework/harbor/blob/26106c6d9c66848a1440faf5724d336affac52aa/docs/content/docs/training-workflows/sft.mdx)가 rollout record를 전송하고 Hub의 [trace viewer](https://huggingface.co/docs/hub/agent-traces)가 지원되는 session format을 엽니다. 각 `.eval_results` 항목의 source field를 통해 점수를 trace에 연결합니다. | 태스크, 비공개 데이터, 제한된 모델 동작을 노출하는 trace. 보호된 분석과 사례 검사를 제공합니다. |
| Reference solution | 공개 예시의 solution method 및 code | 각 태스크와 함께 dataset repository에 게시 | 보호된 항목을 드러내는 solution. 승인된 검토자가 태스크와 대조할 수 있게 합니다. |
| Evaluation harness | Rollout orchestration, model configuration, logging, score aggregation, version record | Dataset과 함께 보관한 run record에 job script, model 및 harness configuration, package version을 기록합니다. Inspect는 log를 `hf://buckets/` 경로에 직접 씁니다. | Credential과 항목을 식별할 수 있는 configuration. 정확한 제한 실행을 검토 가능한 기록으로 남깁니다. |
| Verifier | Rubric, grader code, tolerance, worked example | 각 태스크와 함께 dataset repository에 게시 | 보호된 항목을 드러내는 reference answer와 logic. 승인된 전문가가 이견이 있는 판정을 검사해야 합니다. |
| 주장 및 태스크 설계 | 의도된 사용, 선택 근거, 맥락적 가정, 검증 증거 | Dataset card. Task dataset, trace dataset, harness Space, paper를 하나의 Hub Collection으로 묶습니다. | 민감한 원자료는 redaction이 필요할 수 있습니다. |
| 결과 | Model setup과 기타 재현성 세부 정보가 포함된 점수, 불확실성, 비용, 제외 사항, 한계 | Dataset을 `eval.yaml`과 함께 Hub Benchmark로 등록합니다. 점수는 각 model repo의 `.eval_results/` folder에 넣고 [pull request](https://huggingface.co/docs/hub/eval-results)로 열며, source를 trace dataset에 연결합니다. 선택적으로 [EEE schema](https://github.com/evaleval/every_eval_ever)로 변환할 수도 있습니다. | 결과에 포함된 태스크 식별 정보. 안전한 요약을 공개하고 기본 증거 검토를 마련합니다. |
| 수정 | 영향을 받은 버전을 명시한 수정 이력 | Repository의 [pull request와 discussion](https://huggingface.co/docs/hub/repositories-pull-requests-discussions), card의 change log | 수정의 원인이 된 보호된 example. 그래도 영향을 받은 버전은 기록합니다. |

*표 1. 각 평가 구성 요소에서 무엇을 공개하고, Hugging Face Hub에서 어떻게 공유하며, 무엇을 보호할 수 있는지.*

우리의 예시도 이 접근법을 따릅니다. raman-lab을 [Hub의 dataset](https://huggingface.co/datasets/burtenshaw/raman-lab)으로 공개했습니다. 여기에는 Harbor 태스크 형식의 synthetic graphene spectrum 8개가 포함되며, 각각에 environment, reference fit, verifier, tolerance가 있습니다. 결과의 기반이 된 raw rollout은 [별도의 dataset](https://huggingface.co/datasets/burtenshaw/raman-lab-runs)으로 공유했습니다. 여기에는 6개 모델 모두의 trajectory와 verifier output이 담긴 Harbor trial folder가 포함됩니다. Spectrum generator, analysis code, result는 함께 제공되는 [GitHub repository](https://github.com/burtenshaw/raman-lab-evals)에 있으므로 agent harness와 evaluation harness를 모두 검사하고 다시 실행할 수 있습니다.

더 민감한 분야에서는 clinical evaluation이 environment 정의, harness, tool interface, grading framework를 공개하고, 개발용 synthetic case를 제공할 수 있습니다. 환자 기록과 reference answer는 통제된 접근 아래 두고, 승인된 검토자가 이견이 있는 판정을 검사하도록 합니다. 각 제한에는 재검토 조건이 따라야 합니다.

### 전략적 오픈 소싱 체크리스트

보다 포괄적인 [Agentic Benchmark Checklist](https://huggingface.co/papers/2507.02825)를 일부 참고해 짧은 checklist를 권장합니다. 평가를 공개하거나 폐기하기 전에 다른 구축자가 다음을 할 수 있는지 확인합니다.

- 무엇을 측정하는지, 의도된 사용 목적, 제외되는 조건을 파악할 수 있다.
- 재사용 가능한 artifact, license와 version, 문제를 보고할 방법을 찾을 수 있다.
- 아무 작업도 하지 않는 agent의 점수와, 가능한 경우 human expert의 점수를 확인할 수 있다.
- 각 점수를 불확실성과 함께 읽고, 알려진 결함이 점수에 어떤 의미를 갖는지 안내받을 수 있다.
- 어떤 결함이 발견되었고 어떻게 처리되었으며 결과를 얼마나 바꿀 수 있었는지 알 수 있다.
- 오염과 overfitting을 제한하기 위해 태스크를 언제 어떻게 갱신할지 알 수 있다.

이러한 공개 결정은 벤치마크가 비교 대상으로 삼은 시스템을 더 이상 구분하지 못하게 된 뒤에도 무엇이 유용하게 남을지를 결정합니다.

## 5. 결론: 평가에 유용한 다음 생명을 부여하기 {#section-6}

벤치마크에는 lifecycle이 있습니다. 벤치마크는 구축되고, 채택되고, 최적화의 대상이 되며, 시간이 지나면 비교를 위해 만들어진 시스템들을 더 이상 구분하지 못하게 됩니다. 3절의 [포화 연구](https://huggingface.co/papers/2602.16763)는 60개 언어 모델 벤치마크 중 29개를 높은 포화 상태로 평가했으며, 그 비율은 오래된 벤치마크일수록 높아졌습니다. 포화 자체가 실패를 의미하는 것은 아닙니다. 해당 능력이 도달되었다는 뜻일 수도 있기 때문입니다.

벤치마크가 남기는 것은 얼마나 공개적이었는지에 달려 있습니다. 점수만 공개하면 이후 구축자가 실행하거나 수정할 수 있는 것이 거의 남지 않습니다. 태스크, environment, verifier, trace가 공개되어 있으면 leaderboard가 멈춘 뒤에도 각각 계속 작동합니다. 해결된 태스크는 새 harness를 위한 regression check가 됩니다. Environment와 verifier는 더 어려운 후속 태스크의 기반이 될 수 있습니다. 공개 벤치마크는 수정되고 재발행되기도 합니다. MMLU의 질문은 [MMLU-Redux](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0)가 되었고, Terminal-Bench 2.0은 커뮤니티 기여를 통해 [2.1](https://www.tbench.ai/news/terminal-bench-2-1)이 되었습니다. 우리의 과정도 이러한 다음 생명의 한 사례입니다. raman-lab은 범용 terminal 작업을 위해 구축된 벤치마크의 한 태스크로 시작했습니다.

이것이 평가를 공개적으로 구축하는 것의 약속입니다. 평가에 투입된 작업은 벤치마크 자체보다 오래 남아 다른 사람들이 새로운 질문을 하고, 새로운 능력을 테스트하며, 이전 결과를 개선할 기반을 제공합니다. 우리가 구축하고 배운 것을 더 많이 공유할수록 새로운 평가가 처음부터 시작해야 하는 일은 줄어듭니다.

실제로 어떤 모습인지 궁금하다면 실용적인 예시를 따라가 보세요.

## Appendix A. 실용 예시: 물리학 실험실에서의 에이전트 배포 {#section-7}

Raman spectrum의 peak를 fit하려는 물리학 실험실을 생각해 봅시다. Raman spectroscopy는 sample에 laser를 비추고 산란된 빛의 주파수 변화를 측정하며, 그 결과로 나타나는 peak가 물질의 특성을 보여줍니다. 탄소 원자 한 층 두께의 graphene에서는 1580과 2700 cm⁻¹ 부근의 G 및 2D peak가 [층 수를 식별합니다](https://arxiv.org/abs/cond-mat/0606284). 세 번째 peak인 D는 격자가 무질서할 때 나타나며, G에 대한 intensity는 [결함을 나타낼 수 있습니다](https://arxiv.org/abs/1306.5856). Figure 5는 설정을 보여줍니다.

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig5_raman_setup.html" title="The Raman measurement workflow" width="100%" height="511" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 5. 의도한 설정. Laser가 graphene sample을 비추고 spectrometer가 산란된 빛의 변화를 기록합니다. 에이전트는 해당 spectrum을 file로 받아 각 peak에 curve를 fit하고 fitted value를 저장합니다. Verifier는 저장된 값을 reference fit과 비교합니다. 표시된 curve는 sketch이며 측정 데이터가 아닙니다.*

오픈 벤치마크의 태스크를 기반으로 시작합니다. [Terminal-Bench 2](https://huggingface.co/datasets/harborframework/terminal-bench-2.0)에는 이미 [raman-fitting](https://hub.harborframework.com/tasks/terminal-bench/raman-fitting)이 포함되어 있습니다. 두 peak 각각에 수학적 peak shape인 Lorentzian curve를 fit하고 8개 parameter를 `/app/results.json`에 저장하는 태스크입니다. 이는 더 좁은 범위의 실험실 평가를 실행할 수 있는 출발점을 제공합니다. 이 태스크는 오픈 소스 evaluation harness인 [Harbor](https://docs.harborframework.com/)에서 실행합니다. Table 2는 태스크가 제공하는 것과 각 부분을 재사용하기 전에 확인할 사항을 정리합니다.

| 부분 | raman-fitting에서의 내용 | 재사용 전 확인 사항 |
| --- | --- | --- |
| 태스크 | `instruction.md`에서 두 peak fit을 `/app/results.json`에 저장하도록 요청합니다. | 단위, parameter 정의, output format이 명확한가? |
| 환경 | Spectrum을 포함한 고정 container image이며, `task.toml`가 resource와 limit을 설정합니다. | 에이전트가 설치해야 하는 package는 무엇인가? Resource와 permission이 적절한가? |
| Agent harness 및 도구 | Terminus-2가 모델을 shell command에 연결합니다. | 에이전트가 command의 결과를 확인할 수 있는가? |
| Rollout 및 trace | `agent/trajectory.json`에 기록된 한 번의 시도입니다. | 에이전트가 멈추기 전에 무슨 일이 있었는지 기록에 나타나는가? |
| Reference solution | `solution/solve.sh`가 알려진 fit을 실행합니다. | 제공된 environment에서 통과하는가? |
| Evaluation harness | Harbor가 rollout을 실행하고 score와 trace를 수집합니다. | Version과 execution setting이 기록되는가? |
| Verifier | `tests/test.sh`가 8개 fitted value를 reference tolerance와 비교합니다. | 저장된 file을 채점하고 유효한 대체 fit을 받아들이는가? |

*표 2. 각 구성 요소를 재사용하기 전에 확인할 사항.*

### Step 1. 주장 정의

시작하기 좋은 지점은 *주장*입니다. 주장은 평가가 뒷받침하려는, 에이전트가 할 수 있는 일을 평이하게 표현한 문장입니다. "에이전트는 Raman spectrum을 fit할 수 있다"도 주장이고, "에이전트는 실험실 보고서를 처리할 수 있다"도 주장입니다. 여기서 파일 읽기, peak 찾기, output 작성처럼 한 번에 하나씩 확인할 수 있는 더 작은 skill로 주장을 나눌 수 있습니다.

이 글에서는 주장을 의도적으로 좁게 설정했습니다. 지정된 peak를 fit하고, 요청된 D-to-G ratio를 계산하며, 정확한 결과를 일관되게 저장하는 것입니다. 전체 실험실 보고서를 검토하려면 추가적인 skill과 증거가 필요합니다.

여기서 도메인 전문가의 도움을 받을 수 있습니다. 현실적인 noise level, tolerance, 사용 가능한 software를 선택하도록 요청한 뒤, example을 독립적으로 채점하고 의견 차이를 조사하는 방법을 권장합니다. 그런 다음 주장을 확인 가능한 부분으로 나누고 각 부분을 각 태스크가 테스트하는 내용에 매핑할 수 있습니다.

### Step 2. 시작 태스크 찾기

자신의 도메인에서는 Hugging Face Hub의 Benchmark 및 Environment dataset type을 살펴보는 것이 좋습니다. 후보를 찾았다면 지침, environment, [reference solution](https://huggingface.co/datasets/harborframework/terminal-bench-2.1/blob/main/tasks/raman-fitting/solution/solve.sh), test를 확인할 가치가 있습니다. 모델이 이미 해결할 수 있는 태스크라도 유용한 인프라를 제공할 수 있습니다. raman-fitting에서 작동하는 태스크 형식을 가져왔습니다. 다만 spectrum이 하나뿐이고 실험실의 측정값은 다양하므로 Step 4에서 설명하듯 이 태스크를 확장합니다.

### Step 3. 태스크를 변경 없이 실행

모델을 실행하기 전에 reference solution(oracle이라고도 함)을 먼저 실행하면 도움이 됩니다. 실패한다면 원인은 태스크, environment, verifier에 있으며, 모델 실패를 해석하기 전에 수정해야 합니다. 그 후 설정을 변경하지 않고 모델을 여러 번 실행할 수 있습니다.

runner에 compute를 제공하는 [HF Job](https://huggingface.co/docs/hub/jobs-overview) 안에서 Harbor를 실행했습니다. 각 rollout은 command와 file을 위한 자체 [HF Sandbox](https://huggingface.co/docs/huggingface_hub/guides/sandbox)를 사용했고, [Inference Providers](https://huggingface.co/docs/inference-providers/index)가 모델 응답을 제공했습니다. Harbor의 `hf-sandbox` extra installed와 `HF_TOKEN`를 설정한 뒤, 다음 command는 oracle을 테스트하고 terminal agent harness인 [Terminus-2](https://docs.harborframework.com/agents/pre-integrated-agents)와 OpenAI의 open-weight model인 [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)로 세 번의 rollout을 실행합니다.

```bash
harbor run -t terminal-bench/raman-fitting \
  -a oracle -e hf-sandbox

OPENAI_API_KEY="$HF_TOKEN" harbor run \
  -t terminal-bench/raman-fitting -k 3 -e hf-sandbox \
  -a terminus-2 -m openai/openai/gpt-oss-120b \
  --ak api_base=https://router.huggingface.co/v1
```


이 command에서 `-t`은 태스크, `-a`은 agent, `-e`는 environment, `-m`은 model, `-k`는 태스크당 rollout 수를 지정합니다.

모델은 3개 rollout 중 0개를 통과했습니다. 두 번은 관련 출력을 보기 전에 완료를 선언했고, 세 번째에는 데이터 파일을 읽지 못했습니다. 이 파일은 reference solution이 로드 전에 변환하는 decimal comma를 사용합니다. 점수만으로는 이러한 실패와 잘못된 fit을 구분할 수 없었지만 trace로는 가능했습니다.

### Step 4. 태스크 수정

알려진 parameter와 다양한 noise 및 defect level을 가진 synthetic spectrum 8개로 [raman-lab](https://huggingface.co/datasets/burtenshaw/raman-lab)을 구축했습니다. 수정 버전은 두 개가 아닌 세 개의 peak를 fit하고 defect ratio를 보고합니다. 에이전트의 도움으로 spectrum을 생성하고 수작업으로 검토했으며, 새 지침·reference fit·output check를 작성했습니다. 이 8개 태스크는 평가를 구축하는 동안 사용하는 development set입니다. 더 넓은 범위는 실제 측정값의 변화를 따라야 합니다.

D-to-G intensity ratio는 결함을 나타낼 수 있지만 그 해석은 [laser excitation energy와 disorder regime에 따라 달라집니다](https://pubmed.ncbi.nlm.nih.gov/21696186/). "defect ratio"를 요청하는 평가라면 어떤 peak를 사용할지, intensity가 높이를 의미하는지 아니면 적분 면적을 의미하는지, 결과를 어떻게 확인할지 명시해야 합니다.

원래 [instruction.md](https://huggingface.co/datasets/harborframework/terminal-bench-2.1/blob/main/tasks/raman-fitting/instruction.md)는 이러한 정의가 중요한 이유를 보여줍니다. x0, gamma, amplitude, offset을 요청하지만 이를 정의하지 않고 단위도 제시하지 않습니다. Reference solution은 이를 각각 peak center, half-width at half-maximum, background 위의 height, background level로 취급하고, x-axis를 nanometre에서 Raman shift로 변환합니다. 이러한 선택을 수정된 지침에 넣어 verifier의 요구 사항이 드러나도록 했습니다. Figure 6은 두 file을 나란히 보여줍니다.

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig6_instructions.html" title="Side-by-side instructions for raman-fitting and raman-lab" width="100%" height="677" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 6. [*raman-fitting*](https://huggingface.co/datasets/harborframework/terminal-bench-2.1/blob/main/tasks/raman-fitting/instruction.md)과 [*raman-lab*](https://huggingface.co/datasets/burtenshaw/raman-lab/blob/main/graphene-s01/instruction.md)의 `instruction.md` file을 나란히 비교한 모습. 강조된 추가 내용은 주장의 새로운 부분인 D peak, defect ratio, 저장된 결과의 두 항목을 다룹니다. 다양한 측정값을 처리하는 나머지 부분은 data에 있습니다. 8개 raman-lab 태스크는 모두 이 지침을 공유하고 noise 및 defect level이 포함된 spectrum만 달라집니다.*

Verifier는 세 가지 규칙을 따릅니다.

- 에이전트가 생성한 것을 채점합니다. 완료했다고 주장하는지가 아니라 저장된 output을 검사합니다. [τ-bench](https://huggingface.co/papers/2406.12045)는 최종 database state를 주석이 달린 goal state와 비교하는 방식으로 이를 수행합니다.
- 단위, parameter convention, tolerance, 필수 field를 포함해 확인할 모든 사실을 지침에 명시합니다.
- 알려진 parameter에서 synthetic data를 생성해 reference answer를 확보한 뒤, example이 의도한 문제를 대표하는지 검토합니다.

Data와 check가 바뀌더라도 공유 형식은 그대로 둘 수 있습니다. 다음 command는 task scaffold를 생성합니다.

```bash
harbor task init my-org/task-01 -p datasets/my-evals
```


### Step 5. 평가 확인

추가 모델 실행 전에 평가 자체를 테스트합니다. 여기서 `datasets/my-evals`에는 수정된 태스크가 들어 있습니다.

```bash
harbor run -p datasets/my-evals -a oracle -e hf-sandbox
harbor run -p datasets/my-evals -a nop -e hf-sandbox
```


- Oracle은 모든 태스크를 통과해야 합니다. 즉, 알려진 solution이 제공된 environment에서 작동해야 합니다.
- No-op agent는 아무것도 하지 않으므로 모든 태스크에서 실패해야 합니다. 빈 시험지가 통과해서는 안 됩니다. [에이전트 벤치마크 검토](https://huggingface.co/papers/2507.02825)에서는 빈 응답을 성공으로 센 사례가 발견되었습니다.
- 필수 작업을 건너뛰는 shortcut은 실패해야 합니다. 이를 테스트하는 한 방법은 작업을 하지 않고 그럴듯한 답을 출력하는 의도적으로 게으른 solution을 사용해 verifier가 이를 받아들이는지 확인하는 것입니다. 그럴듯하지만 틀린 답과 올바른 대체 방법도 시도해 false pass와 false failure를 모두 찾아야 합니다.

우리의 shortcut은 spectrum을 fit하지 않고도 8개 태스크 모두에서 예상 defect ratio를 반환했으므로 fit 자체에 대한 검사를 추가했습니다. 3절에서는 이것이 verifier 설계에 어떤 의미인지 설명합니다.

### Step 6. 실행하고 trace 읽기

성공한 rollout과 실패한 rollout을 모두 읽는 것이 유용합니다. Harbor는 [ATIF](https://docs.harborframework.com/agents/atif)라는 공유 trace format을 사용해 `agent/trajectory.json`에 message, tool call, observation을 기록합니다. 실패를 원인별로 묶으면 data parsing, tool 실행, 너무 이른 중단, output 채점과 같은 패턴을 더 쉽게 볼 수 있습니다.

수정된 설정은 24개 rollout 중 11개를 통과했습니다. 19개에서는 모델이 첫 응답에서 `mark_task_complete`를 호출했고, 그중 11개는 results file을 생성하지 않았습니다.

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig7_gptoss_outcomes.html" title="gpt-oss-120b rollout outcomes on raman-lab" width="100%" height="218" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 7. 24개 시도 중 19개에서 에이전트가 첫 응답으로 `mark_task_complete`를 호출했습니다. 11개는 results file을 생성하지 않았고, 7개는 통과했으며, 1개는 fit check에 실패했습니다. 나머지 5개 시도 중 4개는 통과하고 1개는 fit check에 실패했습니다.*

여러 missing-file 실패는 NumPy 또는 SciPy가 없는 상태에서 시작되었습니다. 공개된 태스크는 에이전트가 이를 설치할 것을 요구하며, reference solution은 pip install로 시작합니다. 제공된 도구로 fitting을 수행하는 실험실 평가라면 이를 미리 설치하고 고정할 수 있습니다. Dependency도 변합니다. Terminal-Bench 2.1은 공개 이후 외부 dependency가 변경된 [9개 태스크를 수정했습니다](https://www.tbench.ai/news/terminal-bench-2-1).

미리 설치하면 평가 조건이 바뀌므로 두 설정을 비교할 때 model, harness, task, rollout 수를 고정해야 합니다.

마지막으로 설정을 수정하고 Step 5의 검사를 반복한 뒤 다시 실행합니다. Harbor의 [trace exporter](https://github.com/harbor-framework/harbor/blob/26106c6d9c66848a1440faf5724d336affac52aa/docs/content/docs/training-workflows/sft.mdx)는 Hub에서 공유할 record를 준비할 수 있습니다. 먼저 보호된 콘텐츠가 있는지 검토하고, 다른 사람들이 결과를 조사할 수 있도록 성공과 실패를 모두 보관하는 것이 좋습니다.

### Step 7. 점수 보고

적절한 metric은 내려야 하는 결정에 따라 달라집니다.

- [pass@k](https://huggingface.co/papers/2107.03374)는 k개 rollout 중 하나 이상이 성공하는지를 묻습니다. 우리의 pass@3은 7/8이었습니다. 7개 태스크가 적어도 한 번 성공했습니다.
- [pass^k](https://huggingface.co/papers/2406.12045)는 k개 rollout이 모두 성공하는지를 묻습니다. 우리의 pass^3은 1/8이었습니다. 한 태스크가 매번 성공했습니다.

사람이 output을 검토한다면 재시도가 허용될 수 있습니다. 무인 처리는 일관성이 필요하므로 두 번째 측정값이 더 중요합니다.

평균은 불확실성과 함께 제시할 때 더 유용합니다. 같은 spectrum의 rollout은 난이도를 공유하므로 서로 독립적이지 않으며, standard error는 [task별로 cluster해야 합니다](https://huggingface.co/papers/2411.00640). Clustering 결과 평균 성공률은 0.46, standard error는 약 0.11이었습니다. 반복 실행에서는 24개 rollout 중 10개가 통과했으며, 첫 실행에서는 24개 중 11개가 통과했습니다. Development task가 8개뿐이므로 어느 결과든 새로운 spectrum으로 일반화할 때는 신중해야 합니다.

Harbor는 rollout마다 `result.json`을 기록하며 이를 trial이라고 부릅니다. 이를 task별로 묶고, 각 task에 rollout이 3개씩 있는지 확인하며, error나 exclusion을 기록합니다. Score와 함께 token, elapsed time, tool call을 보고하면 비슷한 결과를 서로 다른 비용으로 내는 설정을 독자가 비교할 수 있습니다.

### Step 8. 모델 비교

단일 모델이 태스크에 실패했을 때 그 실행만으로는 무엇이 잘못되었는지 알 수 없습니다. 모델, agent harness, 평가 자체의 문제일 수 있습니다. 같은 태스크에서 더 많은 모델을 실행하면 가능성을 좁힐 수 있습니다. 여러 모델이 같은 태스크에서 실패하면 공통 원인을 찾기 위해 지침, environment, verifier를 검사합니다. 한 모델이 독특한 방식으로 반복해서 실패한다면 해당 trace를 성공한 시도와 비교합니다.

Inference Providers는 동일한 router를 통해 많은 open-weight model을 제공하므로, 새 모델마다 `-m`에 하나의 값을 추가하면 됩니다. 여러 provider가 서로 다른 quantization 및 context limit으로 같은 모델을 제공할 수 있으므로 `:deepinfra`과 같은 suffix로 하나를 고정하고 점수 옆에 기록하는 것이 좋습니다.

```bash
OPENAI_API_KEY="$HF_TOKEN" harbor run -p datasets/my-evals -k 3 \
  -e hf-sandbox -a terminus-2 -m openai/Qwen/Qwen3.8-27B:deepinfra \
  --ak api_base=https://router.huggingface.co/v1
```


Figure 8은 각 모델의 raman-lab 결과를 보여줍니다.

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig8_rollout_explorer.html" title="Interactive rollout explorer for the six-model comparison" width="100%" height="531" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 8. 모델별 raman-lab 결과. pass@3, pass^3, clustered standard error와 함께 표시한 평균 pass rate를 포함합니다. 사각형을 선택하면 해당 rollout을 검사할 수 있습니다.*

비교를 통해 gpt-oss-120b의 한 패턴이 더 분명해집니다. 이 모델은 command output을 보기 전에 완료를 선언하는 경우가 많습니다. DeepInfra를 provider로 고정한 별도 실행에서는 24개 rollout 중 22개의 첫 응답에서 `mark_task_complete`를 호출했지만, 다른 5개 모델은 아무 rollout에서도 그렇게 하지 않았습니다(Figure 9). Step 6 실행에서는 router의 default provider를 사용했고 24개 중 19개가 이러한 조기 완료였습니다. 이 실행에서는 24개 중 11개가 통과했고, DeepInfra 실행에서는 24개 중 13개가 통과했습니다. 두 수치 모두 모델이 첫 응답에서 `mark_task_complete`를 호출했는지를 같은 방식으로 측정합니다.

<iframe src="https://evijit-blog-assets.static.hf.space/agentic-eval/fig9_outcomes_by_finish.html" title="Rollout outcomes by when each model declared completion" width="100%" height="401" frameborder="0" sandbox="allow-scripts" loading="lazy"></iframe>

*그림 9. 에이전트가 완료를 선언한 시점별 raman-lab rollout 결과. 왼쪽: 에이전트가 먼저 command output을 확인함. 오른쪽: 첫 응답에서 `mark_task_complete`를 호출함.*

비교 결과 development set의 한계도 드러납니다. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)는 모두 24개 raman-lab rollout을 통과했으므로 이 태스크로는 이들을 순위화할 수 없습니다. 더 어려운 태스크가 필요합니다.

Router의 list price 기준으로 6개 모델 전체의 inference 비용은 $5 미만이었습니다. Median raman-lab rollout은 gpt-oss-120b에서 약 6,000 token, MiniMax-M3.1-Flash에서 약 90,000 token을 사용했습니다. [Companion repository](https://github.com/burtenshaw/raman-lab-evals)에는 [trace](https://huggingface.co/datasets/burtenshaw/raman-lab-runs), rollout별 결과, 다른 모델을 추가하는 script가 있습니다.

### 기존 오픈 구성 요소를 재사용한 방법

Runner와 태스크 형식을 재사용함으로써 물리학, 채점, 검토에 집중할 수 있었습니다. Harbor와 Terminal-Bench가 code와 task를 공개하기 때문에 이를 재사용할 수 있었습니다.

- **태스크:** Harbor 태스크 형식과 Terminal-Bench의 raman-fitting 태스크를 재사용했습니다. 새 작업은 더 좁은 주장, skill coverage, raman-lab용 지침이었습니다.

- **태스크 데이터:** 원래 spectrum을 시작 example으로 사용했습니다. 알려진 parameter를 가진 8개 spectrum을 생성하는 generator를 만들고 수동으로 검토했습니다.
- **Environment:** Harbor의 environment 지원과 HF Sandbox로 태스크를 실행했습니다. Package requirement를 확인하고 누락된 dependency를 진단했습니다.
- **Agent harness 및 도구:** shell access가 있는 Terminus-2와 모델 응답을 위한 Inference Providers를 사용했습니다. 이 부분에는 새 작업이 필요하지 않았습니다.
- **Rollout 및 trace:** Harbor의 rollout record와 ATIF가 각 실행을 기록했습니다. 성공과 실패를 검토하고 원인별로 묶었습니다.
- **Reference solution:** `solution/solve.sh`을 실행하는 convention을 유지하고 새 태스크마다 reference fit을 작성했습니다.
- **Evaluation harness:** Harbor를 HF Job에서 실행했습니다. 실행을 구성하고 결과를 task별로 묶었습니다.
- **Verifier:** `tests/test.sh` entry point와 score를 file에 쓰는 방식을 재사용했습니다. 새 작업은 fit 및 ratio check와 shortcut 및 잘못된 output을 위한 test였습니다.

## 참고 문헌 {#section-8}

- Akhtar, M., Reuel, A., Soni, P., et al. (2026). When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation. arXiv:2602.16763. <https://huggingface.co/papers/2602.16763>
- Ansari, A., Sun, H., Liu, A. Z., et al. (2026). How Good Are Frontier Models at Physics? Expert Re-Grading Reveals Broken Evaluations and Near-Saturation of Leading Benchmarks. arXiv:2609.13009. <https://arxiv.org/abs/2609.13009>
- Anthropic. (2026). Demystifying evals for AI agents. Anthropic Engineering, January 2026. <https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents>
- ARC Prize Foundation. (n.d.). ARC-AGI-2. <https://arcprize.org/arc-agi/2>
- Artificial Analysis. (2026). Announcing Artificial Analysis Intelligence Index v5. Post on X, October 2026. <https://x.com/ArtificialAnlys/status/2115265133467635877>
- Barres, V., Barton, A., Foss, N. J., & Nielsen, H. M. (2026). Harder Than the Task: From Technical to Organizational Difficulty in AI Benchmarking. arXiv:2611.03494. <https://arxiv.org/abs/2611.03494>
- Bean, A. M., Kearns, R. O., Romanou, A., et al. (2025). Measuring what Matters: Construct Validity in Large Language Model Benchmarks. arXiv:2511.04703. <https://huggingface.co/papers/2511.04703>
- Buchanan, E. K., Linderman, S., & Ré, C. (2026). Benchmarks as Software: A Case Study on Terminal-Bench. ICML 2026. <https://icml.cc/virtual/2026/82793>
- Cançado, L. G., Jorio, A., Ferreira, E. H., et al. (2011). Quantifying defects in graphene via Raman spectroscopy at different excitation energies. Nano Letters, 11(8), 3190–3196. <https://pubmed.ncbi.nlm.nih.gov/21696186/>
- Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code. arXiv:2107.03374. <https://huggingface.co/papers/2107.03374>
- Ferrari, A. C., & Basko, D. M. (2013). Raman spectroscopy as a versatile tool for studying the properties of graphene. Nature Nanotechnology, 8, 235–246. <https://arxiv.org/abs/1306.5856>
- Ferrari, A. C., Meyer, J. C., Scardaci, V., et al. (2006). Raman spectrum of graphene and graphene layers. Physical Review Letters, 97, 187401. <https://arxiv.org/abs/cond-mat/0606284>
- Gema, A. P., Leang, J. O. J., Hong, G., et al. (2024). Are We Done with MMLU? arXiv:2406.04127. <https://huggingface.co/papers/2406.04127>
- Ghosh, A., Reuel, A., Chim, J., et al. (2026). Evaluation Cards: An Interpretive Layer for AI Evaluation Reporting. arXiv:2606.09809. <https://huggingface.co/papers/2606.09809>
- Harbor Framework. (n.d.). Harbor documentation. <https://docs.harborframework.com/>
- Isaac, W., Messing, S., & Lum, K. (2026). Piloting the world's first double-blind AI evaluations. Google DeepMind, August 2026. <https://deepmind.google/blog/piloting-the-worlds-first-double-blind-ai-evaluations/>
- Longpre, S., Klyman, K., Appel, R. E., et al. (2025). In-House Evaluation Is Not Enough: Towards Robust Third-Party Flaw Disclosure for General-Purpose AI. arXiv:2503.16861. <https://huggingface.co/papers/2503.16861>
- Miller, E. (2024). Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. arXiv:2411.00640. <https://huggingface.co/papers/2411.00640>
- MLCommons. (n.d.). AILuminate: frequently asked questions. <https://mlcommons.org/ailuminate/faq/>
- OpenAI. (2026). Why we no longer evaluate SWE-bench Verified. February 2026. <https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/>
- OpenAI. (2026). Separating signal from noise in coding evaluations. July 2026. <https://openai.com/index/separating-signal-from-noise-coding-evaluations/>
- Paniego, S., & Gosthipaty, A. R. (2026). Harness, Scaffold, and the AI Agent Terms Worth Getting Right. Hugging Face blog, May 2026. <https://huggingface.co/blog/agent-glossary>
- Skarlinski, M., Laurent, J., Bou, A., & White, A. (2025). About 30% of Humanity's Last Exam chemistry/biology answers are likely wrong. FutureHouse, July 2025. <https://www.futurehouse.org/research/hle-exam>
- Terminal-Bench. (2026). Terminal-Bench 2.1. Release notes, May 2026. <https://www.tbench.ai/news/terminal-bench-2-1>
- Wei, K. (2026). A primer on AI evaluations for policymakers: Part I. Matters of Scale, November 2026. <https://ai.scal.ing/p/evals-intro-1>
- XLANG Lab. (2025). Introducing OSWorld-Verified. July 2025. <https://xlang.ai/blog/osworld-verified>
- Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045. <https://huggingface.co/papers/2406.12045>
- Zhu, Y., Jin, T., Pruksachatkun, Y., et al. (2025). Establishing Best Practices for Building Rigorous Agentic Benchmarks. arXiv:2507.02825. <https://huggingface.co/papers/2507.02825>

## 용어집 {#section-9}

**Agent harness:** 모델의 응답을 shell command 같은 action으로 변환하는 모델 주변의 software입니다. 이 글에서는 Terminus-2가 agent harness입니다.

**Agentic evaluation:** 에이전트가 반환한 text만이 아니라 environment에서 변경한 내용을 채점하는 평가입니다.

**ATIF:** Agent Trajectory Interchange Format. 에이전트의 message, tool call, observation을 기록하기 위한 공유 format입니다.

**Canary string:** 모델 trainer가 벤치마크 file을 찾아 training data에서 제외할 수 있도록 하는 고유 marker입니다.

**Clustered standard error:** 같은 태스크의 rollout을 서로 관련된 것으로 취급하는 standard error입니다. 각 rollout을 독립적으로 취급하는 것보다 더 넓고 정직한 margin을 제공합니다.

**Container image:** operating system, file, software를 고정해 묶은 package입니다. 모든 rollout이 같은 environment에서 시작하도록 합니다.

**Contamination:** 테스트 질문이나 solution이 training data에 포함되어 score가 skill이 아니라 memory를 반영할 수 있는 현상입니다.

**Context limit:** 한 요청에서 모델이 읽을 수 있는 text의 최대량으로, token 단위로 계산합니다.

**D, G 및 2D peak:** Graphene의 Raman spectrum에 나타나는 feature입니다. G와 2D는 층 수를 나타내고, D는 carbon lattice에 결함이 있을 때 나타납니다.

**Defect ratio:** D peak의 intensity를 G peak의 intensity로 나눈 값으로, graphene의 결함을 나타내는 데 사용됩니다. D-to-G ratio라고도 합니다.

**Development set:** 평가를 구축하고 debug하는 동안 사용하는 태스크입니다. 최종 score를 위해 보류하는 held-back test set과 대비됩니다.

**Environment:** 태스크 중 에이전트가 작업하는 software, file, resource입니다.

**Evaluation harness:** 태스크를 실행하고 rollout을 반복하며 trace를 수집하고 score를 집계하는 software입니다. 이 글에서는 Harbor가 evaluation harness입니다.

**Gated dataset:** 소유자가 access request를 승인한 뒤에만 다운로드할 수 있는 Hugging Face dataset입니다.

**Harbor:** Agentic task를 위한 오픈 소스 evaluation harness입니다.

**HF Job:** Script를 실행하는 Hugging Face의 compute입니다. 여기서는 Harbor를 실행합니다.

**HF Sandbox:** 하나의 rollout이 command를 실행하고 file을 저장하는 격리된 Hugging Face environment입니다.

**Held-out test set:** 모델이 해당 항목으로 학습하거나 tuning할 수 없도록 비공개로 보관하는 테스트 항목입니다.

**Inference Providers:** 하나의 API인 router를 통해 모델 요청을 타사 provider로 전달하는 Hugging Face service입니다.

**Lorentzian:** Spectral peak의 shape을 모델링하는 데 흔히 사용하는 수학적 curve입니다.

**No-op agent:** 아무 작업도 하지 않는 에이전트입니다. 건전한 평가에서는 score가 0이어야 합니다.

**Open-weight model:** 학습된 parameter가 공개되어 누구나 다운로드하고 실행할 수 있는 모델입니다.

**Oracle:** Reference solution을 실행하는 에이전트입니다. 제공된 environment에서 모든 태스크를 해결할 수 있는지 확인하는 데 사용합니다.

**pass@k:** k개 rollout 중 하나 이상이 성공하는 태스크의 비율입니다.

**pass^k:** k개 rollout이 모두 성공하는 태스크의 비율입니다.

**Pinning:** 실행을 반복할 수 있도록 software, image, model provider의 정확한 version을 고정하는 것입니다.

**Quantization:** 더 빠르고 저렴하게 실행하기 위해 model weight를 낮은 수치 정밀도로 저장하는 것입니다. 출력이 약간 달라질 수 있습니다.

**Raman shift:** laser light와 sample이 산란하는 빛 사이의 주파수 차이로, cm⁻¹ 단위로 측정합니다.

**Raman spectroscopy:** sample에 laser를 비추고 산란된 빛의 shift를 측정해 물질의 특성을 식별하는 기법입니다.

**Reference solution:** 태스크를 완료하는 것으로 알려진 올바른 방법으로, 태스크와 함께 저장됩니다.

**Rollout:** 하나의 태스크에 대한 에이전트의 한 번의 시도입니다. Harbor는 이를 trial이라고 부릅니다.

**Run record:** score를 산출한 실행의 기록입니다. 정확한 task, environment, model configuration, code version을 포함합니다.

**Saturation:** 대부분의 모델이 벤치마크에서 최고 score에 도달해 더 이상 모델을 구분하지 못하는 지점입니다.

**Synthetic data:** 알려진 parameter에서 생성한 data이므로 정답을 미리 알 수 있습니다.

**Terminal-Bench:** 에이전트가 computer terminal에서 완료하는 태스크의 오픈 벤치마크입니다.

**Terminus-2:** 이 글에서 사용한 terminal agent harness입니다. 모델을 shell command에 연결합니다.

**Token:** 모델이 읽고 쓰는 text의 단위로, 대략 짧은 단어 또는 단어의 일부에 해당합니다.

**Trace:** rollout의 전체 기록입니다. 모델의 message, tool call, 각 command가 반환한 내용을 포함합니다. Trajectory라고도 합니다.

**Verifier:** 저장된 output 또는 environment의 최종 상태를 확인해 rollout을 채점하는 code입니다.
