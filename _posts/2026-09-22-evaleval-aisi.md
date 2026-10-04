---
layout: post
title: "UK AISI와 EvalEval이 벤치마크 결과의 재현성을 높이는 방법"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/evaleval-aisi/thumbnail.png
image: assets/images/blog/posts/2026-09-22-evaleval-aisi/thumbnail.png
authors:
  - user: evijit
slug: "evaleval-aisi"
source_url: "https://huggingface.co/blog/evaleval-aisi"
source_published_date: "2026-09-22"
source_published_at: "2026-09-22T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [How UK AISI and EvalEval Are Making Benchmark Results Reproducible](https://huggingface.co/blog/evaleval-aisi)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/evaleval-aisi -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# UK AISI와 EvalEval이 벤치마크 결과의 재현성을 높이는 방법

[EvalEval Coalition](https://evalevalai.com/)은(는) [UK AI Security Institute (AISI)](https://www.aisi.gov.uk/)이(가) EvalEval의 인프라를 활용해 평가 결과를 공개적으로 공유하고, 더욱 재현 가능하고 검증 가능한 평가 과학을 지원하고 있다는 소식을 전하게 되어 매우 기쁘게 생각합니다.

AISI와 EvalEval은 이전에도 [joint workshop alongside NeurIPS 2025](https://evalevalai.com/events/workshop-2025/)에서 시작된 연구를 공동으로 진행했으며, Institute의 피드백은 [Every Eval Ever (EEE) schema](https://evalevalai.com/projects/every-eval-ever/)을(를) 형성하는 데 도움이 되었습니다. 이번 협력의 다음 단계에서는 이러한 공유 인프라를 실제로 활용합니다.

## 재현 가능한 평가 보고가 중요한 이유 {#section-1}

AI 배포가 가속화됨에 따라 평가 결과는 모델과 시스템의 성능에 관한 증거의 원천으로 점점 더 중요해지고 있습니다. 그러나 결과는 충분한 재현 정보 없이 다양한 형식, 플랫폼, 매체를 통해 보고되는 경우가 많습니다. 평가를 다시 실행하는 일 자체가 감당하기 어려울 정도로 비용이 많이 들 수도 있습니다.

EvalEval의 사명은 공유 보고 스키마인 [Every Eval Ever](https://evalevalai.com/projects/every-eval-ever/)과 개방형 플랫폼인 [Evaluation Cards](https://evalcards.evalevalai.com/)을 통해 이러한 생태계를 개선하는 것입니다. 이 플랫폼은 평가 결과와 이를 해석하는 데 필요한 정보를 공통 구조로 통합합니다.

이는 [OptStop](https://arxiv.org/abs/2608.14425)을(를) 통해 평가를 더욱 효율적으로 만들고, [HiBayES](https://www.aisi.gov.uk/blog/hibayes-improving-llm-evaluation-with-hierarchical-bayesian-modelling)을(를) 통해 통계적 엄밀성을 높이며, 트랜스크립트 분석과 능력 도출을 포함한 영역에서 표준화를 추진해 온 AISI의 작업을 자연스럽게 이어갑니다. AISI와 EvalEval은 함께 평가 보고의 격차를 진단하고 이를 해소하기 위한 공유 인프라를 구축하고 있습니다.

## AISI가 공유하는 내용 {#section-2}

트랜스크립트 수준의 투명성은 재현성뿐 아니라 분석과 진단에도 중요합니다. 이번 협력의 새로운 단계에서 AISI는 적절한 경우 공개적으로 보고된 평가 방법과 결과를 Evaluation Cards를 통해 제공합니다. 이번 공개에는 논문의 주요 실험에 포함된 다음 5개 벤치마크에 대한 검증된 결과, 맥락, 구성 정보가 포함됩니다.

- HealthBench
- FrontierMath
- Humanity's Last Exam
- SWE-Bench Pro
- Terminal-Bench 2.0

이 결과는 6개의 프런티어 모델을 대상으로 합니다. Claude Opus 4, Claude Opus 4.5, Claude Opus 4.6, GPT-5, GPT-5.2, GPT-5.4입니다. 또한 이번 공개에는 서로 연관된 두 가지 사이버 평가인 Cyber CTFs와 The Last Ones의 결과도 포함됩니다. 이 평가들은 일부 모델이 겹치지만 서로 다른 모델 집합을 사용합니다. 이 데이터는 [*How Inference Compute Shapes Frontier LLM Evaluation*](https://arxiv.org/abs/2606.17930) 논문과 함께 제공되며, 해당 논문은 추론 시간 컴퓨팅과 평가 프로토콜에 따라 벤치마크 성능이 어떻게 달라지는지를 연구합니다.

<iframe src="https://evaleval-general-eval-card.hf.space/embed/eval/trajectories/aisi-inference-scaling/hle?panel=tokens" frameborder="0" width="100%" height="670px"></iframe>

*Humanity's Last Exam의 성능은 평가 프로토콜과 추론 컴퓨팅에 따라 달라집니다. 각 곡선은 각 과제에서 가장 이른 성공 결과를 사용해, 주어진 토큰 수 이내에 해결된 시도 과제의 누적 비율을 보여줍니다. 모델이 시도할 때마다 오라클로부터 정답 여부에 대한 피드백을 받은 경우, 토큰 사용량이 증가함에 따라 추가 과제를 계속 해결했습니다.*

결과가 설정 정보와 함께 공개되면 연구자와 실무자는 개별 연구를 더욱 면밀히 검토하고 더 넓은 생태계의 결과와 비교할 수 있습니다. 다른 보고서에 이러한 세부 정보가 없는 경우, AISI의 공개 자료와 같은 자료는 평가를 맥락에 맞게 해석하기 위한 검증된 기준점을 제공합니다. 예를 들어 연구자들이 설정 선택이 보고된 성능에 어떤 영향을 미칠 수 있는지 이해하는 데 도움이 됩니다. 더 많은 평가자가 EEE를 채택함에 따라, 이러한 공개 비교는 더 광범위하고 신뢰할 수 있는 메타 연구를 지원할 수 있습니다.

<iframe src="https://evaleval-general-eval-card.hf.space/embed/eval/distribution/aisi-inference-scaling/terminal-bench-2?view=context" frameborder="0" width="100%" height="500px"></iframe>

*서로 다른 평가 설정에서 동일한 모델을 대상으로 보고된 다른 평가 결과와 함께 제시한 AISI의 Terminal-Bench 2.0 결과.*

이러한 도입을 반갑게 생각하며, AISI 및 다른 AI 평가 기관과 함께 평가를 더욱 표준화하고 공유해 나가기를 기대합니다.

## 공동의 사명에 기여하기 {#section-3}

- **모델 개발자:** [Report verified evaluation results](https://evalcards.evalevalai.com/help/get-verified).
- **평가 개발자:** [Every Eval Ever schema](https://github.com/evaleval/every_eval_ever)을(를) 사용해 벤치마크와 실행 데이터를 보고하세요.
- **평가·거버넌스·정책 연구자:** 벤치마크 또는 모델별로 [Explore Evaluation Cards](https://evalcards.evalevalai.com/)하거나, 이를 활용해 평가 보고 전반의 현황을 살펴보세요.

## EvalEval Coalition 소개 {#section-4}

EvalEval Coalition은 평가 생태계를 위한 과학적 근거에 기반한 연구와 견고한 배포 인프라를 개발하는 연구 커뮤니티입니다. 이 연합의 목표는 평가 과학을 개선하고, 평가의 적용 가능성과 유용성을 문서화하는 데 대한 합의가 부족한 문제를 해결하며, 과학 연구와 정책 분석에 중요한 영향에 대한 범위를 넓히는 것입니다.

이 연합의 대표 프로젝트로는 평가 결과를 위한 공유 스키마이자 저장소인 [Every Eval Ever](https://evalevalai.com/projects/every-eval-ever/)과, 벤치마크 메타데이터, 평가 실행 데이터, 모델 메타데이터를 해석 가능한 레코드로 결합하는 [Evaluation Cards](https://evalevalai.com/projects/eval-cards/)이 있습니다. 두 프로젝트는 겉보기에 유사한 점수가 실질적으로 다른 조건에서 산출된 경우를 더 쉽게 파악할 수 있도록 합니다.

## UK AI Security Institute 소개 {#section-5}

[UK AI Security Institute](https://www.aisi.gov.uk/)은(는) 영국 정부의 Department for Science, Innovation and Technology 산하 연구 기관입니다. 이 기관의 사명은 첨단 AI가 초래하는 위험을 과학적으로 이해할 수 있도록 정부를 지원하는 것입니다. AISI는 첨단 AI의 능력과 영향을 이해하고, 완화책을 개발·검증하며, 정책 수립에 정보를 제공하기 위한 연구를 수행하고 인프라를 구축합니다.

## 추가 읽을거리 {#section-6}

- [*How Inference Compute Shapes Frontier LLM Evaluation*](https://arxiv.org/abs/2606.17930)
- [HiBayES: Improving LLM evaluation with hierarchical Bayesian modelling](https://www.aisi.gov.uk/blog/hibayes-improving-llm-evaluation-with-hierarchical-bayesian-modelling)
- [HiBayES paper](https://arxiv.org/abs/2505.05602)
- [OptStop paper](https://arxiv.org/abs/2608.14425)
- [Every Eval Ever](https://evalevalai.com/projects/every-eval-ever/)
- [Evaluation Cards](https://evalcards.evalevalai.com/)
