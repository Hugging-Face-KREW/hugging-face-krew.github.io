---
layout: post
title: "Hugging Face 모델 페이지에서 Every Eval Ever 결과 보기"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-06-30-eee-community-evals/thumbnail.png
authors:
  - user: deepmage121
slug: "eee-community-evals"
source_url: "https://huggingface.co/blog/eee-community-evals"
source_published_date: "2026-06-30"
source_published_at: "2026-06-30T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Featuring Every Eval Ever Results on Hugging Face Model Pages](https://huggingface.co/blog/eee-community-evals)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/eee-community-evals -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Hugging Face 모델 페이지에서 Every Eval Ever 결과 보기

Every Eval Ever (EEE)와 Hugging Face 커뮤니티 Evals는 이제 상호 호환됩니다. 교차 게시 및 평가 결과 해석을 가능하게 하며, 오픈 모델, 리더보드, 그리고 통합 표준 메타데이터 저장소에 연결합니다.

EEE [launched](https://evalevalai.com/infrastructure/2026/02/17/everyevalever-launch/)는 2026년 2월 [EvalEval Coalition](https://evalevalai.com/)의 프로젝트로, 제1자 및 제3자 평가자가 평가 결과를 보고하는 방식을 개선하기 위한 최초의 교차기관 간 노력입니다. Hugging Face는 2026년 2월에 [Community Evals](https://huggingface.co/blog/community-evals)를 출시하여 허브에서 벤치마크 점수의 보고 방식을 탈중앙화했습니다. 이를 합치면 사용자가 연구자, 정책 입안자들이 평가와 모델을 신뢰하고 이해하며 선택하는 데 생기는 간극을 줄여줍니다.

평가 결과는 모델의 역량을 측정하고 서로 다른 모델을 비교하며 안전성과 거버넌스를 판단하는 데 사용되지만, 결과는 흩어져 있어 비교하기 어렵습니다. 논문, 리더보드, 블로그 포스트, 하니스 로그 등 다양한 형식으로 각각 다른 형식으로 존재합니다. 같은 벤치마크의 같은 모델이라도 누가 실행했는지와 실행 방식에 따라 서로 다른 점수를 받는 경우가 많습니다; 예를 들어 LLaMA 65B의 경우 [MMLU](https://huggingface.co/blog/open-llm-leaderboard-mmlu)에서 63.7과 48.8로 보고된 바 있습니다. 이러한 격차는 [evaluation settings that we found are commonly unreported](https://huggingface.co/papers/2606.14516)에서 발생할 수 있습니다.

EEE는 보고 측면의 해결책입니다. 평가 결과를 기록하는 하나의 JSON 스키마이며, 아래를 기록합니다:

- 누가 실행했는지  
- 어떤 모델인지  
- 어떻게 접근했는지  
- 생성 설정  
- 지표가 실제로 의미하는 바  
- [권장] 샘플별 출력을 위한 동반 JSONL 파일.

이 스키마는 연구자 및 정책 연구자의 피드백으로 구축되었으며, 어떤 출처의 결과도 받아들일 수 있도록 설계되어 하니스 로그, 리더보드 스크랩, 논문 수치가 모두 같은 형식으로 모이게 됩니다. [GitHub repository](https://github.com/evaleval/every_eval_ever)에는 변환기(converters), 예제, 그리고 기여자 가이드가 포함되어 있습니다. 출시 이후 [datastore](https://huggingface.co/datasets/evaleval/EEE_datastore)는 Hugging Face에서 약 229,000개의 평가 결과를 22,000개가 넘는 모델과 2,200개 벤치마크에 걸쳐, 31개의 서로 다른 보고 형식에서 끌어모았습니다. 이러한 실행들을 처음부터 재현하려면 수십만 달러의 비용이 들 수 있는데, 이는 누군가 데이터를 생성하기로 비용을 지불한 뒤 데이터를 흩뜨리지 않도록 하는 타당한 주장이 됩니다.

스키마와 기여 방법에 대해 자세히 알아보기 [here](https://evalevalai.com/infrastructure/2026/02/17/everyevalever-launch/).

이제 더 나은 통합과 출처 표기를 제공합니다. 기여자들은 이제 EEE 결과를 Hugging Face 커뮤니티 Evals로 보낼 수 있습니다. 우리는 EEE 기록을 받아 Hugging Face가 기대하는 작은 YAML 파일을 작성하는 변환기를 만들었으므로, 두 형식으로 동일한 결과를 수동으로 유지할 필요가 없습니다.

![Verified Evaluators on Eval Cards](https://cdn-uploads.huggingface.co/production/uploads/6413251362e6057cbb6259bd/czIJDDShvtMBs2M2T7B45.gif)

이것은 평가를 보고하거나 읽는 모든 사람에게 새로운 기능이며, 기존 EEE 기여자들만의 것이 아닙니다. 자사 모델을 보고하는 평가자와 타인의 모델을 보고하는 제3자 평가자 모두 커뮤니티 Evals와 EEE에 제출할 수 있으며, 허브를 둘러보는 누구나 전체 기록으로 연결되는 결과를 얻습니다. 조직의 공식 Hugging Face 계정을 통해 데이터를 제출하면, EvalEval에 [verified](https://evalcards.evalevalai.com/help/get-verified) 확인 표시가 표시되어 독자들에게 숫자가 출처에서 직접 왔음을 알리는 신호가 됩니다. 이 글의 나머지 부분은 Hugging Face 커뮤니티 Evals가 무엇인지와 변환기가 하는 일에 대해 설명합니다.

## Hugging Face 커뮤니티 Evals가 EvalEval와 함께 작동하는 방법 {#section-1}

Hugging Face 커뮤니티 Evals에는 두 가지 측면이 있습니다.

벤치마크는 `eval.yaml`를 추가하여 스스로를 등록하는 데이터셋 저장소에 존재합니다. 등록되면 해당 데이터셋 페이지는 허브 전역에서 보고된 모든 점수의 리더보드를 수집하고 표시합니다. [official benchmarks](https://huggingface.co/datasets?benchmark=benchmark:official&sort=trending)의 목록은 시간이 지남에 따라 증가합니다.

모델의 점수는 모델 저장소 안의 `.eval_results/*.yaml`에 저장됩니다. 모델 카드에 표시되며 해당 벤치마크 리더보드에 반영됩니다. 모델 저자의 결과와 누구든지 PR을 통해 제출한 결과가 모두 집계되며, 각 점수에는 저자 제출 여부, 커뮤니티 제출 여부, 또는 독립적으로 검증되었다는 배지가 따라붙습니다. 누구나 적절한 YAML 파일로 PR을 열어 점수를 추가할 수 있으며, 모델 저자는 자신의 저장소에서 PR을 닫거나 결과를 숨길 수 있습니다.

다음은 이러한 리더보드 중 하나가 어떻게 보이는지:

<iframe src="https://huggingface.co/datasets/cais/hle/embed/leaderboard" frameborder="0" width="100%" height="560px"></iframe>

*Hugging Face 커뮤니티 Evals 리더보드 [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) 허브에서*

여기가 EEE와 커뮤니티 Evals가 함께 작동하는 지점입니다. 두 곳 모두에 결과를 보내면 먼저, 귀하의 점수는 Hugging Face 모델 페이지에 표시되고 벤치마크의 리더보드로 끌려 갑니다. 둘째로, 생성 설정, 하니스 버전, 재현성 노트, 인스턴스 수준 데이터를 포함한 전체 EEE 기록으로의 원천 링크를 가지는 소스 배지가 따라갑니다.

<iframe src="https://evaleval-general-eval-card.hf.space/embed/eval/frontier/mmlu-pro/mmlu-pro" width="100%" height="420" frameborder="0" style="border:1px solid #e5e5e5;border-radius:4px;" loading="lazy" title="Score distribution — Frontier"></iframe>

![EvalEval as source on SmolLM2 Model Page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/eee_commevals/smollm2.png)

*EEE Datastore의 평가(MMLU-Pro) 중 하나가 파일 수준에서 Hugging Face 모델 카드에 교차 연결되어 있습니다(a). 소스 EvalEval 배지는 전체 JSON 기록으로 연결됩니다(b).*

**두 목적지는 같은 목표를 향해 서로 다른 역할을 수행합니다. Hugging Face는 사람들이 모델을 보는 위치에 결과를 놓고 출처로의 링크를 제공합니다. EEE는 결과를 해석 가능하게 만드는 전체 구조화된 기록을 유지하고, 그 위에 [Eval Cards](https://evalevalai.com/projects/eval-cards/)를 담아냅니다.** 데이터를 두 곳 모두에 제출하면 같은 평가가 한꺼번에 보이고 읽기 쉽도록 표시되며, 이것이 하나의 평가를 보고하는 이유이기도 합니다.

아래에서 그 교차 호환성을 확인할 수 있습니다. 위 모델 카드에 표시된 동일한 GPQA 점수는 Eval Cards에서도 렌더링되며, 이는 EEE 실행 데이터를 벤치마크 및 모델 메타데이터와 결합하여 하나의 해석 가능한 기록으로 만듭니다. 같은 평가이지만 표면이 다릅니다:

<iframe src="https://evaleval-general-eval-card.hf.space/embed/eval/leaderboard/openeval/gpqa" frameborder="0" width="100%" height="560px">
</iframe>

## 작동 원리 {#section-2}

Hugging Face는 평가 점수를 `.eval_results/` 아래의 YAML로 모델 저장소에 저장합니다. 필수 필드는 벤치마크 데이터셋, 작업, 값뿐입니다. 소스 블록은 EEE로의 백링크를 생성하는 부분입니다.

```
- dataset:
    id: openai/gsm8k
    task_id: gsm8k
  value: 96.8
  date: '2024-07-16'
  notes: '8-shot CoT'
  source:
    url: https://huggingface.co/datasets/evaleval/EEE_datastore/blob/main/flat/objects/<xx>/<yy>/<uuid>.json
    name: EvalEval
```


**변환기가 기존 기록에서 이를 채웁니다.** 이는 `source_data.hf_repo`를 `dataset.id`로, `evaluation_name`를 `task_id`로, `score_details.score`를 `value`로, 그리고 `evaluation_timestamp`를 `date`로 매핑한 다음, 데이터스토어 객체 URL을 per-record EEE JSON의 소스 링크로 삽입합니다. 현재 네 가지 공식 벤치마크를 다룹니다: MMLU-Pro, GPQA, HLE, GSM8K.

**변환기는 필드를 재구성하는 것 그 이상을 합니다.** 하나의 EEE 데이터스토어 컬렉션을 가리키고 그 컬렉션과 참조되는 기록을 함께 다운로드합니다, 객체 해시를 확인하고 지원되는 벤치마크에 매핑되는 점수를 찾아냅니다. 라이브로 쓰기 전에 이미 존재하는 것을 감사합니다: 모델의 메인 브랜치와 열린 PR들에 있는 모든 `.eval_results` YAML을 읽고, 데이터셋과 작업으로 비교합니다. 점수가 이미 거기에 있으면 `already_present`로 표시하고, 다른 점수가 거기에 있으면 `score_conflict`로 표시하며, 모델 저장소가 허브에서 해결되지 않으면 `missing_hf_model`로 표시합니다. 그 밖의 모든 것은 `ready`로 표시됩니다.

**아무 것도 당신의 서명 없이는 푸시되지 않습니다.** 도구는 로컬 YAML 프리뷰와 검토 파일을 작성하여 확인할 수 있게 하고, 준비된 것과 주의가 필요한 것을 보여주는 보고서를 출력합니다. 커밋 메시지를 입력하고 `OPEN PRS`를 입력한 후에만 PR을 엽니다. 컬렉션에 대해 캐시된 결과를 재실행하면 `--force`를 넘겨주지 않는 한 재사용됩니다.

![TUI of the Converter](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/eee_commevals/terminal%20shot.png)

*변환기의 검토 단계. 제외된 엔트리(여기서는 허브 저장소와 매칭되는 모델이 없는 경우)는 EEE 원본 URL과 함께 나열되며, 준비된 PR은 명시적 OPEN PRS 확인을 기다립니다.*

## 시작하기 {#section-3}

전체 기록을 [the EEE datastore](https://huggingface.co/datasets/evaleval/EEE_datastore)에 제출하세요.

EEE를 활용하려면 한 가지 추가 단계가 필요하며, 이 단계는 변환기가 대부분 자동으로 처리합니다. [community eval converter tool](https://github.com/evaleval/every_eval_ever/tree/main/tools/hf-community-evals)는 GitHub 저장소에서 찾을 수 있습니다. 컬렉션을 처리하려면 다음을 실행하세요:

```shell
uv run tools/hf-community-evals/community_evals_converter.py MMLU-Pro \
  --datastore evaleval/EEE_datastore@main
```


미리보기와 생성된 보고서를 검토한 후 제출 준비가 되면 `OPEN PRS`를 입력하세요. 스키마, CLI, 변환기에 대한 전체 문서는 [evalevalai.com/every\_eval\_ever/hf-community-evals](https://evalevalai.com/every_eval_ever/hf-community-evals/)에 있습니다.
