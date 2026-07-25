---
layout: post
title: "hf CLI를 에이전트에 최적화된 방식으로 Hugging Face Hub와 함께 작동하도록 설계하기"
description: "Hugging Face Hub 작업을 코딩 에이전트에 맞게 단순화한 hf CLI 설계와 벤치마크 결과를 소개합니다."
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/hf-cli-for-agents/thumbnail.png
image: assets/images/blog/posts/2026-06-04-hf-cli-for-agents/thumbnail.png
authors:
  - user: celinah
  - user: Wauplin
slug: "hf-cli-for-agents"
source_url: "https://huggingface.co/blog/hf-cli-for-agents"
source_published_date: "2026-06-04"
source_published_at: "2026-06-04T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Designing the hf CLI as an agent-optimized way to work with the Hub](https://huggingface.co/blog/hf-cli-for-agents)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/hf-cli-for-agents -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# hf CLI를 에이전트에 최적화된 방식으로 Hugging Face Hub와 함께 작동하도록 설계하기

`hf`는 Hugging Face Hub의 공식 명령줄 인터페이스입니다. Hub에서 Python SDK로 할 수 있는 모든 작업을 터미널에서도 수행할 수 있습니다. 모델·데이터셋·Spaces를 다운로드하고 업로드할 수 있으며, 리포지토리·브랜치·태그·풀 리퀘스트를 생성하고 관리할 수 있습니다. 또한 HF 인프라에서 Jobs를 실행하고, Buckets·Collections·webhooks·Inference Endpoints를 관리할 수 있습니다.

`hf` CLI는 수년간 우리의 사용자들을 위해 주로 구축되었습니다. 그러나 이제는 **코딩 에이전트**들: Claude Code, Codex, Cursor 등에게도 점점 더 많이 사용되고 있습니다. 그래서 두 관객 모두를 한꺼번에 활용할 수 있도록 재구축했습니다. 이 블로그 글은 우리가 한 일과 이를 벤치마크한 방법을 요약합니다. 복잡하고 다단계인 작업에서 비-CLI 기준선(에이전트가 `curl`를 수동으로 구성하거나 Python SDK)을 사용할 때 `hf` CLI에 비해 최대 **6배** 더 많은 토큰을 소모하는 것을 발견했습니다.

## Hugging Face Hub의 AI 에이전트 트래픽 {#section-1}

우리는 2026년 4월에 Hugging Face Hub의 에이전트 사용을 추적하기 시작했습니다. `hf` CLI(그리고 그것이 기반하는 `huggingface_hub` Python SDK)는 에이전트가 이를 주도하는지 감지하기 위해 에이전트가 설정한 환경 변수를 읽어 감지합니다: Claude Code용 `CLAUDECODE`/`CLAUDE_CODE`, Codex용 `CODEX_SANDBOX`, 더불어 Cursor, Gemini, Pi, 그리고 범용 `AI_AGENT`. 이 하나의 신호는 두 가지 역할을 합니다: CLI의 출력 형식을 결정하고(아래에서 더 자세히 다룸) 각 Hub 요청에 `agent/<name>` 사용자 에이전트를 태그하여 이를 주도하는 에이전트의 트래픽을 식별할 수 있습니다. 서로 다른 사용자가 가장 많은 두 에이전트는 **Claude Code와 Codex**로, 다른 모든 것들보다 앞서 있으며 이 기사에서 나중에 벤치마크합니다.

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-users.png" alt="Distinct users of the Hugging Face Hub by coding agent since April 2026. Claude Code leads with 39.5k users and 48.6M requests, then Codex with 34.8k users and 36.4M requests, followed by antigravity, cursor-cli, openclaw, cursor, gemini and pi." width="100%"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-users-dark.png" alt="Distinct users of the Hugging Face Hub by coding agent since April 2026. Claude Code leads with 39.5k users and 48.6M requests, then Codex with 34.8k users and 36.4M requests, followed by antigravity, cursor-cli, openclaw, cursor, gemini and pi." width="100%"/>
</div>

막대 그래프는 에이전트별 고유 사용자 수를 보여주며, 요청량은 하위 라벨입니다. Claude Code만 보아도 약 40,000명의 사용자와 거의 49,000,000건의 요청을 기록했고, Codex가 그 뒤를 잇습니다. 이는 초기 수치에 불과하지만(에이전트 트래픽 속성을 확인하기 시작한 시점은 2026년 4월), 규모는 이미 상당하며 코딩 에이전트가 Hub와 함께 작업하는 표준 방식으로 자리잡아 가면서 계속 커질 수 있습니다.

## 사람과 에이전트를 위한 설계 {#section-2}

사람과 코딩 에이전트는 같은 `hf` 명령에 대해 서로 다른 출력 형식을 기대합니다. 사람은 ANSI 색상, 화면에 맞게 잘려진 표, 성공 시 녹색 ✅, 부울 값 `✔`, 진행 표시줄, 서술형 힌트 등을 원합니다. 반면 에이전트는 역으로 원합니다: ANSI 코드 없이, 잘리지 않게 모든 값을 완전하게 표기하며, 에이전트가 인간보다 더 촘촘한 출력을 처리할 수 있도록 간결하고 구조화되어 토큰을 가볍게 유지합니다. 또한 CLI 프롬프트에 답하지 못하고, 시간초과 후에도 기꺼이 명령을 재실행합니다. 이 섹션의 나머지는 `hf`가 각 측에 필요한 것을 어떻게 제공하는지에 대한 내용입니다. 에이전트-모드 출력은 `hf` v1.9.0에서 도입했고, 이후 릴리스에서 CLI의 나머지 부분을 차례로 그 방향으로 마이그레이션해 왔습니다.

### 하나의 명령, 다중 렌더링

위에서 언급한 환경 변수들을 통해 `hf`가 에이전트 사용 여부를 자동 인식하면, **동일한 명령**을 서로 다르게 렌더링합니다. 플래그를 전달하지 않고도 사람용 또는 에이전트용 출력 형식을 최적화합니다:

```text
# human (default in a terminal): aligned table, truncated to fit, with a hint
> hf models ls --author Qwen --sort downloads --limit 3
ID                       CREATED_AT DOWNLOADS LIBRARY_NAME LIKES PIPELINE_TAG    PRIVATE TAGS
------------------------ ---------- --------- ------------ ----- --------------- ------- -------------------------
Qwen/Qwen3-0.6B          2025-04-27  21156913 transformers  1285 text-generation         transformers, safetens...
Qwen/Qwen2.5-1.5B-Ins... 2024-09-17  15143953 transformers   725 text-generation         transformers, safetens...
Qwen/Qwen3-4B            2025-04-27  14808352 transformers   625 text-generation         transformers, safetens...
Hint: Use `--no-truncate` or `--format json` to display full values.

# agent (auto-detected): TSV, full ids + ISO timestamps + every tag, nothing truncated
$ hf models ls --author Qwen --sort downloads --limit 3
id      created_at      downloads       library_name    likes   pipeline_tag    private tags
Qwen/Qwen3-0.6B 2025-04-27T03:40:08+00:00       21156913        transformers    1285    text-generation False   ['transformers', 'safetensors', 'qwen3', 'text-generation', 'conversational', 'arxiv:2505.09388', 'base_model:Qwen/Qwen3-0.6B-Base', 'base_model:finetune:Qwen/Qwen3-0.6B-Base', 'license:apache-2.0', 'text-generation-inference', 'endpoints_compatible', 'deploy:azure', 'region:us']
Qwen/Qwen2.5-1.5B-Instruct      2024-09-17T14:10:29+00:00       15143953        transformers    725     text-generation False['transformers', 'safetensors', 'qwen2', 'text-generation', 'chat', 'conversational', 'en', 'arxiv:2407.10671', 'base_model:Qwen/Qwen2.5-1.5B', 'base_model:finetune:Qwen/Qwen2.5-1.5B', 'license:apache-2.0', 'text-generation-inference', 'endpoints_compatible', 'deploy:azure', 'region:us']
Qwen/Qwen3-4B   2025-04-27T03:41:29+00:00       14808352        transformers    625     text-generation False   ['transformers', 'safetensors', 'text-generation', 'arxiv:2309.00071', 'arxiv:2505.09388', 'base_model:Qwen/Qwen3-4B-Base', 'base_model:finetune:Qwen/Qwen3-4B-Base', 'license:apache-2.0', 'endpoints_compatible', 'deploy:azure', 'region:us']
```


사람은 터미널에 맞춰 정렬된 표와, 화면에 맞게 잘려진 표와 함께 더 보려면 방법에 대한 힌트, 상태를 나타내는 색상 신호(성공 시 초록 `✓`, 오류 시 빨간색)가 제공됩니다. 에이전트는 TSV 형식의 전체 기록을 받습니다: 전체 리포지토리 ID, 전체 ISO 타임스탬프, 모든 태그, ANSI 코드 없음, 잘려진 부분 없음, 파싱하기 쉽고 토큰을 가볍게 유지합니다.

실제로는 원시 데이터를 입력으로 받아 형식을 처리하는 `.table(...)`, `.result(...)`, `.json()` 등의 로깅 방식들을 구현했습니다. 사람 모드와 에이전트 모드 외에도 서로 파이프하기 쉽게 만들어 주는 `--json`와 `--quiet` 옵션을 도입했습니다. 기본 모드는 맥락에 따라 자동으로 선택되지만, 사용자는 언제든지 `--format human | agent | json | quiet`로 원하는 형식을 강제할 수 있습니다.

### Next-command hints

CLI 명령은 드물게 독립적으로 실행됩니다: 한 단계가 보통 다음 단계(`git add`가 먼저, 그다음 `git commit`)를 시사합니다. 많은 `hf` 명령은 이제 끝에 **힌트**를 포함합니다: 방금 사용한 ID로 미리 채워진 정확한 다음 명령으로, 사용자나 에이전트가 처음부터 생각하지 않고도 바로 다음 단계로 연결할 수 있게 합니다. 백그라운드에서 작업을 시작하면 로그로 안내하고, Space를 만들면 부트 상태로 안내합니다:

```text
$ hf jobs run --detach python:3.12 python train.py
✓ Job started
  id: 6f3a1c2e9b
  url: https://huggingface.co/jobs/celinah/6f3a1c2e9b
Hint: Use `hf jobs logs 6f3a1c2e9b` to fetch the logs.
```


사람에게는 편의성입니다. 에이전트에게는 레일입니다: 다음 동작은 이름이 붙고, 올바른 ID로 매개변수화되어 실행 준비가 되어 있어, 무엇을 해야 할지 파악하는 단계를 줄여줍니다. 오류도 같은 방식으로 처리되어 수정 방법의 이름을 붙입니다:

```
Error: Not logged in. Run `hf auth login` first.
```


힌트, 경고 및 오류는 모두 stderr로 보내고 데이터는 stdout으로 보내므로, 이 안내가 에이전트가 파싱하는 출력에 영향을 주지 않습니다.

### 논블로킹 및 재시도 안전성

`hf`은 에이전트가 누를 수 없는 키를 기다리며 대화형 프롬프트에 머무르지 않습니다. 파괴적 명령은 여전히 인간의 확인을 요구하지만, 에이전트 모드에서는 메시지에 수정이 포함된 상태로 빠르게 실패합니다(`Use --yes to skip confirmation.`), 그리고 `-y`/`--yes`는 이를 건너뜁니다. 그리고 에이전트는 시간 초과와 맥락 손실에 대해 재시도하므로, 작업은 재실행하기 쉽게 안전하게 설계되어 있습니다: `hf repos create --exist-ok`는 저장소가 이미 존재하면 무효(무시)이며, 업로드를 다시 실행하면 깔끔하게 재커밋됩니다. 또한 데이터를 실제로 옮기는 명령은 실행 전에 정확히 무엇을 전송할지 보여주는 `--dry-run`를 취합니다. 이는 사람과 에이전트 모두에게 편리합니다. 긴 다운로드나 무작정 동기화를 피할 수 있기 때문입니다:

```text
# agent mode: a destructive command without --yes refuses, with the fix in the message
$ hf repos delete my-org/old-model
Error: You are about to permanently delete model 'my-org/old-model'. Proceed? Use --yes to skip confirmation.

# commands that move data take --dry-run to preview the transfer first
$ hf download deepseek-ai/DeepSeek-V4-Pro config.json --dry-run
[dry-run] Will download 1 files (out of 1) totalling 1.8K.
file         size
config.json  1.8K
```


### 검색 가능하고 예측 가능한 명령

`hf`는 탐색 가능하게 설계되었습니다: 리소스 그룹을 보려면 `hf`를 실행하고, 필요한 항목에서 `--help`를 실행하며, 모든 `--help`은 실제로 복사-붙여넣기 가능한 예제로 끝납니다(에이전트가 설명을 파싱하는 것보다 훨씬 빠르게 매칭합니다):

```
$ hf models ls --help
...
Examples
  $ hf models ls --sort downloads --limit 10
  $ hf models ls --search "qwen" --author Qwen
  $ hf models ls Qwen/Qwen3-4B --tree
```


명령 트리는 일관되게, **리소스 + 동사** 구조이며, 명확한 별칭들(`hf models ls`, `hf repos create`, `hf jobs ps`, `hf collections delete`; `list`/`ls`, `remove`/`rm`)으로 구성되어 있어, 에이전트가 한 명령을 배우면 나머지도 추정할 수 있습니다. 그리고 출력은 이렇게 구성됩니다: `-q`는 한 줄에 하나의 ID를 출력해 다음 명령으로 파이프에 연결하고, `--json`는 [`jq`](https://jqlang.org/)에 넘길 수 있는 무언가를 제공합니다.

```text
$ hf models ls --author Qwen -q | head -3
Qwen/Qwen3-0.6B
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen3-4B
```


## 코딩 에이전트를 위한 hf CLI 벤치마크 {#section-3}

`hf` CLI가 에이전트에 대해 실제로 더 효율적인지 알아보기 위해 이를 측정했습니다. 간단한 평가 하네스를 구축하고 Hub를 구동하는 동일한 작업 집합을 여러 방식으로 반복 실행해 라이브 Hub와 각 실행을 채점했습니다. 방법론에 앞서 간단히 요약하면, 두 에이전트의 모든 경우에서 `hf` CLI가 앞섰고, 특히 다단계의 복잡한 작업에서 토큰을 훨씬 적게 사용했습니다.

| agent                        | tool              | success score | token usage     | self-report error |
| ---------------------------- | ----------------- | ------------- | --------------- | ----------------- |
| **Claude Code (Sonnet 4.6)** | `hf` CLI          | **0.94**      | baseline        | **2 / 163**       |
|                              | curl/SDK          | 0.84          | **1.3-1.6× tokens** | 11 / 163      |
| **Codex (GPT-5.5)**          | `hf` CLI          | **0.93**      | baseline        | **3 / 163**       |
|                              | curl/SDK          | 0.92          | **1.6-1.8× tokens** | 10 / 163      |

*(self-report error = 에이전트가 17개의 해결 가능한 작업에서 성공을 보고했으나 Hugging Face Hub가 다르게 말한 경우입니다. `hf` CLI 행은 스킬이 설치된 CLI이고, 기본 CLI 위에서 스킬이 추가하는 효과(주로 더 적은 도구 호출)는 아래 [스킬 섹션](#the-hf-cli-skill)에서 분리해 다룹니다. 대표 대화 기록은 [이 버킷](https://huggingface.co/buckets/celinah/hf-cli-agent-benchmark)에 게시됩니다.)*

### 설정

우리는 **18개의 비사소적 Hugging Face Hub 작업**을 정의했습니다. 이것은 단순히 "파일 다운로드" 같은 것이 아니라 실제로 요청받을 만한 작업들입니다: 트렌딩 조직의 모델을 집계하고, 리포지토리의 파일과 용량을 검사하고, 포함/제외 규칙으로 폴더를 업로드하고, 파일을 삭제하고, 리포지토리 간 파일을 복사하고, 라이선스를 추가하는 PR을 열고, 브랜치와 태그가 있는 리포지토리를 만들고, 버킷을 동기화하고 정리하고, 컬렉션을 구성합니다. 각 작업은 Hugging Face Hub에 대해 정확히 하나의 대화 방식으로 대화하는 최신 코딩 에이전트로 전달됩니다:

- `hf` CLI, 혹은
- **curl / the Python SDK**: `hf` CLI가 전혀 없으므로 에이전트는 REST API에 대해 `curl`을 사용하거나 `huggingface_hub` Python 라이브러리에 의존합니다.

우리는 `hf` CLI를 두 가지 구성으로 실행합니다: 스킬이 있는 구성과 없는 구성(생성된 명령 참조는 [별도 섹션](#the-hf-cli-skill)에서 다시 다룹니다). 그러나 아래의 헤드라인 비교는 단순히 **`hf` CLI 대 curl/SDK**이며, 스킬의 증가 효과는 충분히 작아서 메인 결과에 포함시키기보다 독립적으로 다룹니다.

구성은 의도적으로 깔끔합니다: 실행당 새 인스턴스, 커스텀 MCP 서버 없음, `CLAUDE.md`나 `AGENTS.md`도 없음, 맥락에서의 동작을 좌우하는 어떠한 nudges도 없음. 작업과 도구를 하나의 프롬프트에 넣고, 에이전트는 `TASK_COMPLETE` 또는 `TASK_FAILED` 마커로 종료하지만, 그 마커를 신뢰하지 않으므로 각 실행은 라이브 Hub를 재질의로 재조회하여 실제로 브랜치가 생성되었는지, 파일이 실제로 사라졌는지, 버킷이 존재하는지 등을 독립적으로 평가합니다. 각 작업/도구 조합은 **10회** 실행되며, 코딩 에이전트가 비결정적이므로 약 **520회 실행**(18개 작업 × 3개 도구 × 10회 반복, 하나의 청구 가능한 Jobs 작업에 대한 상한 제외) 정도의 총 평가 실행이 필요합니다. 이를 두 에이전트(가장 인기 있는 두 코딩 에이전트: Claude Code와 OpenAI Codex)에서 각각 수행했습니다.

### 결과

아래의 두 차트는 위 표를 해석합니다. 먼저, **Sonnet에서의 작업 성공**은 curl과 SDK가 가장 어려움을 겪는 에이전트입니다:

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-success.png" alt="Task success on Claude Code with Sonnet 4.6: hf CLI 94%, curl / Python SDK 84%." width="100%"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-success-dark.png" alt="Task success on Claude Code with Sonnet 4.6: hf CLI 94%, curl / Python SDK 84%." width="100%"/>
</div>

CLI가 없으면 curl과 SDK는 열 점 차로 뒤처집니다. Sonnet의 경우 작업의 일부를 끝내지 못하기 때문인데, 주로 쓰기 작업이 그렇고, `hf` CLI가 이를 해결합니다.

두 번째 이미지는 **GPT-5.5에서의 토큰 영향**을 작업별로 나눈 그림입니다. 각 막대는 동일한 작업에서 curl/SDK의 토큰을 같은 작업의 CLI 토큰으로 나눈 값이므로, `2.4×`은 hf 비포함 버전이 같은 작업을 수행하는 데 토큰을 2.4배 더 사용했다는 뜻입니다:

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-tokens.png" alt="Per-task token ratio of curl/Python SDK divided by the hf CLI on GPT-5.5, sorted high to low. Multi-step tasks cost curl/Python SDK far more: bucket create+sync+prune 6.0x, rank orgs by trending models 4.1x, repo create+branch+tag / delete files / copy files across repos 2.4x each. Simple one-shot reads sit near parity or cheaper: batch model metadata 0.5x, count dataset rows 0.3x." width="80%"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-tokens-dark.png" alt="Per-task token ratio of curl/Python SDK divided by the hf CLI on GPT-5.5, sorted high to low. Multi-step tasks cost curl/Python SDK far more: bucket create+sync+prune 6.0x, rank orgs by trending models 4.1x, repo create+branch+tag / delete files / copy files across repos 2.4x each. Simple one-shot reads sit near parity or cheaper: batch model metadata 0.5x, count dataset rows 0.3x." width="80%"/>
</div>

단일 샷 읽기(데이터 세트 행 수 세기, 배치 메타데이터)에서는 curl과 SDK가 무난하고 때로는 더 가볍습니다. 그러나 작업이 더 복잡하고 여러 종속 단계가 포함되면, 에이전트는 REST 호출의 전체 체인을 수동으로 구성해야 하며 비용이 크게 증가합니다: 리포지토리 생성(브랜치와 태그 포함), 파일 삭제, 리포지토리 간 복사, 버킷 동기화에서 CLI의 토큰 사용은 **2.4×에서 6×**에 달합니다. `hf` CLI는 에이전트가 복잡한 워크플로를 직접 구성하는 대신, 몇 가지 고수준 명령으로 작업을 표현하게 해 줍니다.

### 핵심 발견

- **The `hf` CLI는 curl이나 SDK보다 훨씬 간결합니다.** 같은 작업에서 성공이 같거나 더 나은 경우에도 curl과 SDK는 **약 1.3×에서 1.8×**의 토큰을 소모합니다. 쉬운 읽기에는 문제 없지만, 실제 다단계 작업에서는 **2×에서 6×**를 지불합니다: CLI는 REST 호출 체인을 몇 가지 고수준 명령으로 구성하는 반면, curl이나 SDK는 매 실행마다 체인을 손으로 재구성합니다.
- **더 강력한 모델에서 curl과 SDK는 작동하지만 낭비는 여전합니다.** Sonnet에서는 작업의 일부를 끝내지 못하고(주로 쓰기), GPT-5.5에서는 대부분 성공적으로 REST 호출을 수동으로 구성하거나 SDK를 사용하지만 여전히 CLI의 토큰 비용보다 많이 지출합니다.

## hf-cli 스킬 {#section-4}

`hf`는 **스킬**을 제공합니다: 에이전트가 컨텍스트로 로드하는 전체 명령 표면에 대한 간결한 참조입니다. 이는 라이브 `hf` 명령 트리에서 자동 생성되며, 명령당 한 줄의 서명, 한 줄 설명, 그리고 중요한 플래그가 묶여 있으며, 리소스별로 그룹화되고 일반 옵션의 짧은 용어집이 함께 제공됩니다. 스스로 설명하는 플래그는 의도적으로 생략해 맥락을 간결하게 유지하고, 릴리스마다 재생성됩니다. 출력하려면 `hf skills preview`를 실행하거나 다음으로 설치하십시오:

```bash
# for Codex, Cursor, OpenCode, Pi and other agents that load skills from `.agents/skills`
hf skills add
# includes the above + Claude Code
hf skills add --claude
```


스킬은 무엇을 얻을 수 있게 하나요? 주로 에이전트의 추측을 멈추게 합니다. 가장 명확한 단일 뷰는 각 실행이 얼마나 많은 명령을 필요로 하는지이며, 스킬 사용 여부에 따라 다릅니다:

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-skill.png" alt="Mean commands (tool calls) per run, with and without the hf-cli skill, on both agents. Claude Code (Sonnet 4.6): 10.4 without the skill, 6.9 with it. Codex (GPT-5.5): 10.1 without, 7.3 with. Fewer is better." width="100%"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/huggingface_hub/chart-skill-dark.png" alt="Mean commands (tool calls) per run, with and without the hf-cli skill, on both agents. Claude Code (Sonnet 4.6): 10.4 without the skill, 6.9 with it. Codex (GPT-5.5): 10.1 without, 7.3 with. Fewer is better." width="100%"/>
</div>

두 에이전트 모두 작업당 약 10개의 명령에서 약 7개 정도로 줄어들며, 대략 도구 호출이 30% 정도 감소합니다. 이는 에이전트가 올바른 명령과 인수를 찾기 위해 `--help`를 탐색하지 않기 때문입니다. 스킬은 컨텍스트에 고정된 정보를 앞에 붙이므로 토큰 비용을 크게 줄이지는 않지만, 같은 작업에서 토큰은 대체로 비슷하게 유지되거나 약간 상승합니다. 또한 스킬은 CLI의 신뢰성을 높이지는 않지만, 도구의 작동 방식을 알아보는 대신 작업을 실행하는 데 에이전트가 더 많은 시간을 사용하도록 돕습니다. 로컬 모델과 함께 `hf`를 사용할 때 특히 유용할 수 있습니다.

각 작업을 새로운 세션에서 실행했으므로 스킬은 매 작업마다 컨텍스트 비용을 부담합니다. 다중 작업 세션에서는 이 비용이 상쇄되므로(에이전트가 한 번만 명령 표면을 학습), 토큰 면에서 상황이 개선될 가능성이 있습니다; 다만 그 경우는 측정하지 않았습니다.

## 직접 사용해 보기 {#section-5}

이 모든 것을 벤치마크한 이유는 이것이 중요하다고 생각하기 때문입니다. 에이전트는 Hugging Face Hub의 실제 사용자로 자리잡아 가고 있습니다: 그들은 모델을 훈련하고, 데이터셋을 구성하고 정리하며, Spaces로 데모를 배포합니다. 이는 거의 항상 사람을 대신해 작동합니다. 에이전트에게 잘 작동하는 Hub는 이를 사용하는 사람들에게도 더 잘 작동하는 Hub입니다. 에이전트의 도구가 더 좋아질수록, 에이전트가 당신을 위해 더 많은 일을 할 수 있습니다.

에이전트가 Hugging Face Hub와 상호 작용한다면 `hf` CLI를 제공하는 것을 권장합니다:

```bash
# macOS / Linux
curl -LsSf https://hf.co/cli/install.sh | bash

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```


그다음 스킬을 넘겨 주어, 처음 턴부터 전체 명령 표면을 알 수 있게 하세요:

```bash
hf skills add            # Codex, Cursor, OpenCode, Pi and other agents that load skills from .agents/skills
hf skills add --claude   # the above + Claude Code
```


그다음 에이전트를 Hugging Face Hub로 향하도록 하고 작동하게 하세요. 로그인되어 있는지 확인하고(`hf auth login`), 그런 다음 다음과 같은 프롬프트를 에이전트에게 전달하세요:

```text
Use `hf` to list my Hugging Face Hub models, datasets, and Spaces.
Take a look at how I am currently using the Hub and suggest a few ways you could help me.
```


에이전트가 스스로 명령을 찾아 유용한 결과를 들고 돌아올 것입니다.

전체 명령 참조는 [`hf` CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)에 있습니다.

## 에이전트 하네스 등록 {#section-6}

에이전트 하네스를 구축하고 있나요? **등록하십시오!** 이것이 `hf`가 이를 감지하는 방식이고, Hub가 트래픽을 당신의 하네스에 귀속하는 방법입니다. [`agent-harnesses.ts`](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/agent-harnesses.ts)에 항목을 추가하는 간단한 PR을 열기만 하면 됩니다. 자세한 내용은 [Register your agent harness](https://huggingface.co/docs/hub/agents-overview#register-your-agent-harness) 가이드를 참고하십시오.
