---
layout: post
title: "GitHub CI를 Hugging Face Jobs로 마이그레이션"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-06-09-github-ci-hf-jobs/thumbnail.gif
authors:
  - user: abidlabs
slug: "github-ci-hf-jobs"
source_url: "https://huggingface.co/blog/github-ci-hf-jobs"
source_published_date: "2026-06-09"
source_published_at: "2026-06-09T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Migrating Your GitHub CI to Hugging Face Jobs](https://huggingface.co/blog/github-ci-hf-jobs)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/github-ci-hf-jobs -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# GitHub CI를 Hugging Face Jobs로 마이그레이션

GitHub 저장소가 있고 GitHub Actions가 활성화되어 있다면, 아마도 CI를 위해 GitHub 호스팅 런너를 사용할 가능성이 큽니다. 그것은 많은 프로젝트에서 기본 설정인데, 간단하기 때문입니다: 워크플로를 추가하고 `runs-on: ubuntu-latest`를 작성하면 GitHub이 기계를 제공합니다.

그 기본 설정은 편리하지만 한계도 있습니다. GitHub Actions는 느려지거나 유지 보수로 다운될 수 있고, 호스팅 머신은 일반적이며, GPU 접근은 대부분의 오픈 소스 프로젝트에서 바로 활성화하기 어렵습니다. [Trackio](https://github.com/gradio-app/trackio)의 경우 이러한 한계가 점점 문제로 다가왔습니다. 기본 단위 테스트와 프런트엔드 확인을 위한 안정적인 CPU CI는 물론 실제 CUDA 하드웨어에서 실행해야 하는 테스트를 위한 GPU CI도 원했습니다.

그래서 대안을 만들었습니다: CI는 GitHub Actions가 담당하되, 작업은 [Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs-overview)에서 실행되도록 하는 것입니다.

결과: Trackio의 CI가 이제 Hugging Face Jobs에서 실행되고 실시간 로그를 스트리밍합니다. **CPU 작업의 CI 시간을 약 30% 단축하고 GPU 머신에서 실행되는 완전히 새로운 테스트 스위트를 가능하게 합니다**!

이 글에서는 GitHub 저장소에 대해 동일한 설정을 단계별로 재현하는 방법을 설명합니다. 에이전트를 사용 중이라면 이 글을 참고하실 수 있는데, 인간용으로 브라우저 기반 지침과 함께 CLI 지침이 함께 제공되기 때문입니다.

Hugging Face Jobs에 대한 간단한 소개로 시작하겠습니다!

## Hugging Face Jobs란? {#section-1}

[Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs-overview)는 거의 모든 하드웨어 구성으로 Hugging Face의 서버리스 인프라에서 명령어나 스크립트를 실행할 수 있게 해줍니다. Job은 본질적으로 다음과 같습니다:

- 실행할 명령
- Docker Hub 또는 Hugging Face Space에서의 Docker 이미지
- CPU 또는 `t4-small` 또는 `h200` GPU 와 같은 하드웨어 구성
- 선택적 환경 변수 및 시크릿

예를 들어, 아래를 실행할 수 있습니다:

```bash
hf jobs run python:3.12 python -c "print('Hello world')"
```


또는

```bash 
hf jobs uv run --flavor a10g-small "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" 
```


그로 인해 Jobs는 CI에 자연스럽게 맞습니다. CI 작업은 이미 명령 중심으로 동작하고, 깨끗한 환경에서 실행되며, 정확히 적합한 하드웨어를 선택하는 데서 이점이 있습니다. 머신 러닝 라이브러리의 경우 GPU 쪽은 특히 매력적입니다. 실제 CUDA 하드웨어에서 테스트 스위트를 실행할 수 있습니다. 자체로 항상 작동하는 러너를 유지할 필요가 없습니다.

핵심 단계는 GitHub Actions를 Hugging Face Jobs에 연결하는 것이며, 아래에 설명합니다.

## 아키텍처 {#section-2}

이 설정을 위해 [`huggingface/jobs-actions`](https://github.com/huggingface/jobs-actions)를 만들었는데, 이는 GitHub Actions 작업을 Hugging Face Jobs 안에서 실행되는 일시적인 셀프호스트 러너로 바꿔주는 작은 다리 역할을 합니다.

전체 흐름은 다음과 같습니다:

1. 풀 리퀘스트가 GitHub Actions 워크플로를 트리거합니다.
2. GitHub가 `runs-on` 레이블이 사용 가능하지 않은 모든 작업을 큐에 올리고, 예를 들어 `hf-jobs-cpu-upgrade` 또는 `hf-jobs-t4-small`를 큐에 두고, 서명된 `workflow_job.queued` 웹훅을 GitHub App를 통해 dispatcher Space로 보냅니다.
3. dispatcher Space는 웹훅을 검증하고, `hf-jobs-*` 레이블이 있는지 확인한 뒤, 짧은 수명의 GitHub 러너 등록 토큰을 발급하고 일치하는 하드웨어에서 Hugging Face Job를 시작합니다.
4. Hugging Face Job은 임시 GitHub Actions 러너를 부팅하고, 그 원샷 토큰을 사용해 레포에 등록합니다.
5. GitHub가 대기 중인 워크플로우 작업을 해당 러너에 할당합니다; 러너가 CI 작업을 실행하고 GitHub에 상태를 보고한 뒤 종료합니다.

GitHub 입장에서는 이것은 단지 셀프 호스팅 러너일 뿐입니다. Hugging Face의 입장에서는 그것은 저장소의 GitHub Actions에서 워크플로우 단계를 실행하기 위해 컨테이너를 시작하는 하나의 Job일 뿐입니다.

## 1단계: dispatcher Space 복제 {#section-3}

가장 먼저 필요한 것은 dispatcher Space입니다. 이는 GitHub `workflow_job` 웹훅 이벤트를 수신하고 그에 응답하여 Hugging Face Jobs를 시작하는 작은 Docker Space입니다.

이 Space를 먼저 만드는 이유는 GitHub App에 웹훅 URL이 필요하고 그 URL이 Space에서 나오기 때문입니다. 이 Space는 고유 네임스페이스 아래에 있거나 쓰기 권한이 있는 Hugging Face org 아래에 있어야 합니다.

#### Web 설정

[`huggingface/jobs-actions-dispatcher`](https://huggingface.co/spaces/huggingface/jobs-actions-dispatcher)로 이동하여 **이 Space 복제**를 클릭합니다.

<img width="996" alt="image" src="https://github.com/user-attachments/assets/c8b450c3-b801-43dc-97ff-954d9bbaf975" />

다 following:

```text
Owner: your HF user or org
Name: jobs-actions-dispatcher
Hardware: cpu-upgrade
```


실제 CI를 위해 dispatcher가 GitHub 웹훅에 지속적으로 사용할 수 있도록 `cpu-upgrade`를 사용합니다. `cpu-basic`은 테스트에 적합하고 실행될 가능성이 높지만 비활동 후 잠들 수 있습니다; GitHub의 웹훅이 깨어나는 동안 도착하면 워크플로가 영원히 큐에 남을 수 있습니다.

빌드가 완료되면 복제된 Space를 엽니다. 현재는 무시해도 되는 "Required Space secrets" 섹션이 보일 것입니다. 다음 단계에서 필요한 GitHub App 웹훅 URL이 랜딩 페이지에 표시되어야 하며, 아래와 같은 형태일 것입니다:

```text
https://YOUR-HF-NAMESPACE-jobs-actions-dispatcher.hf.space/webhook
```


#### CLI 설정

대리인으로 dispatcher Space를 설정하거나 CLI 워크플로를 사용하고 싶다면:

```bash
export HF_NAMESPACE=your-hf-user-or-org
export SPACE_ID="$HF_NAMESPACE/jobs-actions-dispatcher"

hf repo duplicate huggingface/jobs-actions-dispatcher "$SPACE_ID" \
  --type space \
  --flavor cpu-upgrade \
  --exist-ok
```


다음과 같이 설정합니다:

```bash
export DISPATCHER_URL="https://${HF_NAMESPACE}-jobs-actions-dispatcher.hf.space"
```


## 2단계: GitHub App 생성 및 설치 {#section-4}

다음으로, dispatcher Space 자체에서 GitHub App을 생성하고 설치합니다. 이 App은 대기 중인 워크플로우 작업을 수신하고 임시 셀프호스드 러너 등록 토큰을 생성할 수 있는 권한이 필요합니다.

### 웹 설정

다음으로, dispatcher Space 자체에서 GitHub App을 생성하고 설치합니다. 이 App은 대기 중인 워크플로우 작업을 수신하고 임시 셀프호스드 러너 등록 토큰을 생성할 수 있는 권한이 필요합니다.

```text
https://YOUR-HF-NAMESPACE-jobs-actions-dispatcher.hf.space
```


설정 양식에서 Hugging Face Jobs에서 CI가 실행되도록 하는 GitHub 리포를 입력합니다:

```text
YOUR-GITHUB-ORG/YOUR-REPO
```


그런 다음 GitHub App 생성을 위한 버튼을 클릭합니다. GitHub은 App의 이름을 선택하라고 묻습니다. 이름은 GitHub 계정이나 조직에서 사용 가능하면 무엇이든 상관없습니다. 제출하면 최종 화면에 `hf` CLI를 사용해 dispatcher Space에 App 자격 증명을 업로드하는 정확한 방법이 표시됩니다.

**중요:** Jobs를 시작할 수 있는 권한이 있는 [Hugging Face token](https://huggingface.co/settings/tokens)를 제공해야 합니다. 이는 귀하의 개인 계정이나 Jobs 요금이 청구될 조직에 해당합니다. 이 토큰은 dispatcher Space의 `HF_TOKEN` 시크릿으로 저장해야 합니다.

마지막으로 Space에 입력한 같은 GitHub 리포지토리에 App을 설치합니다. Trackio 설정에서는 `gradio-app/trackio`에 설치했습니다.

### 에이전트 보조 설정

GitHub App 매니페스트 흐름은 여전히 브라우저 기반이지만, 에이전트도 동일한 Space 주도 경로를 따를 수 있습니다:

```bash
export HF_NAMESPACE=your-hf-user-or-org
export GITHUB_REPO=YOUR-GITHUB-ORG/YOUR-REPO
open "https://${HF_NAMESPACE}-jobs-actions-dispatcher.hf.space"
```


Space에 `$GITHUB_REPO`를 붙여넣고 GitHub App 생성 버튼을 클릭한 뒤 사용할 수 있는 App 이름을 하나 선택하고 생성된 GitHub 지침을 따르면 됩니다.

앱이 생성된 후, App 설정 페이지에서 리포에 설치합니다. GitHub org의 경우 설치 설정은 아래에 있습니다:

```text
https://github.com/organizations/YOUR-GITHUB-ORG/settings/installations
```


## 3단계: 최종 dispatcher 설정 {#section-5}

이 시점에서 dispatcher Space가 구성되어 있어야 합니다. GitHub App 설정 흐름은 Space에 App 자격 증명, 웹훅 시크릿 및 Hugging Face 토큰을 업로드하는 명령을 생성합니다.

<img width="1317" alt="image" src="https://github.com/user-attachments/assets/0fc8ac73-f93a-419b-bd80-70da2756f50c" />

기본적으로 HF Jobs는 dispatcher Space와 동일한 네임스페이스에서 시작됩니다. 원한다면 `HF_NAMESPACE`를 Space 변수로 설정하여 다른 Hugging Face 사용자 또는 org에 비용을 청구하도록 할 수 있습니다:

```bash
export SPACE_ID=YOUR-HF-NAMESPACE/jobs-actions-dispatcher
hf spaces variables add "$SPACE_ID" -e HF_NAMESPACE=your-billing-namespace
hf spaces restart "$SPACE_ID"
```


2단계에서 설정한 토큰은 이 네임스페이스에 해당해야 합니다.

## 4단계: `runs-on` 변경 {#section-6}

실제 워크플로 변경은 작습니다. 대신에:

```yaml
runs-on: ubuntu-latest
```


Dispatcher가 처리하는 레이블 중 하나를 사용합니다:

```yaml
runs-on: hf-jobs-cpu-upgrade
```


GPU 테스트의 경우 GPU 레이블을 사용합니다:

```yaml
runs-on: hf-jobs-t4-small
```


Hugging Face Jobs에서 실행하려는 모든 GitHub Action에 대해 필요한 것은 이 한 줄의 변경뿐입니다!

## 5단계: 테스트해 보기 {#section-7}

CLI에서 최소한의 스모크 테스트 워크플로를 추가하려면:

```bash
mkdir -p .github/workflows
cat > .github/workflows/hf-jobs-test.yml <<'EOF'
name: HF Jobs Test

on:
  pull_request:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: hf-jobs-cpu-upgrade
    steps:
      - uses: actions/checkout@v4
      - run: echo "Hello from Hugging Face Jobs"
EOF

git add .github/workflows/hf-jobs-test.yml
git commit -m "Run CI on Hugging Face Jobs"
git push
```


CLI에서 확인하려면:

```bash
gh run list --repo YOUR-GITHUB-ORG/YOUR-REPO --limit 5
hf jobs ps --namespace "$HF_NAMESPACE"
hf spaces logs "$SPACE_ID"
```


일반적인 GitHub Action처럼 로그를 확인할 수 있어야 합니다—예를 들어 이 [Trackio PR #565](https://github.com/gradio-app/trackio/pull/565)에서.

그게 다입니다!

*적절한 Docker 이미지 선택에 대한 참고*

처음 CPU 구성은 `ubuntu:22.04`를 사용했고 매 실행마다 누락된 시스템 패키지를 설치했습니다. 그것은 작동했지만 필요보다 느렸습니다. GitHub의 `ubuntu-latest` 이미지는 기본적으로 많은 개발 도구를 포함하고 있지만, 순수한 Ubuntu 이미지는 그렇지 않습니다.

Trackio의 UI 테스트는 Playwright 브라우저, Node, ffmpeg, sqlite, git 및 일반적인 Linux 빌드 의존성이 필요합니다. Hugging Face Jobs는 어떤 [Docker image](https://huggingface.co/docs/hub/jobs-popular-images)든 사용할 수 있도록 지원하므로 Microsoft Playwright 이미지를 사용하기로 했고, 잘 작동했습니다:

```text
mcr.microsoft.com/playwright:v1.60.0-jammy
```


GPU 작업의 경우, 우리는 다음을 사용했습니다:

```text
nvidia/cuda:12.4.0-runtime-ubuntu22.04
```


## 결과 {#section-8}

다음은 Trackio CI의 수치입니다:

| 런너 설정 | 실행 시간 | GitHub 평균 대비 |
| --- | ---: | ---: |
| GitHub `ubuntu-latest` 기준 | `1m40s`  | 기준 |
| Hugging Face Jobs CPU, Playwright 이미지 | `1m10s` | `-30s`, 약 `30%` 배 빠름 |
| Hugging Face Jobs GPU, `t4-small` 레이블 | `45s` | GitHub 호스팅 GPU 기준선 없음 |

가장 큰 이점은 GPU CI였습니다. Trackio의 GPU 검사는 Hugging Face Jobs에서 실행되어 `45s`에서 통과했고, 그 기간 동안 `t4-small` 요율로 센트 미만의 비용이 들었습니다.

CPU 결과도 고무적이었습니다. 올바른 이미지로 Linux 테스트 작업은 GitHub 호스팅 기준선보다 빨랐습니다. 이는 Hugging Face Jobs가 특히 맞춤 이미지나 가속기가 필요한 머신 러닝 프로젝트에 실용적인 CI 백엔드가 될 수 있음을 시사합니다.

로그도 또 하나의 즐거운 놀라움이었습니다. GitHub Actions 로그는 유용하지만 대용량 로그에는 웹 UI가 무겁습니다. Hugging Face Jobs 로그는 CLI에서 쉽게 가져올 수 있습니다:

```bash
hf jobs logs <job_id> > logs.txt
```


로컬 도구나 코딩 에이전트로 쉽게 검사할 수 있습니다. 다리에서 GitHub Actions 작업 로그를 Hugging Face Job 로그에 동기화했기 때문에 두 시스템 중 어느 쪽이든 실행을 디버깅할 충분한 정보를 가졌습니다.

마지막으로, Trackio의 CI에 필요하지는 않았지만, Hugging Face Jobs는 또한 [supports mounting volumes](https://huggingface.co/docs/huggingface_hub/en/guides/jobs#mount-a-volume)를 제공하여 CI의 일부로 Hugging Face에서 데이터셋이나 모델을 빠르게 로드해야 할 때 매우 유용할 수 있습니다.

바로 Hugging Face Jobs를 사용해 GitHub Actions를 실행해 보시길 바랍니다!
