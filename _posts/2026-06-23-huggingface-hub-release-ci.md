---
layout: post
title: "매주 AI, 오픈 도구, 그리고 휴먼 인 더 루프가 포함된 huggingface_hub 배포"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-06-23-huggingface-hub-release-ci/thumbnail.png
authors:
  - user: Wauplin
  - user: celinah
slug: "huggingface-hub-release-ci"
source_url: "https://huggingface.co/blog/huggingface-hub-release-ci"
source_published_date: "2026-06-23"
source_published_at: "2026-06-23T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Shipping huggingface_hub every week with AI, open tools, and a human in the loop](https://huggingface.co/blog/huggingface-hub-release-ci)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/huggingface-hub-release-ci -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 매주 AI, 오픈 도구, 그리고 휴먼 인 더 루프가 포함된 huggingface_hub 배포

`huggingface_hub`은 허깅페이스 생태계의 기본인 파이썬 클라이언트입니다. `transformers`, `datasets`, `diffusers`, `sentence-transformers` 그리고 수십 개의 다른 라이브러리들이 Hub와 소통하기 위해 그것에 의존합니다. 우리가 매주 새로운 릴리스를 배포하지 않는 한 주는 `main`에 갇힌 수정사항과 기능의 한 주가 됩니다.

오랜 기간 동안 우리는 4~6주마다 릴리스를 발표했습니다. 이제는 단일 GitHub Actions 워크플로우에서 매주 릴리스를 발표합니다. 오픈 소스 도구와 오픈-가중치 모델을 사용해 이를 구축했고, 판단이 중요한 한 곳에 휴먼 인 더 루프를 두었습니다. 이 글의 어떤 내용도 공급업체 계약, 비공개 모델, 또는 자신이 실행할 수 없는 인프라를 요구하지 않습니다. 이는 시작부터의 설계 목표였으며, 다른 유지 관리자가 가져다 사용하고 조정할 수 있는 워크플로우를 원했기 때문입니다.

본 글이 끝나면 당신이 직접 만들어볼 수 있도록 필요한 모든 것이 제공될 것입니다.

## 시작한 곳 {#section-1}

구식 프로세스는 부분적으로 자동화되어 있었고, 주로 수동이었습니다.

이미 CI에 포함되어 있습니다:

- 태그가 푸시되면 PyPI에 게시합니다.
- 릴리스 후보가 고정된 상태로 다운스트림 라이브러리의 테스트 브랜치를 열고 테스트합니다.

여전히 매번 수동으로:

- 릴리스 브랜치를 만들고, `__init__.py`에서 버전을 올리고, 커밋하고, 태그하고, 푸시합니다.
- 다운스트림 CI 실행을 지켜보고 실패를 분류합니다.
- 마지막 릴리스 이후 병합된 모든 PR을 하나씩 읽고, PR의 맥락과 주제별로 정리하여, `git log` 덤처럼 들리지 않는 어조로 릴리스 노트를 수기로 작성합니다.
- RC 기간이 끝난 후 안정 릴리스를 배포합니다.
- 내부 Slack 발표와 소셜 포스트를 초안합니다.
- 릴리스 후 PR을 열어 `main`를 다음 `dev0`로 올립니다.

새 버전에 대한 좋은 노트를 작성하는 것은 무거운 부분이었고, 서로 다른 주제의 수십 개 PR을 모아놓는 작업이었다. 기술적으로는 어렵지 않지만 몇 시간의 집중이 필요했다. 여기에 발표를 더하면 소형 릴리스도 여러 날에 걸쳐 반나절 정도의 작업으로 끝나곤 했다.

## 두 종류의 작업 {#section-2}

그래서 전체 프로세스를 간소화하기로 했습니다. 그 목록을 보면 작업은 두 가지로 나뉩니다.

일부 단계는 순전히 기계적이며 자동화가 가능하다: 버전 증가, 커밋, 태깅, 푸시, 다운스트림 테스트 브랜치 열기, 포스트 릴리스 PR 열기. 이를 누가 생각할 필요가 없다. 항상 올바른 순서대로 일어나도록 해야 하며, 이것이 CI 워크플로우가 잘하는 일이다.

나머지는 다르다. 릴리스 노트 쓰기, 무엇을 하이라이트로 삼을지 결정, 사람 독자 audience를 위한 발표 문구 작성: 이건 두뇌 노동이다. 수년간 릴리스 매뉴얼을 유지해온 판단의 영역이다. 이때 AI가 들어와 빈 페이지를 몇 초 만에 탄탄한 초고로 바꿔준다. 또한 조심해야 할 점은, 자신감 있어 보이지만 은근히 잘못된 초고가 만들어질 수 있다는 것이다.

## 설계 원칙: 열려 있는 부품, 누구나 재사용 가능 {#section-3}

이 문제를 해결하기로 결정했을 때, 한 가지 제약을 미리 세웠습니다: 모든 가동 부품은 어떤 유지 관리자가 스스로 실행할 수 있어야 한다. 교환할 수 없는 API 뒤의 비공개 모델, 독점적인 릴리스 플랫폼, 비밀 소스는 허용되지 않습니다.

다음은 전체 스택입니다:

| 부분                                                                                              | 하는 일                                    |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **GitHub Actions**                                                                                | 전체 릴리스를 조정합니다                  |
| **[OpenCode](https://opencode.ai/)**                                                              | 모델을 구동하는 에이전트 런타임             |
| **An open-weights model** (현재 [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)에서 온 Z.ai) | 릴리스 노트와 Slack 발표 초안 작성 |
| **[HF Inference Providers](https://huggingface.co/docs/inference-providers/index)**               | 모델의 서비스를 제공합니다                                |
| **[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)**                          | 패키지를 게시합니다                           |

두 번째 원칙: 모델이 초안을 작성하고, 사람은 결정을 내립니다. 언어 모델은 간결한 PR 제목 30개를 읽고 읽기 쉬운 릴리스 노트로 바꾸는 데 능하지만, 맹목적으로 신뢰하기에는 좋지 않습니다. 그래서 이 워크플로우는 사람의 감독 아래 진행됩니다: 모델이 초안을 작성하고, 결정론적 스크립트가 이를 확인하며, 그 후 사람이 검토하고 수정한 뒤에야 어떤 것이든 릴리스됩니다(아래에서 더 자세히 다룹니다).

## 파이프라인 둘러보기 {#section-4}

전체 워크플로우는 단일 파일 [`.github/workflows/release.yml`](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml)로, Actions UI에서 수동으로 트리거됩니다. 입력은 정확히 하나입니다:

```yaml
on:
  workflow_dispatch:
    inputs:
      release_type:
        type: choice
        options:
          - minor-prerelease   # cut an RC from main
          - minor-release      # promote the RC to final
          - patch-release      # bugfix on an existing release branch
```


그다음 작업은 대략 아래 순서로 실행됩니다:

- **Prepare.** 다음 버전을 계산하고, 릴리스 브랜치를 새로 만들거나 재사용하고, `__version__`를 올리고, 커밋하고, 태그하고, 푸시합니다.
- **Publish to PyPI.** `huggingface_hub`를 빌드하고 업로드합니다. 동시에는 `hf` CLI를 PyPI 패키지로도 빌드하고 업로드합니다.
- **Release notes.** 마지막 태그 이후 커밋 범위를 Diff하고, PR 메타데이터를 GitHub API에서 가져와 모델이 구조화된 변경 로그를 초안하게 합니다 ([here's a recent one](https://github.com/huggingface/huggingface_hub/releases/tag/v1.20.0)). *초안* GitHub 릴리스로 저장됩니다.
- **Downstream test branches.** RC용으로, `transformers`, `datasets`, `diffusers`, `sentence-transformers`에서 RC를 고정하고 브랜치를 열어 그들의 CI가 우리가 망가뜨렸는지 빠르게 알려주게 합니다.
- **Slack announcement.** 노트를 읽고 우리 팀의 목소리로 내부 발표를 작성합니다.
- **Archive notes.** 원시 AI 초안과 사람이 편집한 버전을 나란히 허깅페이스 Bucket에 업로드합니다.
- **Post-release bump.** 안정적인 릴리스 후, `main`에서 다음 `dev0`로 올리는 PR을 엽니다.
- **Comment on shipped PRs.** 릴리스의 모든 PR에 "이 버전에서 배포되었습니다"라는 코멘트를 남깁니다.
- **Sync CLI docs.** 재생성된 `hf` CLI 스킬 문서를 포함하는 PR을 우리의 [skills](https://github.com/huggingface/skills) 저장소에 엽니다.
- **Report to Slack.** 각 단계가 상태를 스레드 응답으로 게시하고, 최종 작업이 루트 메시지를 ✅ 또는 ❌로 업데이트합니다.

남은 수동 단계는 초안 릴리스 노트를 검토하고 게시하는 것과 내부 Slack 메시지를 검토하고 게시하는 것입니다. 이 두 단계에는 휴먼 인 더 루프를 유지하려 합니다.

## 신뢰하되 검증하라: 사람의 개입 핵심 {#section-5}

다음은 AI가 생성한 릴리스 노트에서 사람들이 걱정하는 실패 모드입니다: 모델이 조용히 PR 하나를 누락시키거나 이 릴리스에 포함되지 않은 PR을 만들어 버립니다. 거의 맞는 변경 로그는 아무도 재확인하지 않기 때문에, 아무 것도 없는 변경 로그보다 더 나쁩니다.

우리는 생성된 릴리스 노트가 첫 시도에 완전하다고 신뢰하지 않고 결정론적으로 검증합니다. 모델이 실행되기 전에 파이썬 스크립트가 이 릴리스에 속한 모든 PR을 가져와 실제 기준 데이터로 저장합니다.

```python
# Deterministic: extract PR numbers from squash-merge commits in the range.
PR_NUMBER_PATTERN = re.compile(r"\(#(\d+)\)$")

pr_numbers = [
    int(m.group(1))
    for commit in commits_since_last_tag
    if (m := PR_NUMBER_PATTERN.search(commit.title))
]
save_manifest(pr_numbers)  # the source of truth
```


그런 다음 모델이 그들로부터 노트를 초안합니다. 다 작성되면, 초기 PR 목록과 생성된 출력물을 대조해 확인합니다:

```python
expected = set(load_manifest())          # what should be there
found    = extract_pr_refs(notes_md)     # what the model wrote (#1234 -> 1234)

missing = expected - found               # silently dropped
extra   = found - expected               # belongs to a different release
```


무언가 누락되거나 추가되면 실패하지도 않으며 잘못된 파일도 배포하지 않습니다. 불일치를 에이전트에 다시 넘겨 정확히 그 PR들을 수정하도록 요청합니다:

```python
for _ in range(MAX_ITERATIONS):
    missing, extra = validate(notes)
    if not missing and not extra:
        break  # matches the manifest exactly
    run_agent_fix(missing_prs=missing, extra_prs=extra)
```


이 패턴이 전체를 신뢰할 수 있게 만드는 원칙입니다: 결정론적 가드레일로 감싼 비결정적 모델. 모델은 산문 작성에 능하지만 포괄적으로 하는 것은 신뢰할 수 없습니다. 그래서 모델이 초안을 작성하게 두고 코드가 일관성을 강제합니다.

## 모델이 사실을 왜곡하지 않도록 뿌리내리기 {#section-6}

완전성은 한 축이고 정확성은 다른 축이다. 제목만으로 PR을 요약하는 모델은 실제 API와 일치하지 않는 코드 예제를 기꺼이 만들어 냅니다.

이를 방지하기 위해 PR 메타데이터를 가져올 때 PR이 다룬 `docs/` 아래의 `.md` 파일의 실제 문서 차이(diff)를 함께 가져옵니다.

```python
def fetch_doc_diffs(pr):
    return [
        {"filename": f.filename, "status": f.status, "patch": f.patch}
        for f in pr.get_files()
        if f.filename.startswith("docs/") and f.filename.endswith(".md") and f.patch
    ]
```


그 차이는 모델의 컨텍스트에 들어가므로, '다음은 새로운 CLI 명령어입니다'라고 쓸 때 PR 작성자가 문서에 실제로 쓴 예제를 인용하게 됩니다. 이는 이전과 같은 논리입니다: 모델에게 진짜 출처 자료를 주고, 좁은 작업을 부여합니다.

프롬프트 자체는 [Skills](https://github.com/huggingface/huggingface_hub/tree/main/.opencode/skills/hf-release-notes)로 남아 있습니다: 저장소에 커밋된 작은 Markdown 파일들(`SKILL.md`와 참조 템플릿 포함). 릴리스-노트 스킬은 하이라이트를 고르는 방법, 섹션의 구조, 문서 링크를 추가할 시기 등을 명시합니다. 이것은 온보딩 지침처럼 읽히며, 이는 정확히 올바른 사고 모델입니다.

## 사람 체크포인트 {#section-7}

RC가 게시된 후, 초안 GitHub 릴리스는 AI의 첫 번째 초안을 포함한 상태로 남아 있습니다. 여기서 사람이 개입합니다:

1. 리뷰어가 초안을 읽고 어조와 강조를 다듬고 모델이 과대 또는 과소 평가한 부분을 수정합니다.
2. 그때서야 `minor-release` 실행을 트리거하여 RC를 최종으로 승격합니다.

리뷰어의 시간은 다듬기에 쓰여, 글쓰기의 반나절 분량의 작업을 15분의 편집 세션으로 바꿉니다.

또한 시간이 지나도 개선할 수 있도록 문서 기록을 남깁니다. RC 시점에 원시 AI 초안과 최종 릴리스가 잘려질 때의 사람이 편집한 버전을 서로 나란히 허깅페이스 Bucket에 보관합니다.

```bash
# at RC time: straight from the model, untouched
hf cp release_notes_raw.txt    "hf://buckets/huggingface/releases/huggingface_hub/${V}/release_notes_raw.txt"

# at release time: after the human review
hf cp release_notes_edited.txt "hf://buckets/huggingface/releases/huggingface_hub/${V}/release_notes_edited.txt"
```


매주 두 가지를 모두 수집하면 모델이 작성한 것과 우리가 원하던 것이 어떻게 다른지에 대한 점점 커지는 데이터 세트가 생깁니다. 이 데이터 세트는 에이전트의 기술을 업데이트하는 데 재사용할 수 있습니다.

## 개방적이고 안전한 파이프라인 {#section-8}

릴리스 프로세스를 재설계하는 것은 보안을 강화할 좋은 기회였으며, 특히 공급망 공격에 대한 방어에 초점을 맞췄습니다.

**PyPI 토큰 없음.** 게시에는 [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)를 사용합니다: PyPI는 이 워크플로우에 대해 단기간 발급된 OIDC 토큰을 GitHub가 발급해 검증하고, 산출물마다 [PEP 740](https://peps.python.org/pep-0740/) 증명 / Sigstore 원산지를 제공합니다. 장기간 비밀이 누출되거나 회전할 필요가 없습니다.

```yaml
permissions:
  id-token: write       # mint the OIDC token for PyPI
  attestations: write   # generate Sigstore provenance
# ...
- uses: pypa/gh-action-pypi-publish@v1.14.0
  with:
    attestations: true  # no password, no API token, just OIDC
```


**에이전트 런타임은 고정되고 검증됩니다.** 최신 OpenCode를 맹신하지 않고, 버전을 고정하고 실행하기 전에 SHA256을 확인합니다:

```bash
curl -fsSL https://opencode.ai/install | bash -s -- --version "${OPENCODE_VERSION}"
echo "${OPENCODE_SHA256}  $(which opencode)" | sha256sum -c -
```


**오픈 툴링은 부주의한 툴링을 의미하지 않는다.**

## 그래서 비용은 얼마였나? {#section-9}

거의 비용이 들지 않았습니다. 전체 릴리스(노트와 Slack 발표, 20~40개 PR과 몇 차례의 프롬프트 반복)를 Inference Providers에서 약 **$0.25**의 비용으로 처리합니다. 오픈 가중치가 종량제로 청구되므로, 매주 실제로 중요한 질문은 '배포할 가치가 있는가?'이고, 항상 그런 가치가 있습니다.

## 실제로 무엇이 바뀌었나 {#section-10}

주기가 4~6주마다 한 번에서 매주 한 번으로 바뀌었습니다. 2차 효과가 더 흥미로웠습니다:

- **노트가 더 나아졌고, 더 나빠지지 않았다.** 최초 초안은 항상 존재하므로 검토 시간은 다듬기에 집중됩니다. 구성은 더 일관적으로 되며 생략되는 것이 더 적습니다.
- **Breakages가 더 일찍 드러난다.** RC 후보 기간 동안 다운스트림 테스트 브랜치가 통합 이슈를 빠르게 포착합니다.
- **기여자 루프가 짧아졌다.** 자동 "이 버전 vX.Y.Z에 배포되었습니다" 코멘트가 기대 이상으로 중요해졌습니다. 누군가가 닫힌 PR에 이슈를 보고하면, 모두가 어떤 릴리스에 수정이 들어갔는지 즉시 볼 수 있습니다. 예전에는 수동으로 태그를 찾느라 애를 먹었습니다.

## 나의 것으로 만들기 {#section-11}

이 부분이 우리가 가장 중요하게 여겼던 부분입니다. 워크플로우는 `huggingface_hub`를 중심으로 구성되었지만 구조는 일반적입니다.

**거의 그대로 재사용 가능:**

- 트리거 및 버전 증가 로직 (`minor-prerelease` → `minor-release` → `patch-release`).
- 신뢰하되 검증하는 루프: 결정론적 매니페스트, 모델 초안, 검증, 재프롬프트. 이것은 생성하는 어떤 것과도 독립적으로 활용 가능한 아이디어입니다.
- OIDC 신뢰된 게시, 핀되고 체크섬으로 검증되는 런타임, Slack 스레딩.
- 스킬 기반 프롬프트: 템플릿을 바꿔도 구조를 유지합니다.

**당사에 특화된 부분:**

- 다운스트림 저장소 목록과 그 의존성 핀 포맷.
- 스킬에서의 정확한 섹션 분류 체계와 어조.
- Slack 및 Bucket 대상지.

적용하려면: [workflow file](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml)와 [scripts](https://github.com/huggingface/huggingface_hub/tree/main/utils/release_notes)를 포크하고, 당신의 패키지에 맞추어 [skill Markdown](https://github.com/huggingface/huggingface_hub/blob/main/.opencode/skills/hf-release-notes/SKILL.md)를 당신 project's voice에 맞게 재작성하고 두 개의 저장소 변수(모델 ID와 OpenCode 버전)를 설정한 뒤 PyPI에서 Trusted Publishing를 설정하고, 다운스트림이 없다면 다운스트림-테스트 작업을 삭제합니다. 신뢰하되 검증하는 루프는 그대로 재사용할 가치가 있는 부분입니다. 이것이 생성된 산출물을 안전하게 배포하게 만드는 핵심입니다.

## 다음 단계 {#section-12}

- **다운스트림 실패 자동 분류.** 오늘 워크플로우는 테스트 브랜치를 열고 사람이 CI를 읽습니다. 실패 로그를 확인해 내부 Slack 메시지에 보고하는 것이 명백한 다음 단계입니다.
- **패턴 확장.** 이것의 대부분은 일반적입니다. 생태계의 다른 파이썬 라이브러리에서도 큰 부분을 재사용할 수 있을 것으로 기대합니다.

## 시사점 {#section-13}

릴리스의 부분 중 예전에는 반나절의 집중된 인간 작업이 필요했던 것들(노트 작성, 발표 초안 작성, 다운스트림 검사 조정)은 모델이 초안을 작성하기에 적합한 부분입니다. 나머지 모든 것은 기계적이며 YAML 파일에 맞습니다. 비결은 'AI에게 맡기기'가 아니라, 모델이 초안을 작성하게 두고, 결정론적 코드가 검증하게 하며, 사람에게 최종 결정을 맡기는 것입니다. 전적으로 오픈 도구와 오픈 가중치로 구성되었기 때문에 비용은 사실상 0에 가까워 누구나 실행할 수 있습니다.

전체 워크플로우 파일은 공개되어 있습니다. 파이썬 라이브러리를 관리한다면 [fork it](https://github.com/huggingface/huggingface_hub/blob/main/.github/workflows/release.yml)를 포크해 적용해 보고, 어떻게 진행되는지 알려 주세요!
