---
layout: post
title: "효율적이고 안정적인 SEO 모듈 만들기: LLM은 필요한 곳에만"
author: sohyun
categories: [SEO, Agent, Workflow]
---

* TOC
{:toc}
<!--toc-->

# 효율적이고 안정적인 SEO 모듈 만들기: LLM은 필요한 곳에만

Hugging Face KREW는 Hugging Face 블로그의 좋은 글을 매일 한국어로 옮기는 번역 워크플로우를 운영하고 있습니다. 이 워크플로우에는 번역만 있는 것이 아니라, 옮긴 글이 검색과 생성형 응답(GEO)에서 잘 발견되도록 돕는 **SEO 모듈**이 함께 들어갑니다.

SEO 모듈을 만들면서 계속 던진 질문은 하나였습니다. **"이 판단에 정말 LLM이 필요한가?"** 번역이라는 작업 자체는 언어 이해와 생성이 핵심이라 LLM이 중심에 설 수밖에 없습니다. 하지만 "이 글이 발행해도 안전한가", "제목과 본문이 같은 주제를 말하는가", "이미지에 대체 텍스트가 있는가" 같은 SEO 판정은 성격이 제각각입니다. 어떤 것은 규칙으로 정확히 확정되고, 어떤 것은 의미를 읽어야 합니다.

이 글은 그 경계를 어떻게 나눴는지에 대한 기록입니다. 글을 관통하는 생각은 이 한 문장입니다.

> 단순하게 풀 수 있는 문제는 단순하게 풀 수 있도록, 안정성이 중요할 땐 LLM의 유연함이 장점이 아닌 비결정성이라는 리스크임을 신경 써서 설계하는 건 어떨까?

되는 것은 결정적으로(규칙으로) 처리하고, LLM은 의미 판단이 꼭 필요한 곳에만 두었습니다. 그렇게 한 이유 — 효율성과 안정성 — 을 먼저 이야기하고, 그다음에 실제 모듈 구조를 살펴보겠습니다.

## 전체 그림: 매일 도는 번역 파이프라인 속 SEO

SEO 모듈은 홀로 동작하지 않습니다. 매일 실행되는 번역 파이프라인(`daily-translation.yml`)의 한 단계로 들어갑니다.

```text
RSS 수집 ──► EN→KO 번역(LLM) ──► PR + manifest 생성 ──► Discord 알림
   │
   └─► [ SEO 리뷰 · 품질 리뷰 ] ──► reports 커밋 ──► 아티팩트 업로드
```

몇 가지 짚어둘 점이 있습니다.

- 파이프라인에서 LLM 호출은 제한적입니다. **번역**은 작업 성격상 항상 LLM을 부릅니다. **SEO 모듈**은 CI 기본 설정(`SEO_OPENAI_REQUIRED=1`)에서 의미 판정에 LLM을 호출하지만, 설정을 끄면 결정적 경로로 대체됩니다. 반면 파이프라인에 배선된 **품질 리포트(`simple_quality_report.py`)는 한국어 비율·코드블록 균형·TODO·출처 표기 같은 규칙 검사만 수행해 LLM을 호출하지 않습니다.** 나머지 수집·PR 생성·리포트·커밋도 모두 정해진 코드 경로로 움직입니다.
- SEO 리뷰와 품질 리뷰는 별도의 병렬 작업이 아니라 **한 단계 안에서 순차(동기)로** 실행되고, 둘 다 끝난 뒤 리포트를 **한 번** 커밋합니다.
- Discord 알림은 리뷰 결과가 아니라 **"번역 PR이 생성되었다"**는 알림으로, 리뷰보다 앞서 발송됩니다.
- 각 파트가 공유하는 계약은 `manifest.yaml` 하나입니다. SEO 모듈은 이 manifest(또는 파일 경로)를 입력으로 받아 동작합니다.

즉 SEO 모듈은 자율적으로 판단하고 행동하는 에이전트라기보다, 파이프라인이 정해진 순서로 호출하는 **하나의 리뷰 단계**입니다. 이 위치 설정이 이후 설계의 출발점이 됩니다.

## LLM에게 모든 것을 맡기지 않는 이유: 효율성과 안정성

SEO 모듈은 판단을 통째로 모델에 넘기지 않습니다. 대신 어떤 판단이 규칙으로 확정되고 어떤 판단이 의미를 읽어야 하는지를 먼저 나눴습니다. 그 경계를 어디에 그을지 정한 기준은 효율성과 안정성 두 가지였습니다.

### 효율성 — 결정적 판정은 공짜이고, LLM은 그렇지 않다

LLM 호출은 토큰 비용, 지연 시간, 그리고 비결정성을 함께 데려옵니다. 매일 도는 워크플로우에서 글 한 편마다 판정 항목이 십수 개인데, 그 모두를 모델에게 물어보면 비용과 시간이 선형으로 쌓입니다. 반면 "제목이 있는가", "이미지 파일이 실제로 존재하는가", "내부 링크가 깨지지 않았는가" 같은 판정은 **규칙으로 즉시, 무료로, 정확하게** 끝납니다. 여기에 모델을 부르는 것은 낭비일 뿐 아니라 부정확해질 위험까지 더합니다.

이 감각은 우리만의 것이 아닙니다. Anthropic은 에이전트를 다룬 글에서 이렇게 권합니다.

> "We recommend finding the simplest solution possible, and only increasing complexity when needed."
>
> — Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

가장 단순한 해법에서 시작하고, 필요할 때만 복잡도를 올리라는 것입니다. SEO 판정의 대부분은 규칙으로 충분히 정의되므로, 굳이 LLM이라는 복잡도를 얹지 않았습니다.

OpenAI도 에이전트 구축 가이드에서 비슷한 선을 긋습니다. 에이전트가 필요한 조건(복잡한 판단, 비정형 데이터, 규칙 기반 접근이 무너지는 영역)을 먼저 확인하고,

> "Before committing to building an agent, validate that your use case can meet these criteria clearly. Otherwise, a deterministic solution may suffice."
>
> — OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)

라고 말합니다. **"그렇지 않다면 결정적 해법으로 충분하다."** SEO 게이트의 상당 부분이 바로 이 "결정적 해법으로 충분한" 영역입니다.

정리하면, 결정론 우선 원칙은 "LLM을 쓰지 말자"가 아닙니다. 한정된 LLM 예산을 **규칙으로는 잡을 수 없는 의미 판단에 집중**시키자는 것입니다.

### 안정성 — 유연함이 리스크가 되는 자리

리뷰 게이트의 생명은 재현성입니다. 같은 글을 두 번 검사하면 같은 결과가 나와야 하고, CI에서 실패하면 그 실패가 진짜여야 합니다.

여기서 관점을 한 번 뒤집을 필요가 있습니다. LLM의 가장 큰 장점은 유연함입니다. 그런데 **게이트처럼 안정성이 핵심인 자리에서는, 그 유연함이 곧 비결정성이라는 리스크**가 됩니다. 같은 입력에 다른 출력이 나올 수 있다는 성질은 새로운 문제를 풀 때는 강점이지만, 같은 판정을 매일 반복해야 할 때는 약점입니다. 판정을 통째로 모델에 맡기면 게이트가 흔들리고, 게이트가 흔들리면 신뢰가 무너집니다.

Anthropic이 workflow와 agent를 가르는 기준도 정확히 이 축입니다.

> "Workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale."
>
> — Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

잘 정의된 작업에는 **예측 가능성과 일관성**이, 유연한 판단이 필요한 작업에는 **모델 주도의 결정**이 어울린다는 것입니다. SEO 게이트는 명백히 전자에 속합니다. 그래서 안정성을 위해 세 가지를 지켰습니다.

첫째, **사실 판정은 규칙에 맡깁니다.** `robots: noindex`가 걸렸는지, 내부 링크나 로컬 이미지가 깨졌는지는 의견이 아니라 사실입니다. 이런 게시 안정성 항목은 LLM의 판단 없이 규칙으로 확정하고, 실패 시 곧장 `BLOCKED`로 막습니다.

둘째, **의미 판정은 좁은 범위로 격리하고, 스키마로 제한합니다.** 규칙으로 못 잡는 것 — 이를테면 "제목·설명·본문이 같은 주제를 말하는가" — 만 LLM에게 넘기되, 응답을 JSON 스키마로 강제해 환각과 파싱 실패를 줄입니다. 이는 OpenAI가 권하는 **계층형 가드레일**과 같은 구조입니다.

> "Think of guardrails as a layered defense mechanism. [...] Simple deterministic measures (blocklists, input length limits, regex filters) [...]"
>
> — OpenAI, A Practical Guide to Building Agents

규칙 기반 가드레일과 LLM 기반 가드레일을 겹겹이 쌓되 서로 섞지 않는 것 — 이것이 우리 모듈의 Layer 1·2(규칙)와 Layer 3(LLM)의 분리로 이어집니다.

셋째, **LLM 없이도 로직을 검증할 수 있게 만듭니다.** 시맨틱 판정을 담당하는 저지(judge) 함수는 **주입식(injectable)**입니다. 프로덕션에서는 OpenAI를 주입하고, 테스트에서는 스텁을 주입합니다. 저지를 주입하지 않으면 모듈은 결정적 경로만으로 동작합니다. 덕분에 네트워크·API 키 없이 도는 오프라인 테스트(현재 92개)와 golden 스냅샷 회귀가 성립합니다. 결정적 층이 넓을수록 테스트로 고정할 수 있는 표면도 넓어집니다.

마지막으로 한 가지 원칙을 덧붙였습니다. **"조용한 통과"를 허용하지 않는다.** CI에서 OpenAI는 필수 경로로 설정되어 있어, 저지가 어떤 이유로든 실행되지 못하면 게이트를 통과시키지 않고 명시적으로 실패시킵니다("OpenAI rubric required but not run"). LLM을 아끼되, LLM이 담당하기로 한 판정을 슬그머니 건너뛰지는 않도록 한 것입니다.

## SEO 모듈의 구조: Module 1(게이트)과 Module 2(메타데이터)

이제 구조를 봅시다. 엔트리는 `seo_eval.py`이고, **본문(body)만** 평가합니다. frontmatter(제목·설명 등 메타데이터)는 게이트에서 제외합니다. 메타데이터는 게이트를 통과한 뒤 Module 2가 *생성*하는 대상이라, 게이트에 넣으면 "메타가 없어서 실패 → 실패해서 메타를 못 만듦"이라는 교착이 생기기 때문입니다.

SEO 모듈은 두 개의 모듈로 구성됩니다.

- **Module 1 — Eval(게이트):** Layer 1·2·3으로 이루어진 본문 평가. 상태를 판정합니다.
- **Module 2 — 메타데이터 작성:** Module 1의 게이트를 통과(PASS)해야 동작합니다.

### Module 1 — 게이트 (Layer 1·2·3)

Module 1 안의 세 층은 다음과 같습니다.

| 층 | 역할 | LLM |
|---|---|---|
| **Layer 1 — 하드 블로커** | 게시 자체를 막아야 하는 조건. 하나라도 실패하면 즉시 `BLOCKED` | ✗ 규칙 |
| **Layer 2 — 결정적 체크 + 증거** | 구조·키워드·이미지 규칙 체크 + pass/fail을 정하지 않는 관찰 신호(evidence signals) | ✗ 규칙 |
| **Layer 3 — 시맨틱 저지** | 규칙으로 못 잡는 의미 정합만. JSON 스키마 출력, 주입식 | ◆ LLM |

#### Layer 1 — 하드 블로커 (규칙)

품질과 무관하게 게시를 막아야 하는 조건만 담습니다. `--target-root`로 실제 파일과 링크를 해석합니다.

- `body_not_empty` — 본문이 비어 있지 않은가
- `robots_indexable` — frontmatter에 `robots: noindex`가 걸려 있지 않은가
- `internal_links_resolve` — 내부 링크가 저장소 안에서 실제로 해석되는가(깨진 링크 아님)
- `local_images_resolve` — 참조한 로컬 이미지 파일이 실제로 존재하는가

이들은 전부 사실 판정이라 규칙으로 확정합니다.

#### Layer 2 — 결정적 체크 + 증거 신호 (규칙)

구조·키워드·이미지에 대한 규칙 체크입니다. 각 항목의 심각도(severity)는 코드가 아니라 정책 파일(`default-policy.yml`)이 매핑하므로, 팀 정책에 따라 조정할 수 있습니다. 현재 **게이트를 막는 required 항목**은 다음과 같습니다.

- `heading_hierarchy` — heading 레벨을 건너뛰지 않는가(`##` 다음 바로 `####` 금지)
- `primary_keyword` — 주요 키워드가 H1과 도입부에 등장하는가(키워드 미지정 시 건너뜀)
- `alt_text_coverage` — 모든 이미지에 대체 텍스트가 있는가
- `descriptive_alt_text` — 대체 텍스트가 파일명형(`image1.png`)이 아니라 서술적인가
- `image_files_exist` — 참조 이미지 파일이 실제로 존재하는가

`opening_summary`, `h1_count`, `citations`는 `review`(리포트에 남기되 게이트는 막지 않음)로, 내부 링크·secondary 키워드·webp·lazy 로딩 등은 권고(recommended/optional)로 둡니다. Jekyll 레이아웃이 frontmatter의 제목을 페이지 H1로 렌더링하기 때문에, 본문 구조에 대한 판단은 하드 게이트에서 빼고 리포트로만 남기는 쪽을 택했습니다.

Layer 2에는 규칙 체크와 별개로 **증거 신호(evidence signals)**도 있습니다. 이것은 그 자체로 pass/fail을 정하지 않는 **순수 관찰값**입니다. 예를 들면 이런 것들입니다.

```text
frontmatter: title_chars=42, description_present=true, canonical_present=false
opening:     first_real_paragraph_chars=180
headings:    markdown_h1_count=0, rendered_effective_h1=["RTEB: ..."]
images:      total=5, empty_alt_count=1, filename_like_alt_count=0
links:       internal=3, external=8, citation_signal_count=4
```

이 신호들은 판단을 내리지 않고, **Layer 3의 저지와 사람 리뷰어에게 판단 근거**를 넘깁니다. "제목은 있는데 도입부와 주제가 다르다" 같은 판단을 LLM이 내릴 수 있게 하는 입력이 바로 이 신호들입니다. 규칙이 근거를 만들고, 의미는 다음 층이 봅니다.

#### Layer 3 — 시맨틱 저지 (LLM)

규칙으로는 잡을 수 없는 **의미 정합**만 여기서 판정합니다. 좁은 seam 두 개뿐입니다.

- `semantic_metadata` — 제목·설명·렌더된 H1·도입부가 서로 다른 주제를 약속하고 있지 않은가(게이트를 막는 `required`)
- `alt_semantics` — 비어 있지는 않지만 무의미한 대체 텍스트(`그림`, `image`)를 쓰고 있지 않은가(리뷰용, 게이트를 막지는 않음)

이 저지는 `gpt-5-nano`로 호출하며, 앞서 말했듯 주입식이라 미주입 시에는 결정적 경로만으로 게이트가 판정됩니다.

#### 판정 결과: PASS · NEEDS_CHANGES · FAIL · BLOCKED

Module 1은 이진 통과/실패가 아니라 네 가지 상태를 냅니다. 그리고 **판정 단위는 레이어가 아니라 개별 `required` 항목**입니다.

- **PASS** — 통과. Module 2(메타데이터)로 진행
- **NEEDS_CHANGES** — required 항목 **1개** 실패. 좁은 피드백만 반환
- **FAIL** — required 항목 **2개 이상** 실패. 본문 개선 피드백 반환
- **BLOCKED** — Layer 1 블로커 실패(예: `noindex`, 깨진 내부 링크)

### Module 2 — 메타데이터 작성

게이트를 통과하면 Module 2가 제목·설명 후보를 만듭니다. 후보는 증거에서 결정적으로 뽑거나 OpenAI로 생성할 수 있습니다. 다만 **canonical·hreflang은 절대 추측하지 않습니다.** 번역본과 원문의 URL 정책이 입력으로 주어지지 않으면 상태를 `PARTIAL`로 남깁니다. 정책 없이 canonical을 잘못 박으면 원문과 번역본이 검색에서 서로를 잠식할 수 있기 때문입니다.

또 하나 중요한 점은, 파이프라인 안에서 Module 2는 **읽기 전용**이라는 것입니다. 후보(`metadata-suggestion.json`)를 만들 뿐, 실제 포스트 frontmatter에 적용(write-back)하는 단계는 자동화에 포함되어 있지 않습니다. 적용은 사람의 승인(`requires_human`)을 거치도록 되어 있습니다. 자동 생성한 메타데이터를 사람 확인 없이 곧장 반영하는 것은 아직 위험하다고 보았기 때문입니다.

## 실행 흐름: LLM 호출 지도와 실패 처리

### LLM 호출 지도 — 10단계 중 3단계

SEO 모듈 내부의 작업 순서를 펼쳐 보면, LLM을 부르는 지점이 어디인지 한눈에 들어옵니다.

| # | 단계 | 종류 |
|---|---|---|
| 1 | 입력 해석(manifest/파일 → 본문·frontmatter 분리) | 규칙 |
| 2 | Layer 1 블로커 체크 | 규칙 |
| 3 | Layer 2 결정적 D-체크 | 규칙 |
| 4 | 증거 신호 수집(`collect_signals`) | 규칙 |
| 5 | 정책 severity 매핑(`apply_policy`) | 규칙 |
| 6 | **Layer 3 `semantic_metadata` 저지** | **◆ LLM** |
| 7 | **Layer 3 `alt_semantics` 저지** | **◆ LLM** |
| 8 | 게이트·상태 분류(`classify_status`) | 규칙 |
| 9 | **(PASS 시) 메타데이터 후보 생성** | **◆ LLM / 결정적 fallback** |
| 10 | 리포트·JSON 출력 | 규칙 |

열 단계 중 LLM 호출은 **6·7·9 세 곳뿐**입니다. 나머지 일곱 단계는 API 키와 네트워크 없이 재현할 수 있습니다. 그리고 이 세 단계조차 CI 설정(`SEO_OPENAI_REQUIRED=1`)이 없으면 결정적 경로로 대체됩니다.

### 게이트가 실패하면

PR 이벤트 기반 리뷰 경로에서는, 실패 리포트를 근거로 본문을 **최소 수정 → 재검증 → 푸시**하는 자동 복구가 최대 3회까지 시도됩니다. 안전하게 고칠 수 없으면 `hf-agent:needs-human` 라벨을 달고 멈춰 사람에게 넘깁니다. 매일 도는 데일리 경로에서는 리포트만 남기고, 본문을 고치면 다음 실행에서 다시 검사합니다.

## 어떻게 여기까지 왔나

이 구조는 처음부터 있던 것이 아닙니다. 단일한 통과/실패 게이트에서 출발해 지금의 3층 구조로 나뉘어 왔습니다.

- **1단계 — 단일 이진 게이트.** 결정적 required 체크만으로 PASS/FAIL을 냈습니다. 오프라인 테스트 하네스와 body-only 게이트라는 기본 계약을 이때 확립했습니다. 이후 리뷰를 거치며, Jekyll이 제목을 H1로 렌더링하는 점 때문에 본문 구조 체크(H1·계층·도입부·인용)를 게이트에서 빼고 리포트로만 남기도록 조정했습니다.
- **2단계 — 측정 기반 마련.** 런타임 로직은 그대로 두고, 평가를 측정할 데이터셋(curated·mutated·실제 글 픽스처)만 확장했습니다. "테스트가 통과한다"와 "정책이 SEO 관점에서 충분하다"를 구분하기 위한 토대입니다.
- **3단계 — 층의 분화(현재).** 게시 안정성 블로커(Layer 1), 결정적 증거(Layer 2), 스키마 기반 LLM 저지(Layer 3)를 분리하고, 상태를 네 가지로 나누었습니다. 정책 설정 파일, 정책 인지 메타데이터, OpenAI 배선이 이때 들어왔고 테스트는 92개로 늘었습니다.

핵심 변화는 두 가지입니다. **결정적 층이 하나에서 세 겹으로 분화**되었고, 규칙으로 못 잡는 의미 판정을 위해 **LLM 호출 지점이 좁고 명확하게 신설**되었습니다.

## 정리

SEO 모듈을 만들며 지킨 원칙은 네 가지로 요약됩니다.

1. **층을 섞지 않는다.** 게시 안정성(하드 블로커)·결정적 증거·의미 판정을 분리한다. 렌더링 방식이나 문서 종류에 따라 뜻이 달라지는 항목은 하드 게이트에서 뺀다.
2. **LLM은 스키마로만, 최소로.** 시맨틱 저지는 좁은 seam과 JSON 스키마로 제한하고 주입식으로 만들어 오프라인에서 결정적으로 재현한다.
3. **메타는 정책 없이 적용하지 않는다.** canonical·hreflang은 추측하지 않고 `PARTIAL`로 남기며, 적용은 사람 승인을 거친다.
4. **조용한 통과를 허용하지 않는다.** LLM이 담당하기로 한 판정이 실행되지 못하면 게이트를 통과시키지 않는다.

멀티 에이전트 시스템에서 모든 모듈이 똑같이 LLM을 많이 써야 하는 것은 아닙니다. 번역처럼 언어 생성이 핵심인 모듈은 LLM 농도가 높고, SEO나 품질처럼 대부분이 사실·규칙 판정인 모듈은 낮은 것이 오히려 자연스럽습니다. SEO 모듈은 자율적으로 판단하는 에이전트가 아니라 **예측 가능하고, 무료로 재현되며, 오프라인에서 검증되는 결정적 workflow 노드**로 설계했습니다. 그리고 그 위에 꼭 필요한 의미 판정에만 LLM을 얹었습니다.

결국 하고 싶은 말은 처음의 그 문장으로 돌아갑니다. **단순하게 풀 수 있는 문제는 단순하게 풀고, 안정성이 중요한 자리에서는 LLM의 유연함이 장점이 아니라 비결정성이라는 리스크가 된다는 점을 설계에 반영하는 것.** 효율성과 안정성은 바로 여기에서 나옵니다.

## 참고 자료

- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
