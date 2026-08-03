---
layout: post
title: "에이전트 기반 리소스 탐색: 에이전트에게 도구·스킬·다른 에이전트 검색을 맡기다"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-06-17-agentic-resource-discovery-launch/thumbnail.png
authors:
  - user: burtenshaw
  - user: evalstate
slug: "agentic-resource-discovery-launch"
source_url: "https://huggingface.co/blog/agentic-resource-discovery-launch"
source_published_date: "2026-06-17"
source_published_at: "2026-06-17T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Agentic Resource Discovery: Let agents search](https://huggingface.co/blog/agentic-resource-discovery-launch)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/agentic-resource-discovery-launch -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 에이전트 기반 리소스 탐색: 에이전트에게 도구·스킬·다른 에이전트 검색을 맡기다

오늘날 에이전트를 사용해 개발한다면, 아마도 세 가지 프로토콜을 알고 있을 것입니다. MCP는 에이전트가 도구를 호출하는 표준 방법을 제공합니다. 스킬은 에이전트가 지시를 소비하는 방법을 제공합니다. A2A는 에이전트가 다른 에이전트를 호출하는 방법을 제공합니다. 이 세 가지 모두 사용자가 필요한 도구, 지시, 또는 에이전트를 이미 알고 있다고 가정합니다. 사용자는 여전히 이러한 기능을 발견하고, 통합하고, 유지 관리해야 합니다.

에이전트 주도 자원 발견(ARD) 사양은 그것들 앞에 자리한 발견 계층입니다. 이는 마이크로소프트, 구글, GoDaddy, 허깅페이스 등 다양한 기여자들이 개발한 초안형 공개 사양이며, 업계 전반에 걸친 폭넓은 참여가 있습니다. 이 사양은 에이전트와 도구가 연합 레지스트리 전체에서 어떻게 카탈로그화되고 색인화되며 검색되는지 정의합니다. 이를 통해 에이전트는 런타임에 필요한 기능을 미리 설치할 필요 없이 찾을 수 있습니다. 이는 제품이나 마켓플레이스가 아닙니다. 어떤 기업이 독립적으로 구현할 수 있는 공유 표준이며, 어떤 에이전트나 도구도 참여할 수 있습니다.

이번 글에서는 사양과 허깅페이스가 이를 어떻게 구현했는지, 그리고 ARD에서 시작해 구축하는 방법에 대해 살펴보겠습니다.

## 발견 문제 {#section-1}

에이전트 기능의 현재 모델은 설치를 먼저 하고, 사용은 나중에 하는 방식입니다. 개발자가 MCP 서버 URL을 구성 파일에 하드코딩합니다. 사용자는 플러그인을 통해 자신의 AI 앱에 서비스를 연결하고 이를 재사용합니다. 이는 에이전트가 매일 사용하는 몇 가지 도구에는 작동하지만, 수천 개의 임의 표면에는 확장되지 않습니다.

대체 방법은 사용 가능한 모든 도구 설명을 LLM의 컨텍스트 창에 모두 넣고 모델이 선택하게 하는 것입니다. 이는 컨텍스트 예산의 제약을 받습니다. 여기에도 검색 기반 전략이 있지만 설명이 충분히 구분하기에 얇아 모호성을 해결하기 어렵습니다.

ARD는 선택을 LLM 밖으로 이동합니다. 레지스트리는 게시자 식별, 대표 쿼리, 규정 준수 증명, 태그 등과 같은 풍부한 신호로 기능을 색인화합니다. REST 엔드포인트를 노출합니다. 클라이언트는 자연어로 검색하고, 모델은 검색 결과를 실행합니다. 이 전환은 수동으로 설치된 정적 카탈로그에서 의도 기반 검색으로의 이동으로, 에이전트가 동적으로 올바른 기능을 찾아 MCP 도구, A2A 에이전트, 그리고 사전에 구성 없이 점점 성장하는 다른 서비스 생태계에 도달할 수 있게 합니다.

## 사양이 정의하는 두 가지 {#section-2}

- `ai-catalog.json`: 게시자가 잘 알려진 URL에서 그들의 역량을 호스팅할 수 있도록 하는 정적 매니페스트 형식.
- `POST /search`에 있는 동적 레지스트리 API는 실시간으로 순위가 매겨진 검색 결과를 제공합니다.

## ARD 허깅페이스 허브에서 {#section-3}

허깅페이스 [Discover Tool](https://github.com/huggingface/hf-discover)는 ARD의 참조 구현체입니다. 허깅페이스에서 및 다른 ARD 발견 서비스 전반에 걸쳐 수천 개의 스킬, ML 애플리케이션, MCP 서버에 대한 검색 접근을 제공합니다.

허깅페이스 허브의 Spaces에 대한 기존 시맨틱 검색과 우리의 에이전트 스킬을 결합하고, 그 결과를 ARD 카탈로그 항목으로 제공합니다. 허깅페이스 허브는 이미 Gradio 앱을 실행하는 Spaces, MCP 서버, 데모의 카탈로그를 호스팅합니다. 그 시맨틱 검색은 에이전트 지향 메타데이터로 Spaces를 순위 매겨 반환하는 `agents=true` 플래그를 지원하고, Discover가 그 검색을 ARD 사양으로 변환합니다.

어댑터는 두 가지 필터를 적용합니다. 첫째, 응답은 런타임 단계가 `RUNNING`인 Spaces만 포함합니다. 둘째, 응답 미디어 유형은 요청에 의해 결정됩니다. 세 가지 미디어 유형이 지원됩니다:

- `application/ai-skill`: 기본값. Space의 `agents.md`를 감싸는 생성된 `SKILL.md`입니다.
- `application/mcp-server+json`: `mcp-server` 태그가 붙은 Spaces의 MCP 서버 카탈로그 항목.
- `application/vnd.huggingface.space+json`: 자체적으로 Space 데이터를 처리하려는 클라이언트를 위한 원시 Space 메타데이터.

스킬 타입은 추가 변환을 수반합니다. 많은 Spaces가 에이전트가 그들과 상호 작용하는 방법을 설명하는 `agents.md` 파일을 제공합니다. Discover가 그 파일을 읽고 스킬 소비자가 기대하는 프런트매터: `name`, `description`, 그리고 Space ID, Hub URL, 앱 URL, 그리고 원래 `agents.md` URL를 포함하는 소스 메타데이터로 래핑합니다. 그 결과는 일반 스킬 흐름을 통해 설치하거나 로드할 수 있는 스킬이 됩니다.

MCP 태그가 붙은 Spaces의 경우, 어댑터는 Space의 Gradio MCP 엔드포인트를 HTTP 전송으로 가리키는 카탈로그 항목을 생성합니다. Hub가 런타임 도메인을 제공하는 경우 그 URL은 해당 도메인을 사용하고, 그렇지 않으면 표준 `.hf.space` 슬러그 규칙을 사용합니다.

## 사용하기 {#section-4}

`discover`는 [Hugging Face CLI](https://github.com/huggingface/huggingface_hub)(`hf`)에 내장되어 있습니다. 시작하고 귀하나 귀하의 에이전트에 접근 권한을 부여하려면:

```bash
# Install the Hugging Face CLI tool:
uv tool install huggingface_hub

# Search for resources to train a model
hf discover search "Fine tune a language model"

# Find MCP Servers to generate an image
hf discover search "Generate an image" --json --kind mcp

# Search other registries
hf discover search "Purchase aeroplane tickets" --registry-url <catalog-url>
```


### REST API 및 MCP 도구

REST API 또는 MCP 서버를 사용하여 카탈로그를 직접 검색할 수도 있습니다.

허깅페이스 카탈로그는 잘 알려진 URL에서 공개됩니다:
```
https://huggingface.co/.well-known/ai-catalog.json
```


직접 검색을 호출하려면:
```
POST https://huggingface-hf-discover.hf.space/search
```


```bash
curl -s https://huggingface-hf-discover.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "text": "fine tune a sentence transformer",
      "filter": {
        "type": ["application/ai-skill"]
      }
    },
    "pageSize": 5
  }'
```


MCP 서버 검색

```bash
curl -s https://huggingface-hf-discover.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "text": "transcribe some audio",
      "filter": {
        "type": ["application/mcp-server-card+json"]
      }
    },
    "pageSize": 5
  }'
```


또는 어떤 MCP 클라이언트든 MCP 엔드포인트를 통해 카탈로그를 검색하도록 https://huggingface-hf-discover.hf.space/mcp를 사용해 검색할 수 있습니다.

## 이 사양에 대한 의미 {#section-5}

ARD는 발견과 실행을 분리합니다. 정적 매니페스트 형식은 미디어 타입에 의해 구동되므로 어떤 아티팩트 프로토콜이든 같은 래핑으로 동작할 수 있으며, 사양 수준의 변경이 필요하지 않습니다. 레지스트리 API는 일반 HTTP REST이며, 어떤 클라이언트도 이를 기반으로 연합할 수 있습니다. Discover는 생태계 전반에 걸친 이 사양의 여러 참조 구현 중 하나이며, 프로토콜에 연합이 내장되어 있기 때문에 한 서비스의 검색이 다른 서비스가 호스팅하는 기능을 노출할 수 있습니다.

Discover 도구는 그 설계의 작동 테스트입니다. 새로운 아티팩트 형식을 발명하지 않습니다. 기존 검색 백엔드인 허깅페이스 허브를 사양의 래핑으로 감싸고, 클라이언트가 요청한 것에 따라 동일한 Spaces를 스킬이나 MCP 서버로 노출하게 합니다.

다음 단계는 사양의 연합 모드(`auto`, `referrals`, `none`)와 사용자 및 조직 프로필에 대한 정적 `ai-catalog.json` 매니페스트를 허깅페이스 허브 측에서 지원하는 것의 더 긴밀한 통합입니다. 일단 이 기능이 구현되면, 모든 Space 게시자는 표준의 잘 알려진 URI 메커니즘을 통해 자신의 기능을 광고할 수 있게 될 것입니다.

## 더 알아보기 {#section-6}

- 에이전트 기반 자원 발견 사양: [https://agenticresourcediscovery.org/](https://agenticresourcediscovery.org/)
- 허깅페이스 Discover 도구: [https://github.com/huggingface/hf-discover](https://github.com/huggingface/hf-discover)  
- 허깅페이스 CLI: [https://github.com/huggingface/huggingface\_hub](https://github.com/huggingface/huggingface_hub)  
- 허깅페이스 허브의 에이전트 스킬: [https://huggingface.co/docs/hub/agents-skills](https://huggingface.co/docs/hub/agents-skills)  
- 허깅페이스 Spaces: [https://huggingface.co/spaces](https://huggingface.co/spaces)
