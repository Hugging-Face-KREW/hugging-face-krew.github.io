---
layout: post
title: "MCP 서버 설계 전략 및 개발 도구 선정"
author: hyeonseo
categories: [MCP]
image: assets/images/blog/posts/2025-11-17-hf_translation_hub_mcp_design_and_tooling/thumbnail.png
---
* TOC
{:toc}
<!--toc-->
> Hugging Face 번역 MCP 서버 설계 전략 및 개발 도구는 다음과 같은 **프로젝트 목표와 운영 원칙**을 중심으로 결정되었습니다.
>
> **(1) 문서 번역 자동화 및 범용성 확보**
>
> 번역 MCP 서버의 특정 문서에만 국한되지 않고 다른 오픈소스 문서에도 재사용 가능한 번역 도구를 만드는 것을 목표로 합니다. 이를 위해 기존의 단일 에이전트 기반 방식 대신 **각 기능(문서 검색, 번역, PR 등록, 리뷰)을 모듈화한 MCP 서버 아키텍처**를 도입했습니다.
>
> **(2) 필요한 기능만 구현하는 단순성 원칙 준수**
>
> 번역 문서 검색·번역·PR 생성·리뷰 자동화는 독립된 단일 작업이며, 복잡한 상호작용이나 세션 유지가 필요 없습니다. 이에 따라 MCP 서버는 **불필요한 기능을 배제하고 최소한의 통신 및 상태 관리 방식만 적용하는 단순 설계**를 채택했습니다.
>
> **(3) 빠른 배포와 반복 개발이 가능한 방식 선택**
>
> HuggingFace KREW·OSSCA 활동을 통해 매년 문서 번역이 이루지고 있기 때문에, 실제 번역 참여자들로부터 즉각적인 피드백을 받을 수 있습니다. 따라서 초기에는 **개발 속도가 빠른 Gradio 및 FastMCP 기반으로 프로토타입을 구성**하고, 추후 기능 확장 시 Python SDK로 전환하는 전략을 적용했습니다.
>
> 위와 같은 원칙에 따라 Hugging Face 블로그의 **[Building the Hugging Face MCP Server](https://huggingface.co/blog/building-hf-mcp)**(한국어 번역: **[Hugging Face 서버 구축기](https://hugging-face-krew.github.io/building-hf-mcp-ko/)**) 내용을 기반으로 전송 방식, 통신 패턴, 상태 관리, 개발 도구를 비교·검토하고 MCP 서버 설계 전략을 수립했습니다.
>
> 이 글은 HuggingFace 번역 MCP 서버를 구축하며 정립한 설계 전략과 그 과정에서 도출된 의사결정 기준을 이해하고자 하는 분들께 추천합니다.

# 1. MCP 서버 의사결정 흐름에 따른 설계 전략

## 1-1. 전송 방식 결정 (Transport Selection)

- **STDIO (Standard Input/Output)**
    - 로컬 환경에서 가장 단순하고 안정적이며 설정과 디버깅이 빠른 개발용 전송 방식

- **HTTP with SSE (Server-Sent Events)**
    - 과거 표준이었지만 단방향 스트림 특성·연결 유지 부담으로 현재는 비권장되는 방식

- **Streamable HTTP (최신 표준, 권장)**
    - 단일 엔드포인트에서 요청/응답/스트리밍을 통합 처리하고, 서버리스·OAuth 인증까지 지원하는 MCP 공식 원격 표준 방식

> 💡 **번역 서버에 대한 전송 방식 선택** : **Streamable HTTP** 전송 방식 사용
- 장기적으로는 MCP 최신 표준과의 정합성을 위해 Streamable HTTP가 가장 적합함
- 초기 개발 단계에서는 빠른 반복 개발을 위해 STDIO로 시작 후 안정화되면 Streamable HTTP로 전환하는 방향으로 접근 가능


## 1-2. 통신 패턴 결정 (Communication Pattern Selection)

- **Direct Response (직접 응답)**
    - 클라이언트 요청에 대해 즉시 JSON 결과를 반환하는 가장 기본적인 패턴
    - 구현이 간단하고 서버 부하가 낮으며, 번역 도구, 검색 엔진, 데이터 조회 서비스 등에 적합함

- **Request Scoped Streams (요청 단위 스트리밍)**
    - 단일 요청 범위 내에서 진행 상황이나 부분 결과를 실시간으로 전달함
    - LLM 출력, 대용량 문서 처리, 장시간 연산이 필요한 경우에 유용함

- **Server Push Streams (서버 푸시 스트림)**
    - 장기 연결을 유지하며 서버가 클라이언트에 능동적으로 알림을 전송함
    - 실시간 모니터링, 알림 시스템, 이벤트 기반 아키텍처에 적합하지만 연결 관리 복잡도가 높음


> 💡 **번역 서버에 대한 통신 패턴 선택** : **Direct Response** 기반 구현
- 요청 처리 시간이 짧고 상태가 독립적이므로 Direct Response가 가장 적합함
- 문서 검색은 GitHub 저장소에서 텍스트를 즉시 반환하고, 번역 서버 역시 LLM API 응답을 즉시 반환하는 구조라 스트리밍 패턴의 필요성이 낮음
- 장기 연결·진행률 스트리밍이 요구되지 않으므로 Request Scoped Streams 및 Server Push Streams는 현재 단계에서 불필요함


## 1-3. 상태 관리 결정 (State Management Strategy)

- **Stateless (무상태)**
    - 각 요청을 독립적으로 처리하며 서버 간 세션 정보를 공유하지 않음
    - 확장성이 뛰어나고 로드밸런싱이 용이하며, Redis나 세션 저장소가 불필요함

- **Stateful (상태 유지)**
    - 세션 ID를 통해 클라이언트 세션을 추적하고 상태를 유지함
    - 세션 어피니티 또는 Redis 같은 공유 저장소가 필요하며, 대화형 에이전트에 적합함


> 💡 **번역 서버에 대한 상태 관리 선택** : **Stateless** 기반 구현
- 번역 문서 검색과 번역 호출은 모두 독립적인 단일 트랜잭션으로, 요청 간 컨텍스트를 서버가 유지할 필요가 없음
- 세션 관리 인프라(세션 스토어, 어피니티 설정 등)를 추가할 필요가 없어서, 초기 구현 복잡도와 운영 부담을 모두 줄일 수 있음


## 1-4. 상호작용 기능 결정 (Interactive Features)

- **Sampling (샘플링)**
    - 여러 후보 결과를 제공하고 클라이언트가 선택하는 상호작용 패턴

- **Elicitation (추가 정보 요청)**
    - 서버가 클라이언트에게 추가 정보를 요청해 입력을 보완하는 패턴

- **Progress Notification (진행 상황 알림)**
    - 장시간 작업의 진행 상황을 실시간으로 클라이언트에 전달하는 패턴

> 💡 **번역 서버에 대한 상호작용 기능 선택** : 현재 단계에서는 **상호작용 기능 미구현 예정**
- 모두 짧은 요청–응답 흐름을 가지므로 추가 상호작용 기능이 필요하지 않음
- 상호작용은 향후 번역 파이프라인 관련 추가 기능 개발 시 선택적으로 확장하는 것이 적합함


## 2. MCP 서버 개발 도구 비교 및 선정

## 2-1. Gradio

- **특징**
    - `demo.launch(mcp_server=True)`로 MCP 서버 자동 생성
    - 함수 기반 MCP 도구 스키마 자동 구성
    - Hugging Face Spaces 무료 배포 가능

- **장단점**
    - 장점: 초기 설정이 거의 불필요하며 프로토타입 구축 용이
    - 단점: MCP 스펙의 세밀한 제어가 어렵고, 복잡한 아키텍처에 부적합

- **적합한 환경**: PoC 및 데모 단계, UX가 필요한 MCP 서비스

- **서버 구축 가이드** : [Building an MCP Server with Gradio](https://www.gradio.app/guides/building-mcp-server-with-gradio)

## 2-2. FastMCP
- **특징**
    - Python 기반의 경량 MCP 서버 프레임워크
    - 데코레이터 기반 MCP 도구 정의 용이
    - 인증·배포 도구·고급 패턴(리소스·툴·프록시) 내장

- **장단점**
    - 장점: Python SDK보다 설정이 간단하고, 인증·배포·테스트 등 프로덕션 기능을 내장하여 Gradio보다 유연함 
    - 단점: Python에만 한정되며, [대규모 환경에서는 권장되지 않음](https://www.jlowin.dev/blog/stop-converting-rest-apis-to-mcp)

- **적합한 환경**: 빠른 개발이 필요한 실험적 MCP 서버

- **서버 구축 가이드** : [FastMCP](https://gofastmcp.com/getting-started/welcome)

## 2-3. Python SDK (공식)

- **특징**
    - MCP 스펙 전 요소(도구/리소스/프롬프트/상태/세션)를 직접 제어
    - 고급 상태 관리·비동기 처리·커스텀 인증 구현 가능

- **장단점**
    - 장점: 제어권 최상위 (모든 MCP 기능 자유롭게 사용 가능) 및 높은 확장성, 커스텀 아키텍처 설계 가능
    - 단점: 초기 구조 설계와 반복적으로 사용되는 표준 코드 및 템플릿 필요, 개발 시간이 김

- **적합한 환경**: 정식 상용 서비스, 대규모 MCP 서버 클러스터

- **서버 구축 가이드** : [Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

## 2-4. TypeScript/Node.js SDK (공식)

- **특징**
    - MCP 서버·클라이언트 모두 TS로 구현 가능
    - 자동 타입 생성 및 ESM 기반 비동기 프로그래밍

- **장단점**
    - 장점: 서버리스 환경(Cloudflare Workers, Vercel)에서 배포가 쉬움, TypeScript의 타입 안전성
    - 단점: Python 생태계보다 MCP 관련 라이브러리와 예제가 적음, 개발 시간이 김

- **적합한 환경**: 웹 플랫폼, VSCode MCP 확장 개발, 서버리스 중심 MCP 서버

- **서버 구축 가이드** : [Build an MCP server](https://modelcontextprotocol.io/docs/develop/build-server)

## 2-5. 도구 비교

| 항목 | Gradio | FastMCP | Python SDK | TypeScript SDK |
| --- | --- | --- | --- | --- |
| **개발 속도** | 매우 빠름 | 빠름 | 느림 | 느림 |
| **학습 곡선** | 매우 낮음 | 낮음 | 높음 | 중간 |
| **유연성** | 낮음 | 중간 | 매우 높음 | 매우 높음 |
| **MCP 스펙 지원** | 부분 | 부분 | 전체 | 전체 |
| **확장성** | 제한적 | 중간 | 매우 높음 | 매우 높음 |
| **적합 단계** | 프로토타입 | 초기~중기 개발 | 프로덕션 | 프로덕션 |

- 개발 속도 → Gradio > FastMCP > TS SDK ≈ Python SDK
- 유연성/제어권 → Python SDK ≈ TS SDK > FastMCP > Gradio
- 프로덕션 적합성 → Python SDK ≈ TS SDK > FastMCP > Gradio
- 서버리스 적합성 → TS SDK 우수
- 인증/배포 편의성 → FastMCP 우수

## 2-6. 도구 선택
- **프로토타입(현재) 단계 : FastMCP 또는 Gradio**
    - 빠른 개발 + 기본 기능 검증에 최적
    - Streamable HTTP 기본 지원으로 별도 설정 불필요

**프로덕션 전환 시 (사용자 증가 시) : Python SDK**
    - Python SDK 또는 FastMCP 고급 기능 활용
    - 추가 기능·확장성·커스텀 로직 구현 용이


## 3. 설계 결정 요약

| 항목 | 선택 | 비고 |
| --- | --- | --- |
| **전송 방식** | Streamable HTTP | - |
| **통신 패턴** | Direct Response | - |
| **상태 관리** | Stateless | - |
| **상호 작용** | 미구현 | 향후 확장 가능 |
| **개발 도구** | GradioMCP | - |
| **Gateway** | 미사용 | - |

## 마무리
이번 글에서는 Hugging Face 번역 MCP 서버를 설계하면서 고려한 전송 방식, 통신 패턴, 상태 관리, 상호작용 기능, 개발 도구를 전체적으로 정리했습니다. 

특히 이번 프로젝트에서는 **필요한 기능만 구현하는 단순성의 원칙** 아래, 초기 개발 속도와 구조적 명확성을 최우선으로 삼아 의사결정을 진행했습니다. 그 결과, 프로토타입 단계에서는 Gradio MCP 기반의 빠른 구축이 가능했으며, 향후에는 Python SDK로 확장할 수 있는 기반도 확보하게 되었습니다.

본 글이 MCP 기반 도구를 설계하거나 번역 자동화 파이프라인을 구축하려는 분들께 하나의 참고 자료가 되기를 바랍니다.