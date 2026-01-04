---
layout: post
title: "Hugging Face 번역 MCP 서버 총정리" 
author: minju
categories: [Robotics, AI, Community]
image: assets/images/blog/posts/2025-12-28-translation-mcp-project-overview/thumbnail.png
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 문서 번역 프로젝트의 MCP server 소개글입니다._

---

## 들어가기 전

이 글은 Hugging Face 번역 MCP 서버 프로젝트의 글들과 함께 읽으시면 좋습니다.

현서님께서 MCP 서버 전략 및 개발 도구 선정 과정을 소개해주셨어요. 
👉 [MCP 서버 설계 전략 및 개발 도구 선정](https://hugging-face-krew.github.io/hf_translation_hub_mcp_design_and_tooling/)

그리고 각 MCP 서버의 사용법도 소개해주셨습니다.
👉 [HuggingFace 번역 MCP 서버 사용법](https://hugging-face-krew.github.io/hf-translation-hub-mcp-server-usage-guide/)

---

## MCP server 배포 완료 🎉

프로젝트를 기획할 당시에 설계했던 3개의 MCP server와 현서님께서 추가해주신 PR 리뷰 MCP server까지 총 **4개의 MCP server**들의 배포가 완료되었습니다!

허깅페이스 스페이스에 각각 배포를 완료하였고, Hugging Face Chat, Claude Desktop, Gemini CLI에 MCP server를 연결하고 테스트를 진행하였어요.

이 글에서는 각 MCP server 안에 등록된 tool들과 각 tool들에 대한 설명을 소개드리려고 합니다.

번역할 문서 검색, 문서 번역, 번역 PR 등록, 번역 PR 리뷰의 총 4가지 MCP server를 차례대로 살펴보겠습니다.

![mcp-list](../assets/images/blog/posts/2025-12-28-translation-mcp-project-overview/mcp_list.png)

---

## Hugging Face MCP 서버와 도구 소개

우리 팀은 번역 문서 관리와 PR 자동화, 리뷰를 위해 총 4개의 Hugging Face MCP 서버를 운영하고 있습니다.

각 서버는 특화된 도구들을 제공하여 번역 작업의 검색, 생성, 관리, 리뷰를 자동화할 수 있습니다.

아래에서 각 서버와 그 안의 도구들을 자세히 소개합니다.

---

## 1️⃣ hf-translation-docs-explorer

### 개요
- **주요 역할:** 번역 프로젝트 및 문서 탐색, 상태 확인
- **URL:** `https://hyeonseo-hf-translation-docs-explorer.hf.space/gradio_api/mcp/`

### 설명
이 서버는 번역해야 할 문서 파일을 찾고, 누락되거나 오래된 파일을 확인하는 데 특화되어 있습니다. 사용자는 프로젝트별로 번역 진행 상태를 쉽게 파악할 수 있습니다.

예를 들어, `transformers` 프로젝트의 한국어 번역에서 어떤 파일이 아직 번역되지 않았는지, 어떤 파일이 원본 대비 오래되어 업데이트가 필요한지 한눈에 확인할 수 있어요.

### 🛠️ 도구 목록

| Tool 이름 | 설명 | 입력 파라미터 |
|-----------|------|---------------|
| `hf_translation_docs_explorer_list_projects` | 번역 프로젝트 목록 조회 | 없음 |
| `hf_translation_docs_explorer_search_files` | 특정 프로젝트/언어의 파일 검색 | `project` (`smolagents`, `transformers`)<br/>`lang` (`ko`, `ja`, `zh`, `fr`, `de`)<br/>`limit`<br/>`include_status_report` |
| `hf_translation_docs_explorer_list_missing_files` | 누락된 번역 파일 목록 조회 | `project`<br/>`lang`<br/>`limit` |
| `hf_translation_docs_explorer_list_outdated_files` | 오래된 번역 파일 목록 조회 | `project`<br/>`lang`<br/>`limit` |

**지원하는 프로젝트:** `smolagents`, `transformers`  
**지원하는 언어:** `ko` (한국어), `ja` (일본어), `zh` (중국어), `fr` (프랑스어), `de` (독일어)

---

## 2️⃣ hf-translation-docs

### 개요
- **주요 역할:** 번역 문서 처리 전체 파이프라인
- **URL:** `https://jmj-minju-hf-translation-docs.hf.space/gradio_api/mcp/`

### 설명
문서 번역 과정 전체를 다루는 핵심 서버입니다. 번역 파일 검색부터 내용 조회, 번역 프롬프트 생성, 품질 검증, 결과 저장까지 모든 단계를 지원합니다.

이 서버를 통해 번역 워크플로우를 일관성 있게 관리할 수 있습니다. 특히 번역 프롬프트 생성 기능은 번역자가 일관된 스타일과 품질로 번역할 수 있도록 도와주며, 검증 기능은 번역 결과가 원본 포맷을 유지하고 있는지 자동으로 확인해줍니다.

### 🛠️ 도구 목록

| Tool 이름 | 설명 | 입력 파라미터 |
|-----------|------|---------------|
| `hf_translation_docs_get_project_config` | 프로젝트 설정 정보 조회 | `project` |
| `hf_translation_docs_search_translation_files` | 번역이 필요한 파일 검색 | `project`<br/>`target_language`<br/>`max_files` |
| `hf_translation_docs_get_file_content` | 특정 파일 내용 조회 | `project`<br/>`file_path`<br/>`include_metadata` |
| `hf_translation_docs_generate_translation_prompt` | 번역 프롬프트 생성 | `target_language`<br/>`content`<br/>`additional_instruction`<br/>`project`<br/>`file_path` |
| `hf_translation_docs_validate_translation` | 번역 품질/포맷 검증 | `original_content`<br/>`translated_content`<br/>`target_language`<br/>`file_path` |
| `hf_translation_docs_save_translation_result` | 번역 결과 저장 | `project`<br/>`original_file_path`<br/>`translated_content`<br/>`target_language` |

이 서버는 번역 작업의 중심축 역할을 하며, 번역의 시작부터 끝까지 모든 과정을 체계적으로 관리할 수 있게 해줍니다.

---

## 3️⃣ hf-translation-pr-generator

### 개요
- **주요 역할:** 번역 PR(Pull Request) 자동 생성
- **URL:** `https://jmj-minju-hf-translation-pr-generator.hf.space/gradio_api/mcp/`

### 설명
번역 작업을 PR 형태로 GitHub에 자동 생성하고 관리하는 서버입니다. 

번역이 완료되면 수동으로 GitHub에 접속해서 PR을 생성하는 것이 아니라, 이 서버를 통해 자동으로 PR을 생성할 수 있어요. PR 생성 전 설정 검증, 참조 PR 검색, 번역 분석, PR 초안 생성, 최종 PR 생성까지 일련의 과정을 모두 지원합니다.

특히 참조 PR 검색 기능은 이전에 승인된 번역 PR들을 참고하여 일관성 있는 번역 스타일을 유지할 수 있도록 도와줍니다.

### 🛠️ 도구 목록

| Tool 이름 | 설명 | 입력 파라미터 |
|-----------|------|---------------|
| `hf_translation_pr_generator_validate_pr_config_ui` | PR 생성 설정 검증 (UI) | `github_token`<br/>`owner`<br/>`repo_name`<br/>`project` |
| `hf_translation_pr_generator_search_reference_pr` | 참조용 PR 검색 | `target_language`<br/>`context` |
| `hf_translation_pr_generator_analyze_translation` | 번역 내용 분석 및 메타데이터 생성 | `filepath`<br/>`translated_content`<br/>`target_language`<br/>`project` |
| `hf_translation_pr_generator_generate_pr_draft` | PR 초안/메타데이터 생성 | `filepath`<br/>`translated_content`<br/>`target_language`<br/>`reference_pr_url`<br/>`project` |
| `hf_translation_pr_generator_create_github_pr` | GitHub PR 생성 | `github_token`<br/>`owner`<br/>`repo_name`<br/>`filepath`<br/>`translated_content`<br/>`target_language`<br/>`reference_pr_url`<br/>`project` |

이 서버를 사용하면 번역 작업이 끝난 후 PR 생성에 들어가는 시간과 노력을 크게 줄일 수 있습니다.

---

## 4️⃣ hf-translation-reviewer

### 개요
- **주요 역할:** 번역 리뷰 및 피드백 자동화
- **URL:** `https://hyeonseo-hf-translation-reviewer.hf.space/gradio_api/mcp/`

### 설명
번역 PR을 검토하고 피드백을 생성하는 자동화 서버입니다.

리뷰 준비, 응답/피드백 생성, 제출, 엔드투엔드 리뷰 등 모든 과정을 자동화할 수 있어 리뷰 업무의 부담을 크게 줄여줍니다. 특히 `e2e_proxy` 도구를 사용하면 리뷰 준비부터 피드백 생성, 제출까지 전 과정을 한 번에 실행할 수 있어요.

리뷰어는 이 서버가 생성한 피드백을 바탕으로 최종 검토만 하면 되기 때문에, 리뷰 시간을 대폭 단축할 수 있습니다.

### 🛠️ 도구 목록

| Tool 이름 | 설명 | 입력 파라미터 |
|-----------|------|---------------|
| `hf_translation_reviewer__prepare_proxy` | 리뷰 준비 (프록시) | `pr_url_`<br/>`original_path_`<br/>`translated_path_` |
| `hf_translation_reviewer__review_emit_proxy` | 검토 응답/피드백 생성 | `pr_url_`<br/>`translated_path_`<br/>`translated_text_`<br/>`raw_response_` |
| `hf_translation_reviewer__submit_proxy` | 리뷰 제출 (프록시) | `pr_url_`<br/>`translated_path_`<br/>`payload_json_` |
| `hf_translation_reviewer__e2e_proxy` | 엔드투엔드 리뷰 및 제출 | `pr_url_`<br/>`original_path_`<br/>`translated_path_`<br/>`save_review_`<br/>`save_path_`<br/>`submit_flag_`<br/>`e2e_raw_response_` |

---

## 🔹 전체 워크플로우 요약

4개의 MCP 서버가 어떻게 연결되어 작동하는지 살펴볼까요?

### 단계별 프로세스

1. **hf-translation-docs-explorer** → 번역할 문서 탐색 및 상태 확인
   - 프로젝트에서 번역이 필요한 파일들을 찾습니다
   - 누락되거나 오래된 파일들을 확인합니다

2. **hf-translation-docs** → 파일 조회, 번역 생성, 검증, 저장
   - 파일 내용을 가져옵니다
   - 번역 프롬프트를 생성합니다
   - 번역 결과를 검증하고 저장합니다

3. **hf-translation-pr-generator** → GitHub PR 생성 및 관리
   - 번역 내용을 분석합니다
   - PR 초안을 생성합니다
   - GitHub에 PR을 자동으로 생성합니다

4. **hf-translation-reviewer** → PR 리뷰 및 제출 자동화
   - PR을 검토합니다
   - 자동으로 피드백을 생성합니다
   - 리뷰를 제출합니다

### 시퀀스 다이어그램

![mcp-list](../assets/images/blog/posts/2025-12-28-translation-mcp-project-overview/image.png)


이 4개의 MCP 서버를 연계하면 **번역 문서 관리 → PR 생성 → 리뷰 제출**까지 완전한 자동화 워크플로우를 구축할 수 있습니다.

![mcp-list](../assets/images/blog/posts/2025-12-28-translation-mcp-project-overview/image2.png)

예를 들어, Hugging Face chat에 mcp server를 연결한 모습입니다.

---

## 💭 고민거리

프로젝트를 진행하면서 가장 큰 고민은 역시 **LLM 모델 사용 비용**입니다.

Tool 사용이 가능하고 tool 선택을 잘 하는 모델은 Claude나 Gemini 같은 상용 모델들이에요. 이런 모델들을 사용하면 MCP 서버와의 연동이 매우 원활하고 정확합니다.

하지만 프로젝트를 장기적으로 운영하려면 비용 문제를 고려하지 않을 수 없어요. 현재는 각 모델들의 무료 사용량 범위 내에서 테스트를 진행하고 있지만, 앞으로는 오픈소스 모델 중에서도 좋은 성능을 내는 모델을 찾아 활용하는 노력이 필요할 것 같습니다.

---

## 마치며

Hugging Face 번역 MCP 서버 프로젝트는 번역 작업의 전 과정을 자동화하여 효율성을 크게 높일 수 있는 시스템입니다.

4개의 서버가 각자의 역할을 충실히 수행하면서도 유기적으로 연결되어 하나의 완전한 워크플로우를 만들어내는 것이 이 프로젝트의 핵심이에요.

앞으로도 계속해서 개선하고 발전시켜 나갈 예정이니, 많은 관심 부탁드립니다! 🚀

---

### 📚 관련 자료

- [MCP 서버 설계 전략 및 개발 도구 선정](https://hugging-face-krew.github.io/hf_translation_hub_mcp_design_and_tooling/)
- [HuggingFace 번역 MCP 서버 사용법](https://hugging-face-krew.github.io/hf-translation-hub-mcp-server-usage-guide/)