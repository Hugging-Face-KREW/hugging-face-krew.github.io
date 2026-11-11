---
layout: post
title: "Nemotron-Personas-Finance-Japan: 일본의 인구 통계와 금융 맥락을 담은 합성 데이터셋"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/604ad59e2fad27aca4515dab/-hyd03JaQoRhMbmlw9-uJ.jpeg
image: assets/images/blog/posts/2026-11-09-nemotron-personas-finance-japan/thumbnail.jpeg
authors:
  - user: nvidia
slug: "nemotron-personas-finance-japan"
source_url: "https://huggingface.co/blog/nvidia/nemotron-personas-finance-japan"
source_published_date: "2026-11-09"
source_published_at: "2026-11-09T15:05:57+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Nemotron-Personas-Finance-Japan: A Synthetic Dataset Capturing Japan's Demographics and Financial Context](https://huggingface.co/blog/nvidia/nemotron-personas-finance-japan)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/nemotron-personas-finance-japan -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Nemotron-Personas-Finance-Japan: 일본의 인구 통계와 금융 맥락을 담은 합성 데이터셋

![nemotron-personas-finance-japan-approach](https://cdn-uploads.huggingface.co/production/uploads/604ad59e2fad27aca4515dab/P649AGmEfc-8C0I5qjqv_.png)

실제 분포를 기반으로 한 일본어 페르소나용 복합 AI 접근 방식

### 금융 AI의 길을 열기 위해 개인정보 보호 장벽을 허무는 오픈 데이터

금융 부문에서는 엄격한 개인정보 보호 규정, 컴플라이언스 요건, 고도의 도메인 전문성 필요성으로 인해 실제 데이터의 활용이 크게 제약됩니다. 이러한 상황에서 금융 도메인의 에이전틱 AI를 개발하려면 실제 데이터 사용에 따른 위험을 피하면서도 인구 통계, 가구 구조, 자산 현황을 반영해 다양하고 현실적인 금융 시나리오를 재현할 수 있는 합성 데이터가 필요합니다.

이러한 과제를 해결하기 위해 [Elith](https://www.elith.ai/)와 NVIDIA는 오픈 소스 페르소나 데이터셋 "[Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan)"의 금융 도메인 확장판인 [Nemotron-Personas-Finance-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Finance-Japan)을 출시했습니다. 저희가 아는 한, 이는 일본의 공식 통계와 가계 금융 및 금융 리터러시 데이터를 결합한 최초의 백만 건 규모 오픈 페르소나 데이터셋입니다. 공식 정부 통계를 기반으로 구축된 이 데이터셋은 합성 데이터 생성(SDG)을 위한 확률 그래프 모델 프레임워크인 [NeMo SDG-PGMs](https://github.com/NVIDIA-NeMo/SDG-PGMs)와 엔터프라이즈 합성 데이터 생성 시스템인 [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)를 사용해 개발되었습니다. 또한 금융 사용 사례에서 데이터셋의 실용성을 높이기 위해 Sompo Japan, Tokio Marine Holdings, Bank of Fukuoka, Mizuho Financial Group, Meiji Yasuda Life Insurance, Financial Data Utilization Association(FDUA)을 비롯한 주요 국내 금융 기관 및 업계 협회가 제공한 인사이트와 피드백을 반영해 설계했습니다.

이 데이터셋은 실제 고객 데이터를 사용하지 않고도 일본의 개발자와 기업이 현지 및 산업별 요구사항에 맞춘 AI를 개발하고 평가할 수 있는 환경을 확장합니다. CC BY 4.0 라이선스로 공개되어 상업적 사용이 가능합니다. 대화형 에이전트 구축부터 고객 시뮬레이션과 모델 평가에 이르기까지 실용적인 금융 AI 개발을 지원합니다. Nemotron-Personas-Finance-Japan을 포함한 [Nemotron-Personas collection](https://huggingface.co/collections/nvidia/nemotron-personas)은 사용자 시뮬레이션 프레임워크인 [NeMo User Sim](https://github.com/NVIDIA-NeMo/UserSim)과 원활하게 통합되도록 설계되었으며, 실제 세계와 일치하는 다양한 분포에서 도출된 시뮬레이션을 즉시 실행할 수 있습니다.

### 데이터셋에는 무엇이 포함되어 있나요?

[![image2](https://cdn-uploads.huggingface.co/production/uploads/604ad59e2fad27aca4515dab/qEMManizZ8vV73X4Ynh1P.png)](https://cdn-uploads.huggingface.co/production/uploads/604ad59e2fad27aca4515dab/qEMManizZ8vV73X4Ynh1P.png)

- 총 900만 개 페르소나(페르소나당 100만 건의 레코드 x 9개), 자연스러운 일본어로 작성

- 레코드당 50개 필드: 페르소나 필드 9개, 페르소나 속성 11개, 공식 인구 통계 및 노동 통계를 기반으로 한 맥락 필드 30개

- 총 약 15억 토큰(페르소나 토큰 약 7억 5천만 개 포함)

- 고유 이름 약 95만 개

- 일본 노동시장을 반영하는 1,600개 이상의 직업 범주

- 인구 통계, 지역, 자산 분포, 성격 특성 축 전반을 포괄

- 다양한 페르소나 유형: 직업, 스포츠, 예술, 여행, 요리, 금융, 금융 행동, 금융 관련 우려

- 자연어 페르소나 속성: 문화적 배경, 기술 및 전문성, 경력 목표와 포부, 취미와 관심사, 가구·부모 역할·금융 배경, 돈에 대한 태도, 재무 계획

- CC BY 4.0 라이선스에 따라 상업적 및 비상업적 용도로 사용 가능

### 구축 방법

#### 데이터 생성 파이프라인

이 데이터셋은 NVIDIA의 합성 데이터 생성 프레임워크인 NeMo Data Designer를 사용해 구축되었습니다. 이 복합 AI 시스템은 복잡한 Jinja 템플릿 작성, Pydantic 검증, 구조화된 출력, 자동 재시도, 여러 생성 백엔드 지원을 제공합니다. 이는 이처럼 대규모 합성 데이터셋을 생성하는 데 필요한 핵심 도구 모음입니다. 또한 다음 모델을 활용했습니다:

- NeMo SDG-PGMs (Apache-2.0): 통계적으로 근거 있는 데이터 생성을 위한 확률 그래프 모델 프레임워크

- gemma-4-31B-it (Apache-2.0): 일본어 텍스트 생성

#### 문화적 맥락 강화

Nemotron-Personas-Finance-Japan은 일본의 공식 인구 통계 및 노동 통계에 부합하도록 설계되었으며, AI 학습에 필수적인 다음과 같은 주요 고려사항을 반영했습니다:

- 교육: 국가 통계가 학위 수준을 대체로 집계하는 경우, 다양한 교육 경로를 모델이 반영할 수 있도록 더욱 세분화된 수준을 도입했습니다.

- 직업: 학습에 사용되는 직업의 범위를 넓히기 위해 사업주와 전문직 등 추가 범주를 포함했습니다.

- 생애 단계: 학생, 은퇴자, 실업 기간처럼 통계에서 두드러지게 나타나는 경우가 드문 상황을 모델링해 더욱 현실적인 페르소나를 표현했습니다.

- 가계 금융: 개인 소득과 지출 성향에 더해 가구 수준의 금융 자산과 실물 자산을 포함해 다양한 경제적 배경을 재현했습니다.

- 디지털 및 금융 리터러시: 금융 리터러시와 함께 인터넷 및 스마트폰 사용을 포함해 금융 사용 사례에서 소비자 프로필을 정확하게 포착할 수 있도록 했습니다.

### 프라이버시를 기본으로 설계

모든 페르소나는 완전히 합성된 것입니다. 실제 개인을 나타내는 페르소나는 없으며, 데이터셋은 개인정보 보호 위험을 줄이도록 설계되었습니다. 인구 통계와 가계 금융은 공식 통계 및 공개 설문조사 데이터를 따르고, 이름은 실제 일본의 작명 패턴을 따릅니다. 페르소나는 실제 사람에 관한 레코드가 아니라 이러한 분포를 기반으로 구축됩니다.

### 이 데이터는 누구를 위한 것인가요?

Nemotron-Personas-Finance-Japan은 일본에서 금융 AI를 개발하는 팀, 즉 은행과 보험사, 그리고 이들을 지원하는 스타트업 및 시스템 통합업체를 우선 대상으로 구축되었습니다. 각 페르소나는 인구 통계와 함께 가계 금융 및 금융 리터러시 정보를 포함합니다. 팀은 이를 사용해 고객 서비스 에이전트를 구축하고, 영업 대화 및 마케팅 이니셔티브를 평가하며, NeMo User Sim에서 사용자 시뮬레이션을 실행하고, 고객 대상 AI를 레드팀 테스트할 수 있습니다.

이번 릴리스는 일본 사용자를 대상으로 무언가를 구축하는 모든 이에게도 유용합니다. 50개 필드에 걸쳐 900만 개의 페르소나를 포함하며, 이는 22개 필드였던 Nemotron-Personas-Japan보다 50% 더 많습니다. 따라서 일본어 프로젝트라면 무엇이든 학습 및 테스트에 사용할 수 있는 더욱 세분화된 인구 집단 구간을 확보할 수 있습니다.

### 실용적인 AI 애플리케이션

이 데이터셋의 합성 페르소나는 다음과 같은 용도로 활용할 수 있습니다:

- 다중 턴 대화: 페르소나를 "시드"로 사용해 사람과 유사한 대화 데이터셋 생성

- 도메인별 AI 개발: 금융 사용 사례의 현실적인 시나리오에 맞춘 AI 어시스턴트를 구축하기 위한 데이터셋 생성

- 편향 테스트 및 공정성: 농촌과 도시 간 격차, 다양한 연령대, 교육 배경, 경제적 상황 전반에서 모델과 AI 에이전트 시스템이 어떻게 작동하는지 평가하여 AI가 일본 사회 전반에서 일관되게 작동하는지 확인

### 중요한 이유

실제 세계의 인구를 반영하는 다양하고 고품질인 학습 데이터에 대한 접근은 오랫동안 AI 개발의 과제였습니다. 엔터프라이즈 AI 개발은 대부분 비공개 데이터에 의존해 왔으며, 이로 인해 연구자와 스타트업, 특히 사용 가능한 데이터가 제한적인 지역의 AI 개발자에게 장벽이 생겼습니다.

- 데이터 다양성: 일본 인구의 전체 스펙트럼을 반영하여 편향된 학습과 [모델 붕괴](https://medium.com/data-science/addressing-concerns-of-model-collapse-from-synthetic-data-in-ai-7cd380208d14) 위험을 줄이는 데 도움을 줍니다.

- 문화적 진정성: 서구 중심 데이터셋에 대한 의존을 줄이고 [Sovereign AI systems](https://www.nvidia.com/en-us/lp/industries/global-public-sector/sovereign-ai-technical-overview/) 개발을 지원합니다.

- 개인정보 보호 및 거버넌스: AI 거버넌스에 대한 일본의 혁신 우선 접근 방식에 부합합니다. 모든 페르소나는 합성된 것이므로 이를 사용해 구축하거나 테스트할 때 고객 데이터가 필요하지 않습니다.

Nemotron-Personas-Finance-Japan을 CC BY 4.0으로 공개함으로써 엔터프라이즈급 고품질 합성 데이터에 대한 접근성을 제공하고, 누구나 더 낮은 비용과 적은 개인정보 보호 위험으로 문화적 맥락에 기반한 AI 시스템을 구축할 수 있도록 지원합니다.

### Nemotron-PII-Japan과 함께 사용

[Nemotron-PII-Japan](https://huggingface.co/datasets/nvidia/Nemotron-PII-Japan)도 같은 날 출시됩니다. 이 데이터셋에는 레이블이 지정된 개인정보를 포함하는 합성 일본어 비즈니스 문서 10만 개가 포함되어 있어, 개인정보 탐지 및 비식별화 모델의 학습과 테스트에 사용할 수 있습니다. 해당 문서는 Nemotron-Personas-Japan을 기반으로 구축되었습니다. Nemotron-Personas-Finance-Japan으로 고객을 시뮬레이션하고, Nemotron-PII-Japan으로 파이프라인이 고객 데이터를 어떻게 보호하는지 테스트해 보세요.

### Nemotron-Personas-Finance-Japan으로 구축 시작

아래 명령어를 사용해 이 데이터셋을 다운로드하면 일본의 문화와 언어를 진정으로 이해하는 AI를 개발하는 데 도움이 됩니다.

```
from datasets import load_dataset
ds = load_dataset("nvidia/Nemotron-Personas-Finance-Japan")
```


프로덕션 환경에 바로 사용할 수 있는 애플리케이션 구축을 위한 사용 사례:

- 대화 생성의 시드로 페르소나 사용

- 문화적 맥락에 기반한 데이터로 모델 미세 조정

- 일본의 전반적인 인구 구성을 반영하는 개인화 엔진 구축

- 일본의 맥락을 갖춘 도메인별 AI 에이전트 개발

일본에서 AI를 구축하는 팀부터 더 넓은 지역에 배포하려는 글로벌 개발자에 이르기까지, [Nemotron-Personas-Finance-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Finance-Japan) 데이터셋은 애플리케이션에 필요한 진정성 있고 개인정보 보호를 고려한 기반을 제공합니다.
