---
layout: post
title: "Project ProtoStar 발표: 공공 자금으로 수행된 과학을 AI-ready하게 만들기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/announcing-project-protostar/thumbnail.jpg
image: assets/images/blog/posts/2026-10-29-announcing-project-protostar/thumbnail.jpg
authors:
  - user: evijit
slug: "announcing-project-protostar"
source_url: "https://huggingface.co/blog/announcing-project-protostar"
source_published_date: "2026-10-29"
source_published_at: "2026-10-29T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Announcing Project ProtoStar: Making Publicly Funded Science AI-Ready](https://huggingface.co/blog/announcing-project-protostar)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/announcing-project-protostar -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Project ProtoStar 발표: 공공 자금으로 수행된 과학을 AI-ready하게 만들기

*대학, 연구 기관, 산업계, 오픈 사이언스 커뮤니티를 한데 모아 공공 자금으로 수행된 연구를 더 쉽게 발견하고, 재사용하고, 그 위에 새로운 성과를 쌓을 수 있도록 하는 이니셔티브.*

![Project ProtoStar: Public research, put to work](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/announcing-project-protostar/header.jpg)

오늘, 공공 자금으로 수행된 과학 연구의 잠재력을 발휘하기 위한 이니셔티브인 **[Project ProtoStar](https://huggingface.co/project-protostar)**을 발표합니다. 특히 AI 기반 연구를 위해 데이터셋, 모델, 코드를 더욱 쉽게 접근하고 재사용할 수 있도록 하는 것이 목표입니다. ProtoStar는 연구 기관, 산업계, 오픈 사이언스 커뮤니티를 한데 모아 과학적 산출물을 AI-ready하고 machine-actionable한 형식으로 공개하도록 지원하고, 일련의 오픈 챌린지를 통해 그 가치를 보여줍니다. 첫 번째 챌린지는 [Carnegie Mellon University](https://www.cmu.edu/)이 주최하며 11월 2일에 시작됩니다. 자세한 내용은 아래의 [How to Participate](#the-first-challenge-climate-risk-intelligence)를 참조하세요.

이 이니셔티브의 창립 파트너로는 [Carnegie Mellon University](https://www.cmu.edu/), [Purdue University](https://www.purdue.edu/), [Georgia Institute of Technology](https://www.gatech.edu/), [Arizona State University](https://www.asu.edu/), [The George Washington University](https://www.gwu.edu/)가 있으며, 대학 연합 파트너인 [Open Forum for AI (OFAI)](https://www.openforumai.org/)를 통해 함께 활동합니다. 또한 [Astera Institute](https://astera.org/), 인프라 파트너인 [Hugging Face](https://huggingface.co/), 그리고 커뮤니티 파트너인 [Open Research Community Accelerator (ORCA)](https://orcaopen.org/)가 [Fostering Innovation Network](https://orcaopen.org/work/fostering-innovation-through-open-science-fin)를 통해 참여합니다.

참여 조직은 적합한 과학 데이터셋, 모델, 코드를 오픈 인프라에서 AI-ready 형식으로 공개하는 것을 지원하는 양해각서(memorandum of understanding)를 통해 참여를 공식화합니다. Project ProtoStar 참여에 관심이 있는 조직은 [protostar@orcaopen.org](mailto:protostar@orcaopen.org)로 연락할 수 있습니다.

## 공공 자금으로 수행된 과학을 더 유용하게 만들기 {#section-1}

미국 연방 정부는 약 [$190 billion annually in research and development](https://www.congress.gov/crs-product/R48694)를 투자하고 있으며, 연방 정책은 그 결과를 점점 더 공개적으로 공유할 것을 요구하고 있습니다. [National Institutes of Health (NIH)](https://www.nih.gov/) 및 [National Science Foundation (NSF)](https://www.nsf.gov/)와 같은 기관은 자금 지원을 받은 연구 출판물에 대한 공개 접근 요건을 강화했으며, 연구자에게 기초 데이터를 공유할 계획을 세우도록 요구하고 있습니다. Hugging Face는 연구 데이터 공유를 위한 [ten generalist repositories listed by NIH](https://www.nlm.nih.gov/NIHbmic/generalist_repositories.html) 중 하나입니다.

**하지만 연구를 공개적으로 이용할 수 있게 만든다고 해서 자동으로 사용 가능한 상태가 되는 것은 아닙니다.** 과학 데이터셋, 모델, 코드는 서로 호환되지 않는 시스템에 흩어져 있거나, 정적 형식 또는 레거시 형식으로 저장되어 있거나, 이를 이해하고 재사용하는 데 필요한 문서, 메타데이터, 출처 정보 없이 공개되는 경우가 많습니다. 이러한 한계는 연구자가 기존 연구를 기반으로 작업하기 어렵게 만들며, AI 시스템이 분야를 넘어 과학 리소스를 발견하고 해석하고 결합하는 일은 더욱 어렵게 만듭니다.

Project ProtoStar는 오픈 인프라와 실용적인 게시 가이드의 지원을 바탕으로, 공공 자금으로 수행된 과학적 산출물을 AI-ready 형식으로 체계적으로 공유하도록 장려함으로써 이러한 격차를 줄이고자 합니다. 이는 [National Cancer Institute's AI Data-Readiness Challenge](https://datacommons.cancer.gov/news/nci-crdc-artificial-intelligence-data-readiness-aidr-challenge) 및 [Hugging Face's guide to preparing public-sector data for AI](https://huggingface.co/blog/evijit/public-org-data-ai)과 같은 활동을 기반으로 합니다. 목표는 연구 기관이 기존의 공개 접근 기대에 부응하는 동시에 공개하는 연구의 과학적 가치를 높이도록 돕는 것입니다.

재사용 가능한 과학 리소스에 대한 수요는 이미 분명하게 나타나고 있습니다. **[Hugging Face](https://huggingface.co/)에는 핵심 과학 분야 전반에 걸쳐 17,000개가 넘는 공개 데이터셋이 호스팅되어 있으며**, 여기에는 유전체학, 단백질 설계, 화학, 날씨 및 기후, 지구 관측, 천문학이 포함됩니다. 의학과 수학을 포함하면 그 수는 약 29,000개에 이르며, 과학 모델도 비슷한 규모로 제공됩니다.

공유 속도 또한 빨라지고 있습니다. **연구자들은 2026년 첫 9개월 동안 2025년 같은 기간보다 3배가 넘는 새로운 과학 데이터셋을 Hugging Face에서 공유했으며**, 가장 많은 데이터를 업로드한 단일 업로더를 제외한 경우에도 증가세는 상당한 수준으로 유지되었습니다.

<iframe src="https://evijit-hf-science-growth.static.hf.space/embed.html?chart=quarterly-datasets&theme=dark" title="New science datasets per quarter (Hugging Face Hub)" width="100%" height="640" style="border:0" loading="lazy"></iframe>

*그림 1. 분기별 Hugging Face의 새로운 공개 과학 데이터셋, 분야별 현황(Q1 2023–Q3 2026). 생성일을 기준으로 현재도 공개 상태인 데이터셋 수를 집계했습니다. 출처: Hugging Face Hub 데이터, 2026년 10월. [Explore the data](https://huggingface.co/spaces/evijit/hf-science-growth).*

조직이 연구를 재사용 가능한 형식으로 공유하면 과학 커뮤니티는 이를 대규모로 기반 삼아 발전시킬 수 있습니다. Hugging Face에서 NIH의 [National Center for Biotechnology Information (NCBI)](https://www.ncbi.nlm.nih.gov/)가 관리하는 18개 공개 리포지터리는 총 1,100만 회가 넘는 다운로드를 기록했으며, [NASA IMPACT's](https://huggingface.co/nasa-impact)의 27개 리포지터리는 2026년 10월 26일 기준 약 770,000회의 다운로드를 기록했습니다. 이러한 리소스를 유용하게 만드는 데 문서화 역시 중요한 부분입니다: [90% of dataset downloads on Hugging Face are from fully documented datasets](https://arxiv.org/abs/2401.13822).

공개적으로 이용 가능한 유전체 데이터는 생물학 연구와 같은 분야를 더욱 발전시키는 연구에 기여하는 데 중요한 역할을 해왔습니다. Hugging Face의 연구자들은 [HuggingFaceBio](https://huggingface.co/HuggingFaceBio)의 최근 공개 자료인 [Carbon-A](https://huggingface.co/blog/HuggingFaceBio/carbon-annotator-genbank-genome-annotation) 및 [Carbon-X](https://huggingface.co/spaces/HuggingFaceBio/Carbon-X-Release)를 비롯해 공개적으로 이용 가능한 과학 리소스를 기반으로 연구를 진행해 왔습니다. 그 결과로 만들어진 모델, 데이터셋, 도구를 공개함으로써 이러한 활동은 기존 과학 리소스가 다른 사람들이 기반으로 삼을 수 있는 새로운 연구를 어떻게 지원할 수 있는지 보여줍니다.

참여 조직이 채택한 기관 정책을 통해 Project ProtoStar로 더욱 촉진하고자 하는 활동이 바로 이러한 것입니다. 기관 차원의 약속을 보여주는 초기 사례로, CMU's Open Source Programs Office는 연구자들이 적합한 데이터셋, AI 모델 및 기타 연구 산출물을 Hugging Face에서 공유하도록 장려하는 한편, 분야별 리포지터리의 지속적인 중요성도 인정하고 있습니다.

## 오픈 사이언스 챌린지 {#section-2}

[Making scientific artifacts AI-ready](https://huggingface.co/blog/evijit/public-org-data-ai)는 중요한 첫걸음입니다. 또한 이러한 리소스를 재사용할 수 있게 했을 때 연구자, 개발자, 더 넓은 과학 커뮤니티가 무엇을 달성할 수 있는지 보여주고자 합니다.

ProtoStar의 핵심 질문은 다음과 같습니다. **AI-ready 형식으로 공개된 공공 자금 지원 과학 데이터셋, 모델, 코드를 쉽게 발견하고, 재사용하고, 결합하고, 확장하여 과학적 발견과 공익을 가속할 수 있을까요?**

이를 탐구하기 위해 ProtoStar는 일련의 오픈 챌린지를 개최합니다. 각 챌린지는 참여 연구 기관이 주최하며, 해당 기관이 공개적으로 출시한 과학 리소스를 중심으로 구성됩니다. 챌린지에는 서로 보완적인 두 가지 목표가 있습니다.

- **과학적 재사용을 보여줍니다.** 공공 자금으로 수행된 과학적 산출물을 연구자와 개발자가 기반으로 삼는 구체적인 사례를 만들고, 기관이나 분야를 넘어 리소스를 결합하는 애플리케이션도 포함합니다.
- **재사용을 가능하게 하는 요소를 파악합니다.** 향후 연구 인프라, 자금 지원 결정, 공공 정책에 반영할 수 있도록 기술적 장벽, 문서화의 공백, 상호운용성 문제, 성공적인 관행을 파악합니다.

그 결과로 만들어지는 프로젝트와 얻어진 교훈은 공공 자금으로 수행된 과학을 더욱 접근 가능하고 유용하게 만들기 위한 실질적인 근거 기반을 구축하는 데 도움이 될 것입니다.

### 첫 번째 챌린지: 기후 위험 인텔리전스

![CMU Climate Risk Intelligence Challenge, November 2–13, 2026](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/announcing-project-protostar/climate-challenge.jpg)

**[Carnegie Mellon University](https://www.cmu.edu/) 주최 | 2026년 11월 2–13일**

첫 번째 ProtoStar 챌린지는 210개 미국 도시의 과거 날씨 데이터를 사용해 AI가 도시 기후 조건 예측을 개선할 수 있는지 참가자들이 시험하도록 합니다. [research by Carnegie Mellon University](https://doi.org/10.1184/R1/7890488.v6)를 기반으로 팀은 2018년까지의 기후 기록을 사용해 2019년부터 2026년까지의 실제 조건을 예측합니다. 이를 통해 가상의 미래 시나리오가 아니라 관측된 날씨와 비교하여 모델을 평가할 수 있습니다.

참가자는 도시 계획과 관련된 네 가지 지표를 예측합니다. 연평균 기온, 연간 총강수량, 연중 최고 기온이 기록된 날, 그리고 냉방 수요를 나타내는 냉방도일입니다. 예측 결과는 과거 평균과 선형 추세를 포함한 단순 통계적 기준선과 비교되어 AI 접근법이 정확도에서 의미 있는 개선을 제공하는지 평가받습니다.

예측 외에도 참가자는 데이터에서 얻은 내용을 바탕으로 짧은 기후 적응 정책 아이디어를 제안합니다. 이 구성 요소는 팀이 과학적 분석을 극심한 더위에 대비한 도시 준비, 에너지 효율 개선, 변화하는 강수 패턴 관리와 같은 실질적인 공익과 연결하도록 장려합니다.

팀은 이러한 결과물이 담긴 Hugging Face 데이터셋을 제출합니다. 예측 성능은 확립된 기준선과 비교한 예측 오차를 측정하는 공개 리더보드를 통해 평가되며, 정책 아이디어는 별도로 검토됩니다. 전체 2026년 날씨 기록을 이용할 수 있게 되는 2027년 초에 최종 예측 결과가 공개됩니다.

**[Explore the CMU Climate Risk Intelligence Challenge](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence)**

CMU 챌린지는 이 시리즈의 첫 번째 챌린지입니다. 다른 참여 기관들도 차례로 자체 챌린지를 개최하며, 각 챌린지는 서로 다른 공공 자금 지원 과학 리소스와 재사용 기회를 조명합니다. 향후 챌린지는 [Project ProtoStar page on Hugging Face](https://huggingface.co/project-protostar)를 통해 발표됩니다.

## Project ProtoStar에 참여하는 방법 {#section-3}

학술 연구자, 학생, 독립 개발자, 오픈 과학 인프라를 기반으로 구축하는 기업 중 어느 쪽이든 ProtoStar를 통해 공공 자금으로 수행된 연구를 실제로 활용하고, AI-ready 오픈 사이언스를 더욱 폭넓게 유용하게 만드는 데 필요한 요소를 알아볼 기회를 얻을 수 있습니다.

과학 리소스의 접근성과 재사용성을 높이는 데 관심이 있는 대학, 연방 연구 기관, 국립 연구소, 비영리 단체 및 기타 연구 조직도 환영합니다. **참여 또는 협력 가능성을 논의하려면 [protostar@orcaopen.org](mailto:protostar@orcaopen.org)로 연락해 주세요.**

ProtoStar를 통해 공공 자금으로 수행된 과학 연구가 이를 기반으로 발전시킬 수 있는 사람과 도구에 더 쉽게 도달하고, 동시에 오픈 사이언스를 대규모로 지원하는 데 필요한 인프라와 관행에 관한 실질적인 근거를 만들어낼 수 있기를 바랍니다.
