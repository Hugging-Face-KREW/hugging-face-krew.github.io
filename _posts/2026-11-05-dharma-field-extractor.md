---
layout: post
title: "Dharma Field Extractor: 어떤 문서든 입력하고, 구조화된 필드를 JSON으로 출력"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/c1ik58RednXjcqrx5s2Fo.png
image: assets/images/blog/posts/2026-11-04-dharma-field-extractor/thumbnail.png
authors:
  - user: Dharma-AI
slug: "dharma-field-extractor"
source_url: "https://huggingface.co/blog/Dharma-AI/dharma-field-extractor"
source_published_date: "2026-11-04"
source_published_at: "2026-11-04T17:50:10+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Dharma Field Extractor: Any Document In, Structured Fields in JSON Out](https://huggingface.co/blog/Dharma-AI/dharma-field-extractor)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/Dharma-AI/dharma-field-extractor -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Dharma Field Extractor: 어떤 문서든 입력하고, 구조화된 필드를 JSON으로 출력

모든 문서 유형에서 요청한 필드만 정확히 추출하며, 테스트한 모든 frontier LLM을 능가하고, 비용은 일부에 불과한 20억 파라미터 모델입니다.

[![cover_combined_1](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/JIRHcND5LFS_JhgAe-ObV.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/JIRHcND5LFS_JhgAe-ObV.png)
4월에 계약서와 법원 제출 서류부터 청구서, 신분증, 양식에 이르기까지 폭넓은 문서에서 OCR을 수행하도록 특화된 모델 [DharmaOCR-LITE](https://huggingface.co/Dharma-AI/Dharma-OCR-LITE)를 출시했습니다. 이 모델의 기반이 된 [논문](https://arxiv.org/abs/2604.14314)은 도메인 특화에 대한 좁고 정량적으로 검증된 사례를 제시했습니다. 즉, 작업에 더 가깝게 학습된 소형 모델이 해당 작업에서 범용 frontier API를 능가한다는 것입니다.

OCR은 원본 서식에 충실한 페이지의 텍스트를 제공합니다. 계약서 전체를 처음부터 끝까지 읽거나 디지털화할 원고처럼 문서 전체가 작업 대상일 때 바로 이것이 필요합니다.

하지만 다른 사용 사례도 있습니다. 페이지 전체가 아니라 시스템이 후속 단계에서 사용할 수 있는 형태로 특정 필드를 추출하려는 경우입니다. 청구서의 모든 품목을 목록으로 추출하거나, 의료 의뢰서에서 환자 이름, 검사 유형, 날짜를 추출해 바로 예약할 수 있게 만들 수 있습니다. 신분증의 필드를 직원 기록의 열에 매핑할 수도 있습니다.

문서를 OCR로 처리하면 문서에 인쇄된 모든 내용이 줄 단위로 반환되지만, 어떤 문자열이 검사 유형인지, 어떤 문자열이 날짜인지, 어떤 숫자가 합계인지에 대한 정보는 없습니다. 각 값이 데이터베이스에 들어가기 전에 올바른 필드에 매핑하는 코드를 누군가는 여전히 작성해야 합니다.

오늘날에는 신중한 프롬프트를 사용하면 범용 LLM으로 이 작업을 수행할 수 있습니다. 하지만 많은 기업이 대규모로 실행해야 하는 작업을 frontier API의 가격과 지연 시간으로 처리하면, 잘 정의된 문제를 해결하는 방식 중 빠르게 비용이 가장 많이 드는 방법이 됩니다.

그래서 이 작업을 위한 모델을 직접 만들었습니다. Field Extractor는 문서 이미지와 함께 문서에서 찾을 수 있는 모든 것을 요청하거나 채울 정확한 schema를 입력으로 받아 구조화된 JSON을 반환합니다. 각 필드는 타입이 지정되고, 문서의 중첩 구조에 따라 중첩됩니다. 청구서는 모든 품목이 순서가 지정된 목록으로 반환되며, 각 품목에는 고유한 수량과 값이 있고 합계도 함께 제공됩니다. 신분증은 시스템에서 이미 사용하는 이름 아래 저장할 필드 형태로 반환됩니다. 문서 유형별 미세 조정 없이 동일한 체크포인트가 이 모든 작업을 처리합니다.

직접 [여기](https://ocr-portal.prd-us-east-2.dharma-ai.com/login)에서 사용해 볼 수 있습니다.

## 모델의 기능 {#section-1}

매일 아침 공급업체 청구서가 한 묶음씩 도착하고, 각 청구서에서 공급업체, 청구서 번호, 납기일, 합계라는 네 가지 정보를 시스템에 입력해야 한다고 해 보겠습니다. 스케줄러로 전달되는 의료 의뢰서나, 당사자, 계약 기간, 위약금, 벌금 금액이 중요한 컴플라이언스 시스템으로 전달되는 계약서도 마찬가지입니다. 페이지 전체가 필요한 것이 아니라 특정 필드가 필요합니다. Field Extractor에 이를 요청하는 방법은 두 가지입니다.

[![field-extractor-1280](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/j-k3pMh-GwLH6bgsRZ-4u.gif)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/j-k3pMh-GwLH6bgsRZ-4u.gif)

- Open: schema를 제공하지 않습니다. 모델은 고정된 필드명 목록이 아니라 문맥과 레이아웃을 바탕으로 페이지를 읽고 구조화된 필드에 해당하는 부분(레이블 뒤에 값이 오는 형태나, 열 위에 헤더 행이 있는 표)을 파악한 다음, 찾은 모든 항목을 키-값 쌍으로 반환합니다.

- Schema-directed: 원하는 정확한 형태를 필드별로 지정하고, 각 필드의 타입(string, number, boolean)을 설정합니다. 시스템에서 이미 사용하는 키 이름 아래에 결과가 반환됩니다. "service_provider"와 "client_company"를 요청하면 정확히 해당 키가 반환됩니다. 표를 요청하면 행과 열을 따라 각 셀을 해당 키에 맞춰 채웁니다. 문서에 없는 필드를 요청하면 값을 만들어 내지 않고 null을 반환합니다.

선택은 이미 무엇을 알고 있는지에 따라 달라집니다. 문서에 어떤 내용이 들어 있는지 모를 때는 open 요청을 보냅니다. 출력이 다른 곳에 이미 존재하는 형태와 일치해야 할 때는 schema를 보냅니다.

어느 경우든 반환되는 것은 JSON입니다. 필드는 키가 되고, 값은 요청한 타입으로 지정되며, 문서에 요청한 내용이 없으면 `null`이(가) 반환됩니다. schema를 보내면 출력은 동일한 키를 사용하고 추가 필드 없이 schema를 정확히 따르므로 다음 시스템에 바로 전달할 수 있습니다.

## 학습 방법 {#section-2}

기반 모델은 DharmaOCR-LITE v1.1입니다. 이는 논문에 사용된 모델에서 최적화된 새로운 20억 파라미터 버전으로, supervised fine-tuning과 DPO를 통해 이미 OCR에 특화되어 있습니다. 이 체크포인트 위에서 두 요청 모드를 모두 다루는 43,384개의 training example로 LoRA adapter를 학습했습니다.

데이터는 세 가지 흐름으로 구성됩니다.

- English multi-domain data: 공개 웹에서 수집한 다양한 문서로, 여러 문서 유형을 포괄합니다.

- Brazilian domain data: 내부적으로 수집한 실제 브라질 문서로, 포르투갈어로 작성되었습니다.

- Multilingual data: 영어와 포르투갈어를 넘어 더 폭넓게 일반화할 수 있도록 다른 유럽 및 아시아 언어를 다루는 소규모 데이터입니다.

세 흐름 모두 필드 수준으로 주석이 지정되어 있습니다. 데이터 구성은 의도적으로 여러 도메인을 아우릅니다. 하나의 서류 유형에 집중하는 대신 영수증, 상업용 청구서, 양식, 은행 거래 내역서, 신분증 문서, 합성 문서를 포함합니다.

[![03_training_data_mix_v2](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/SuMRjXc_gcp3LZm3K_mWs.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/SuMRjXc_gcp3LZm3K_mWs.png)

## 벤치마크의 한계선 {#section-3}

전체 벤치마크는 세 가지 요청 시나리오에 걸쳐 546개 문서, 총 1,638개 사례로 구성됩니다. schema 없이 모든 필드를 추출하는 경우, 실제 필드만 지정하는 부분 schema, 의도적으로 존재하지 않는 필드까지 포함하는 schema입니다. 점수는 필드 수준 F1(표 내부 위치를 포함해 경로별로 매칭)에 매칭된 값의 Levenshtein 유사도와 유효한 JSON으로 반환된 출력의 비율을 곱해 계산합니다.

동일한 벤치마크를 네 가지 범용 frontier 구성에 대해서도 실행했습니다.

[![image](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/rGnjzOfdYJrtG8gG_6kZc.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/rGnjzOfdYJrtG8gG_6kZc.png)

범용 LLM과 비교했을 때 품질 격차는 1.8포인트(가장 근접한 Claude Opus 5)에서 9.3포인트(격차가 가장 큰 GPT-5.6 Luna)까지 나타났습니다.

[![image](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/bHPhh4WTqoSI2Aqg0NAii.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/bHPhh4WTqoSI2Aqg0NAii.png)

비용은 약 6배 저렴한 경우(GPT-5.6 Luna)부터 약 147배 저렴한 경우(GPT-5.6 Sol)까지입니다. 두 측의 가격 책정 방식은 다릅니다. frontier 수치는 각 제공업체의 공개 정가를 기준으로, reasoning token을 포함해 페이지당 실제 평균 입력 및 출력 토큰을 측정해 산출했습니다. 모든 frontier 모델은 reasoning을 low로 설정해 실행했고, Claude Opus 5는 adaptive effort를 사용했습니다. adaptive effort는 간단한 페이지에서는 reasoning을 건너뛰면서도 low 수준을 유지할 수 있습니다. 저희 모델은 토큰당 과금되지 않습니다. self-hosted로 실행되므로, 수치는 시간당 $2.24인 단일 L40S에서의 GPU 시간입니다.

이는 판매 가격이 아닙니다. self-hosted 비용은 하드웨어, 처리량, 각 배포에 필요한 지연 시간에 따라 달라집니다. 이 비교의 목적은 자체 인프라에 배포하는 sovereign SSLM이 frontier API의 실질적인 대안임을 보여주는 것입니다.

## 일반화 테스트 {#section-4}

SROIE와 FUNSD는 벤치마크에 포함된 공개 데이터셋 두 가지이며, 학습에는 사용하지 않았습니다. 아래에 각각의 예시를 하나씩 소개합니다.

SROIE 영수증에서 이번에는 존재하지 않는 필드 없이 인쇄된 18개 필드를 모두 요청했고, 하나도 빠짐없이 정확하게 반환되었습니다. 여러 줄로 구성되고 레이블이 없는 청구지 주소는 하나의 필드로 캡처되었고, 비어 있는 두 필드(salesperson과 ref)는 정확히 null로 반환되었으며, 영수증의 모든 숫자는 센트 단위까지 일치했습니다.

[![sroie_combined](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/m3CfHgYlaDZ9srzt2X99_.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/m3CfHgYlaDZ9srzt2X99_.png)

FUNSD 복사본 편차 양식에는 5개의 하위 옵션으로 이루어진 중첩 그룹 하나와 12개의 boolean 필드 체크리스트가 있습니다. 모델은 체크된 상자 하나(품질이 낮은 원본이며 옆에 손글씨로 "faded"라고 적혀 있음)만 true로 표시하고 나머지 11개는 false로 표시해 양식과 정확히 일치시켰습니다. 또한 페이지에서 손글씨로 작성된 Bates ID를 정확히 읽었으며, 비어 있는 두 필드에는 null을 반환했습니다.

[![funsd_combined](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/GRHwR5tiDzDKVUtMotsAb.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/GRHwR5tiDzDKVUtMotsAb.png)

## 품질과 비용을 동시에 {#section-5}

20억 파라미터 모델은 0.834점을 기록해 테스트한 모든 frontier 구성보다 높은 점수를 얻었습니다. Claude Opus 5는 0.816점, GPT-5.6 Luna는 0.741점이었습니다. 또한 단일 L40S에서 실행하면서 비교 대상의 모든 API보다 추출당 비용이 낮았습니다. 동일한 벤치마크에서 동일한 실행으로 더 높은 품질과 더 낮은 비용을 측정한 것입니다.

이를 측정한 것이 이번이 처음은 아닙니다. 저희 OCR 논문에서는 첫 번째 DharmaOCR-LITE인 30억 파라미터 모델이 브라질 포르투갈어에서 테스트한 모든 상용 frontier API보다 높은 성능을 기록했습니다. 3개월 후 더 새롭고 더 많은 자원을 사용하는 모델들이 등장했지만 격차는 유지되었습니다. 이제 20억 파라미터 모델이 필드 추출에서도 같은 결과를 냅니다. 각각의 결과는 Dharma-AI를 시작할 때 세운 논리를 뒷받침하는 증거를 더합니다. 즉, 범위가 제한된 작업에서는 해당 작업에 가깝게 학습된 소형 모델이 같은 일을 범용적으로 수행하도록 요청받은 frontier 모델을 능가한다는 것입니다. 이 벤치마크에서는 실행 비용도 6배에서 147배 더 저렴했습니다.

파라미터 수는 상한선을 정하고, 학습은 모델이 해당 작업에서 그 상한선에 얼마나 도달하는지를 결정합니다. 비즈니스에 필요한 것은 사용 가능한 가장 큰 모델이 아닙니다. 여러분의 문제를 위해 학습된 모델입니다.

직접 문서에서 [여기](https://ocr-portal.prd-us-east-2.dharma-ai.com/login)로 테스트해 보세요. Dharma-AI와 저희가 만드는 특화 모델에 대해 더 알아보려면 [dharma-ai.com](https://dharma-ai.com)을 방문하세요.

### 출처

- BID. Brazilian Identity Document Dataset. Soares, Neves Jr., Bezerra (2020), "BID Dataset: a challenge dataset for document processing tasks," Anais Estendidos XXXIII SIBGRAPI. [github.com/ricardobnjunior/Brazilian-Identity-Document-Dataset](https://github.com/ricardobnjunior/Brazilian-Identity-Document-Dataset)

- CORD. Park et al., NAVER CLOVA (2019). [huggingface.co/datasets/naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

- DocVQA. Mathew et al., WACV 2021 (ICDAR 2019). [huggingface.co/datasets/lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)

- WildReceipt. Sun et al., IJCAI 2021. [huggingface.co/datasets/Theivaprakasham/wildreceipt](https://huggingface.co/datasets/Theivaprakasham/wildreceipt)

- SROIE (ICDAR 2019). 학습에서 제외했으며 평가에만 사용했습니다. [huggingface.co/datasets/darentang/sroie](https://huggingface.co/datasets/darentang/sroie) (mirror); upstream [rrc.cvc.uab.es/?ch=13](https://rrc.cvc.uab.es/?ch=13)

- FUNSD. Jaume et al. 학습에서 제외했으며 평가에만 사용했습니다. [huggingface.co/datasets/nielsr/funsd](https://huggingface.co/datasets/nielsr/funsd) (mirror); [guillaumejaume.github.io/FUNSD](https://guillaumejaume.github.io/FUNSD/)

- Zenodo Invoices (PT). Cruz & Castelli, NOVA IMS (2022). [zenodo.org/records/6371710](https://zenodo.org/records/6371710)

- line_items. Sagarjha456, Hugging Face. [huggingface.co/datasets/Sagarjha456/line_items](https://huggingface.co/datasets/Sagarjha456/line_items)

- Katanaml Invoices (Donut). Kozłowski & Weichbroth (2021), Mendeley Data [doi:10.17632/tnj49gpmtz.2](https://doi.org/10.17632/tnj49gpmtz.2). [huggingface.co/datasets/katanaml-org/invoices-donut-data-v1](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1)

- HumynLabs Brazilian Bills and Invoices. [huggingface.co/datasets/HumynLabs/Brazilian_Bills_and_Invoices_Dataset](https://huggingface.co/datasets/HumynLabs/Brazilian_Bills_and_Invoices_Dataset)

- Tech4Humans BR-Doc Extraction. [huggingface.co/datasets/tech4humans/br-doc-extraction](https://huggingface.co/datasets/tech4humans/br-doc-extraction)

- Cardoso, Gabriel Pimenta de Freitas, et al. "DharmaOCR: Specialized Small Language Models for Structured OCR that outperform Open-Source and Commercial Baselines." arXiv preprint [arXiv:2604.14314](https://arxiv.org/abs/2604.14314) (2026).

### 추가 읽기

- [SLM의 어휘를 특화하면 실제 애플리케이션에서 추론 비용을 77% 절감할 수 있는 방법](https://huggingface.co/blog/Dharma-AI/vocabx-vocabulary-specialization)

- [동일한 클러스터, 33포인트 더 높은 활용률: 바뀐 것은 순서였습니다](https://huggingface.co/blog/Dharma-AI/gpu-management-pt2)

- [GPU Management: 유휴 GPU가 새로운 Grounded Aircraft인 이유](https://huggingface.co/blog/Dharma-AI/gpu-management)

- [더 새로운 모델, 동일한 이점](https://huggingface.co/blog/Dharma-AI/newer-models-same-advantages)

- [특화가 필연적인 이유](https://huggingface.co/blog/Dharma-AI/why-specialization-is-inevitable)

- [특화가 규모를 능가한다: 대부분의 AI 조달 의사결정이 간과하는 전략적 변수](https://huggingface.co/blog/Dharma-AI/specialization-beats-scale)

- [Text Degeneration: 대부분의 벤치마크가 추적하지 않는 프로덕션 장애 모드](https://huggingface.co/blog/Dharma-AI/text-degeneration-a-production-failure-mode)

- [챗봇을 넘어선 Direct Preference Optimization](https://huggingface.co/blog/Dharma-AI/direct-preference-optimization-beyond-chatbots)

[Hugging Face의 Dharma AI](https://huggingface.co/Dharma-AI)를 둘러보고 [대화형 데모를 사용해 보고](https://huggingface.co/spaces/Dharma-AI/DharmaOCR-demo), [오픈 소스 모델을 다운로드하고](https://huggingface.co/Dharma-AI/Dharma-OCR-LITE), 실제 엔터프라이즈 애플리케이션에서 특화 AI 시스템이 범용 모델을 어떻게 능가하는지 알아보세요.
