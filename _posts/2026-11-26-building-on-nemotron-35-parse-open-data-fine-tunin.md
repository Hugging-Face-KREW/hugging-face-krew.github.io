---
layout: post
title: "Nemotron 3.5 Parse 기반 구축: 공개 데이터, 미세 조정, 더 빠른 추론"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6365c3f398da81987e2753f2/zArpuP9uLsLz9luMW0K2g.png
image: assets/images/blog/posts/2026-11-26-building-on-nemotron-35-parse-open-data-fine-tunin/thumbnail.png
authors:
  - user: nvidia
slug: "building-on-nemotron-35-parse-open-data-fine-tunin"
source_url: "https://huggingface.co/blog/nvidia/building-on-nemotron-35-parse-open-data-fine-tunin"
source_published_date: "2026-11-26"
source_published_at: "2026-11-26T05:05:36+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Building on Nemotron 3.5 Parse: Open Data, Fine-Tuning, and Faster Inference](https://huggingface.co/blog/nvidia/building-on-nemotron-35-parse-open-data-fine-tunin)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/building-on-nemotron-35-parse-open-data-fine-tunin -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Nemotron 3.5 Parse 기반 구축: 공개 데이터, 미세 조정, 더 빠른 추론

최근 [NVIDIA Nemotron 3.5 Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse)를 출시했습니다. 이는 페이지 이미지를 검색 증강 생성(RAG) 및 에이전트를 위한 구조화된 데이터로 변환하는 소형 문서 모델입니다.

이제 합성 학습 데이터셋, 미세 조정 레시피, 더 빠른 추론을 위한 DFlash2 speculator를 통해 이 릴리스를 발전시키고 있습니다. 또한 Parse 3.5를 학습한 방법도 공유합니다. 먼저 지도 미세 조정을 통해 문서 이해를 가르친 다음, 강화 학습을 사용해 정확도, 일관성, 안정적 완료를 개선했습니다.

## Parse 3.5가 생성하는 결과 {#section-1}

Parse는 문서 페이지에서 콘텐츠를 추출해 reading order로 정렬하고, 제목, 문단, 표, 차트와 같은 content class를 할당하며, 콘텐츠의 위치를 나타내는 bounding box를 반환합니다. 표와 차트는 후속 문서 처리 애플리케이션에서 사용할 수 있는 구조화된 레코드로 변환됩니다.

[![An operations brief with predicted regions on the left and its structured extraction on the right.](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-01-page-structure.png)](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-01-page-structure.png)

그림 1. 합성 운영 브리프에서 얻은 실제 모델 출력으로, 페이지와 구조화된 표현 사이의 연결을 보여줍니다.

Parse의 이번 최신 버전은 특히 더 어렵고 모호한 경우에 차트와 표를 읽는 능력을 크게 확장했습니다. 예를 들어 표시된 숫자 값이 전혀 없는 막대 차트나 기타 플롯을 처리할 수 있습니다. 모델은 축과 막대의 높이를 살펴 값을 추론합니다. 표는 구조가 매우 중요하기 때문에 특히 까다로울 수 있습니다. 행이나 열이 한 칸만 이동해도 해석과 의미가 완전히 달라질 수 있습니다. 이러한 발전은 알려진 ground truth를 사용해 차트와 표의 다양한 렌더링 변형을 생성한 다음, 이 과정을 역으로 수행하도록 모델을 학습하는 합성 데이터 생성에 크게 힘입었습니다.

Parse 3.5 학습에 사용한 세 가지 데이터셋을 오픈 소스로 공개합니다.

- [Synthetic Charts V1 — 9.98M examples](https://huggingface.co/datasets/nvidia/Synthetic-Charts-V1): 카테고리, 시리즈, 수치 값을 학습하기 위한 렌더링된 차트와 구조화된 주석입니다.

- [Synthetic Tables V1 — 1M examples](https://huggingface.co/datasets/nvidia/Synthetic-Tables-V1): 행, 열, 헤더, 셀 간 관계를 보존하는 target이 포함된 렌더링된 HTML 표입니다.

- [OCR Synthetic Multilingual v2 — 7.9M examples](https://huggingface.co/datasets/nvidia/OCR-Synthetic-Multilingual-v2): 다양한 글꼴과 레이아웃에 걸쳐 Parse 형식의 target을 포함하는 합성 문서 페이지로, 다국어 OCR 및 구조화된 페이지 읽기를 위한 데이터입니다.

각 릴리스에는 이미지, 주석, 생성 문서, 학습 형식이 포함됩니다. 이 쌍을 사용해 문서 모델을 미세 조정하거나, 대상 문서 유형 또는 언어에 맞는 새로운 변형을 만들 수 있습니다.

[![A bar chart without printed bar values, alongside the released model’s extracted values of 18, 27, 39, and 54.](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-02-chart-geometry.png)](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-02-chart-geometry.png)

그림 2. Parse가 차트 기하 구조에서 18, 27, 39, 54를 복원합니다.

## SFT는 기능을 가르치고, RLVR은 일관성을 개선합니다 {#section-2}

Parse 3.5는 두 단계로 학습했습니다. 텍스트, 레이아웃, 표, 차트, 다국어 문서를 아우르는 지도 미세 조정(SFT)을 먼저 수행했습니다. 구조화된 target을 예측하도록 학습하면서 페이지를 읽고 그 내용을 표현하는 일반적인 기능을 확립했습니다.

그런 다음 Group Relative Policy Optimization (GRPO)을 사용해 검증 가능한 보상을 통한 강화 학습(RLVR)을 계속 진행했습니다. 보상은 샘플링된 추출 결과의 품질을 확인합니다. 예를 들어 차트 추출은 올바른 시리즈와 값을 복원하고, 사용할 수 있는 구조를 생성하며, 동일한 콘텐츠를 반복해서 생성하지 않고 완료될 때 더 높은 보상을 받을 수 있습니다.

어려운 페이지에는 여러 가지 유효한 표현이 있을 수 있습니다. 이러한 검사는 정확히 하나의 토큰 시퀀스를 지정하지 않고도 모델이 올바른 답을 찾을 수 있도록 유연성을 제공하며, 반복과 미완성 출력을 억제합니다.

[![A matched training comparison showing extraction gains after GRPO and fewer repetition flags and token-cap hits.](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-03-sft-grpo-gains.png)](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-03-sft-grpo-gains.png)

그림 3. 동일한 평가 설정을 적용해 SFT와 GRPO를 통제된 방식으로 비교한 결과입니다. 이는 최종 공개 모델과는 별개인 개발 체크포인트입니다. 루프 수는 반복 검사를 사용합니다. [방법](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/blob/main/measurement-notes.md#training-and-adaptation-figures).

동일 조건 비교에서 효과가 확인됩니다. 루프로 표시된 출력은 126페이지에서 29페이지로 감소해 약 77% 줄었고, 9,000토큰 한도에 도달한 출력은 185개에서 35개로 감소했습니다. 전체 점수는 66.24에서 71.38로 향상되었으며, 표에서 +5.18점, 차트에서 +4.15점을 기록했습니다.

## 아키텍처와 DFlash2를 통한 더 빠른 추론 {#section-3}

파서는 모든 페이지에 대해 상당한 양의 텍스트와 구조를 생성해야 합니다. 학습 작업과 함께 Parse 3.5가 이미지를 처리하고 해당 출력을 디코딩하는 방식도 최적화했습니다.

| 구성 요소 | Parse 2.0 | Parse 3.5 |
| --- | --- | --- |
| 페이지 전처리 | 고정된 2048 × 1664 캔버스 | NaViT 동적 직사각형 입력 |
| 이미지-텍스트 연결 | 인코더–디코더 cross-attention | 인과적 prefix로서의 압축된 시각 토큰 |
| 디코더 self-attention | Multi-head attention | GQA-8: 16개 query head, 8개 key/value head |
| Speculative decoding |   | One-token lookahead (MTP); 선택적 DFlash2 |

NaViT 전처리는 페이지에 맞게 입력 형태를 조정합니다. visual adapter가 이미지 시퀀스를 압축하고, 텍스트 디코더는 이를 prefix로 처리하므로 encoder–decoder cross-attention을 대체합니다. 8개의 key/value head를 사용하는 grouped-query attention(GQA-8)은 디코더 attention-cache 메모리를 줄입니다.

내장된 multi-token prediction(MTP) head는 one-token lookahead를 제공합니다. [DFlash2 speculator](https://huggingface.co/nvidia/Nemotron-3.5-Parse/tree/main/dflash2)는 여기서 더 나아갑니다. 소형 draft model이 최대 32토큰 앞까지 제안하고, Parse가 제안들을 함께 검증합니다. 측정된 32토큰 설정에서는 평균 약 14개의 draft token을 수용하며, target token을 포함하면 verification step당 진행되는 출력이 약 15토큰입니다.

[![Throughput curves comparing Parse 2.0 with Parse 3.5 base, MTP, and DFlash2.](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-04-parse-concurrency.png)](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-04-parse-concurrency.png)

그림 4. 동일한 2,078페이지를 사용해 H200 한 대에서 BF16으로 측정한 native single-pass throughput입니다. Parse 2.0은 각 지점에서 한 번의 trial을 사용하고, Parse 3.5는 세 번의 trial에서 얻은 중앙값을 보여줍니다. 화살표는 Parse 3.5 base에서 DFlash2로 얻은 향상을 나타냅니다. [측정 세부 정보](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/blob/main/measurement-notes.md).

H200 한 대에서 BF16으로 수행한 native single-pass benchmark에서 DFlash2는 speculative decoding을 비활성화한 Parse 3.5와 비교해 concurrency 1에서 throughput을 7.62배, concurrency 512에서 1.52배 높였습니다. 512에서 Parse 3.5는 speculation 없이 13.48 pages/s, DFlash2 사용 시 20.44 pages/s에 도달하며, Parse 2.0은 2.98 pages/s입니다. 이는 각각 Parse 2.0 throughput의 4.53배와 6.86배입니다.

빠른 파싱을 통해 어려운 영역을 다시 살펴볼 여지가 생깁니다. 먼저 전체 페이지를 예측한 다음, 문서에 필요한 부분을 집중적으로 재처리합니다.

- Single pass - 텍스트, 레이아웃, 표, 차트에 대해 전체 페이지를 한 번 파싱합니다.

- Double pass - 밀집된 표와 작은 차트 세부 정보에 사용합니다. 파서가 찾은 표와 차트 영역을 crop하고, 더 높은 세부 수준으로 다시 파싱한 다음, 개선된 결과를 페이지 출력에 병합합니다.

- Triple pass - 모델의 full page parsing bounding box만 사용하는 대신, detection mode에서 모델을 두 번째 pass로 실행해 모든 box가 정확하게 발견되었고 텍스트가 누락되지 않았는지 확인한 다음, 발견된 차트나 표를 세 번째 pass에서 다시 추출합니다.

[![Single-pass page parsing, double-pass crop refinement, and triple-pass refinement with an independent region detector.](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-05-multipass.png)](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-05-multipass.png)

그림 5. 추가 단계는 어려운 영역에 연산을 집중합니다. 여러 crop을 사용하면 여러 요청이 필요할 수 있습니다.

공개된 reference evaluation에서 double pass와 triple pass는 single pass보다 ParseBench 점수를 각각 2.18점과 2.46점 높였습니다. 추가 작업을 통해 애플리케이션은 throughput 일부를 더 나은 어려운 영역 추출과 교환할 수 있습니다. [Reference results and serving measurements](https://huggingface.co/nvidia/Nemotron-3.5-Parse#single-double-or-triple-pass).

## 동일한 SFT/RLVR 접근 방식을 문서에 적용하기 {#section-4}

개발자가 paired data generation, NeMo Automodel SFT, NeMo RL GRPO를 단계별로 수행할 수 있는 [Parse fine-tuning recipe](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/parse/index.md)를 공유합니다. 이 과정에서는 vLLM을 사용해 후보 답변을 생성합니다.

예제에서는 odontogram이 포함된 합성 치과 양식을 사용합니다. odontogram은 치아, 예정된 시술, 영향을 받은 치아 표면을 보여주는 차트입니다. 적응 전 Parse는 인쇄된 치아 번호는 읽지만 예정된 시술 표를 놓칩니다. SFT는 각 표시된 치아를 시술 및 치아에서의 위치와 연결하도록 학습합니다.

[![A synthetic dental page, the base model’s tooth-number transcription, and the intended tooth/procedure/surface table.](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-06-dental-adaptation.png)](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-06-dental-adaptation.png)

그림 6. 기록된 recipe 예시입니다. SFT 후 차트의 6개 행이 모두 target과 일치합니다. [전체 이미지 및 native training label](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/blob/main/examples/dental-adaptation.md).

이후 RLVR은 정확하고 일관된 완료를 강화합니다. 치과 reference run에서 추정된 첫 시도 성공률(샘플링된 pass@1)은 base model의 0.00%에서 SFT 후 95.85%, RLVR 후 96.58%로 상승했습니다.

[![Dental sampled pass@1: 95.85% after SFT and 96.58% after RLVR, with the y-axis shown from 90% to 100%](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-07-dental-sft-rlvr.png)](https://huggingface.co/datasets/nvidia/Nemotron-Parse-Blog-Assets/resolve/main/images/figure-07-dental-sft-rlvr.png)

그림 7. 동일한 합성 생성기에서 만든 보류 양식 256개에 대해 페이지당 8개의 샘플로 수행한 동일 조건 평가입니다. 성공하려면 자연스러운 완료, 정확한 차트 콘텐츠, 오류 없는 정규화된 페이지 텍스트가 필요합니다. [평가 프로토콜](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/parse/workflow.md#evaluation-and-dental-example-results).

이 레시피를 적용하려면 자체 image/label 쌍과 verifier를 제공하면 됩니다. verifier는 추출 결과가 올바른지 판단하는 검사입니다. 포함된 [document reward components](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/parse/document_rewards)는 텍스트 충실도, bounding box, 표, 차트 정보, 완료를 다룹니다. 작업과 관련된 구성 요소를 결합하고, schema 검사와 함께 completion 및 repetition 검사를 추가해 루프나 미완성 출력을 억제할 수 있습니다.

이미 커뮤니티가 [Thai](https://huggingface.co/iapp/OpenThai2.1-OCR-Markdown-20261118), [Persian/Farsi](https://huggingface.co/Reza2kn/Bina-0.4-Koochik), [Arabic](https://huggingface.co/g3ida/absir-arabic-ocr-mini)에 맞게 Parse를 확장하고 개선했다는 점을 기쁘게 생각합니다. 이러한 추가 데이터셋과 미세 조정 레시피가 이러한 노력과 향후 모델 개발을 촉진하는 데 도움이 되기를 바랍니다.

vLLM 통합과 독립적인 성능 검증을 수행한 Denis Ovchinnikov, 미세 조정 레시피의 런타임 설정과 재현성을 개선한 Oliver Holworthy, 기술 검토를 맡은 Bo Liu에게 감사드립니다.
