---
layout: post
title: 'Indic 사전 학습 확장: 2.8조 토큰과 개방형 데이터 생성 파이프라인'
author: dailybot
categories:
- Translation
- HuggingFace
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/x0aObe_0mduBjWFuKBGoV.png
image: assets/images/blog/posts/2026-10-16-scaling-indic-pretraining/thumbnail.png
authors:
- user: nvidia
slug: scaling-indic-pretraining
source_url: https://huggingface.co/blog/nvidia/scaling-indic-pretraining
source_published_date: '2026-10-16'
source_published_at: '2026-10-16T01:44:47+00:00'
locale: ko
translation_status: draft
translator: openai
description: '* TOC {:toc} <!--toc--> _이 글은 Hugging Face 블로그의 [Scaling Indic Pretraining:
  2.8 Trillion Tokens and an Open Data Generation Pipeline](https://huggingface.'
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Scaling Indic Pretraining: 2.8 Trillion Tokens and an Open Data Generation Pipeline](https://huggingface.co/blog/nvidia/scaling-indic-pretraining)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/scaling-indic-pretraining -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Indic 사전 학습 확장: 2.8조 토큰과 개방형 데이터 생성 파이프라인

22개 언어의 네이티브 스크립트 데이터, 12개 언어의 로마자 표기 데이터, 그리고 번역·품질 평가·복구를 위한 재사용 가능한 GPU 가속 파이프라인.

저자: AI4Bharat — Mohammed Safi Ur Rahman Khan, Mitesh Khapra, Anoop Kunchukuttan. NVIDIA — Niket Agarwal, Raviraj Joshi, Anusha Kamath, Markus Kliegl, Yev Meyer, Sriharsha Niverty, Kiran Praveen, Utkarsh Vaidya.

약 2.8조 개 토큰으로 구성된 22개 인도 언어 데이터셋 Nemotron-pretraining-Indic-v1을 소개하게 되어 기쁩니다. 이 데이터셋을 만드는 데 사용한 개방형 GPU 가속 데이터 생성 파이프라인도 곧 함께 공개될 예정입니다. 이 22개 언어는 인도 헌법에 명시된 언어로, 2011년 인구 조사 기준 인도인의 97%, 즉 약 12억 명에 해당하는 사람들이 처음 사용하는 언어입니다. NVIDIA가 [IIT Madras](https://www.iitm.ac.in/) 및 인도의 대표적인 공과대학인 [AI4Bharat](https://ai4bharat.iitm.ac.in/) 연구소와 함께 개발한 이번 공개는 Indic 사전 학습을 위한 새로운 대규모 리소스를 모델 개발자에게 제공합니다. 이 파이프라인은 프롬프트, LLM workflow, 품질 검사를 포함한 전체 생성 레시피를 담고 있어, 개발자가 이를 자체 소스 자료에 맞게 조정하고 필요에 따라 더 많은 데이터를 생성할 수 있습니다.

AI4Bharat는 IIT Madras의 인도 언어 연구소로, 인도 언어를 위한 오픈 소스 데이터셋과 모델을 구축합니다. 22개 인도 언어를 위한 개방형 사전 학습 코퍼스인 [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha)과 같은 리소스는 개방형 Indic 언어 모델 연구의 기반을 마련하는 데 기여해 왔습니다. 이번 공동 오픈 소스 공개는 커뮤니티에서 활용할 수 있는 개방형 합성 Indic 학습 데이터의 규모를 크게 확장합니다. 수학, 코드, 질의응답 콘텐츠를 포함한 다양한 선별 영어 자료를 인도 언어로 제공하여, 팀이 해당 언어로 기술 개념을 설명하고 문제를 해결하며 질문에 답할 수 있는 모델을 구축하는 데 사용할 학습 자료를 늘립니다. 모든 레코드는 선별된 소스 문서로 추적할 수 있으며, 각 출력은 입력과 대조해 검사하므로 합성 텍스트가 실제 정보에 기반을 유지합니다.

AI4Bharat를 이끌며 [2025 TIME100 AI list](https://time.com/collections/time100-ai-2025/7305866/mitesh-khapra/)을 만든 Mitesh Khapra는 핵심 문제를 다음과 같이 지적했습니다. 인도 언어 기술이 영어에 뒤처지는 이유는 "인도 언어를 위한 데이터가 충분하지 않기 때문"입니다. 우리는 더 많은 데이터와 더 많은 데이터를 생성할 수 있는 개방형 레시피로 이에 답하고 있습니다.

이 코퍼스는 사람들이 글을 쓰는 방식도 반영합니다. 22개 언어 모두에 대한 네이티브 스크립트 데이터와 12개 언어에 대한 로마자 표기 텍스트를 포함하여, 총 34개의 언어-스크립트 구성을 다룹니다. workflow는 문서 단위 번역, 품질 평가, 대상별 복구를 결합하며, 수식·표·코드와 같은 콘텐츠를 보존하기 위한 검사를 포함합니다. 이 데이터는 지역 지식과 문화적 맥락에 필수적인, 자연적으로 작성된 Indic 텍스트를 보완합니다.

커뮤니티가 데이터를 재현하는 데 그치지 않고 이 작업을 발전시킬 수 있도록 [open generation recipe](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/tree/main/plugins/data-designer-nemotron-indic-translation)을 공유합니다. 팀은 코퍼스를 직접 사용하거나 프롬프트와 품질 검사를 조정하거나 자체 소스 문서를 workflow에 가져올 수 있습니다. 데이터와 프로세스를 모두 공개함으로써 대규모 Indic 데이터 생성을 더 쉽게 이용할 수 있도록 하고, 더 많은 개발자가 인도 언어 전반에서 더욱 강력한 모델을 만드는 데 기여할 수 있도록 하는 것이 우리의 목표입니다.

[Nemotron 3.5 Super VL](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)에 대한 계속된 사전 학습에는 이 코퍼스의 12개 언어 하위 집합과 자연 Indic 텍스트에서 추출한 100억 개의 Indic 토큰이 추가되었습니다. 모델에 대한 기술 보고서에서 전체 내용을 다룹니다. 향후 Nemotron 모델에서도 이 코퍼스를 기반으로 발전시켜 나갈 계획입니다.

## 코퍼스에는 무엇이 포함되어 있나요? {#section-1}

[![Corpus composition by native-script and romanized token counts.](https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/VmVFMKjOsoNkEKfQ7cRnh.png)](https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/VmVFMKjOsoNkEKfQ7cRnh.png)
그림 1. 코퍼스에는 2,601.0 billion개의 네이티브 스크립트 토큰과 224.6 billion개의 로마자 표기 토큰이 포함되어 있습니다. 개수는 Gemma-4 토크나이저를 사용해 계산했습니다. 문서 레코드에는 공유된 소스의 번역본과 스크립트 변형이 포함되며, 고유한 영어 문서의 개수가 아닙니다.

코퍼스에는 두 그룹의 언어 구성이 있습니다.

- 네이티브 스크립트 및 로마자 표기 데이터: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Tamil, Telugu, Urdu.

- 네이티브 스크립트 데이터: Assamese, Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri, Sanskrit, Santali, Sindhi.

여기서 네이티브 스크립트는 문자 체계를 의미하며, 텍스트 자체는 여전히 영어에서 번역된 것입니다. 로마자 표기는 라틴 스크립트로 작성된 Indic 언어 텍스트를 뜻하며, 영어 번역이 아닙니다. 이중 스크립트를 사용하는 12개 언어 각각에서 로마자 표기는 허용된 네이티브 스크립트 번역의 일부를 대상으로 합니다.

학습 혼합물을 구성할 때 이 구분은 중요합니다. 네이티브 및 로마자 표기 레코드는 서로 다른 스크립트 노출을 제공하지만, 동일한 underlying 콘텐츠를 표현할 수 있습니다. 토큰 수는 학습 실행의 토큰 예산이 아니라, 허용된 재작성본을 포함해 유지된 코퍼스를 설명합니다. 또한 개수는 사용한 토크나이저에 따라 달라지므로 코퍼스 목록과 학습 토큰 수는 별도로 해석해야 합니다.

## 영어 소스 문서에서 Indic 학습 데이터로 {#section-2}

이 코퍼스를 구축하기 위해 웹 텍스트, 질의응답 자료, 수학, 코드 및 기타 전문 콘텐츠를 다루는 15개 [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) 하위 집합의 문서로 시작합니다. NVIDIA의 합성 데이터 생성 오픈 소스 프레임워크인 [NeMo Data Designer](https://docs.nvidia.com/nemo/datadesigner/)을 사용해 번역, 품질 검사, 대상별 복구를 실행합니다. workflow 전반에서 각 소스 식별자를 유지하므로 개발자는 출력 결과를 원본 문서까지 추적할 수 있습니다. 이처럼 명확한 provenance는 개방형 데이터를 감사하고 기반으로 발전시키는 데 실용적인 기반이 됩니다. 그림 2는 12개 언어를 위한 workflow를 보여줍니다.

[![Corpus composition by native-script and romanized token counts.](https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/PCtm5uAGrdrp8kgHXb6y3.png)](https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/PCtm5uAGrdrp8kgHXb6y3.png)
그림 2. NeMo Data Designer와 Gemma-4-31B-it을 사용해 구현한 12개 언어용 4단계 Indic 합성 데이터 파이프라인입니다. 영어 문서를 번역하고, 번역 품질을 평가하며, 낮은 점수의 출력을 복구하고, 허용된 번역의 일부를 라틴 스크립트로 변환합니다. 나머지 10개 언어에는 아래에서 설명하는 별도의 Indic Translate workflow를 사용합니다.

### 12개 언어 workflow

이 12개 언어의 경우 모델이 문단, 표, 수식 전반의 맥락을 활용할 수 있도록 문서를 전체 단위로 번역합니다. 완전한 문서를 대상으로 모델을 비교하고 번역 품질과 기술 콘텐츠 보존 여부를 확인하여 [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)을 선정했습니다. 프롬프트는 번역할 산문과 보존할 콘텐츠를 구분하며, 코드·수식·URL·서식 등이 보존 대상에 포함됩니다.

번역 후에는 동일한 모델을 사용해 각 출력을 영어 소스와 대조하여 평가합니다. 번역 오류를 유형과 심각도별로 분류하는 개방형 프레임워크인 [Multidimensional Quality Metrics](https://themqm.org/)(MQM)에서 영감을 받은 평가 기준은 의미, 유창성, 용어, 스타일, 서식을 다룹니다. 100점 만점에 75점 이상을 받은 출력만 유지합니다(이는 정확도 백분율이 아니라 평가 기준 점수입니다). 낮은 점수의 출력에는 평가 피드백을 활용해 복구를 유도한 뒤 동일한 기준으로 다시 평가합니다. 이 과정을 통해 네이티브 스크립트 및 로마자 표기 데이터에서 438.4 billion개의 토큰을 복구했으며, 이는 이미 코퍼스 총량에 포함되어 있습니다. 샘플링한 사람의 검토와 의도적으로 오류를 삽입한 테스트를 통해 누락과 잔존 영어를 파악했고, 이를 바탕으로 프롬프트와 검사를 개선했습니다.

허용된 네이티브 스크립트 번역은 로마자 표기 데이터의 출발점이기도 합니다. 그중 일부를 라틴 스크립트로 변환하고 각 결과를 네이티브 스크립트 입력과 대조해 검사합니다. 이를 통해 번역과 스크립트 변환을 분리합니다. 초기 사람 대상 비교에서 이 2단계 접근 방식은 영어에서 직접 로마자 표기 텍스트를 생성하는 것보다 환각으로 추가된 내용과 잔존 영어가 적었습니다.

### 10개 언어 workflow

나머지 10개 언어에서는 비교 과정에서 확인된 장문 문서 품질 및 스크립트 안정성 문제에 대응하도록 workflow를 조정합니다. IIT Madras에서 인큐베이팅된 교육 분야 AI 비영리 센터인 [Bodhan AI](https://bodhan.ai/)이 제공하는 22개 인도 언어와 영어를 위한 문서 단위 번역 모델 [Indic Translate](https://huggingface.co/bodhan-ai/indic-translate)으로 번역합니다. 그런 다음 Gemma-4-31B-it이 각 번역을 영어 소스와 대조해 평가합니다. 8개 언어에는 전체 문서 접근 방식을 유지합니다. 더 긴 Santali 문서는 겹치는 window로 번역한 뒤 문서를 재구성하고, Manipuri는 Bengali 스크립트로 번역한 후 네이티브 스크립트인 Meitei Mayek으로 변환합니다. 이러한 workflow는 100점 만점에 75점 이상을 받은 출력만 유지하며, 피드백 기반 재작성 없이 네이티브 스크립트 데이터만 생성합니다.

## Indic 계속된 사전 학습 이후 무엇이 달라졌나요? {#section-3}

Nemotron 3.5 Super VL 기술 보고서는 Indic 계속된 사전 학습(CPT) 직전의 체크포인트와 해당 단계 이후의 체크포인트를 비교합니다. 설정된 예산은 학습 토큰 100 billion개였으며, 그중 10%(10 billion개 토큰)를 합성 및 자연 Indic 텍스트 혼합물에 할당했습니다. 나머지 90%는 이전 학습 혼합물의 상대적 비율을 유지했습니다.

보고된 5개 Indic 지식 및 추론 benchmark인 MILU, Indic MMLU-Pro, Indic ARC-Challenge, Indic ARC-Easy, Indic GSM8K 모두에서 점수가 상승했습니다.

[![figure3-cpt-before-after@2x](https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/ZPR2A2wDoNW6OVb4-GX63.png)](https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/ZPR2A2wDoNW6OVb4-GX63.png)
그림 3. Indic CPT 단계 전후의 Nemotron 3.5 Super VL입니다. 향상 폭은 4.98~9.82 percentage points입니다. 이 비교는 합성 및 자연 Indic 데이터를 결합한 학습 단계 전체의 변화를 보여주며, 합성 코퍼스의 기여만을 분리해 보여주는 것은 아닙니다. 언어 범위는 benchmark마다 다르며, 이는 최종 모델 릴리스의 비교가 아니라 단계별 체크포인트의 비교입니다.

출처: Nemotron 3.5 Super VL 기술 보고서, 표 2. 평가가 serving context limit을 초과했기 때문에 GSM8K에서는 Odia를 제외했습니다.

이러한 향상은 합성 및 자연 Indic CPT 레시피를 함께 사용하는 방식을 뒷받침하지만, 다른 모델과 언어에 사용할 혼합물의 선택은 추가 연구 과제로 남겨 둡니다.

동반 Indic 보고서는 12개 언어 그룹에서 1차로 허용된 데이터를 사용해 소스 혼합물, 합성 데이터 풀 크기, 스크립트 범위를 살펴봅니다. 복구된 재작성본과 나머지 10개 언어를 포함한 전체 코퍼스의 평가는 이러한 통제 연구의 다음 단계로 남아 있습니다.

### Nemotron 3.5 Super VL의 더 폭넓은 Indic 역량

CPT 단계 결과를 넘어, 공개된 모델은 이전 세대와 비교해 어떨까요? 그림 4는 Indic task 성능과 언어 품질 전반에서 Nemotron 3.5 Super VL과 [Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)을 비교합니다. 이 결과는 Indic CPT와 post-training을 포함한 공개 모델의 전체 학습 파이프라인을 반영합니다.

Nemotron 3.5 Super VL은 지식, 추론, 수학, 코드, 번역을 아우르는 6개 task benchmark 중 5개에서 더 높은 점수를 기록했습니다. IFEval 1.5의 지시 따르기 점수는 2.04포인트 하락했습니다. 또한 4개의 [IndiVibe](https://huggingface.co/datasets/sarvamai/indivibe) 차원인 유용성, 유창성, 언어 및 스크립트 충실도, 적절한 장황성 모두에서 향상되었습니다. IndiVibe는 인도 언어의 assistant-style 프롬프트를 대상으로 Sarvam AI가 만든 개방형 benchmark입니다. 이러한 평가는 모델이 무엇을 할 수 있는지와 인도 언어로 얼마나 잘 소통하는지를 함께 살펴봅니다.

[![figure4-super-generations-HD](https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/YQO_-yZWrhhQgueXuxzuM.png)](https://cdn-uploads.huggingface.co/production/uploads/68d2fec8856b85d927e44d32/YQO_-yZWrhhQgueXuxzuM.png)
그림 4. Nemotron Super 세대별 Indic task 성능과 언어 품질입니다. 왼쪽: 6개 task benchmark. 오른쪽: 예비 IndiVibe conditional preference 점수이며, 직접적인 head-to-head 승률이 아닙니다. WMT24++를 제외한 레이블은 percentage points 단위의 변화를 나타내며, WMT24++는 chrF points를 사용합니다. benchmark 및 평가 세부 사항은 동반 Indic 기술 보고서를 참조하세요.

## 시작하기 {#section-4}

### 코퍼스 로드

모델에 필요한 언어와 스크립트로 시작하고, 샘플을 검사하며, 학습 전에 dataset card와 라이선스를 확인하세요. 구성은 언어, 스크립트, 지역 태그를 사용합니다. 예를 들어 `hi-Deva-IN`은 Devanagari의 Hindi를, `hi-Latn-IN`은 로마자 표기의 Hindi를 나타냅니다.

아래 예제는 dataset card에서 repository ID를 요청한 다음 사용 가능한 split을 확인하고 샘플을 stream합니다.

```
from datasets import get_dataset_split_names, load_dataset

dataset_id = input("Repository ID from the dataset card: ").strip()
config = "hi-Deva-IN"

splits = get_dataset_split_names(dataset_id, config_name=config)
print("Available splits:", splits)

data = load_dataset(
    dataset_id, name=config, split=splits[0], streaming=True
)
print(next(iter(data)))
```


[Hugging Face Datasets](https://huggingface.co/docs/datasets/stream)은 streaming을 지원하므로 전체 코퍼스를 다운로드하지 않고도 독자가 레코드를 검사할 수 있습니다. 릴리스가 제공되면 재현 가능한 실험을 위해 dataset revision을 고정하세요.

첫 학습 연구에서는 고정된 토큰 예산으로 혼합물을 비교하고 자연/합성 및 네이티브/로마자 표기 비율을 기록하세요. 영어 역량 유지와 언어/스크립트 품질을 함께 추적하면서 언어별 결과를 기록하세요. held-out split을 만들 때는 여러 언어와 스크립트에 걸쳐 동일한 소스에서 파생된 레코드를 고려하세요.

더 완전한 레시피를 구축하려면 코퍼스를 [Chat & Instruction](https://huggingface.co/collections/nvidia/nemotron-chat-and-instruction-following) 또는 [Post-Training Blends](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) 컬렉션과 같은 Nemotron post-training 데이터와 결합하세요. 인도에 특화된 합성 데이터의 경우, 인도의 실제 분포에 기반한 영어 및 Hindi 합성 페르소나인 [Nemotron-Personas-India](https://huggingface.co/datasets/nvidia/Nemotron-Personas-India)을 사용해 NeMo Data Designer를 시작하세요.

### 생성 레시피 조정

NeMo Data Designer용 오픈 소스 [Nemotron Bharat translation plugin](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/tree/main/plugins/data-designer-nemotron-indic-translation)은 번역 및 평가 프롬프트, 선택적 피드백 기반 재작성, 문서 windowing, 스크립트 remapping 지원을 제공합니다. [configuration reference](https://github.com/NVIDIA-NeMo/DataDesignerPlugins/blob/main/plugins/data-designer-nemotron-indic-translation/docs/index.md)을 사용해 모델과 언어를 선택하고, 도메인에 맞게 프롬프트를 조정하며, 재작성 파라미터를 설정하세요. 함께 제공되는 [developer note](https://github.com/NVIDIA-NeMo/DataDesigner/pull/959)에서는 workflow와 엔지니어링 선택 사항을 설명합니다.

먼저 자체 소스 문서의 작은 샘플로 시작하세요. 코드, 수학, 서식의 번역 품질과 보존 여부를 검사한 다음, 규모를 확대하기 전에 네이티브 언어 검토자와 함께 프롬프트와 검사를 개선하세요. 생성과 코퍼스 수용을 별도의 단계로 다루세요. plugin은 후보와 점수를 반환하고, 어떤 레코드를 학습 코퍼스에 포함할지는 필터링 정책이 결정합니다.

## 범위와 한계 {#section-5}

번역은 소스 자료에 대한 접근성을 넓히지만, 원래 인도 언어로 작성된 텍스트를 대체하지 않으며 지역 지식과 문화를 충분히 다룬다고 보장하지도 않습니다. 이 코퍼스를 적절한 출처에서 확보한 자연 Indic 텍스트와 결합하고, 의도한 사용 사례에 맞게 그 결과 혼합물을 평가할 것을 권장합니다.

품질 평가는 주로 자동화되며, 언어별 범위가 서로 다른 샘플링된 사람의 검토로 보완됩니다. 12개 언어 workflow(그림 2)에서는 번역기가 judge 역할도 수행합니다. 유지된 텍스트에도 오류가 남아 있을 수 있으며, 10개 언어 workflow에는 더 폭넓은 원어민 검증이 필요합니다. source-ID 검사는 처리 중 발생한 중복을 제거하지만, 유사한 콘텐츠에 대한 의미적 deduplication은 아닙니다.

코퍼스와 함께 IIT Madras의 AI4Bharat는 NVIDIA의 지원을 받아 Indic 모델 역량을 위한 개방형이자 계속 발전하는 benchmark 모음인 Indic-LLM-Eval-Suite를 구축하고 있습니다. 이는 모델 개발자가 task 성능과 언어 품질을 모두 평가하는 데 도움이 됩니다. 동반 블로그 [Better Models Need Better Benchmarks: Introducing Indic-LLM-Eval-Suite](https://ai4bharat.github.io/indic-llm-eval-suite-blog/)을 참조하세요.

언어 구성을 살펴보고, 학습 혼합물을 테스트하거나, 새로운 도메인에 개방형 레시피를 적용해 보세요. 언어별 결과와 생성 workflow 개선 사항을 공유하면 커뮤니티가 이 작업을 발전시키는 데 도움이 될 수 있습니다.

## 추가 읽을거리 및 감사의 말 {#section-6}

동반 보고서인 [Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs](https://huggingface.co/datasets/nvidia/nemotron-indic-pretraining-report/blob/main/Scaling%20Synthetic%20Pre-training%20Data%20and%20Evaluation%20for%20Indic%20LLMs.pdf) 및 [Nemotron 3.5 Super VL and Lightning technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf)은 코퍼스 집계, 생성 방법, 실험 및 평가 세부 사항을 제공합니다. 평가는 이 작업의 또 다른 절반입니다. benchmark 측면에서는 AI4Bharat와 NVIDIA의 동반 블로그 [Better Models Need Better Benchmarks: Introducing Indic-LLM-Eval-Suite](https://ai4bharat.github.io/indic-llm-eval-suite-blog/)를 참조하세요.

이번 작업에는 NVIDIA, IIT Madras, AI4Bharat, Bodhan AI가 함께 참여했습니다. 또한 파이프라인에 사용된 모델 및 도구 기여에도 감사를 표합니다. 여기에는 Google's Gemma, Bodhan AI's Indic Translate, NeMo Data Designer가 포함됩니다. 추가 NVIDIA 데이터 리소스는 [Nemotron Pre-Training Datasets collection](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets)에서 이용할 수 있습니다.
