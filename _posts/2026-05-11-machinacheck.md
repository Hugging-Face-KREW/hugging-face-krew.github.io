---
layout: post
title: "MachinaCheck: AMD MI300X에서 다중 에이전트 CNC 제조 가능성 시스템 구축"
author: minju
categories: [Translation, HuggingFace]
slug: "machinacheck"
source_url: "https://huggingface.co/blog/lablab-ai-amd-developer-hackathon/machinacheck"
source_published_date: "2026-05-10"
source_published_at: "2026-05-10T18:44:11+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

> Source: https://huggingface.co/blog/lablab-ai-amd-developer-hackathon/machinacheck

* TOC
{:toc}
<!--toc-->

_이 글은 Hugging Face 블로그의 [MachinaCheck: Building a Multi-Agent CNC Manufacturability System on AMD MI300X](https://huggingface.co/blog/lablab-ai-amd-developer-hackathon/machinacheck)를 한국어로 번역한 글입니다._

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# MachinaCheck: AMD MI300X에서 다중 에이전트 CNC 제조 가능성 시스템 구축

lablab.ai에서 열린 AMD Developer Hackathon — 2026년 5월

## 우리가 해결한 문제

작은 CNC 기계 공방에 들어가 매니저에게 고객 작업을 수주할지 어떻게 결정하는지 물어보면 대답은 거의 항상 같다.

대답은 거의 언제나 같다: 도면을 인쇄하고 모든 치수를 손으로 읽고, 공방을 한 바퀴 돌며 어떤 도구가 사용 가능한지 확인하고, 기계가 필요한 공차를 지킬 수 있을지 추정한 뒤, 클립보드에 메모를 남긴다. 전체 과정은 도면당 30~60분이 걸린다. 주당 10~20 RFQ를 받는 바쁜 공방의 경우 피인성 분석에 숙련된 매니저의 시간이 5~20시간에 달한다.

가끔은 잘못 판단하기도 한다. 일을 수주하고 생산에 들어갔는데 도중에 필요한 나사탭이 없거나 밀링기가 핵심 특징의 공차를 지킬 수 없다는 것을 알게 된다. 부품은 버려지고, 고객은 불만하며, 기계 시간도 낭비된다.

우리는 이 문제를 완전히 없애기 위해 MachinaCheck를 만들었다.

## MachinaCheck가 하는 일

MachinaCheck는 다중 에이전트 AI 시스템이다. 고객들이 기계공방에 보내는 표준 CAD 포맷인 STEP 파일과 함께, 세 가지 간단한 입력값인 재료 유형, 필요한 공차, 그리고 나사 규격을 업로드한다. 30초 뒤에 완전한 제조 가능성 리포트를 얻을 수 있다. 이 보고서는 부품을 만들 수 있는지, 어떤 도구가 필요한지, 무엇이 누락되었는지, 생산을 시작하기 전에 어떤 조치를 취해야 하는지를 정확히 알려준다.

수작업으로 도면을 읽는 일도, 공방을 돌아다니는 일도, 추측으로 인해 생기는 위험도도 없다.

## 왜 AMD MI300X에서 이를 구축했는가

아키텍처를 설명하기 전에 이 점은 단지 기술적 선택이 아니라 비즈니스 요구사항이므로 별도로 다룰 가치가 있다.

제조 고객은 NDA를 체결한다. 그들의 STEP 파일은 수년 간의 엔지니어링 작업과 수백만 달러의 R&D를 상징하는 독점 기하를 담고 있다. 의료 기기의 구멍 패턴이나 항공우주 부품의 포켓 기하학은 기밀 지적 재산이다.

그 데이터를 OpenAI, Anthropic 또는 어떤 상용 API 엔드포인트로 보내는 것은 기밀 위반이다. 단호하게.

AMD Instinct MI300X가 이 방정식을 완전히 바꾼다. 192GB의 HBM3 VRAM과 5.3 TB/s의 메모리 대역폭으로, 우리는 Qwen 2.5 7B Instruct를 완전히 온프레미스에서 실행한다. 데이터가 매장의 인프라를 떠나지 않는다. STEP 기하 데이터가 제3자 서버로 전송되지 않는다. 고객의 IP는 자체에 머문다.

이것이 제조 맥락에서의 “privacy by design”이 실제로 의미하는 바다 — 체크박스가 아니라, 실제 엔터프라이즈 고객에게 제품의 실행 가능성을 확보하는 근본적인 아키텍처 결정이다.

## 에이전트 아키텍처

MachinaCheck는 LangChain으로 구축된 다섯 개 구성 요소의 파이프라인을 FastAPI로 조정한다.

### 구성요소 1 — STEP 파일 파서 (Pure Python, LLM 없음)

우리는 cadquery, OpenCASCADE를 기반으로 한 Python 라이브러리를 사용해 STEP 파일을 직접 파싱한다. 이를 통해 수학적으로 정확한 피처 추출을 얻는다:

- 지름과 깊이가 있는 모든 원기둥 구멍
- 평면 및 면적
- 모따기(Chamfer) 및 필렛(Fillet)
- 경계 박스 치수
- 전체 부피 및 표면적

이 추출은 수학적 기하를 직접 읽기 때문에 100% 정확하다 — 비전 모델도, OCR도, 근사도도 없다. Ø6.0mm 구멍은 출력에서 정확히 Ø6.0mm이다.

```
def
 
extract_features
(
step_file_path: 
str
) -> 
dict
:
    model = cq.importers.importStep(step_file_path)
    shape = model.val()
    bb = shape.BoundingBox()
    
    holes = {}
    
for
 face 
in
 model.faces().vals():
        adaptor = BRepAdaptor_Surface(face.wrapped)
        
if
 adaptor.GetType() == GeomAbs_Cylinder:
            radius = adaptor.Cylinder().Radius()
            diameter = 
round
(radius * 
2
, 
3
)
            holes[diameter] = holes.get(diameter, 
0
) + 
1

    
    
return
 {
        
"bounding_box_mm"
: {
"length"
: 
round
(bb.xlen, 
3
), ...},
        
"holes"
: [...],
        
"flat_surfaces_count"
: 
len
(flat_surfaces),
    }
```

### 에이전트 1 — 작업 분류기 (Qwen 2.5 7B)

추출된 기하 데이터와 사용자 입력 — 재료, 공차, 나사 규격 — 를 vLLM을 통해 AMD MI300X에서 실행되는 Qwen 2.5 7B로 전달한다.

에이전트는 다음과 같이 대답한다: "이 부품을 제조하기 위해 필요한 CNC 공정 및 도구는 무엇인가?"

제조 도메인 지식을 적용한다: Steel 304는 카바이드 공구가 필요하다. 원통형 구멍은 드릴이 필요하고 엔드밀이 필요하지 않다. ±0.005mm의 공차는 일반 밀링기가 아닌 고정밀 기계가 필요하다.

### 에이전트 2 — 도구 매칭기 (Pure Python)

이 에이전트는 LLM을 사용하지 않는다. 매장의 도구 재고 데이터베이스를 질의하고 필요한 도구를 사용 가능 여부와 대조한다. 순수 결정론적 로직 — 데이터베이스 조회, 비교, 결과. 데이터베이스 질의에 LLM이 필요하지 않으며, 여기에서 이를 사용하면 불필요한 지연과 환각 위험이 증가한다.

### 에이전트 3 — 타당성 결정 에이전트 (Qwen 2.5 7B)

일치 결과를 다시 Qwen으로 보낸다. 에이전트는 전체 상황을 고려해 구조화된 결정을 산출한다:

```
{

  
"decision"
:
 
"CONDITIONAL"
,

  
"confidence"
:
 
"HIGH"
,

  
"reason"
:
 
"All tools available except M10x1.5 tap"
,

  
"action_items"
:
 
[
"Purchase M10x1.5 tap ($15)"
]
,

  
"risk_flags"
:
 
[
"Verify spindle speed for Steel 304"
]
,

  
"estimated_setup_hours"
:
 
2.5


}
```

### 에이전트 4 — 보고서 생성기 (Qwen 2.5 7B)

최종 에이전트는 모든 것을 전문적인 제조 가능성 보고서로 종합한다. 총괄 상태, 경영진 요약, 부품 분석, 도구 상태, 기계 상태, 최종 권고안이 포함된다.

## AMD 스택

ROCm과 vLLM를 통해 AMD MI300X에서 Qwen 2.5 7B를 실행하는 것은 간단했다. AMD Developer Cloud의 vLLM Quick Start 이미지에는 모든 것이 미리 구성되어 있다.

```
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.5
```

`gpu-memory-utilization 0.5`를 적용하면 사용 가능한 192GB 중 약 96GB를 사용하게 되며 여유가 충분하다. 에이전트 호출에 대한 추론 지연은 평균 3초 미만이다.

LangChain은 OpenAI 호환 엔드포인트를 통해 vLLM에 연결한다:

```
from
 langchain_community.llms 
import
 VLLMOpenAI

llm = VLLMOpenAI(
    openai_api_base=
"http://localhost:8000/v1"
,
    openai_api_key=
"EMPTY"
,
    model_name=
"Qwen/Qwen2.5-7B-Instruct"
,
    temperature=
0.1
,
    max_tokens=
1000

)
```

## 결과

GrabCAD의 실제 STEP 파일로 테스트:

- 피처 추출: 최대 50 피처인 부품에서 1초 미만
- 전체 파이프라인(4개 에이전트 모두): 엔드투엔드 25~40초
- 결정 정확도: 모든 테스트 부품에 대해 제조 가능성 평가가 정확
- 프라이버시: STEP 기하 데이터가 외부로 전송되지 않음

## 우리가 배운 것

사고가 필요한 곳에서만 LLM을 사용하라. 에이전트 2(도구 매칭)는 순수 Python이다. 거기에 LLM을 두면 속도도 느려지고 비용도 커지며 신뢰도도 낮아진다. 데이터베이스 조회에 최적의 도구는 데이터베이스 쿼리이다.

구조화된 출력을 위한 프롬프트 엔지니어링이 중요하다. Qwen이 유효한 JSON을 안정적으로 출력하도록 하려면 프롬프트에 신중한 규칙이 필요했다 — 원통형 구멍은 드릴이 필요하고 엔드밀이 아니라는 점, 직경은 정확히 일치해야 한다는 점, 나사는 스레드가 지정될 때만 등장한다는 점 등을 명시적으로 밝히는 것.

AMD MI300X는 이 사용 사례에 대해 정말 인상적이다. 192GB VRAM은 필요하다면 훨씬 더 큰 모델을 실행할 수 있음을 의미한다. 프로덕션 배치를 위해서는 Qwen 2.5 72B가 충분히 잘 맞고 훨씬 더 나은 추론 품질을 제공한다.

## 사용해 보기

- HF Space: [huggingface.co/spaces/lablab-ai-amd-developer-hackathon/MachinaCheck](https://huggingface.co/spaces/lablab-ai-amd-developer-hackathon/MachinaCheck)

- GitHub: [github.com/SyedMuhammadSarmad/Manufacturing-Agent](https://github.com/SyedMuhammadSarmad/Manufacturing-Agent)

임의의 STEP 파일을 업로드하고 전체 파이프라인이 작동하는 모습을 확인해 보세요.

Syed Muhammad Sarmad와 Sabari Doss R이 AMD Developer Hackathon에서 제작, 2026년 5월.

Stack: Qwen 2.5 7B · AMD Instinct MI300X · ROCm · vLLM · LangChain · cadquery · FastAPI · Next.js · Hugging Face Spaces
