---
layout: post
title: "사실만이 아니라 출처를 올바르게: MCP 에이전트를 위한 출처 인식 검증"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/phhhj_BLOJbvnTFhVaZ5y.png
image: assets/images/blog/posts/2026-09-29-getting-the-source-right-not-just-the-fact-source/thumbnail.png
authors:
  - user: MultiverseComputingCAI
slug: "getting-the-source-right-not-just-the-fact-source"
source_url: "https://huggingface.co/blog/MultiverseComputingCAI/getting-the-source-right-not-just-the-fact-source"
source_published_date: "2026-09-29"
source_published_at: "2026-09-29T13:07:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Getting the Source Right, Not Just the Fact: Source-Aware Verification for MCP Agents](https://huggingface.co/blog/MultiverseComputingCAI/getting-the-source-right-not-just-the-fact-source)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/MultiverseComputingCAI/getting-the-source-right-not-just-the-fact-source -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 사실만이 아니라 출처를 올바르게: MCP 에이전트를 위한 출처 인식 검증

도구를 사용하는 LLM 에이전트는 더 이상 검색된 단일 구절만 읽지 않습니다. [Model Context Protocol (MCP)](https://modelcontextprotocol.io)을 통해 에이전트는 검색 도구를 호출하고, 구조화된 환자 또는 계정 기록을 확인하고, 데이터베이스를 조회하고, 메타데이터를 가져온 다음 이 모든 정보를 하나의 답변으로 엮을 수 있습니다. 따라서 사실성에 관한 일반적인 질문은 겉보기보다 더 미묘해집니다. [RAGAS faithfulness](https://github.com/explodinggradients/ragas)부터 MiniCheck, AlignScore, SummaC와 같은 세밀한 검사기에 이르기까지 LLM 답변을 검증하도록 구축된 대부분의 시스템은 증거를 하나로 모은 뒤, 해당 증거가 주장을 뒷받침하는지를 묻습니다. 일반적인 형태로는 각 주장을 뒷받침하는 MCP 도구 출력이 무엇인지, 또는 그것이 답변에서 지목한 출처인지 알려주지 않습니다.

최신 논문인 ProvenanceGuard: Source-Aware Factuality Verification for MCP-Based LLM Agents([Hugging Face](/blog/MultiverseComputingCAI/%5BHF-PAPER-LINK%5D)에서 읽거나, 그동안에는 [arXiv](https://arxiv.org/abs/2606.18037)에서 읽을 수 있음)는 이러한 공백을 다룹니다. 우리가 주목하는 실패 유형은 cross-source conflation이라고 부르는 현상입니다. 즉, 증거 어딘가에서는 참이지만 잘못된 출처에 귀속된 주장입니다. 출처를 구분하지 않는 검증기는 해당 사실이 증거 풀에 실제로 존재하기 때문에 이를 통과시킬 수 있습니다. 출처를 인식하는 검증기는 통과시켜서는 안 됩니다.

## 문제: 어딘가에서 뒷받침된다는 것과 올바른 출처가 뒷받침한다는 것은 다르다 {#section-1}

고객 지원 에이전트가 다음과 같이 답한다고 생각해 보겠습니다. "계정 기록에 따르면 이 요금제에는 30일 환불 기간이 포함되어 있습니다." 환불 기간 자체는 실제로 존재할 수 있지만, 답변이 지목한 계정 기록이 아니라 정책 문서에 명시되어 있을 수 있습니다. 두 문서를 하나로 합치면 해당 주장은 뒷받침되는 것처럼 보입니다. 그러나 둘을 분리하면 출처 귀속이 잘못되며, 데이터에 민감한 환경에서는 잘못된 출처 귀속이 잘못된 사실만큼이나 큰 피해를 줄 수 있습니다. 임상 에이전트에서도 같은 패턴이 나타납니다. 환자 이력 도구에서 가져온 환자별 약물 정보가 답변에서 의학 문헌의 발견으로 제시되는 순간 오해를 일으킬 수 있습니다.

[![Diagram contrasting source-blind pooled support with source-aware verification, using a refund-window claim that is supported by a policy document but attributed to the account record.](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/svnUpzhFgTpLqOF_bRDHl.png)](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/svnUpzhFgTpLqOF_bRDHl.png)

하나의 MCP 출처가 주장을 뒷받침하지만 답변은 이를 다른 출처에 귀속할 수 있습니다. 출처를 구분하지 않는 점수화는 통합된 증거에서 뒷받침을 확인하고 이를 통과시키지만, ProvenanceGuard는 지원하는 출처가 답변에서 명시하거나 암시한 출처와 일치하는지를 별도로 확인합니다. 출처: 논문 Figure 1.

이 때문에 유용하기는 하지만 faithfulness 점수만으로는 MCP 에이전트에 충분하지 않습니다. 답변에는 출처 정보가 포함되며, 때로는 명시적으로("계정 기록에 따르면") 드러나고 때로는 암묵적으로 드러납니다. ProvenanceGuard는 주장과 출처 사이의 연결을 검사할 수 있는 상태로 유지합니다.

## ProvenanceGuard의 기능 {#section-2}

ProvenanceGuard는 블랙박스 MCP 에이전트 위에 배치되는 생성 후 검증 계층입니다. 에이전트가 답변을 생성한 뒤 실행되며, 증거를 하나의 익명 컨텍스트로 합치지 않습니다. 대신 파이프라인 전체에서 출처 식별자를 유지합니다. 에이전트를 재학습하지 않고 캡처된 MCP trace를 읽으며, 여기에는 도구 출력과 해당 source ID가 포함됩니다. 그런 다음 다섯 가지 작업을 순서대로 수행합니다. 답변을 구체적인 주장으로 나누고, 각 주장과 가장 관련성이 높은 출처를 찾고, 해당 출처가 실제로 주장을 뒷받침하는지 확인하고, 그 출처를 답변이 명시하거나 암시한 출처와 비교한 뒤, 주장별 출처 판정과 답변 전체 수준의 허용 또는 차단 결정을 모두 출력합니다.

[![Pipeline diagram showing the agent producing a draft answer and trace, then ProvenanceGuard decomposing claims, routing to sources, checking support with NLI, calibrating, checking attributions, and either allowing the answer or sending it to repair.](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/nDchWzRhBSmscjaa8w5G5.png)](https://cdn-uploads.huggingface.co/production/uploads/668e37fd9c9aa124a3c867e8/nDchWzRhBSmscjaa8w5G5.png)

검증 흐름. 출처 식별자는 증거를 하나로 모으는 대신 분해, 라우팅, 지원 점수화, 귀속 확인, 복구 과정 전체에서 유지됩니다. 차단된 답변은 RARR 스타일 복구를 거친 뒤 다시 검증할 수 있습니다. 출처: 논문 Figure 2.

몇 가지 설계 선택은 짚고 넘어갈 가치가 있습니다. 논문의 실험에서는 캡처된 trace를 통제된 오프라인 환경에서 처리할 수 있도록 로컬 모델을 사용했습니다. [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)은 관련 출처를 찾고, [DeBERTa NLI verifier model](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)은 해당 출처가 주장을 뒷받침하는지 확인하며, 로컬 언어 모델은 답변을 주장 단위로 분해합니다. 검증기는 리터럴 값도 면밀히 확인합니다. 숫자, 날짜 또는 식별자가 출처에 없으면 문장이 그럴듯하게 들린다는 이유만으로 통과할 수 없습니다. 보정된 결정 단계는 이러한 신호를 결합합니다. 답변이 차단되면 [RARR](https://arxiv.org/abs/2210.08726) 스타일의 복구 단계에서 출처에 근거한 수정이나 안전한 대체 답변을 시도할 수 있으며, 검증기는 그 결과를 다시 확인합니다.

앞서 언급한 모델은 우리가 평가한 설정일 뿐 ProvenanceGuard의 필수 요건은 아닙니다. 동일한 주장, 출처 및 결정 단계는 팀이 클라우드 서비스를 선호하는 경우 호스팅 모델에 맞게 조정할 수 있지만, 새로운 설정에는 별도의 테스트와 보정이 필요합니다. 보고된 결과는 로컬 구성에서 얻은 것입니다. 이 구성의 보수적인 결정 정책은 가능한 한 빠른 답변을 생성하는 것보다 출처를 올바르게 확인하는 것이 중요한 데이터 민감 검토에 적합합니다.

## 결과 {#section-3}

환자 기록, 연구 논문 및 기타 도구를 사용한 의료 에이전트의 답변을 대상으로 ProvenanceGuard를 테스트했습니다. 이를 통해 실제 trace 281개를 연구할 수 있었습니다. 환자 기록에서 나온 사실과 일반 연구에서 나온 사실을 동일한 출처로 취급할 수 없기 때문에 의학은 유용한 테스트 환경입니다. 에이전트가 도구 출력과 source ID를 기록하는 경우 이 방법은 다른 분야에도 사용할 수 있습니다. 주요 테스트에서는 시스템 개발에 사용된 데이터에서 따로 분리한 40개 답변의 주장 361개를 인간 전문가가 확인했습니다.

가장 직접적인 결과는 다음과 같습니다. 전문가들은 139개의 주장을 통과시켜서는 안 된다고 판단했고, ProvenanceGuard는 이 중 138개를 찾아냈습니다. 하나는 통과시켰습니다. 또한 전문가들이 뒷받침된 것으로 판단한 주장 67개를 보류하여 검토 또는 복구 대상으로 보냈습니다. 이는 우리가 테스트한 신중한 설정을 반영합니다. 즉, 뒷받침되지 않은 주장을 통과시키기보다 뒷받침된 주장 일부를 재검토 대상으로 보내는 방식을 선호합니다. 식별 가능한 출처가 있는 주장에 대해서는 이 테스트에서 약 86%의 확률로 올바른 출처를 선택했습니다.

동일한 주장에 대해 네 가지 다른 지원 검사기도 실행했습니다. ProvenanceGuard는 차단해야 할 주장을 얼마나 잘 포착하면서 불필요한 차단은 피하는지를 측정하는 논문의 지표에서 가장 높은 점수를 기록했습니다. 이 비교에 포함된 다른 검사기들은 각 주장을 뒷받침하는 도구 출력이 무엇인지 알려주지 않았습니다. ProvenanceGuard는 이 연결을 기록하므로 검토자는 각 주장에 대해 어떤 출처를 확인했으며 어떤 결정을 내렸는지 볼 수 있습니다.

| 검증기 | Reject/block F1 | 주장-출처 ID 출력 |
| --- | --- | --- |
| ProvenanceGuard (ours) | 0.802 | Yes |
| MiniCheck | 0.783 | No |
| RAGAS Faithfulness | 0.758 | No |
| AlignScore | 0.662 | No |
| SummaC-ZS | 0.436 | No |

동일한 홀드아웃 주장 묶음에 대한 이진 지원 지표입니다. ProvenanceGuard는 차단 성능에서 출처를 구분하지 않는 baseline과 대등하거나 더 나은 성능을 보이는 동시에 주장별 출처 판정을 생성합니다. 출처: 논문 초록 및 Table III.

## 출처가 비슷해 보일 때 주장 확인하기 {#section-4}

서로 유사한 출처가 여러 개 있는 별도의 더 어려운 테스트에서 ProvenanceGuard는 차단할 주장을 결정하는 F1 0.846을 기록했지만, 주장의 50.3%에서만 정확한 출처를 식별했습니다. 유사한 출처를 구분하는 일은 여전히 중요한 개선 과제로 남아 있습니다.

잘못된 출처 귀속에 초점을 맞춘 통제 테스트도 진행했습니다. 지원하는 증거는 그대로 둔 채 50개 사례에서 지목된 출처를 변경했습니다. ProvenanceGuard는 이 50건의 변경을 모두 찾아냈습니다. 이는 명확한 출처 오류를 감지할 수 있음을 보여주지만, 더 어려운 테스트는 여러 개의 그럴듯한 출처 중 하나를 선택하는 일이 얼마나 어려운지를 보여줍니다.

## 차단된 답변 복구하기 {#section-5}

차단된 답변에 대해 조치할 수 없다면 차단은 유용하지 않습니다. RARR 스타일 복구 루프를 연결한 전체 trace 실행에서는 차단된 답변 173개를 모두 해결했습니다. 다만 이 중 144개는 실질적인 재작성 대신 fallback 텍스트로 끝났습니다. 이는 검증할 수 없는 답변을 만들어내기보다 피하도록 시스템이 선택한 결과입니다. 재구성된 다중 출처 테스트 trace에서는 새로운 복구 실행을 통해 처음 차단된 답변 59개를 모두 해결했으며, 최종 fallback은 두 건뿐이었습니다. 오프라인 게이트로서의 오버헤드는 크지 않습니다. 보고된 로컬 구성에서 답변당 대략 0.5초이며, NLI 및 라우팅 호출 자체는 수십 밀리초 수준입니다.

## 이것이 Multiverse Computing에 적합한 이유 {#section-6}

에이전트가 단일 구절 RAG에서 다중 도구 MCP 설정으로 이동함에 따라, 어떤 출처에서 사실이 실제로 나왔는지에 관한 질문은 더 이상 각주가 아니라 사실성이 의미하는 바의 일부가 됩니다. ProvenanceGuard는 이러한 출처 연결을 주장별로 확인할 수 있도록 드러냅니다. Multiverse Computing에 이는 필요할 때 민감한 trace를 통제된 환경에 유지하면서 기존 에이전트를 점검할 수 있는 방법을 의미합니다. 의료 연구는 하나의 사용 사례이며, 에이전트의 trace에 도구와 출처가 보존되는 곳이라면 동일한 접근 방식을 적용할 수 있습니다.

이러한 적용은 이미 [NVIDIA NVFlow](https://github.com/NVIDIA/nvflow/pull/9)에서 확인할 수 있습니다. 이 프로젝트는 finance agent에 선택적 grounding-verification 단계를 통합했습니다. 에이전트가 검색한 SEC 발췌문을 기준으로 완료된 답변을 확인하고, 기존 rollout이나 training data를 변경하지 않은 채 별도의 결정을 저장합니다. NVFlow의 기여는 ProvenanceGuard의 출처 인식 검증 접근 방식을 사용하며, 앞서 설명한 복구 루프는 더 광범위한 연구 시스템에 속합니다.

ProvenanceGuard는 또한 [presented as a poster at the Agentic AI Summit 2026 at UC Berkeley](https://github.com/aalvsz/provenanceguard/blob/main/poster/ProvenanceGuard_Agentic_AI_Summit_2026_Berkeley.pdf).

라우팅 및 NLI 도출 과정, calibration ablation, 다중 출처 stress slice, 전체 결과 표를 포함한 기술적 세부 사항을 모두 확인하고 싶으신가요? [Hugging Face](https://huggingface.co/papers/2606.18037)에서 전체 논문을 읽거나, 자체 에이전트에 출처 인식 검증을 적용하는 방법을 논의하려면 저희 팀에 문의해 주세요.
