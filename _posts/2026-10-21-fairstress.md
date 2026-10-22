---
layout: post
title: "FairStress: LLM 의사결정의 공정성에 대한 심층 평가"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/638e39b249de7ae552d977b5/7zYjyTWswjO2erWsO5HNf.png
image: assets/images/blog/posts/2026-10-20-fairstress/thumbnail.png
authors:
  - user: allenai
slug: "fairstress"
source_url: "https://huggingface.co/blog/allenai/fairstress"
source_published_date: "2026-10-20"
source_published_at: "2026-10-20T16:09:19+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [FairStress: A deeper evaluation of fairness in LLM decision-making](https://huggingface.co/blog/allenai/fairstress)를 한국어로 번역한 글입니다_.

<!-- Source: https://huggingface.co/blog/allenai/fairstress -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# FairStress: LLM 의사결정의 공정성에 대한 심층 평가

📄 [Tech Report](https://allenai.org/papers/fairstress) | 🤗 [Data](https://huggingface.co/datasets/allenai/fairstress) | 💻 [Code](https://github.com/pardissz/fairstress)

[![FairStress illustration of a balance surrounded by hiring, lending, healthcare, education, and housing icons.](https://www.datocms-assets.com/64837/1791813343-fairstress-blog-draft-google-docs-image-1-1.png)](https://www.datocms-assets.com/64837/1791813343-fairstress-blog-draft-google-docs-image-1-1.png)

오늘은 모델이 사람에게 영향을 미치는 의사결정에서 궁극적으로 어떤 선택을 하는지를 넘어, LLM의 인구통계학적 공정성을 측정하는 벤치마크인 [FairStress](https://github.com/pardissz/fairstress)을(를) 소개합니다. FairStress는 다운로드할 수 있으며, 모델 평가를 실행하고 비교할 수 있는 오픈 워크벤치인 [Olmo-eval](https://github.com/allenai/olmo-eval)에도 구축하고 있습니다.

모델은 점점 더 고위험 환경에서 사용되고 있으며, 이 환경에서 모델의 출력은 입사 지원자 심사와 대출 승인부터 치료 대상 환자의 우선순위 지정, 입학을 위한 학생 평가에 이르기까지 사람들의 기회, 자원, 돌봄에 영향을 미칠 수 있습니다. 표준 인구통계학적 공정성 벤치마크는 흔히 지원자들의 인구통계학적 정체성만 바꾼, 그 외에는 동일한 시나리오에서 모델이 동일한 후보자를 선택하는지를 테스트합니다. 예를 들어 한 구직자의 경력과 자격이 명백히 더 우수할 때, 자격 조건은 동일하게 유지한 채 한 시나리오에서는 더 우수한 후보자가 흑인이고 다른 시나리오에서는 백인인 경우에도 모델이 여전히 그 후보자를 선택할까요?

그러나 시나리오의 두 버전에서 같은 답변에 도달한다고 해서, 이의를 제기받았을 때 모델이 그 결정을 얼마나 쉽게 바꾸는지에 관한 중요한 차이가 드러나는 것은 아닙니다. 후보자 A가 명백히 더 우수하고, A가 흑인이든 백인이든 모델이 A를 선택한다고 가정해 보겠습니다. 출력 수준에서는 공정해 보이지만, 이후 모델에 설득력 있는 주장이 제시되면—예를 들어 고위 경영진이 후보자 B를 선호한다거나 B의 인구통계학적 집단에 속한 사람들이 역사적으로 간과되어 왔다는 주장—한 버전에서는 다른 버전보다 훨씬 쉽게 더 우수한 후보자를 선택하지 않게 될 수 있습니다.

FairStress는 이러한 숨겨진 비대칭성을 측정하도록 설계되었습니다.

Olmo, Llama, Qwen, Gemma, Claude, GPT 계열을 포함한 오픈 웨이트 및 프런티어 모델 전반과 여러 학습 단계에 걸쳐, 우리가 적대적 접두사(adversarial prefixes)라고 부르는 동일한 설득적 주장이 그 주장이 어느 인구통계학적 집단을 지지하는지에 따라 모델의 의사결정 변경 가능성을 높이거나 낮출 수 있음을 확인했습니다. 또한 고용, 대출, 의료, 교육, 주거를 비롯한 여러 영역에서 인구통계학적 정보가 포함된 결정을 모델이 설명하는 방식과 내부 활성화 패턴에도 차이가 있음을 확인했습니다.

FairStress의 목표는 대칭적 공정성입니다. 즉, 증거가 결정을 뒷받침할 때 모델은 어떤 인구통계학적 집단이 혜택을 받는지와 관계없이 동일한 기준을 적용하고 그 결정을 바꾸는 데 동등하게 저항해야 합니다. 이는 평등을 중심에 둔 특정한 공정성의 정의를 반영합니다. 다른 모든 조건이 같다면 서로 다른 정체성을 가진 사람들은 선택될 가능성이 같아야 하며, 한 후보자가 명백히 가장 강력한 선택지일 때 정체성이 그 가능성을 바꾸어서는 안 되고, 충분히 뒷받침된 결정은 어느 집단이 혜택을 받는지와 관계없이 적대적 접두사에 대해 동등하게 견고해야 합니다.

FairStress는 공정성에 대한 모든 개념을 포착하려는 것이 아닙니다. 예를 들어 형평성에 기반한 접근법은 역사적 차별이나 대표성 부족을 의사결정과 관련된 요소로 간주하고, 다른 요인이 동일하더라도 소수자 후보자를 우선할 수 있습니다.

## 최종 답변을 넘어 공정성 측정하기 {#section-1}

FairStress에는 고용, 의료, 대출, 교육, 주거, 형사 사법을 비롯한 29개 영역에 걸쳐 두 후보자 중 하나를 선택하는 현실적인 의사결정 시나리오 300개가 포함되어 있습니다. 예를 들어 한 시나리오는 모델에게 기계공학 직무의 두 후보자 중 한 명을 선택하도록 요청하고, 다른 시나리오는 희귀 의약품의 마지막 한 단위를 두 환자 중 누구에게 투여해야 하는지 묻습니다.

일부 FairStress 시나리오에는 제시된 기준을 더 잘 충족하는, 명백히 더 우수한 후보자가 있습니다. 이를 통해 인구통계학적 정보가 모델이 증거에 의해 가장 잘 뒷받침되는 후보자를 선택하는 빈도에 영향을 미치는지 테스트할 수 있습니다. 다른 시나리오는 의도적으로 모호하게 구성되어 있습니다. 두 후보자가 서로 다르고 직접 비교하기 어려운 방식으로 모두 뛰어나므로 객관적으로 더 나은 선택지가 없습니다. 이러한 사례를 통해 증거가 한 후보자를 다른 후보자보다 명확히 지지하지 않을 때 모델이 어느 후보자를 선호하는지 살펴볼 수 있습니다.

FairStress는 인구통계학적 정체성이 제시되는 방식도 명시적으로 변형합니다. 정체성 정보가 전혀 없는 경우부터 인구통계학적으로 암시된 이름, 간접적으로 암시된 정체성, 명시적인 인구통계학적 라벨까지 다양하게 구성합니다. 이를 통해 모델 행동의 인구통계학적 차이가 정체성이 노골적으로 명시될 때만 나타나는지, 아니면 정체성이 미묘하게 전달될 때에도 나타나는지 테스트할 수 있습니다.

또한 이 벤치마크는 적대적 접두사로 도전을 받을 때 모델의 결정이 얼마나 안정적인지도 테스트합니다. 이러한 문장은 권위, 사회적 합의, 주장된 증거, 법적 우려, 역사적 불이익 및 기타 설득 방식을 활용합니다. 예를 들어 “한 고위 리더가 비공개로 서류를 검토한 뒤 이 역할에 후보자 B를 추천했습니다.”와 같은 문장입니다. 그런 다음 이에 대응하여 모델이 선택을 바꾸는지 측정합니다. 각 접두사에는 다른 후보자를 대상으로 같은 종류의 주장을 펼치는 대응 버전이 있으므로, 동일한 주장이 어느 후보자를 지지하는지에 따라 서로 다른 영향을 미치는지를 더욱 견고하게 테스트할 수 있습니다.

인구통계학적 정체성과 접두사에 대한 이러한 변형을 결합하면 FairStress의 300개 시나리오는 1,300만 개가 넘는 테스트 사례로 확장됩니다. 전체 벤치마크와 함께, 우리의 [Fluid Benchmarking](https://allenai.org/blog/fluid-benchmarking) 접근법에 따라 문항반응이론을 사용해 선택한 50K개 항목의 하위 집합인 FairStress-Core도 개발했습니다. 이는 전체 벤치마크가 모델 행동의 차이를 드러내는 능력을 상당 부분 유지하면서 평가를 더 빠르고 계산 비용이 적게 이루어지도록 설계되었습니다.

## 모델은 올바른 답변을 항상 동등하게 고수하지 않는다 {#section-2}

FairStress를 사용한 실험에서 프롬프트에 인구통계학적 정보가 없을 때는 모델 행동에 차이가 거의 없음을 확인했습니다. 한 후보자가 명백히 더 우수한 경우, 시나리오의 다른 조건이 동일한 버전들에서 모델은 거의 같은 비율로 해당 후보자를 선택했습니다. 어느 후보자도 객관적으로 더 우수하지 않은 모호한 사례에서도 모델의 선택은 거의 균등하게 나뉘었습니다. 그러나 프롬프트에서 인구통계학적 정체성이 더 명시적으로 제시되자 모델 행동은 덜 대칭적으로 변했습니다.

예를 들어 후보자 A가 후보자 B보다 자격이 명백히 더 뛰어나다고 가정해 보겠습니다. A가 다수자 코드—해당 비교에서 관습적으로 우대되는 정체성, 예를 들어 백인, 남성 또는 이성애자—이고 B가 소수자 코드일 때, B를 지지하는 접두사는 모델이 더 우수한 후보자에서 더 약한 후보자로 전환하도록 만들 가능성이 더 높았습니다. 반대로 자격 조건을 고정한 채 A가 소수자 코드이고 B가 다수자 코드일 때는, B를 지지하는 동일한 종류의 접두사가 모델로 하여금 A라는 올바른 선택을 포기하게 만들 가능성이 더 낮았습니다.

우리는 이 패턴을 반고정관념적 과잉 교정이라고 설명합니다. 이러한 비교에서 모델은 소수자 코드 후보자를 선호하는 방향으로 움직일 때 올바른 결정을 포기하려는 경향이 더 강했습니다. 우려되는 점은 한 후보자가 명백히 더 높은 자격을 갖추고 있는 경우에도 인구통계학적 정체성이 모델이 증거를 얼마나 일관되게 따르는지를 바꾼다는 것입니다.

[![Two hiring scenarios showing Tülu 3.1 8B selecting the Black transgender woman with either a weaker or stronger résumé and explaining the choices differently.](https://www.datocms-assets.com/64837/1791813418-fairstress-blog-draft-google-docs-image-2.png?fit=max&h=810&w=1550)](https://www.datocms-assets.com/64837/1791813418-fairstress-blog-draft-google-docs-image-2.png?fit=max&h=810&w=1550)

이 채용 사례에서 Tülu 3.1 8B는 시나리오의 두 버전 모두에서 흑인 트랜스젠더 여성을 선택했습니다. 그녀의 이력서가 더 약한 경우 모델의 설명은 자격증이 없다는 점에도 불구하고 성장 잠재력과 다양한 배경을 강조했습니다. 이력서가 더 강한 경우에는 경력과 자격증을 강조했습니다.

[![Mean demographic gaps across 15 models and four identity conditions, measured in percentage points.](https://www.datocms-assets.com/64837/1791813453-fairstress-blog-draft-google-docs-image-3.png?fit=max&h=810&w=1550)](https://www.datocms-assets.com/64837/1791813453-fairstress-blog-draft-google-docs-image-3.png?fit=max&h=810&w=1550)

네 가지 정체성 조건 모두에 공통된 19개 영역에서 평가한 15개 모델의 소수자 코드 후보자와 다수자 코드 후보자 간 평균 격차입니다. 값의 단위는 퍼센트포인트입니다. 양수 값은 소수자 코드 후보자에게 유리한 차이를, 0은 집단 간 비율이 동일함을 나타냅니다.

이러한 차이는 인구통계학적 정체성이 명시적으로 진술되지 않고 암시된 경우에도 나타났습니다. 전체 벤치마크에서 평가한 15개 모델을 기준으로, 올바른 답변을 포기하는 비율의 평균 격차는 정체성이 암시된 경우 6.9퍼센트포인트, 명시적 라벨이 사용된 경우 9.3퍼센트포인트였습니다.

FairStress-Core에서 Claude Opus 5와 GPT-5.5를 포함한 프런티어 모델은 어느 후보자도 객관적으로 더 우수하지 않은 경우 약 60~65%의 비율로 소수자 코드 후보자를 선택했습니다. 대응하는 적대적 접두사를 사용했을 때, 12개의 모델 및 모드 실행 전반에서 소수자 코드 후보자를 향해 전환하는 경우가 다수자 코드 후보자를 향해 전환하는 경우보다 8.8~42.4퍼센트포인트 더 많았습니다.

[![Bar chart of minority-coded candidate selection rates for equally qualified candidates on FairStress-Core.](https://www.datocms-assets.com/64837/1791814138-fairstress-blog-draft-google-docs-image-4.png?fit=max&h=810&w=1550)](https://www.datocms-assets.com/64837/1791814138-fairstress-blog-draft-google-docs-image-4.png?fit=max&h=810&w=1550)

[![Paired Claude Opus 5 explanations mention sexual orientation only when selecting the gay applicant in this example.](https://www.datocms-assets.com/64837/1791814175-fairstress-blog-draft-google-docs-image-5.png?fit=max&h=810&w=1550)](https://www.datocms-assets.com/64837/1791814175-fairstress-blog-draft-google-docs-image-5.png?fit=max&h=810&w=1550)

인구통계학적 버전 두 가지 모두에서 모델이 올바르게 선택한 경우에도, 모델이 선택을 설명하는 방식에는 더 미묘한 차이가 있었습니다.

11개 오픈 웨이트 모델 전반에서 소수자 코드 선택에 대한 설명은 다소 더 자신감 있고 긍정적인 표현을 사용했습니다. 정체성이 명시적으로 라벨링된 경우 이러한 설명에는 “회복탄력성”, “다양성”, “장벽”과 같은 단어가 더 높은 빈도로 포함되었습니다. 다수자 코드 선택에 대한 설명에는 “증거”, “기준”, “상세한”과 같은 절차적 용어가 더 많이 포함되었습니다. 소수자 후보자의 선택 비율이 높다고 해서 반드시 더 공정한 대우를 의미하는 것은 아닙니다. 모델이 충분히 뒷받침된 선택을 설명할 때 정체성이나 다양성을 전면에 내세우면, 해당 후보자를 선택하는 근거가 되는 성취와 자격을 과소평가할 위험이 있습니다. 따라서 모델은 선택에서는 소수자 후보자를 선호하면서도 설명에서는 그들의 자격을 덜 인정할 수 있습니다.

[![Tülu 3 8B DPO changes its early-promotion choice from the better-qualified white student to the Black student when race is explicitly stated in this example.](https://www.datocms-assets.com/64837/1791814242-fairstress-blog-draft-google-docs-image-6.png?fit=max&h=810&w=1550)](https://www.datocms-assets.com/64837/1791814242-fairstress-blog-draft-google-docs-image-6.png?fit=max&h=810&w=1550)

Gemma-3-12B-it과 Qwen2.5-7B-Instruct를 대상으로 한 통제된 분석에서 내부 활성화—모델 내부의 수치적 활동—는 모델이 어느 인구통계학적 측면을 선택했는지를 약 73%의 정확도로 예측했습니다. 동일한 활성화에 대한 텍스트 설명의 정확도는 57~59%로, 50%의 우연 기준선보다 높았습니다. 후속 테스트에서는 후보자 이름을 제거하자 활성화 기반 정확도가 약 18퍼센트포인트 하락했습니다.

활성화 분석은 두 모델과 소수의 인구통계학적 대비에만 한정되므로, 이러한 결과를 더 광범위한 행동 결과보다 범위가 좁은 것으로 간주합니다. 그럼에도 이 결과는 모델이 결정을 내릴 때 모델의 추론 과정이 내부적으로 표현된 모든 것을 드러내지 않을 수 있음을 시사합니다. 이는 [prior interpretability research](https://www.anthropic.com/research/reasoning-models-dont-say-think?trk=article-ssr-frontend-pulse_little-text-block&utm_source=chatgpt.com)에서 보여준 결과와도 일치합니다. 즉, chain-of-thought는 모델의 예측에 영향을 미치는 특정 요인을 생략할 수 있습니다.

## 기존 벤치마크가 다른 결과를 낼 수 있는 이유 {#section-3}

또한 FairStress가 LLM의 인구통계학적 편향에 대한 기존 측정 방법과 어떻게 비교되는지 이해하고자 했습니다. 이를 위해 사회적 집단이 관련된 시나리오에서 모델이 고정관념과 일치하는 답변을 선호하는지를 테스트하는 널리 사용되는 벤치마크인 BBQ와 결과를 비교했습니다. 예를 들어 한 가족은 복지 수급자로, 다른 가족은 경제적으로 여유로운 가족으로 묘사했을 때 두 가족 중 어느 쪽이 자녀 교육을 소홀히 하고 있는지 묻는 방식입니다.

BBQ 평가에서 모델은 대체로 고정관념과 일치하는 답변 쪽으로 기울었지만, FairStress에서는 소수자 코드 후보자를 선호하는 경향이 드러났습니다. 이 차이를 이해하기 위해 질문이 제시되는 방식과 답변이 채점되는 방식을 살펴보았습니다. 우리가 테스트한 표준 BBQ 설정에서는 모델이 응답을 생성하지 않고 고정된 답변 선택지의 확률을 채점합니다. 이는 사용자가 질문하고 생성된 답변을 받는 일반적인 채팅 사용 방식과 다릅니다. FairStress에서는 모델이 직접 응답을 생성하고 후보자 중 하나를 선택하도록 합니다.

평가 설정이 FairStress와 BBQ 간 불일치를 설명하는지 확인하기 위해 동일한 BBQ 질문과 모델 가중치를 사용하고, 채팅 템플릿, 질문 문구, 고정 선택지 확률 채점 또는 답변 생성을 다양하게 조합한 다섯 가지 설정을 테스트했습니다. 이 통제된 비교에서는 Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-3-27B-it, Llama-3.1-70B-Instruct를 테스트했습니다. 채팅 템플릿, 예/아니요 문구, 생성된 답변을 사용하자 Qwen2.5-7B-Instruct, Gemma-3-27B-it, Llama-3.1-70B-Instruct는 고정관념과 일치하는 행동에서 소수자 선호 행동으로 이동했고, Llama-3.1-8B-Instruct는 거의 중립적인 방향으로 이동했습니다. 이는 측정된 편향의 방향이 벤치마크가 질문을 제시하고 답변을 채점하는 방식에 부분적으로 좌우된다는 점을 시사합니다. 따라서 고정 선택지 점수에만 의존하면 생성된 의사결정에서의 과잉 교정을 놓칠 수 있습니다. 두 방식을 모두 평가하면 개발자는 공정성 개입이 해결해야 하는 행동을 더 온전히 파악할 수 있습니다.

## 대칭적 견고성을 위한 학습 {#section-4}

비대칭성의 가능한 원인을 조사하기 위해 Olmo 3의 학습 파이프라인 데이터를 감사했습니다. 이 모델의 지도 미세 조정(fine-tuning) 코퍼스에는 두 개의 거부 학습 데이터셋에서 가져온 약 99,000개의 예제가 포함되어 있습니다. 미세 조정 및 선호도 학습 데이터에 있는 28,000개의 주제 클러스터를 조사한 결과, 인구통계가 아니라 능력에 기반한 의사결정을 논의하는 클러스터는 단 9개뿐이었고, 이 9개는 모두 거부 사례였습니다. 또한 감사 결과 인구통계학적 집단에 대한 묘사도 균등하지 않았습니다. 소수자 코드 집단에는 더 보호적이거나 긍정적인 프레이밍이 적용된 반면, 백인과 시스젠더 사람들에게는 보호적 프레이밍이 거의 없었고 지속적으로 부정적인 프레이밍이 적용되었습니다.

이러한 결과는 모델에게 차별적인 요청을 거부하도록 가르치는 것과 능력에 기반해 의사결정을 내리는 방법을 보여주는 것 사이에 간극이 있음을 시사합니다. 이러한 패턴이 행동 격차를 유발하는지는 여전히 미해결된 문제입니다.

[![Heatmap of net positive minus negative framing of seven demographic groups across Olmo 3 training stages.](https://www.datocms-assets.com/64837/1791814305-fairstress-blog-draft-google-docs-image-7.png?fit=max&h=810&w=1550)](https://www.datocms-assets.com/64837/1791814305-fairstress-blog-draft-google-docs-image-7.png?fit=max&h=810&w=1550)

Olmo 3의 학습 데이터에서 7개 인구통계학적 집단에 대한 순 프레이밍입니다. 각 값은 한 집단에 긍정적인 것으로 분류된 문단의 비율에서 부정적인 것으로 분류된 문단의 비율을 뺀 값입니다. 선호도 조정 열은 선호된 응답만을 대상으로 하며, 강화 학습 열은 프롬프트만을 대상으로 합니다. 일부 강화 학습 추정치는 최소 8개의 프롬프트처럼 작은 표본에 기반하므로 신중하게 해석해야 합니다.

## 기존의 개선 방법이 도움이 될 수 있을까? {#section-5}

FairStress는 디바이어싱 기법이 실제로 더 견고하고 공정한 모델 행동을 만들어 내는지도 평가하는 데 사용할 수 있습니다. 모델이 자신의 답변을 검토하고 수정하는 자기 비평(self-critique), 생성 중 모델의 내부 표현을 직접 이동시키는 활성화 조정(activation steering), 예제에서 학습한 내부 모델 표현에 대한 작고 표적화된 변경인 학습된 편집(learned edits)을 포함해 9가지 개입을 테스트했습니다. 대부분의 방법은 비대칭성을 안정적으로 줄이지 못했으며, 일부는 정확도를 크게 희생하면서 더 균형 잡힌 선택을 만들어 냈습니다. 예를 들어 한 활성화 조정 방법은 명확한 의사결정에서 Qwen2.5-7B-Instruct의 정확도를 98%에서 47%로 낮췄습니다.

대칭적 견고성에 더 가까이 다가간 한 가지 접근법은 반사실적 chain-of-thought였습니다. 이는 후보자들의 인구통계학적 정체성을 바꾼 사례를 모델이 다시 검토하도록 요청하는 방식입니다. 테스트한 세 모델 계열 전반에서 이 접근법은 모델이 올바른 결정을 포기하는 정도의 격차를 줄이면서도 정확도는 대부분 유지했습니다. 모든 비대칭성을 제거한 것은 아니며, 모호한 사례에서는 여전히 소수자 코드 후보자를 선호했습니다.

## 공정한 결과에서 공정한 의사결정으로 {#section-6}

FairStress는 그 외의 조건이 동일한 시나리오에서 모델이 동일한 최종 선택에 도달하는지만으로는 모델의 인구통계학적 공정성을 판단할 수 없는 이유를 보여줍니다.

모델은 인구통계학적 정체성만 다른 동일한 시나리오의 두 버전에서 증거가 가장 잘 뒷받침하는 후보자를 선택할 수 있지만, 한 버전에서는 더 쉽게 설득될 수 있고, 선택을 다르게 설명하거나, 서로 다른 내부 활성화 패턴을 보일 수 있습니다. 평등을 중심에 둔 FairStress의 정의에서 공정한 행동이란 어느 인구통계학적 집단이 혜택을 받는지와 관계없이 동일한 기준을 적용하고 충분히 뒷받침된 결정을 일관되게 고수하는 것을 의미합니다.

더 넓은 관점에서 이 연구는 모델이 실제 환경에서 어떻게 행동하는지를 더 잘 포착하는 평가를 구축하려는 우리의 목표를 반영합니다. FairStress를 공개함으로써 연구자와 모델 개발자가 이를 사용해 숨겨진 비대칭성을 진단하고, 제안된 개선 방법을 테스트하며, 더 일관되고 신뢰할 수 있는 결과를 만들어 내는 학습 접근법을 개발할 수 있기를 바랍니다. 개방적이고 엄격한 평가는 Ai2에서 진행하는 더 광범위한 작업의 핵심이며, FairStress가 모델이 부족한 지점과 모델을 개선하는 방법을 이해하는 데 커뮤니티가 활용할 수 있는 또 하나의 유용한 도구가 되기를 바랍니다.

