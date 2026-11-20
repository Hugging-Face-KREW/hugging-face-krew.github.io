---
layout: post
title: "미세 조정 작업이 FLOPs가 말하는 것보다 오래 걸리는 이유"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/J6DIf4Len_31Tj-CQlQXY.png
image: assets/images/blog/posts/2026-11-18-predicting-fine-tuning-time/thumbnail.png
authors:
  - user: Dharma-AI
slug: "predicting-fine-tuning-time"
source_url: "https://huggingface.co/blog/Dharma-AI/predicting-fine-tuning-time"
source_published_date: "2026-11-18"
source_published_at: "2026-11-18T18:53:43+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Why your fine-tuning job takes longer than the FLOPs say](https://huggingface.co/blog/Dharma-AI/predicting-fine-tuning-time)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/Dharma-AI/predicting-fine-tuning-time -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 미세 조정 작업이 FLOPs가 말하는 것보다 오래 걸리는 이유

272개의 실제 운영 실행을 바탕으로 공유 GPU 클러스터에서 엔드투엔드 미세 조정 시간을 예측합니다.

곧 발표할 논문 Predicting End-to-End Fine-Tuning Time for Generative Language Models의 미리보기입니다. 전체 버전은 곧 arXiv에 공개될 예정입니다.

오늘날 AI에서 GPU는 가장 비싸고 부족한 리소스 중 하나입니다. 공유 클러스터를 운영하는 사람은 매일 어떤 작업을 어떤 GPU에서 어떤 순서로 실행할지 결정해야 합니다. 이 계획은 각 작업에 걸리는 시간을 얼마나 정확히 추정하느냐에 달려 있습니다. 작업이 예상보다 몇 시간 일찍 끝나면 다음 작업이 할당될 때까지 GPU가 유휴 상태로 남습니다. 몇 시간 늦게 끝나면 뒤에서 대기 중인 모든 작업이 지연되고, 실험은 기다리게 되며, 납기 일정도 미뤄집니다.

더 나은 소요 시간 추정은 이를 추측이 아닌 계획의 문제로 바꿉니다. 무엇을 언제 몇 개의 GPU에서 실행할지 정하고, 그 사이의 유휴 용량을 줄일 수 있습니다. 하지만 대부분의 팀이 사용하는 추정치는 대략적인 FLOP 계산입니다. 9시간이라고 예측했는데 작업은 다음 날 오후에 끝나는 식입니다.

우리는 그 계산이 실제로 얼마나 빗나가는지, 그리고 작업이 할당되기도 전에 더 나은 예측을 할 수 있는지 알고 싶었습니다. 그래서 NVIDIA H200 GPU의 운영 풀에서 실제 지도 미세 조정(SFT) 작업 272개를 가져왔습니다. 총 약 7,382 GPU-시간에 해당하며, 분석적 추정치와 학습된 예측기를 비교했습니다.

요약하면, 일반적인 FLOP 기반 추정치는 우리가 관찰한 현상을 거의 설명하지 못했습니다(R² = 0.31, MAE 12.41 h). 할당 전에 이용 가능한 정보만 사용하는 CatBoost 회귀 모델은 5-fold 교차 검증에서 R² = 0.98, 평균 절대 오차 1.71 h, 중앙값 오차 0.51 h를 달성했습니다. 이 차이는 폐쇄형 추정으로는 처리할 수 없는 두 가지 요인에서 비롯됩니다. 회귀 모델은 평가 주기, 체크포인트 정책, GPU당 부하, 미세 조정 방식과 같은 학습별 신호를 더 많이 보고, 과거 실행에서 이들 사이의 비선형 상호작용을 학습합니다.

## 모두가 사용하는 추정치는 계산만 고려한다 {#section-1}

Dense Transformer의 학습에는 대략 6·P·D 부동소수점 연산이 필요합니다. 여기서 P는 파라미터 수이고 D는 학습 토큰 수입니다. 이를 GPU가 제공할 수 있는 성능으로 나누면 학습 루프 시간은 다음과 같습니다:

```
T_theoretical = (6 · P · D) / (N_GPU · FLOPS_peak · MFU)
```


여기서 N_GPU는 GPU 수이고, FLOPS_peak는 사양표에 기재된 GPU 한 개의 최대 처리량이며, MFU(Model FLOPs Utilization)는 학습이 실제로 달성하는 최대 성능의 비율입니다. 메모리 트래픽, 통신, 커널 비효율성 때문에 실제 작업은 최대 성능으로 실행되지 않으므로 MFU는 일반적으로 30%에서 60% 사이입니다. 또한 특정 작업에 적합한 값은 사전에 알 수 없습니다.

이는 최적화 루프를 설명하는 데는 좋은 방식입니다. 하지만 작업은 루프 그 이상입니다. 첫 단계 전에 환경 설정, 모델 다운로드 및 로딩, 토큰화가 이루어집니다. 학습 중에는 주기적인 평가와 체크포인트 업로드가 진행됩니다. 마지막에는 최종 모델이 객체 스토리지로 전송됩니다. 멀티테넌트 클러스터에서는 이 모든 과정이 네트워크와 스토리지 I/O를 두고 다른 작업과 경쟁합니다.

이러한 오버헤드는 구성에 따라 달라지며, 작업량이나 GPU 수에 따라 깔끔하게 확장되지 않습니다. 명목상 계산량이 같은 두 작업도 완료 시점이 몇 시간씩 차이 날 수 있습니다.

비교의 공정성을 유지하기 위해, 우리가 테스트하는 FLOP 기준선은 학습 토큰뿐 아니라 예약된 평가 토큰도 계산하며 MFU를 넓은 범위에서 변화시켰습니다.

## 우리가 구축한 것 {#section-2}

엔드투엔드 소요 시간을 지도 회귀 문제로 다룹니다. 작업 구성과 모델 메타데이터로 구성한 특성 벡터가 주어지면, 작업이 GPU를 할당받은 시점부터 완료되는 시점까지의 실제 경과 시간을 예측합니다. 대기 시간은 제외합니다.

두 가지 설계 선택이 작업의 방향을 결정했습니다:

- 모든 정보는 할당 전에 알려져 있습니다. 모든 입력은 제출된 구성에서 가져옵니다. 하이퍼파라미터, 데이터셋 크기, 모델 식별자, GPU 할당 등이 이에 해당합니다. 런타임 로그나 부분 진행 상황은 사용하지 않습니다. 이것이 예측을 스케줄링에 활용할 수 있게 합니다.

- 오버헤드를 드러내는 특성입니다. 원시 구성 위에 `num_eval_data`(모든 평가 주기에 걸쳐 처리된 샘플), `num_saves` 및 `num_params_saved`(체크포인트 I/O), GPU당 토큰 볼륨과 같은 값을 파생합니다. MFU를 0.45로 고정한 FLOP 추정치 자체도 하나의 특성(`baseline_time`)으로 입력합니다. 따라서 모델은 계산만 고려한 수치에서 출발해 보정값을 학습합니다.

Linear Regression, Ridge, XGBoost 및 CatBoost를 비교했습니다. 특성 선택에는 18개의 후보 특성 세트에 대한 상관 필터링, 상호 정보량, 순열 중요도를 결합하고, MAE를 기준으로 그리드 검색을 수행했습니다. 최종 CatBoost 모델은 24개의 특성을 사용합니다.

## 데이터 {#section-3}

- 기업 운영 환경의 운영 로그에서 수집한 272개의 SFT 실행입니다. 최대 150억 개 파라미터를 사용하는 여러 모델 계열, LoRA와 전체 미세 조정, 폭넓은 데이터셋 크기와 하이퍼파라미터 범위를 포함합니다.

- 공유 인프라입니다. 작업은 1~4개 노드에 걸쳐 8~32개의 H200으로 구성된 풀에서 1개, 2개 또는 4개의 GPU를 사용해 실행되었으며, 다른 학습 및 추론 작업과 함께 진행되었습니다. 측정 시간에는 실제 리소스 경합이 포함됩니다.

- 소요 시간은 1.6~172.4시간이며, 평균은 23.43 h, 중앙값은 15.29 h입니다.

- 스택: Hugging Face Transformers 및 Unsloth. 스택은 범주형 특성으로 모델링하며, 두 스택 사이에 고정된 속도 향상이 있다고 가정하지 않습니다.

[![fig-duration-distribution](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/uD4Td4N-bvoPStQBNx8yy.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/uD4Td4N-bvoPStQBNx8yy.png)

272개 실행에서 엔드투엔드 소요 시간의 분포입니다. 평균 23.43 h, 중앙값 15.29 h입니다.

## 결과 {#section-4}

모든 수치는 층화 5-fold 교차 검증에서 얻었습니다. 처음 네 행은 분석적 기준선이며 피팅을 수행하지 않습니다.

| 모델 | R² | RMSE (h) | MAE (h) |
| --- | --- | --- | --- |
| FLOP 기준선 (MFU 30%) | 0.31 | 20.45 | 12.41 |
| FLOP 기준선 (MFU 45%) | 0.18 | 22.38 | 13.01 |
| FLOP 기준선 (MFU 60%) | 0.01 | 24.56 | 14.62 |
| Calculon | −0.51 | 30.40 | 20.36 |
| Linear Regression | 0.69 ± 0.14 | 13.52 ± 4.23 | 7.52 ± 0.49 |
| Ridge | 0.73 ± 0.09 | 12.72 ± 3.70 | 7.23 ± 0.68 |
| XGBoost | 0.97 ± 0.01 | 4.18 ± 1.17 | 1.95 ± 0.32 |
| CatBoost | 0.98 ± 0.01 | 3.52 ± 0.78 | 1.71 ± 0.33 |

몇 가지가 눈에 띕니다.

어떤 단일 MFU 값도 FLOP 기준선을 바로잡지 못합니다. MFU를 0.25에서 0.65까지 변화시켰습니다. 가장 좋은 설정인 30%에서도 평균 23시간이 걸리는 작업의 평균 오차가 여전히 12시간을 넘습니다.

더 상세한 분석 모델도 격차를 줄이지 못합니다. 두 번째 기준선으로 선행 연구의 LLM 학습용 분석적 성능 모델인 Calculon(Isaev et al., SC '23)을 사용했습니다. 이 모델은 명시적인 레이어 그래프를 구축하고 계산, 메모리 트래픽, 디바이스 간 통신을 고려합니다. 공개 구현을 확장해 LoRA, grouped-query attention, fused attention, bfloat16 및 SwiGLU를 지원하도록 했습니다. 그럼에도 추정치는 관찰된 실제 경과 시간의 중앙값 11%에 불과했습니다. 이는 단계별 도구로서의 Calculon에 대한 비판이 아닙니다. Calculon은 애초에 그런 용도로 설계되었습니다. 이 결과는 운영 작업의 상당 부분이 최적화 단계 외부에서 발생한다는 점을 보여줍니다.

선형 모델은 개선을 가져오지만 곧 한계에 도달합니다. Ridge는 MAE를 7.23 h까지 낮춥니다. 구성과 소요 시간 사이의 관계는 순수한 선형 모델로는 잘 포착되지 않습니다.

Gradient boosting이 핵심적인 역할을 합니다. CatBoost와 XGBoost는 모든 지표에서 서로의 표준편차 한 개 이내에 들어옵니다. 모든 지표에서 평균 오차가 가장 낮은 CatBoost를 이후 모델로 선택했습니다.

## 예측기가 의존하는 것 {#section-5}

CatBoost의 특성 중요도는 소수의 특성에 집중됩니다. 예상대로 모델 및 작업량 변수도 포함되지만, 평가 및 체크포인트 변수도 포함됩니다. 작업 전체에서 평가 중 처리된 총 샘플 수인 `num_eval_data`가 1위를 차지하고, 그 뒤를 `num_trainable_params`, `batch_eval_size_percent`, `num_all_params`, `num_gpus*batch_train_percent`가 따르며 `num_params_saved`도 근소한 차이로 뒤따릅니다. SHAP 값도 같은 이야기를 들려줍니다. 평가량이 많을수록 예측 시간이 일관되게 증가하는 반면, 학습 가능한 파라미터 수는 가장 큰 모델에서만 큰 효과를 보이는 비선형적 영향을 미칩니다.

[![fig-feature-importance](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/MHyGZSFJZ6HXmwAUCTAnF.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/MHyGZSFJZ6HXmwAUCTAnF.png)

각 fold에 대한 CatBoost 모델의 평균 특성 중요도(PredictionValuesChange)입니다.

특성 중요도는 인과관계에 대한 주장이 아닙니다. 그럼에도 이 패턴은 작업이 실제로 시간을 어떻게 사용하는지와 일치합니다. 평가를 얼마나 자주 수행하는지, 평가 세트가 얼마나 큰지, 저장하는 체크포인트가 몇 개인지는 모델 및 데이터셋 규모와 함께 실제 경과 시간에 영향을 줍니다. 이는 학습 구성을 작성할 때 사람들이 습관적으로 선택하는 경우가 많은 요소입니다.

플롯을 해석할 때 한 가지 주의할 점이 있습니다. 더 큰 GPU 할당은 예측 시간에 양의 기여를 보입니다. 우리 데이터에서는 더 큰 할당이 더 큰 작업과 함께 나타나므로 `num_gpus`는 부분적으로 작업 규모의 대리 변수로 작동합니다. GPU를 추가하면 작업이 느려진다는 의미는 아닙니다.

## 얼마나 틀리며, 어디에서 틀리는가 {#section-6}

272개의 모든 out-of-fold 예측을 살펴보면 중앙값 절대 오차는 0.51 h입니다. 예측의 80%는 2.20 h 이내, 90%는 4.31 h 이내에 있습니다. 작업 소요 시간별로 나누면 다음과 같습니다:

| 작업 소요 시간 (h) | 실행 수 | 중앙값 절대 오차 (h) | P90 (h) |
| --- | --- | --- | --- |
| 1.56~5 | 60 | 0.30 | 1.55 |
| 5~10 | 63 | 0.18 | 1.27 |
| 10~30 | 69 | 0.73 | 4.17 |
| 30~60 | 67 | 1.06 | 5.50 |
| 60~175 | 13 | 5.19 | 18.51 |

60시간까지는 중앙값 오차가 약 1시간 이하로 유지됩니다. 모델이 어려움을 겪는 부분은 긴 꼬리 구간이며, 60시간을 넘는 실행이 13개뿐이라는 점을 고려하면 이를 데이터의 한계로 해석합니다.

[![fig-error-by-duration](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/gQbjzLv37ag5kX7nZuuMh.png)](https://cdn-uploads.huggingface.co/production/uploads/6a725daa5ca641a21ab61956/gQbjzLv37ag5kX7nZuuMh.png)

실제 소요 시간에 대한 out-of-fold 절대 오차를 소요 시간 범위별로 색칠했습니다. 오차는 60 h까지 낮게 유지되며 데이터가 희소한 꼬리 구간에서 커집니다.

데이터가 더 많아지면 여전히 도움이 될 것입니다. 고정된 20% 홀드아웃에서 학습 파티션의 85%에서 100%로 늘리면 MAE가 1.63 h에서 1.50 h로 감소하며, 뚜렷한 포화점은 나타나지 않았습니다.

## 어디에 유용한가 {#section-7}

실제로 예측기는 작업이 시작되기 전에 스케줄러가 계획을 세울 수 있는 소요 시간을 제공합니다. 어떤 작업을 함께 실행할지, GPU가 언제 해제될지, 작업이 마감 시간에 맞춰 완료될지를 판단할 수 있습니다.

## 한계 {#section-8}

- SFT만 다룹니다. RLHF, DPO 또는 RLVR과 같은 다른 학습 및 정렬 전략은 포함하지 않았습니다.

- 한 가지 GPU 유형만 사용했습니다. 모든 실행은 H200에서 수행되었습니다. 학습된 모델은 해당 인프라에 맞게 보정되었으므로 재학습 없이 다른 GPU 세대에 수치적으로 그대로 적용될 것이라고 기대해서는 안 됩니다. 혼합 GPU 풀용 예측기에는 계산 처리량, 메모리 용량 및 대역폭, 인터커넥트 토폴로지와 같은 하드웨어 설명자도 필요합니다.

- 매우 긴 실행이 적습니다. 60시간을 넘으면 오차가 커집니다.

전이할 수 있는 것은 방법입니다. 오버헤드를 고려한 특성, 다중공선성을 고려한 선택 방식, 비교 프로토콜이 이에 해당합니다. 전체 논문에서는 특성 공식, 선택 절차, 탐색 공간, 최종 하이퍼파라미터를 모두 상세히 보고할 예정입니다. 따라서 유사한 구조를 가진 어떤 실행 로그 데이터셋에서도 파이프라인을 재구축할 수 있습니다.

다음 계획은 하드웨어별 설명자와 다른 GPU 유형에서의 검증, 관측치가 적은 구성에 대한 전이 학습, 다른 학습 목적 함수, 장시간 실행에 대한 추가 데이터입니다.

## 다음 단계 {#section-9}

전체 논문은 곧 arXiv에 공개될 예정입니다. Hugging Face에서 Dharma-AI 조직을 팔로우하면 논문 공개 알림을 받을 수 있습니다. 아래 댓글에 질문을 남겨 주세요. 여러분의 클러스터에서 작업 소요 시간을 어떻게 추정하는지 듣고 싶습니다.

## 추가 읽을거리 {#section-10}

- [GPU 관리: 유휴 GPU가 새로운 지상 대기 항공기가 된 이유](https://huggingface.co/blog/Dharma-AI/gpu-management)

- [같은 클러스터, 33포인트 더 높은 활용률: 달라진 것은 순서였습니다](https://huggingface.co/blog/Dharma-AI/gpu-management-pt2)

- [SLM의 어휘를 전문화하면 실제 애플리케이션에서 추론 비용을 77% 줄일 수 있는 이유](https://huggingface.co/blog/Dharma-AI/vocabx-vocabulary-specialization)

- [규모보다 전문화: 대부분의 AI 조달 결정이 간과하는 전략적 변수](https://huggingface.co/blog/Dharma-AI/specialization-beats-scale)

- [전문화가 불가피한 이유](https://huggingface.co/blog/Dharma-AI/why-specialization-is-inevitable)

- [더 최신인 모델, 동일한 이점](https://huggingface.co/blog/Dharma-AI/newer-models-same-advantages)

- [챗봇을 넘어선 직접 선호 최적화](https://huggingface.co/blog/Dharma-AI/direct-preference-optimization-beyond-chatbots)

- [텍스트 퇴화: 대부분의 벤치마크가 추적하지 않는 운영 장애 모드](https://huggingface.co/blog/Dharma-AI/text-degeneration-a-production-failure-mode)

Dharma-AI R&D · AI를 여러분의 모델로 만듭니다
