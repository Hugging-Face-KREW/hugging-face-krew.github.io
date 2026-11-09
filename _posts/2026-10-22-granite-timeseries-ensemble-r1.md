---
layout: post
title: "IBM Granite 시계열 모델을 활용한 앙상블 예측"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6806091ee92bd22fd8ce9719/P6P8ZzHR8yL7GOOO4hQOg.png
image: assets/images/blog/posts/2026-10-22-granite-timeseries-ensemble-r1/thumbnail.png
authors:
  - user: ibm-research
slug: "granite-timeseries-ensemble-r1"
source_url: "https://huggingface.co/blog/ibm-research/granite-timeseries-ensemble-r1"
source_published_date: "2026-10-22"
source_published_at: "2026-10-22T07:51:31+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Ensemble forecasting with IBM Granite time-series models](https://huggingface.co/blog/ibm-research/granite-timeseries-ensemble-r1)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-research/granite-timeseries-ensemble-r1 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# IBM Granite 시계열 모델을 활용한 앙상블 예측

4개의 오픈 Granite 모델, 하나의 확장 가능한 예측 인터페이스, 그리고 어떤 모델을 사용할지에 대한 추측을 줄이는 방법.

사전 학습된 시계열 모델은 데이터셋에 대해 학습하지 않고도 예측할 수 있지만, 여전히 모델 하나를 선택해야 합니다. 매끄러운 산업 신호를 잘 처리하는 모델이 잡음이 많은 소매 수요나 긴 계절성 예측 구간에서는 어려움을 겪을 수 있으며, 최선의 선택은 시계열과 작업에 따라 달라집니다.

Granite Timeseries Ensemble은 이러한 선택을 합리적인 기본값 하나로 줄여 줍니다. 하나의 레시피는 서로 보완적인 4개의 Granite 아키텍처를 결합하고, 재학습이나 학습된 앙상블 가중치 없이 중앙값 및 분위수 예측을 반환합니다. Probabilistic Ensemble(균일 확률 풀링)으로 시작하고, 적응형 Inter Quartile Range(IQR) Ensemble과 비교한 다음, 호환 가능한 예측 모델을 사용해 동일한 인터페이스를 확장할 수 있습니다. 앙상블 레시피와 기본 Granite 구성원 모델은 허용적 라이선스로 제공됩니다.

이 아이디어는 일기 예보에서 익숙하게 접했을 것입니다. [Ensemble forecasts](https://www.ecmwf.int/en/about/media-centre/focus/2017/fact-sheet-ensemble-weather-forecasting)은 가능한 결과의 범위를 탐색하고, [North American Multi-Model Ensemble](https://www.cpc.ncep.noaa.gov/products/NMME/NMME_description.html)과 같은 시스템은 계절성 예측을 위해 서로 다른 모델을 결합합니다. 시계열 예측도 이와 유사한 이점을 얻습니다. 서로 보완적인 모델은 다음에 일어날 일에 대해 서로 다른 관점을 제공하고, 앙상블은 이러한 관점들을 하나의 예측으로 통합합니다.

## Granite Timeseries Ensemble 사용해 보기 {#section-1}

Granite TSFM을 설치하고 Hugging Face에서 권장하는 4개 모델 레시피를 직접 불러옵니다:

```
pip install "granite-tsfm>=0.3.11"
```


```
import pandas as pd

from tsfm_public.models.ensemble.modeling_ensemble import QuantileEnsembleForecaster

df = pd.read_csv(
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    parse_dates=["date"],
)

ensemble = QuantileEnsembleForecaster.from_pretrained(
    "ibm-granite/granite-timeseries-ensemble-r1",
)

result = ensemble(
    df.tail(1024),
    timestamp_column="date",
    target_columns=["OT"],
    context_length=1024,
    prediction_length=24,
)

print(result.prediction_outputs) # point forecast
print(result.quantile_outputs)   # quantiles
```


이 레시피는 병합된 모델 가중치가 아니라 앙상블 구성을 저장합니다. 런타임에 4개의 구성원 체크포인트를 다운로드하므로 첫 실행에는 추가 시간이 필요합니다. 전체 구성, 구성원 체크포인트, 평가 세부 정보 및 제한 사항은 [model card](https://huggingface.co/ibm-granite/granite-timeseries-ensemble-r1)을 참조하세요.

`result.prediction_outputs`에는 점 예측이 포함되고, `result.quantile_outputs`에는 기본적으로 0.1부터 0.9까지의 범위를 갖는 분위수 예측이 포함됩니다. 애플리케이션은 점 값을 중심 예측으로 사용하고, 선택한 분위수를 통해 예측 범위를 표현할 수 있습니다. 예를 들어 0.1 및 0.9 분위수는 명목상 80%의 커버리지를 갖는 중심 예측 구간을 정의합니다.

## 앙상블 구성원 모델 {#section-2}

기본 레시피는 서로 다른 아키텍처를 사용하는 4개의 사전 학습된 Granite 모델을 결합합니다:

| 모델 | 아키텍처 |
| --- | --- |
| [Granite PatchTST-FM-r1](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r1) | Patched Transformer |
| [Granite PatchTST-FM-r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2) | Patched Conformer |
| [Granite FlowState-r1.1](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1) | State-space model |
| [Granite TTM-r3](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3) | Tiny Time Mixers (MLP mixer) |

배경 정보는 [PatchTST-FM-r1 paper](https://arxiv.org/abs/2602.06909),
[PatchTST-FM-r2 architecture overview](https://huggingface.co/blog/ibm-research/ibm-releases-sota-granite-time-series),
[FlowState paper](https://arxiv.org/abs/2508.05287) 및
[Tiny Time Mixers paper](https://arxiv.org/abs/2401.03955)을 참조하세요.

이러한 아키텍처는 추세, 계절성 및 국소적 변동을 서로 다른 방식으로 포착하므로 오류가 반드시 일치하지는 않습니다. [50-year review of forecast
combinations](https://arxiv.org/abs/2205.04216)은 예측을 결합하면 단일 모델을 잘못 선택할 위험을 줄일 수 있음을 설명하면서도, 구성원 모델의 품질과 다양성이 여전히 중요하다는 점을 강조합니다.

[![Four Granite forecasting models feed a common quantile interface and one of two aggregation methods.](https://huggingface.co/buckets/rry003/blogcomponents/resolve/ensemble_overview.svg?download=true)](https://huggingface.co/buckets/rry003/blogcomponents/resolve/ensemble_overview.svg?download=true)

## 예측을 결합하는 두 가지 방법 {#section-3}

각 구성원 모델은 분위수 예측을 반환합니다. 두 가지 주요 레시피는 이러한 예측을 하나의 예측 분포와 그 중앙값으로 변환하는 서로 다른 규칙을 사용합니다.

### Probabilistic Ensemble: 권장되는 기본값

Probabilistic Ensemble(균일 확률 풀링)은 각 구성원 모델에 동일한 영향력을 부여하고, 확률 공간에서 이들의 분위수 예측을 풀링한 다음, 결합된 값에서 요청된 분위수를 추출합니다. 가중치가 고정되어 있으므로 새로운 시계열에 대해 피팅 단계가 필요하지 않고, 조정할 가중치 하이퍼파라미터도 없습니다.

[forecast combination puzzle](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0084.2008.00541.x)은 유용한 동기를 제공합니다. 점 예측에서는 단순한 동일 가중치 결합이 추정된 가중치를 사용하는 방법에 비해 좋은 성능을 유지하는 경우가 많았습니다. 이 레시피는 동일한 영향력이라는 아이디어를 확률적 예측에 적용합니다. 위 예제에서 기본으로 로드되는 방식입니다.

### IQR Ensemble: 적응형 대안

IQR Ensemble은 예측 분산에 따라 구성원 모델에 가중치를 부여하여 더 좁은 구간에 더 큰 영향력을 줍니다. 25번째 및 75번째 백분위수에 가까운, 사용 가능한 분위수로부터 각 구성원의 사분위 범위를 근사합니다. 가중치는 샘플, 대상 및 예측 구간 단계에 따라 달라지며, Probabilistic Ensemble과 마찬가지로 과거 예측 오류를 사용해 피팅하지 않습니다. 단조 회귀는 결합된 분위수가 서로 교차하지 않도록 합니다.

더 좁은 구간이 반드시 더 정확하거나 더 잘 보정된 예측을 의미하는 것은 아니며, IQR 가중치는 과도한 확신을 가진 구성원 모델을 선호할 수 있습니다. 실험해 보려는 경우, 레시피는 temperature 설정(값이 클수록 가중치가 균일한 방향으로 이동)과 단일 구성원 모델의 가중치에 대한 선택적 상한을 제공합니다.

하위 폴더만 변경하여 평가된 IQR 레시피를 로드합니다:

```
ensemble = QuantileEnsembleForecaster.from_pretrained(
    "ibm-granite/granite-timeseries-ensemble-r1",
    subfolder="recipes/granite-iqr",
)
```


## 어떤 방법을 사용해야 할까요? {#section-4}

| 목표 | 다음으로 시작할 방법 | 이유 |
| --- | --- | --- |
| 단순한 기본값 | Probabilistic Ensemble | 동일한 영향력, 학습된 가중치 없음 |
| 예측 분산에 기반한 적응형 가중치 | IQR Ensemble | 각 구성원 모델의 예측 분산에 따라 영향력이 달라짐 |

Probabilistic Ensemble으로 시작하세요. 애플리케이션에서 예측 분산이 유용해 보인다면, 대표성 있는 데이터에서 IQR Ensemble과 비교해 보세요. 정확도와 보정은 시계열과 예측 구간에 따라 달라지기 때문입니다.

## 앙상블은 구성원 모델보다 성능이 좋은가요? {#section-5}

두 레시피를 97개의 GIFT-Eval 데이터셋 구성에서 평가했습니다. 모델은 미세 조정 없이 예측했으며, 테스트 대상은 집계 방법을 선택하거나 피팅하는 데 사용하지 않았습니다. 여기서는 2026년 10월 20일에 수집한 공개 리더보드 점수와 순위를 사용하여 공식 앙상블 제출 결과를 4개의 구성 모델과 비교합니다.

GIFT-Eval은 점 예측에 대해 MASE를, 확률적 예측에 대해 CRPS/MWQL(벤치마크의 `mean_weighted_sum_quantile_loss` metric)을 보고합니다. 점수는 Seasonal Naive를 기준으로 정규화되며 구성 전체에서 기하 평균됩니다. 낮을수록 좋습니다.

| 예측 | MASE ↓ | MASE Rank ↓ | CRPS / MWQL ↓ | CRPS Rank ↓ |
| --- | --- | --- | --- | --- |
| Granite Ensemble R1 — Prob | 0.683★ | 47.175★ | 0.471 | 50.948 |
| Granite Ensemble R1 — IQR | 0.684 | 48.773 | 0.471 | 47.619★ |
| Granite PatchTST-FM-r2 | 0.685 | 51.371 | 0.467★ | 48.918 |
| Granite FlowState-r1.1 | 0.701 | 66.474 | 0.490 | 69.485 |
| Granite PatchTST-FM-r1 | 0.717 | 74.794 | 0.488 | 68.134 |
| Granite TTM-r3 | 0.724 | 83.062 | 0.520 | 88.959 |

★ 표시된 예측 중 최상의 값입니다. 낮을수록 좋습니다.

순위 열은 이 6개의 예측만이 아니라 전체 GIFT-Eval 참조 리더보드에 대해 구성별 순위의 평균을 나타내므로, 리더보드에 항목이 추가되면 순위 값이 변할 수 있습니다.

두 앙상블 모두 종합 MASE와 평균 MASE Rank에서 4개의 개별 구성원 모델을 모두 앞섭니다. Probabilistic Ensemble은 이 6개 예측 중 가장 낮은 MASE인 0.683을 달성했으며, PatchTST-FM-r2는 0.685였습니다. 평균 MASE Rank가 낮다는 것은 GIFT-Eval의 다양한 데이터셋 구성 전반에서 상대적으로 더 강력한 점 예측 성능을 보인다는 의미입니다. 이러한 결과는 개발자가 사전에 가장 강력한 개별 아키텍처를 선택할 필요 없이 앙상블을 실용적인 시작점으로 사용할 수 있음을 뒷받침합니다. 최상의 단일 구성원 모델에 비한 MASE 차이는 작으므로, 주된 이점은 견고성입니다. 어떤 아키텍처가 데이터에 적합할지 미리 알지 못해도 비슷하거나 더 나은 종합 정확도를 얻을 수 있습니다.

PatchTST-FM-r2는 종합 CRPS/MWQL 점수가 가장 낮은 상태를 유지합니다. IQR Ensemble은 이 6개 예측 중 평균 CRPS Rank가 가장 낮은 반면, Probabilistic Ensemble은 해당 지표에서 PatchTST-FM-r2보다 낮은 순위를 기록했습니다. 따라서 종합 오류와 평균 순위는 서로 다른 선택을 선호합니다.

2026년 10월 20일 비교에서 앙상블 접근법은 MASE Rank와 CRPS Rank에서 적격 시스템 중 3위를 기록했습니다. 아래 그림은 각 metric에 대해 두 레시피를 적격 참조 시스템 상위 10개와 함께 보여 줍니다. 미세 조정되었거나 허용적이지 않은 라이선스를 보유하여 제외된 항목을 포함해 일부 시스템은 두 레시피보다 더 높은 점수를 기록합니다. 따라서 앙상블의 가치는 리더보드 정상에 오르는 데 있지 않습니다. 그보다는 구성원 모델을 선택하지 않고도 하나의 기본값으로 여러 시계열에서 강력하고 일관된 결과를 제공한다는 데 있습니다.

[![GIFT-Eval average rank comparison for both Granite ensembles and eligible reference systems. Lower is better.](https://huggingface.co/buckets/rry003/blogcomponents/resolve/gift_eval_ranks_granite.png?download=true)](https://huggingface.co/buckets/rry003/blogcomponents/resolve/gift_eval_ranks_granite.png?download=true)

그림: 2026년 10월 20일의 GIFT-Eval 평균 순위 현황. 각 metric에 대해 두 Granite 앙상블을 적격 참조 시스템 상위 10개와 함께 보여 줍니다. 적격 시스템은 재현 코드가 있고 테스트 데이터 누출이 선언되지 않은 시스템입니다. 미세 조정된 항목과 비허용적 라이선스 또는 구성원 종속성이 알려진 시스템은 제외됩니다. 순위 값은 공개된 전체 리더보드 스냅샷에서 직접 가져왔으며, 필터링은 표시되는 시스템을 변경할 뿐 순위 값은 변경하지 않습니다.

## 다른 예측 모델로 레시피 확장하기 {#section-6}

기본 레시피는 4개의 Granite 모델을 사용하지만, 공통 `Forecaster` 인터페이스는 호환 가능한 외부 모델도 허용합니다. 구성원 모델의 조합이나 집계 방법을 변경하면서도 구성원 체크포인트를 병합하거나 재학습하지 않고 동일한 후속 예측 형식을 유지할 수 있습니다. 따라서 기존 예측 스택에 앙상블을 통합하고, 필요가 변화함에 따라 도메인에 적합한 예측 모델을 추가하기가 더 쉬워집니다.

이 저장소에는 IBM Research 체크포인트를 추가하는 확장 레시피도 포함되어 있습니다. 여기에는 비상업적 사용 및 연구 목적 사용 제한을 비롯한 별도의 라이선스 조건이 적용되므로, 사용하기 전에 각 구성원 모델의 모델 카드를 검토하세요.

새로운 조합을 사용할 때는 예측 호환성, 정확도, 보정, 지연 시간 및 라이선스를 확인하세요.

## 애플리케이션에 맞게 레시피 평가하기 {#section-7}

GIFT-Eval은 폭넓은 벤치마크이지만, 종합적인 성능 향상이 모든 시계열에 이점을 준다는 의미는 아니며 평균 순위만으로 보정 성능을 입증할 수도 없습니다. 레시피가 애플리케이션에서 어떻게 작동하는지 알아보려면 대표성 있는 데이터에 대해 백테스트를 수행해야 합니다.

- 사용 사례에 맞추세요. 각 예측 시작 시점에 사용할 수 있었던 데이터를 반영하고, 필요한 예측 구간과 실제로 제공할 것으로 예상되는 시계열을 포함하는 과거 예측 구간을 사용하세요.

- 동일한 조건끼리 비교하세요. 동일한 구간에서 Probabilistic Ensemble, IQR Ensemble 및 강력한 개별 구성원 모델(예: PatchTST-FM-r2)을 실행하세요.

- 점 오류만 보지 마세요. MASE와 함께 분위수 손실을 평가하고, 0.1~0.9 구간의 경험적 커버리지를 명목상 80%와 비교해 확인하세요.

- 비용을 측정하세요. 4개 모델을 결합하는 이점이 작업 부하에 비해 정당한지 판단할 수 있도록 런타임과 메모리 사용량을 기록하세요.

두 가지 제한 사항을 염두에 두세요. 현재 레시피는 채널 간 정보를 사용하지 않고 다변량 채널을 독립적으로 예측하며, 4개의 구성원 체크포인트를 로드하므로 단일 구성원 모델에 비해 추론 시간과 메모리 사용량이 증가합니다. 구성원 모델의 컨텍스트 제한, 예측 구간, 전처리 및 결측값 처리 방식도 서로 다릅니다. 또한 어느 집계 방법도 분포 이동 상황에서 보정이나 견고성을 보장하지 않으므로, 배포 후에도 예측을 계속 모니터링하세요.

## 내 데이터로 사용해 보기 {#section-8}

권장되는 시작점은 기본 4개 모델 레시피입니다:

- [Granite Timeseries Ensemble on Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ensemble-r1)

- [Getting-started notebook](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/probabilistic_ensemble_getting_started.ipynb)

- [Granite TSFM repository](https://github.com/ibm-granite/granite-tsfm)

- [GIFT-Eval benchmark](https://huggingface.co/spaces/Salesforce/GIFT-Eval)

기본 레시피를 로드하면 한 번의 호출로 4개의 Granite 아키텍처 모두에서 중앙값 및 분위수 예측을 얻을 수 있으며, 필요에 따라 레시피를 조정할 수 있습니다.

## 참고 문헌 {#section-9}

- [ECMWF: Ensemble weather forecasting](https://www.ecmwf.int/en/about/media-centre/focus/2017/fact-sheet-ensemble-weather-forecasting)

- [NOAA: North American Multi-Model Ensemble](https://www.cpc.ncep.noaa.gov/products/NMME/NMME_description.html)

- [FlowState: Sampling-Rate-Equivariant Time-Series Forecasting](https://arxiv.org/abs/2508.05287)

- [Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series](https://arxiv.org/abs/2401.03955)

- [Revisiting the Generic Transformer: Deconstructing a Strong Baseline for Time Series Foundation Models](https://arxiv.org/html/2602.06909v1)

- [Forecast combinations: an over 50-year review](https://arxiv.org/abs/2205.04216)

- [Smith and Wallis (2009), A Simple Explanation of the Forecast Combination Puzzle](https://doi.org/10.1111/j.1468-0084.2008.00541.x)
