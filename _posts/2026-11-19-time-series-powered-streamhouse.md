---
layout: post
title: "Streamhouse의 시계열 모델: 새로운 개척지와 사용 사례"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/yf6E9vJHIkGHG_toGSasb.gif
image: assets/images/blog/posts/2026-11-18-time-series-powered-streamhouse/thumbnail.gif
authors:
  - user: ibm-research
slug: "time-series-powered-streamhouse"
source_url: "https://huggingface.co/blog/ibm-research/time-series-powered-streamhouse"
source_published_date: "2026-11-18"
source_published_at: "2026-11-18T15:00:23+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Time-Series Models in the Streamhouse: A New Frontier and Use Cases](https://huggingface.co/blog/ibm-research/time-series-powered-streamhouse)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/ibm-research/time-series-powered-streamhouse -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Streamhouse의 시계열 모델: 새로운 개척지와 사용 사례

50 million회 이상의 다운로드를 기록한 IBM 시계열 모델로 구동되는 실시간 예측, 이상 탐지, 최적화. 이제 클라우드와 온프레미스에서 CPU로 실행되는 형태로 Confluent에서 네이티브로 사용할 수 있습니다.

모든 비즈니스는 변화의 연속적인 기록을 생성합니다. 기계 판독값, 결제, 판매, 주문 또는 재고 변경 등이 그 예입니다. 이러한 관측값은 시계열을 구성하며, 팀이 다음에 일어날 일을 예측하는 데 도움이 되는 신호를 담고 있습니다.

과제는 이러한 신호가 여전히 유효할 때 의사결정으로 전환하는 것입니다. 생산 실행, 품절 또는 결제 승인 이후에 전달된 예측은 정확할 수 있지만 결과에 영향을 미치기에는 너무 늦습니다.

IBM과 Confluent는 데이터가 흐르는 곳에 시계열 파운데이션 모델을 더 가까이 가져오고 있습니다. Apache Flink를 사용하면 팀은 SQL로 스트리밍 데이터에 예측, 이상 탐지 및 시맨틱 인텔리전스를 적용하고, 다음 조치를 담당하는 애플리케이션과 사람에게 모델 출력을 제공할 수 있습니다.

파운데이션 모델은 조직이 텍스트, 이미지, 문서를 다루는 방식을 바꿨습니다. 시계열 파운데이션 모델은 다양한 과거 데이터에서 일반적인 패턴을 학습한 다음 그 지식을 새로운 시계열과 작업에 적용함으로써 이와 유사한 개념을 수치 데이터로 확장합니다. 이러한 모델은 팀이 시계열마다 별도의 모델을 학습하지 않아도 시작점을 제공할 수 있습니다. 작업과 배포 방식에 따라 팀은 모델을 파인튜닝하거나 모델 출력값을 다른 데이터 및 비즈니스 로직과 결합할 수도 있습니다.

[![More than 50 million downloads and the family of four models. Click to try the models on Confluent.](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/uNQkq4JD8ABh729KrjG0A.webp)](https://events.confluent.io/early-access-flink-features)

그림 1. 50 million회 이상의 다운로드와 이를 뒷받침하는 4개 모델 제품군

이제 시계열 파운데이션 모델은 클라우드와 온프레미스의 Confluent 내부에서 기본 제공 함수로 네이티브 실행됩니다. Confluent Cloud에서는 완전 관리형으로 제공되므로 배포할 것이 없습니다([documentation](https://docs.confluent.io/cloud/current/ai/builtin-functions/forecast.html)). 온프레미스에서는 Confluent Platform for Apache Flink를 사용해 자체 인프라 내에서 모델을 실행할 수 있습니다.([documentation](https://docs.confluent.io/cp-flink/current/jobs/sql-statements/ai/set-up-ai-functions.html)). 모델은 GPU가 없어도 CPU에서 실행됩니다.

모델은 생산 라인을 사양 내로 유지하고 매장의 이상 활동을 탐지하는 일부터 제품 수준의 계획 개선에 이르기까지 다양한 미션 크리티컬 사용 사례를 지원할 수 있습니다. 또한 카드 결제를 평가하고 거래량을 예측하는 신호를 제공해 대규모 주문을 배치하는 데 도움을 줄 수 있습니다. 이러한 구성 요소는 제조, 소매, 은행, 항공 산업 전반에 적용할 수 있어 의사결정이 이루어지는 곳에 인사이트를 더 가까이 가져옵니다.

시계열과 실시간 컨텍스트가 결합되어 가치를 창출하는 방식을 살펴보세요.

## 에이전트 시대를 위한 실시간 데이터와 인텔리전스, Streamhouse {#section-1}

비즈니스는 끊임없이 운영됩니다. 주문이 업데이트되고, 결제가 들어오며, 배송이 지연되고, 하루 종일 재고가 변합니다. 그러나 이러한 이벤트를 뒷받침하는 정보는 서로 다른 시간에 업데이트되거나 주기적인 배치 프로세스에 의존하는 여러 시스템에 분산되어 있는 경우가 많습니다. 애플리케이션과 AI 에이전트가 오래되었거나 일관되지 않은 정보에 기반해 작동하면 현재 상황을 더 이상 반영하지 않는 의사결정을 내릴 위험이 있습니다.

Streamhouse는 현재의 비즈니스 컨텍스트를 지속적으로 수집, 처리, 거버넌스 적용 및 서빙함으로써 이 문제를 해결합니다. Confluent’s Data Streaming Platform은 연결, 스트리밍, 처리, 거버넌스, 쿼리라는 5가지 기능을 결합해 이러한 컨텍스트를 애플리케이션, 분석 시스템, 에이전트에서 사용할 수 있도록 합니다. Apache Flink는 데이터가 흐르는 동안 데이터를 변환하고 보강하며, Lightning Tables는 현재 상태에 대한 지연 시간이 짧은 액세스를 제공합니다.

[![The Streamhouse on Confluent](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/M2Rn5ft2uBCZcTkyUxNfx.png)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/M2Rn5ft2uBCZcTkyUxNfx.png)

그림 2. Confluent의 Streamhouse: 기존 데이터 소스에서 에이전트, 애플리케이션, 분석 시스템, 레이크하우스가 어디서나 사용하는 현재의 비즈니스 컨텍스트로

시계열 모델은 이 아키텍처에 예측 기능을 더합니다. Flink는 데이터를 준비하고, 예측 및 이상 탐지를 호출하며, 결과를 결과를 생성한 이벤트와 함께 사용할 수 있도록 합니다. 대규모 언어 모델, 통계 모델, 특화된 머신러닝 시스템을 비롯한 다른 모델도 적절한 경우 기여할 수 있습니다.

[![The layers of the Streamhouse: agentic experience, end-to-end solutions, stream-native intelligence and the Streamhouse](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/ELIbCPJpgpZn1X58rxs7p.png)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/ELIbCPJpgpZn1X58rxs7p.png)

그림 3. Streamhouse의 인텔리전스 계층: 시계열 모델, 대규모 언어 모델, ML 및 통계 모델과 이를 아우르는 오케스트레이션

이 접근 방식은 팀에 다음 네 가지 방식으로 도움을 줄 수 있습니다.

-모델 개발을 확장합니다. 사전 학습된 모델은 여러 시계열에 공통적인 시작점을 제공하여 각 시계열마다 별도의 모델을 개발하고 유지 관리해야 할 필요를 줄일 수 있습니다.

-데이터 변화에 따라 예측을 새로 고칩니다. 모델의 요구 사항과 파이프라인 구성에 따라 스트리밍 워크플로의 일부로 최근 관측값을 예측에 반영할 수 있습니다.

-신호를 의사결정에 더 가깝게 가져옵니다. 예측과 이상 징후를 운영 애플리케이션으로 더 빠르게 전달하여 팀이 이를 비즈니스 컨텍스트, 규칙, 사람의 판단과 결합할 수 있습니다.

-CPU에서 소형 모델을 실행합니다. 모델은 표준 CPU에서 스트리밍 파이프라인과 함께 클라우드 또는 온프레미스로 실행할 수 있을 만큼 작으므로, 전용 GPU 인프라의 필요성을 줄이고 배포를 간소화할 수 있습니다.

[![05-architecture-dynamic](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/Rcda03VYTPUiq9sh5zW1B.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/Rcda03VYTPUiq9sh5zW1B.webp)

그림 4. Confluent에서 네이티브로 실행되는 스트림 기반 시계열 인텔리전스

## 실제 기업 전반의 미션 크리티컬 사용 사례 {#section-2}

다음 시나리오는 조직이 시계열 모델, 스트리밍 데이터, 애플리케이션 로직을 어떻게 결합할 수 있는지 보여줍니다.

### 사용 사례 1 | 생산 최적화: 모든 제품을 사양 내로

아이스크림 라인의 품질은 혼합물에 휘저어 넣는 공기의 양인 오버런에 따라 달라집니다. 현재 공정 엔지니어와 작업자는 통이 채워진 후에 측정한 QA 실험실 결과에 의존합니다. 새로운 크림 로트나 덥고 습한 날씨로 공정이 바뀌면 팀이 그 결과를 확인할 때 이미 몇 시간 분량의 생산품을 재작업해야 할 수 있습니다.

문제를 사전에 파악하기 위해 사전 학습된 TTM 모델은 10분마다 SCADA 텔레메트리와 QA 측정값을 읽습니다. 이 모델은 4시간 후 사양을 충족할 것으로 예상되는 통의 비율과 예상 범위를 예측합니다.

예측값이 공장의 목표 아래로 떨어지면 최적화 프로세스가 냉동고 흡입 온도와 공기 흐름의 조합을 평가합니다. 모델이 이러한 제어 변수를 입력으로 받기 때문에 라인에서 무엇이든 변경하기 전에 각 옵션의 점수를 계산할 수 있습니다. 작업자는 권장 설정값을 승인하고, 일상적인 조정은 사전 정의된 가드레일 내에서 자동화할 수 있습니다.

그 결과 더욱 선제적인 공정이 구현됩니다. 재작업이 줄고, 톤당 에너지 사용량이 감소하며, 품질이 더욱 일관되게 유지됩니다. 동일한 접근 방식은 오븐, 건조기, 가마, 밀에도 적용할 수 있습니다.

[![07-production-optimization](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/B1eCtmZ6M0ufgJ9u_P_oU.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/B1eCtmZ6M0ufgJ9u_P_oU.webp)

그림 5. 제품 품질 예측을 통해 작업자는 조정하기 전에 냉동고 설정값을 평가할 수 있습니다.

### 사용 사례 2 | 이상 탐지: 플래그에서 올바른 조치까지

매장 운영팀은 수백 개 매장의 수천 개 제품에 대한 판매를 모니터링합니다. 고정 임계값은 예측 가능한 주말 판매 피크를 자주 플래그하지만, 예상보다 적게 판매되는 제품과 같은 조용한 변화는 놓치는 경우가 많습니다. 알림이 쌓일수록 팀은 어디에 집중해야 할지 판단하기 어려워집니다.

AI_DETECT_ANOMALIES는 최근 이력을 기반으로 각 판매 시계열에 예상 범위를 제공합니다. 팀이 선택한 신뢰 수준에서 각각의 새 판독값을 해당 범위와 비교하므로 제품과 매장마다 별도의 모델을 학습할 필요가 없습니다.

판매량이 예상 범위를 벗어나면 이 신호를 재고 및 매장 데이터와 결합해 가능한 원인을 파악하고 다음 단계를 권장할 수 있습니다. 예를 들어 진열대를 확인하거나 재고를 보충하는 방식입니다. 동일한 접근 방식은 기계 센서, API 텔레메트리, 미터 판독값, 결제량에도 적용할 수 있습니다.

[![08-anomaly-detection](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/vbMeXLhmrNv_BP4tSHwl7.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/vbMeXLhmrNv_BP4tSHwl7.webp)

그림 6. 매장 판매량을 예상 범위와 비교합니다. 표시된 화면에서는 반복되는 피크가 범위 안에 있어 이상 징후가 플래그되지 않습니다.

### 사용 사례 3 | 공급망 계획: 모든 SKU, 모든 매장, 매시간

카테고리 플래너와 보충 팀은 과잉 재고를 보유하지 않으면서 모든 매장에서 제품을 판매 가능 상태로 유지해야 합니다. 하지만 예측은 일주일에 한 번만 새로 고쳐질 수 있고, 대개 가장 많이 팔리는 제품에 집중하는 반면 롱테일 제품은 더 단순한 규칙으로 관리됩니다. 계획 주기 사이에 판매 패턴이 바뀌면 한 매장은 품절되고 다른 매장은 재고가 너무 많아질 수 있습니다.

AI_FORECAST를 사용하면 사전 학습된 모델이 제품과 매장 전반의 수요를 예측할 수 있습니다. 이 예제에서는 P10–P90 범위와 함께 28일 예측을 생성하고, 새로운 판매 데이터가 들어오면 이를 새로 고칩니다. 계획 애플리케이션은 예측을 재고, 리드 타임, 플래너가 선택한 서비스 수준과 결합해 재주문점을 계산합니다.

재고가 해당 지점 아래로 떨어질 것으로 예상되면 에이전트가 검토를 위한 구매 주문서를 작성할 수 있습니다. 동일한 접근 방식은 예비 부품, 인력 배치, ATM 현금, 에너지 수요 계획도 지원할 수 있습니다.

[![09-supply-chain-planning-v2](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/CjaQoYxJfRy89ahsPN2P3.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/CjaQoYxJfRy89ahsPN2P3.webp)

그림 7. 매장 수준의 수요 예측은 귀리 우유의 재주문점과 제안된 구매 주문서를 결정하는 데 활용됩니다.

### 사용 사례 4 | 사기 및 보안: 모든 카드에는 고유한 정상 상태가 있습니다

사기 대응팀은 이미 정교한 규칙과 모델을 사용해 거래를 평가합니다. 지속적인 과제 중 하나는 여러 카드 소유자에게 적용되는 임계값이 각 개인의 정상적인 지출 방식을 반영하지 못할 수 있다는 점입니다.

시계열 신호는 이러한 개인별 컨텍스트를 추가할 수 있습니다. 이 예제에서는 각 결제를 해당 카드의 최근 이력을 기반으로 한 예상 지출 범위와 비교합니다. 범위를 벗어난 거래는 다른 신호 및 비즈니스 규칙과 함께 추가 입력값으로 은행의 기존 사기 탐지 시스템에 전달됩니다.

그러면 은행은 결제를 승인할지, 추가 인증을 요청할지, 아니면 추가 조사를 진행할지 결정할 수 있습니다. 결정과 그 근거가 된 신호는 검토를 위해 보존할 수 있습니다.

[![10-fraud-and-security](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/5rPcP75nAZTLx56KSASo1.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/5rPcP75nAZTLx56KSASo1.webp)

그림 8. 카드의 예상 지출 범위를 벗어난 결제가 예시적인 사기 워크플로에서 단계적 인증을 요청하도록 합니다.

### 사용 사례 5 | 트레이딩: 시장 흐름에 맞춰 주문 속도 조절

체결 데스크는 시장에 미치는 영향을 관리하면서 대규모 주문을 소규모 거래로 나누어야 합니다. 과거 거래량 곡선은 시작점을 제공하지만 오늘의 거래 활동은 평균과 크게 다를 수 있습니다.

각 1분 바에서 PatchTST-FM은 P10–P90 범위와 함께 다음 30분의 거래량을 예측합니다. 체결 알고리즘은 이 예측을 사용해 데스크가 설정한 참여 한도 내에서 주문 속도를 조절합니다.

뉴스 이벤트로 시장 활동이 바뀌면 알고리즘은 일시 중지하고 계획을 다시 수립할 수 있습니다. 예상보다 거래가 조용하면 각 분할 주문의 규모를 줄이고 주문 완료에 더 오래 걸릴 수 있음을 알릴 수 있습니다. 각 예측과 그에 따른 체결 결정은 감사 추적에 보존할 수 있습니다.

[![12-trading](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/wZtPpIN2ujDQEfOzfUBwm.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/wZtPpIN2ujDQEfOzfUBwm.webp)

그림 9. 거래량 예측은 100,000주 주문에 대한 제안된 체결 일정을 결정하는 데 활용됩니다.

### 사용 사례 6 | 시맨틱 검색: 이와 비슷한 모든 신호 찾기

이상한 센서 패턴을 조사하는 신뢰성 엔지니어는 공장 내 다른 곳에서도 유사한 동작이 나타났는지 확인하고 싶을 수 있습니다.

TSPulse는 유사도 검색을 지원하는 시계열 윈도우 표현을 생성할 수 있습니다. 별도로 구축한 애플리케이션에서는 이러한 표현을 인덱싱하고 장비 전반의 과거 패턴을 검색하는 데 사용할 수 있습니다. 엔지니어는 급격한 상승 후 회복, 점진적인 상승, 단계적 변화와 같은 형태를 검색한 다음 관련 유지보수 기록과 운영 조건을 살펴볼 수 있습니다.

이 애플리케이션에는 임베딩 파이프라인, 인덱스, 관련 컨텍스트에 대한 액세스가 필요합니다. 유사도는 조사 범위를 좁히는 데 도움이 될 수 있지만, 그 자체로 근본 원인을 입증하지는 않습니다.

[![11-semantic-search](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/8xO2-4i9W0BDyJ4TVrsGC.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/8xO2-4i9W0BDyJ4TVrsGC.webp)

그림 10. 시계열 임베딩은 장비와 기타 신호 전반에서 유사한 패턴을 검색하는 데 활용할 수 있습니다.

## 산업별 애플리케이션 {#section-3}

동일한 모델과 구성 요소를 산업 전반의 다양한 워크플로에서 지원할 수 있습니다.

- 산업용 AI: 센서에서 설정값까지. 공장 관리자는 모든 배치가 사양을 충족하고 예상치 못한 문제로 라인이 멈추지 않기를 원합니다. SCADA, QA 실험실, 에너지 미터 스트림은 들어오는 즉시 예측됩니다. 생산 최적화는 품질이 흔들리기 시작하면 설정값을 조정하고, 이상 탐지는 어떤 규칙에도 작성되지 않은 판독값을 플래그하며, 시맨틱 검색은 고장 전에 동일한 패턴을 보인 모든 펌프를 찾습니다. 설정값, 작업 주문, 알림은 다음 교대조 보고서가 아니라 해당 사이클 내에 생산 현장에 전달됩니다.

[![13-twin-plant](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/q8GaODH5na0inQAQ5NsSy.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/q8GaODH5na0inQAQ5NsSy.webp)

그림 11. 센서에서 설정값까지 이어지는 공장 트윈. 새로운 크림 로트로 인해 오버런이 목표에서 벗어납니다. 예측이 라인을 분석하고, 에이전트가 새로운 흡입 온도와 공기 흐름을 제안하며, 작업자가 승인하면 통이 다시 사양을 충족합니다. 시뮬레이션 데이터.

- 소매: 매장과 공급망. 매장 관리자는 월요일에 과잉 재고가 발생하지 않으면서 토요일에 진열대를 가득 채우기를 원합니다. 모든 매장의 모든 제품에 대해 판매, 날씨, 프로모션 스트림을 예측합니다. 공급망 계획이 주문을 설정하고, 이상 탐지가 빈 진열대나 누락된 배송을 포착하며, 사기 및 보안 시스템이 의심스러운 카드를 보류합니다. 진열대가 비기 전에 주문, 매장 작업, 보류 조치가 전달됩니다.

[![14-twin-store](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/H2fM8uPxwgGowSbE81se3.webp)](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/H2fM8uPxwgGowSbE81se3.webp)

그림 12. 매장과 공급망을 나타내는 매장 트윈. 귀리 우유 프로모션이 예정되어 있습니다. 예측이 프로모션을 반영해 재주문점을 설정하고, 에이전트가 구매 주문서를 생성하며, 플래너가 승인하면 프로모션 당일 진열대가 가득 찹니다. 시뮬레이션 데이터.

- 은행: 결제와 트레이딩 데스크. 사기 대응 책임자는 정상 고객을 차단하지 않으면서 카드 테스트를 막기를 원하고, 트레이더는 가격을 움직이지 않고 대규모 주문을 완료하기를 원합니다. 모든 카드와 모든 종목은 이벤트마다 고유한 예상 패턴과 비교해 점수가 계산됩니다. 사기 및 보안 시스템은 승인 전에 이상한 카드 사용을 보류하고, 이상 탐지는 비정상적인 흐름을 포착하며, 트레이딩은 예측된 거래량에 맞춰 각 주문의 속도를 조절합니다.

- 항공: 허브 운영, 항공기, 네트워크. 운영 관제팀은 조치를 취할 시간이 아직 있을 때 장애가 다가오고 있음을 파악하고 싶어 합니다. 엔진, 게이트, 네트워크 스트림을 실시간으로 예측하고 점검합니다. 이상 탐지는 변화하는 엔진 판독값을 포착하고, 생산 최적화는 게이트와 승무원의 균형을 조정하며, 공급망 계획은 항공기가 필요로 하는 곳에 예비 부품을 배치합니다.

모든 산업에서 Streamhouse는 동일한 작업을 수행합니다. Flink는 Granite 모델을 실행해 예측, 탐지, 최적화를 수행하고, Streaming Agents는 실시간 컨텍스트에 따라 작동하며, 사람은 예외 상황을 승인합니다.

Confluent에서 모델을 살펴보고 시계열 인텔리전스가 실시간 의사결정을 어떻게 지원할 수 있는지 확인해 보세요.

[![Try the models on Confluent](https://cdn-uploads.huggingface.co/production/uploads/64e8143f6de557454220921e/_UuP1FMzinS6oJu3XiXUW.webp)](https://events.confluent.io/early-access-flink-features)

### 자세히 알아보기

- 코드 및 노트북: [github.com/ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm)

- [Confluent의 IBM 시계열 모델을 활용한 실시간 인텔리전스](https://huggingface.co/blog/ibm-research/real-time-intelligence)

- [IBM Granite 시계열 모델을 활용한 앙상블 예측](https://huggingface.co/blog/ibm-research/granite-timeseries-ensemble-r1)

- [상업적 친화적 라이선스를 적용한 SOTA Granite Time Series PatchTST-FM-r2 모델을 IBM이 출시](https://huggingface.co/blog/ibm-research/ibm-releases-sota-granite-time-series)

- [Confluent Intelligence 및 AI Tools의 새로운 기능: 스트림에 에이전트를 네이티브로 구현하고, 모델 지원을 확장하며, 새로운 에이전트 스킬과 Confluent Assistant 제공](https://www.confluent.io/blog/2026-q3-confluent-intelligence-ai-update/)

- [Confluent Data Streaming Platform Q4'26의 새로운 기능](https://www.confluent.io/blog/2026-q4-confluent-bundle-launch/)

- [Confluent Platform for Apache Flink의 새로운 기능, 2026년 11월](https://docs.confluent.io/cp-flink/current/changelog.html#november-2026)
