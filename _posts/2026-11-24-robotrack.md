---
layout: post
title: "RoboTrack: 실제 로봇 조작을 위한 포인트 추적"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/638e39b249de7ae552d977b5/0KlbITzhyUGtGPyvi09um.png
image: assets/images/blog/posts/2026-11-23-robotrack/thumbnail.png
authors:
  - user: allenai
slug: "robotrack"
source_url: "https://huggingface.co/blog/allenai/robotrack"
source_published_date: "2026-11-23"
source_published_at: "2026-11-23T17:20:37+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [RoboTrack: Point tracking for real-world robotic manipulation](https://huggingface.co/blog/allenai/robotrack)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/allenai/robotrack -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# RoboTrack: 실제 로봇 조작을 위한 포인트 추적

🤖 [모델](https://huggingface.co/collections/allenai/robotrack) | 📄 [기술 보고서](https://allenai.org/papers/robotrack) | 📊 [데이터](https://huggingface.co/datasets/allenai/RoboTrack-15M) | 💻 [코드](https://github.com/allenai/robotrack) | 🌐 [프로젝트 페이지](https://robotrack.allen.ai/) | 📈 [벤치마크](https://huggingface.co/datasets/allenai/RoboTrack-Bench)

[![RoboTrack wordmark logo on a dark teal-to-magenta gradient background](https://www.datocms-assets.com/64837/1795407244-robotrack-blog-draft-google-docs-image-1.png)](https://www.datocms-assets.com/64837/1795407244-robotrack-blog-draft-google-docs-image-1.png)

실제 환경에 로봇을 배치하려면 상호작용 전반에 걸쳐 신뢰할 수 있는 인식이 필요합니다. 무언가를 집으려면 로봇은 그리퍼와 물체가 이동할 때 두 요소의 위치를 추적해야 합니다. 실험실에서 피펫을 다루는 로봇은 튜브 입구에 맞추는 동안 팁의 위치를 계속 추적해야 합니다. 주변 물체가 시야를 복잡하게 만들거나, 그리퍼가 조작 중인 물체를 가리거나, 시점이 바뀌면 이러한 작업은 더 어려워집니다.

로봇이 이러한 상호작용을 따라가도록 돕는 한 가지 방법은 로봇이 다루는 물체의 특정 위치를 추적하는 것입니다. 이것이 포인트 추적의 핵심 아이디어입니다. 한 비디오 프레임에서 위치 하나가 주어지면 tracker는 이후 프레임에서 동일한 포인트가 어디에 나타나는지, 가려졌을 때는 어디에 있는지, 그리고 다시 나타날 때 어디에 나타나는지를 추정합니다. 이러한 궤적은 로봇이 시연을 통해 학습하고, 움직임을 계획하고 안내하며, 물체가 동작에 어떻게 반응하는지 모델링하는 데 도움을 줄 수 있습니다.

로보틱스 시스템은 비디오에서 움직임을 자동으로 주석 처리하고 실행 중 피드백을 제공하기 위해 상용 tracker에 의존하는 경우가 많습니다. 또한 현재 tracker는 기존 로보틱스 벤치마크에서 높은 점수를 기록하고 있어, 실제 시스템에서 더 폭넓게 사용할 준비가 되었을 가능성을 시사합니다. 우리는 이러한 결과가 로봇이 실제로 마주치는 다양한 장면, 시점, 물체 상호작용에서도 유지되는지 확인하고자 했습니다.

우리는 로봇 조작에서 포인트 추적을 위한 벤치마크 및 학습 데이터 모음인 [RoboTrack](https://robotrack.allen.ai/)을 소개합니다. [RoboTrack-Bench](https://huggingface.co/datasets/allenai/RoboTrack-Bench)에는 DROID, Ego-Exo4D, [MolmoAct 2](https://allenai.org/blog/molmoact2), EgoVerse에서 수집한 사람이 주석 처리한 비디오 307개가 포함되어 있으며, 다양한 로봇과 사람의 손, 복잡한 장면, 1인칭 및 3인칭 시점을 다룹니다. 이러한 환경에서 학습 데이터가 추적에 미치는 영향을 연구하기 위해, 로봇 학습을 위한 Ai2의 시뮬레이션 생태계인 [MolmoSpaces](https://allenai.org/blog/molmospaces)를 확장하여 시뮬레이션된 로봇 조작 궤적에서 조밀한 포인트 트랙을 생성했습니다. 이 파이프라인은 로봇 조작을 대상으로 하는 새로운 합성 학습 데이터셋 [RoboTrack-15M](https://huggingface.co/datasets/allenai/RoboTrack-15M)을 생성합니다.

[![Overview diagram of RoboTrack showing RoboTrack-15M simulated data generation from MolmoSpaces and RoboTrack-Bench human-annotated real-world manipulation footage](https://www.datocms-assets.com/64837/1795407698-robotrack-blog-draft-google-docs-image-2.png)](https://www.datocms-assets.com/64837/1795407698-robotrack-blog-draft-google-docs-image-2.png)

RoboTrack은 로봇 조작에서 포인트 추적을 위한 벤치마크 및 학습 데이터 모음입니다. MolmoSpaces를 기반으로 하는 RoboTrack-15M(왼쪽 위)은 학습을 위한 포인트 트랙을 생성합니다. 이 시뮬레이션 데이터로 학습하면 실제 조작 영상에서 추적 성능이 향상됩니다. RoboTrack-Bench(오른쪽 위)는 실제 조작 비디오에서 사람이 주석 처리한 포인트 트랙을 제공합니다.

평가 결과, 주요 tracker는 기존 로보틱스 벤치마크보다 RoboTrack-Bench에서 훨씬 낮은 점수를 기록했습니다. 기존 tracker의 학습에 RoboTrack-15M을 추가하면 실제 조작 비디오에서 성능이 향상되며, 이러한 향상은 다른 추적 벤치마크에도 이어집니다. 또한 이러한 요소들을 실제 로봇에 적용하여, 결과 tracker로 습식 실험실 작업에서 피펫 조작을 안내했습니다.

오늘 우리는 RoboTrack의 두 구성 요소와 새로운 포인트 추적 모델 두 가지인 [RoboTracker 및 RoboTracker-Warp](https://huggingface.co/collections/allenai/robotrack)를 모두 공개합니다. RoboTracker는 우리의 기법과 데이터를 사용해 fine-tuning한 AllTracker 모델이며, RoboTracker-Warp는 CoWTracker에서 학습되었습니다. 두 모델 모두 각각 공개된 기준 모델과 비교해 세 가지 로보틱스 벤치마크에서 평균 AJ를 향상시키며, RoboTracker-Warp는 평가한 다섯 개 벤치마크 중 네 개에서 state-of-the-art 성능을 달성합니다. 또한 작업과 평가 결과를 설명하는 [논문](https://allenai.org/papers/robotrack)도 공개합니다.

## RoboTrack-Bench: 실제 조작을 위한 포인트 추적 벤치마크 {#section-1}

[![Examples of human-verified point tracks in RoboTrack-Bench across DROID, Ego-Exo4D, MolmoAct 2, and EgoVerse](https://www.datocms-assets.com/64837/1795407981-robotrack-blog-draft-google-docs-image-3-1.png)](https://www.datocms-assets.com/64837/1795407981-robotrack-blog-draft-google-docs-image-3-1.png)

RoboTrack-Bench 한눈에 보기. 각 소스 데이터셋의 예시(총 307개 클립과 사람이 검증한 2,522개 트랙). MolmoAct 2와 EgoVerse는 패널마다 하나의 embodiment를 보여줍니다. DROID와 EgoExo4D는 동기화된 ego(위) 및 exo(아래) 뷰를 보여줍니다.

RoboTrack-Bench는 통제된 테이블탑 설정을 넘어 평가 범위를 확장합니다. DROID, Ego-Exo4D, MolmoAct 2, EgoVerse에서 수집한 비디오 307개와 총 2,522개의 포인트 트랙이 포함되어 있습니다. 비디오에는 복잡한 장면과 1인칭 및 3인칭 시점에서 로봇과 사람의 손이 물체를 조작하는 모습이 담겨 있습니다. 작은 물체는 복잡한 배경 속에서 추적하기 어려울 수 있으며, 그리퍼는 자신이 잡고 있는 물체를 가리는 경우가 많습니다.

RoboTrack-Bench의 모든 트랙은 사람이 주석을 달고 검증합니다. 로봇 비디오에 존재하는 수많은 정적 배경 포인트보다 움직이거나 가려지는 포인트를 우선하도록 주석 작성자에게 요청합니다. TAP-Vid 주석 방식을 따라 주석 작성자는 움직임 추정치를 사용해 비디오 전체에서 각 포인트의 궤적을 추적합니다. 향후 각 그룹의 성능을 분석할 수 있도록 각 트랙이 로봇 또는 사람의 신체, 조작되는 물체, 정적 물체 중 어디에 속하는지도 표시합니다. 두 번째 주석 작성자 그룹은 선별된 데이터에 포함하기 전에 모든 트랙을 프레임 단위로 확인합니다.

다중 뷰 포인팅, 대응 관계, 추적 연구를 지원하기 위해 RoboTrack-Bench에는 두 대의 카메라로 동시에 촬영한 비디오 쌍 33개가 포함되어 있습니다. 두 뷰에서 동일한 물리적 포인트에 주석을 달고 프레임 간 동기화하여, 동일한 상호작용을 서로 다른 시점에서 연결합니다.

## 실제 조작에서 포인트 tracker가 어려움을 겪는 이유 {#section-2}

RoboTrack-Bench와 네 가지 다른 벤치마크에서 네 가지 주요 tracker인 CoTracker3, BootsTAPNext, AllTracker, CoWTracker를 평가했습니다. 다른 벤치마크는 [RoboTAP](https://arxiv.org/abs/2308.15975)(실제 테이블탑 조작), RGB-Stacking(시뮬레이션 조작), DAVIS 및 Kinetics(일반 비디오)입니다. 추적 정확도와 가시성 예측을 측정하는 Average Jaccard(AJ)를 보고하며, 점수가 높을수록 좋습니다.

[![Evaluation table comparing CoTracker3, BootsTAPNext, AllTracker, and CoWTracker in Average Jaccard across RoboTrack-Bench, RoboTAP, RGB-Stacking, DAVIS, and Kinetics](https://www.datocms-assets.com/64837/1795408316-robotrack-blog-draft-google-docs-image-4.png)](https://www.datocms-assets.com/64837/1795408316-robotrack-blog-draft-google-docs-image-4.png)

RoboTrack-Bench는 기존 로봇 벤치마크보다 어렵습니다. 평가한 모든 최종 tracker는 유일하게 다른 실제 비디오 로보틱스 벤치마크인 RoboTAP보다 RoboTrack-Bench에서 낮은 점수를 기록했습니다. 점수는 3.0~6.6 AJ 하락했습니다. 가장 강력한 상용 tracker인 CoWTracker조차 RoboTAP의 70.5 AJ에서 RoboTrack-Bench의 63.9로 하락했습니다. 서로 다른 아키텍처와 학습 데이터를 사용했음에도 모델 전반에서 이러한 하락이 나타났으며, 이는 RoboTrack-Bench가 여러 tracker의 약점을 드러낸다는 것을 보여줍니다.

RGB-Stacking은 실제 환경의 능력보다 합성 학습을 더 크게 반영합니다. BootsTAPNext는 1,500만 개의 실제 비디오로 추가 학습한 TAPNext입니다. 이러한 추가적인 실제 환경 학습은 RoboTrack-Bench 점수를 5.1 AJ(55.1에서 60.2) 높였지만, RGB-Stacking 점수는 9.5 AJ(75.0에서 65.5) 낮췄습니다. RGB-Stacking은 시뮬레이션된 블록 쌓기로 구성되며 Kubric만 사용한 모델을 더 높게 평가하는 반면, 실제 조작 비디오에서는 순위가 반대로 나타납니다.

기존 벤치마크에서의 우위가 그대로 이어지지는 않습니다. CoWTracker는 RoboTAP에서 CoTracker3보다 4.7 AJ 앞섰지만, RoboTrack-Bench에서는 차이가 2.1 AJ에 불과했습니다. RoboTAP에서 AllTracker가 CoTracker3보다 2.7점 앞섰던 차이는 0.8로 줄었습니다. DAVIS에서 가장 강력한 모델인 BootsTAPNext는 RoboTrack-Bench에서 CoWTracker보다 3.7 AJ 뒤처졌으며, 1,500만 개의 실제 비디오로 학습했음에도 Kubric만 사용한 AllTracker(60.2 대 61.4 AJ)보다도 낮았습니다.

이러한 결과는 기존 벤치마크가 실제 조작에서의 추적 성능을 불완전하게 보여준다는 것을 시사합니다. 더 다양한 장면에서 평가하면 네 tracker 모두가 공유하는 어려움이 드러나며, 각 tracker의 강점 비교도 달라집니다.

참고: 각 tracker는 해당 저자가 설정한 방식으로 실행했으며 모두 동일한 방식으로 점수를 계산했습니다. 공개된 평가 코드의 점수 계산 버그를 수정했기 때문에 AllTracker의 점수는 공개된 점수와 약간 다릅니다. 또한 AllTracker의 프로토콜을 따르는 대신 주석 처리된 모든 포인트에서 RoboTAP과 Kinetics를 평가했습니다. 전체 평가 프로토콜은 논문을 참조하세요.

## RoboTrack-15M: MolmoSpaces의 시뮬레이션 데이터 {#section-3}

이 tracker는 왜 실제 조작에서 어려움을 겪을까요? 많은 tracker는 [Kubric](https://github.com/google-research/kubric)으로 생성한 합성 비디오로 학습하며, Kubric은 장면에 물체를 던져 넣는 모습을 보여줍니다. 이러한 데이터는 움직임과 가림에 대한 예시를 제공하지만, 로봇이 물체를 잡고 이동시키는 상황에 초점을 맞추지는 않습니다. 우리는 이러한 상호작용에서 얻은 궤적을 추가하면 실제 조작 비디오에서의 추적 성능을 향상시킬 수 있는지 테스트하고자 했습니다.

여기서 Ai2의 로봇 시뮬레이션 환경인 MolmoSpaces가 사용됩니다. 기존 조작 시뮬레이션을 확장하여 로봇과 로봇이 다루는 물체에서 조밀한 포인트 트랙을 생성했습니다. 각 에피소드에서 시뮬레이션된 로봇 팔은 컵, 접시, 블록과 같은 일상적인 물체를 집습니다. 두 대의 카메라가 상호작용을 촬영합니다. 한 대는 손목에, 다른 한 대는 로봇의 베이스에 장착되며, MuJoCo에서 렌더링됩니다. 시뮬레이터의 형상, 물체 포즈, 카메라 정보를 사용하여 사람의 주석 없이 모든 프레임에서 각 포인트의 위치와 가시성을 계산하고, 배경과 로봇의 고정된 베이스에 있는 정적 포인트를 제거합니다.

이 파이프라인은 학습을 위한 RoboTrack-15M을 생성합니다. RoboTrack-15M에는 300개의 시뮬레이션된 집에서 생성한 비디오 3,000개와 비디오당 5,000개의 트랙이 포함되어 총 1,500만 개의 트랙을 제공합니다.

[![Diagram of the RoboTrack-15M synthetic data generation pipeline built on MolmoSpaces and MuJoCo](https://www.datocms-assets.com/64837/1795408512-robotrack-blog-draft-google-docs-image-5.png)](https://www.datocms-assets.com/64837/1795408512-robotrack-blog-draft-google-docs-image-5.png)

RoboTrack-15M의 데이터 생성 파이프라인. 물체, 실내 장면, 로봇의 파지는 MolmoSpaces에서 가져옵니다. 시뮬레이터의 형상과 카메라 정보를 사용하여 포인트의 위치와 가시성을 계산합니다.

## RoboTracker: 시뮬레이션 조작으로 학습한 포인트 tracker {#section-4}

RoboTrack-15M으로 학습한 두 모델 RoboTracker와 RoboTracker-Warp를 소개합니다. 두 모델 모두 기존 tracker를 원래 학습 데이터와 함께 RoboTrack-15M으로 fine-tuning합니다. RoboTracker는 1,700만 개의 파라미터를 가진 경량 모델 AllTracker를 기반으로 하며, RoboTracker-Warp는 약 10억 개의 파라미터를 가진 훨씬 큰 모델 CoWTracker를 기반으로 합니다.

다른 연구자들이 이 작업을 살펴보고 발전시킬 수 있도록 두 모델의 체크포인트를 모두 공개합니다. RoboTracker-Warp의 가중치는 비상업적 용도로 제공됩니다.

RoboTrack-15M을 추가하면 로보틱스 벤치마크 점수가 향상됩니다. 공개된 AllTracker 및 CoWTracker 모델과 각각 비교했을 때, RoboTracker는 RoboTrack-Bench에서 0.8 AJ(62.6에서 63.4), RoboTracker-Warp는 2.1 AJ(63.9에서 66.0) 향상되었으며, 세 가지 로보틱스 벤치마크의 평균에서는 최대 1.2 AJ의 향상을 보였습니다. 일반 비디오 벤치마크에서는 두 모델 모두 기반 모델과 동등한 성능을 보였습니다. 전체적으로 RoboTracker-Warp는 평가한 다섯 개 벤치마크 중 네 개에서 모델 중 가장 높은 AJ를 기록했습니다. 예외는 DAVIS로, 여기서는 BootsTAPNext와 CoWTracker가 약간 더 높은 점수를 기록했습니다.

[![Comparison of Average Jaccard scores before and after fine-tuning AllTracker (RoboTracker) and CoWTracker (RoboTracker-Warp) on RoboTrack-15M](https://www.datocms-assets.com/64837/1795408662-robotrack-blog-draft-google-docs-image-6.png)](https://www.datocms-assets.com/64837/1795408662-robotrack-blog-draft-google-docs-image-6.png)

AllTracker(RoboTracker)와 CoWTracker(RoboTracker-Warp)를 RoboTrack-15M으로 fine-tuning했을 때의 향상. 로보틱스 평균은 RoboTrack-Bench, RoboTAP, RGB-Stacking의 평균입니다.

[![Chart of RoboTrack-Bench results broken down by video source dataset](https://www.datocms-assets.com/64837/1795408710-robotrack-blog-draft-google-docs-image-7.png)](https://www.datocms-assets.com/64837/1795408710-robotrack-blog-draft-google-docs-image-7.png)

비디오 소스별 RoboTrack-Bench 결과.

소스별로 향상을 나누어 보면 향상이 어디에서 비롯되는지 확인할 수 있습니다. RoboTrack-15M에는 로봇 팔만 포함되어 있음에도 RoboTracker-Warp는 사람 영상인 EgoVerse와 Ego-Exo4D를 포함한 모든 소스에서 대략 1~3 AJ 향상되었습니다. RoboTracker의 향상은 더 작으며 대부분 로봇 영상에서 나타났습니다. 특히 DROID Franka와 EgoVerse EVA에서 뚜렷했습니다.

## 포인트 추적에 적합한 시뮬레이션 데이터란 무엇인가? {#section-5}

RoboTrack-15M은 합성 데이터에 대해 여러 차례 반복과 실험을 거친 결과입니다. 실제로 첫 번째 시도에서는 RoboTrack-Bench에서 어떠한 향상도 나타나지 않았습니다. 그 이유를 알아보기 위해 간단한 질문을 살펴보았습니다. 추적된 포인트는 실제 조작 비디오와 비교했을 때 얼마나 멀리 이동하는가? 데이터의 각 버전에 대해 이를 히스토그램으로 표시하고 RoboTrack-Bench와의 거리를 측정했습니다(아래 그림).

[![Histograms comparing point motion distributions and Jensen-Shannon divergence across training data iterations and RoboTrack-Bench for third-person and first-person views](https://www.datocms-assets.com/64837/1795408926-robotrack-blog-draft-google-docs-image-8.png)](https://www.datocms-assets.com/64837/1795408926-robotrack-blog-draft-google-docs-image-8.png)

3인칭(위) 및 1인칭(아래) 뷰에서 RoboTrack-Bench와 비교한 각 학습 데이터 버전의 포인트 이동 거리. 거리는 두 히스토그램 간 Jensen–Shannon divergence(JSD)이며, 0은 히스토그램이 동일함을 의미합니다. Kubric에는 카메라 유형이 하나뿐이므로 동일한 클립이 두 행에 모두 나타납니다. AJ는 전체 RoboTrack-Bench에서 측정했습니다.

히스토그램을 통해 무엇이 잘못되었는지 명확히 알 수 있었습니다. 첫 번째 버전은 프레임 어디에서나 포인트를 선택했으며, 벽, 바닥, 가구도 포함되었습니다. 따라서 고정된 3인칭 카메라에서 보이는 대부분의 포인트가 움직이지 않았습니다. 이를 해결하기 위해 다음 버전에서는 로봇과 로봇이 다루는 물체에 있는 포인트만 샘플링했고, 그 결과 손목 뷰가 실제 비디오에 가까워지면서 점수가 상승했습니다. 흥미롭게도 Kubric도 손목 뷰에서 동일한 불일치를 보였으며, 이것이 Kubric만으로는 로봇 조작의 다양한 시점을 충분히 다루지 못하는 이유일 가능성이 높습니다. 마지막으로 로봇의 고정된 베이스와 장착 플랫폼에서 포인트를 이동하는 팔로 옮겼습니다. 그 결과 3인칭 뷰도 실제 비디오와 일치하게 되었고, ablation 점수는 63.0 AJ로 상승했습니다. 이는 Kubric만 사용한 대조군보다 1.5점 높은 수치입니다.

## 응용: 포인트 추적을 이용한 로봇 조작 {#section-6}

[![Diagram of the single-demonstration wet-lab pipetting pipeline guided by RoboTracker point tracks](https://www.datocms-assets.com/64837/1795409178-robotrack-blog-draft-google-docs-image-9-4.png)](https://www.datocms-assets.com/64837/1795409178-robotrack-blog-draft-google-docs-image-9-4.png)

단일 시연의 포인트 트랙으로 로봇을 안내하는 습식 실험실 파이프라인.

더 나은 포인트 추적은 로보틱스에 어떤 의미가 있을까요? 많은 로봇 시스템은 포인트 트랙을 물체가 어떻게 움직이는지를 나타내는 압축된 표현으로 활용하며, 이를 통해 정책을 직접 안내할 수 있습니다. 이를 실제로 확인하기 위해 습식 실험실 작업과 유사한 시스템을 구축했습니다. 한 바이알에서 다른 바이알로 액체를 피펫팅하는 작업입니다.

이 작업에서 로봇 팔은 전자 피펫을 집어 첫 번째 바이알에서 액체를 흡입하고, 두 번째 바이알에 분주한 뒤 피펫을 홀더로 되돌립니다. 피펫 팁을 바이알 입구에 수 밀리미터 이내로 맞춰야 하고, 각 시행에서 바이알 배치가 달라지므로 로봇이 이에 맞게 조정해야 합니다.

RoboTap을 따라 로봇은 단일 시연에서 작업을 학습하며, 이 시연은 원격 조작으로 기록한 뒤 단계별로 나눕니다. 테스트 시 로봇은 카메라에 보이는 모습과 시연을 비교하고 RoboTracker를 사용해 바이알에 맞추는 데 필요한 움직임을 추정합니다. 이후의 추종 동작은 추가 피드백 없이 실행되므로, 추정이 정확해야만 작업이 성공합니다.

바이알이 어디로 이동했는지 추정하는 세 가지 방법을 비교했습니다.

- Optical flow(RAFT)는 시연 이미지와 실시간 카메라 이미지 사이의 픽셀을 직접 대응시킵니다.

- Object masks(SAM3)는 새로운 각 이미지에서 바이알을 찾고 마스크의 중심을 시연 이미지와 정렬합니다.

- Point tracking(RoboTracker)은 비디오 전체에서 바이알 위의 포인트를 추적하여 시연 이미지의 포인트를 실시간 이미지의 포인트와 대응시킵니다.

바이알의 위치가 서로 다른 20개 배치에서 각 방법을 실행했습니다. 이 중 10개는 시야가 명확했고 10개는 부분적으로 가려져 있었습니다. 포인트 추적은 20회 시행 중 17회에서 전체 이송을 완료했으며, object masks는 8회, optical flow는 5회였습니다. 시야가 명확한 10회에서는 모두 성공했고, 가림이 있는 경우에는 10회 중 7회 성공했습니다. 반면 마스크 기반 방법은 한 번만 성공했습니다.

다른 방법들은 서로 다른 방식으로 실패합니다. 바이알이 시연 당시의 위치에서 멀리 떨어져 있으면 optical flow가 제대로 작동하지 않으며, 마스크는 비슷하게 생긴 다른 물체에 달라붙거나 바이알이 가려졌을 때 바이알을 놓칠 수 있습니다. 반면 포인트 추적은 프레임마다 동일한 바이알 물체를 따라가므로 큰 이동과 부분적인 가림을 모두 처리합니다. 이러한 결과는 포인트 추적이 정밀한 로봇 조작 작업을 위한 유망한 도구임을 시사합니다.

[![Bar chart showing wet-lab pipetting trial success rates for optical flow, object masks, and RoboTracker point tracking with and without occlusion](https://www.datocms-assets.com/64837/1795408826-robotrack-blog-draft-google-docs-image-9.png)](https://www.datocms-assets.com/64837/1795408826-robotrack-blog-draft-google-docs-image-9.png)

가림이 있을 때와 없을 때 각 방법의 피펫팅 시행 성공률(각 10개 배치). 로봇이 전체 이송을 완료하면 해당 시행을 성공으로 계산합니다.

## 한계와 향후 과제 {#section-7}

tracker는 여전히 긴 가림을 처리하고, 포인트가 다시 나타났을 때 이를 찾으며, 복잡한 배경 속에서 작은 물체를 따라가는 데 어려움을 겪습니다. RoboTrack-15M으로 학습하면 성능이 향상되지만, 이러한 실패는 실제 사용에서 여전히 과제로 남아 있습니다.

또한 현재 시뮬레이션은 일상적인 물체를 집는 작업에 초점을 맞추고 있습니다. 아직 피펫 조작과 같은 작업에 필요한 정밀한 움직임을 포착하지는 못합니다. 다음 단계는 이러한 작업을 더 가깝게 시뮬레이션하고, 해당 궤적으로 학습하면 실제 배치 중 추적 성능이 향상되는지 테스트하는 것입니다.

RoboTrack은 이러한 실패를 측정하고 이를 해결하기 위한 학습 데이터를 생성할 방법을 제공합니다. 이 벤치마크, 시뮬레이션 파이프라인, 공개 모델이 다른 연구자들이 로봇이 신뢰할 수 있는 tracker를 구축하는 데 도움이 되기를 바랍니다.
