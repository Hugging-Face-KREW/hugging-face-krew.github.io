---
layout: post
title: "빛의 속도로 텍스트 생성을 향한 Nemotron-Labs Diffusion Language Models"
author: dailybot
categories: [Translation, HuggingFace]
slug: "nemotron-labs-diffusion"
source_url: "https://huggingface.co/blog/nvidia/nemotron-labs-diffusion"
source_published_date: "2026-05-23"
source_published_at: "2026-05-23T00:02:03+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

> Source: https://huggingface.co/blog/nvidia/nemotron-labs-diffusion

* TOC
{:toc}
<!--toc-->

_이 글은 Hugging Face 블로그의 [Towards Speed-of-Light Text Generation with Nemotron-Labs Diffusion Language Models](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion)를 한국어로 번역한 글입니다._

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 빛의 속도로 텍스트 생성을 향한 Nemotron-Labs Diffusion Language Models

대형 언어 모델(LLMs)은 코드 생성, 수학 문제 해결, 요약, 문서 이해 및 기타 개발자 워크플로의 기본 인터페이스가 되었습니다. 하지만 속으로는 여전히 많은 LLM들이 같은 방식으로 텍스트를 한 토큰씩 생성하고, 각 토큰은 앞서 등장한 토큰에 의존합니다. 따라서 이들 모델은 자신의 출력을 소비하는 자기회귀(AR) 모델로 불립니다.

그 autoregressive(AR) 접근 방식은 놀라울 정도로 성공적이었습니다. 학습이 안정적이고, 서비스를 단순하게 제공하며, 현대 언어 모델링의 진전에 큰 기여를 해 왔습니다. 그러나 동시에 강한 제약을 만듭니다: 매 새 토큰마다 전체 모델 패스가 필요하고, 계산이 시작되기 전에 메모리에서 모든 가중치를 로드해야 합니다. 지연 시간에 민감한 애플리케이션을 구축하는 개발자, 더 작은 배치 크기를 운영하거나 현대 GPU를 더 잘 활용하려는 경우, 토큰-대-토큰 생성은 GPU의 시간 대부분이 계산이 아닌 메모리 작업에 소모되기 때문에 성능이 최대한 발휘되지 않을 수 있습니다.

또한 autoregressive 모델이 토큰을 생성하면 그 토큰은 최종적이며, 이전 토큰을 수정하는 능력이 본질적으로 내재되어 있지 않습니다. 결과적으로 생성 과정에서 실수가 생성 과정 내내 확산될 수 있습니다.

Nemotron-Labs Diffusion은 새로운 전진 방향을 제시합니다: 병렬로 여러 토큰을 생성한 다음, 여러 단계에 걸쳐 생성된 토큰을 반복적으로 다듬는 diffusion language models(DLM)입니다. 이러한 모델은 현대 GPU의 계산 패턴을 더 잘 활용하고 실행 시간 측면에서 상당한 성능 이점을 제공할 수 있을 뿐만 아니라, 생성된 토큰을 수정할 수 있어 기존 텍스트를 개정하고 중간에 채워넣는 목표를 다루는 데 더 적합합니다. 또한 이 생성-정제 특성은 추론 예산을 제어하는 내장 방식을 제공합니다. 정제 단계를 줄이면 런타임에서 이들 모델의 계산 요구량을 줄일 수 있습니다.

## 모델, 학습 레시피 및 기술 보고서에 대한 빠른 링크

Nemotron-Labs Diffusion 계열에는 3B, 8B, 14B 규모의 텍스트 모델이 모두 포함되어 있으며, 상업적으로 친화적인 NVIDIA Nemotron Open Model License 하에 이용 가능하고, 폭넓은 연구 유연성을 부여하는 NVIDIA Source Code License 하에 이용 가능한 8B 규모 비전-언어 모델(VLM)도 있습니다. 라인업 전반에서 NVIDIA는 기본 모델과 지시-튜닝된 채팅 변형을 함께 출시하고 있습니다. 또한 NVIDIA는 이들 모델의 학습 코드를 [NVIDIA Megatron Bridge framework](https://github.com/NVIDIA-NeMo/Megatron-Bridge/)를 통해 공개하고 있습니다.

- [NVIDIA Nemotron-Labs Diffusion model collection on HuggingFace](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion)

- [Training recipe and code on GitHub](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/diffusion/recipes/nemotron_labs_diffusion)

- [Technical report](http://bit.ly/Nemotron-Labs-Diffusion-Report)

## 한 모델에서의 세 가지 생성 모드

Nemotron-Labs Diffusion은 간단한 아이디어를 바탕으로 설계되었습니다: autoregressive와 diffusion 생성은 서로 다른 모델 계열이 되어서는 안 됩니다. 같은 모델의 기능이어야 합니다. 이 모델은 세 가지 생성 모드를 지원합니다:

- 자기회귀 모드(Autoregressive mode)가 표준의 좌에서 우로 진행되는 LLM처럼 작동합니다. 개발자가 이미 알고 있는 생성 워크플로우와의 호환성을 유지합니다.

- Diffusion 모드는 블록 단위로 생성하며, 여러 단계에 걸쳐 토큰을 점진적으로 생성합니다.

- Self-speculation 모드(Self-speculation mode)는 확산을 사용해 다수의 후보 토큰을 초안 작성한 뒤, 이를 자기회귀 디코딩으로 검증합니다. 확산 방식의 초안 작성 속도와 AR 검증의 신뢰성을 결합합니다.

이 유연한 설계는 속도와 정확도가 모두 중요한 개발자 친화적 핵심 특징으로, 예측할 수 없는 배치 크기에서도, 혹은 배치 크기가 1인 단일 쿼리에서도 작동합니다. 원하는 추론 모드를 선택하는 데 애플리케이션 수준에서 거의 변경이 필요 없으며, 이는 배포 시 설정으로 결정되기 때문입니다. 따라서 개발자들은 오늘 쓰는 모델과 Nemotron-Labs Diffusion의 다양한 추론 모드 간에 ultra-fast 생성 속도를 위해 매끄럽게 전환할 수 있습니다.

## 성능 하이라이트

Nemotron-Labs Diffusion 8B는 Qwen3 8B에 비해 평균 정확도가 1.2% 향상되었습니다. 토큰을 앞으로 한 번에 처리하는 속도(TPF, 토큰 디코딩 효율의 하드웨어 독립적 지표)로 비교하면, diffusion 모드는 AR 모델보다 2.6배 높은 TPF에 도달하고, Self-speculation은 선형(Self-speculation)에서 6배, 이차(Quadratic Self-speculation)에서 6.4배까지 더 향상되며, 평가된 작업 전반에서 비슷한 정확도를 유지합니다.

## Nemotron-Labs Diffusion은 어떻게 학습되었나

Diffusion 언어 모델은 수년간 가능성을 보여 왔지만, 역사적으로 실용적 장벽이 남아 있었습니다: 강한 AR 모델에 비해 낮은 정확도, 더 어려운 학습 과정, KV 캐싱과의 제한된 호환성.

최근 연구가 그 방향을 바꿨습니다. [Efficient-DLM](https://arxiv.org/abs/2512.14067)은 사전 학습된 AR 모델을 지속적 사전학습과 주의 메커니즘을 블록 단위로 바꿔 diffusion language models로 변환할 수 있음을 보여주었습니다. 이 설계는 AR 모델의 기능을 보존하면서 KV-캐시 친화적인 병렬 디코딩을 가능하게 합니다.

Nemotron-Labs Diffusion은 같은 실용적 인사이트를 바탕으로 기존 AR 모델에 확산 능력을 추가합니다. 이 모델은 AR와 diffusion 목표를 함께 학습하는 방식으로 훈련되어, 초기 AR 학습 중에 이미 학습한 것을 보존하는 동시에 diffusion은 병렬 초안 작성 능력을 추가했습니다. 이 모델은 [NVIDIA Nemotron Pretraining datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets)의 1.3조 토큰으로 사전 학습되었고, [NVIDIA Nemotron Post-training datasets](https://huggingface.co/collections/nvidia/nemotron-post-training-v3)의 45B 토큰을 사용한 추가 감독 미세 조정 단계를 거쳤습니다.

## SGLang을 통한 배포 및 추론

 Nemotron-Labs Diffusion 모델의 배포는 곧 SGLang의 메인 브랜치에서 지원될 예정입니다. 이 글을 쓰는 시점에서, 추론 지원은 [GitHub의 이슈 트래커 요청](https://github.com/sgl-project/sglang/pull/25803)을 통해 이용 가능합니다.

 흥미로운 점은 이 통합을 통해 하나의 체크포인트를 세 가지 다른 방식으로 서비스할 수 있다는 것입니다. 알고리즘 구성에서 한 줄의 설정으로 선택됩니다:

- Plain autoregressive - `ar_mode=true`를 설정하면 모델은 다른 모든 causal LM처럼 동작합니다. 정합성 기준으로 유용하거나 순수 AR 출력에 대한 건전성 확인이 필요할 때 유용합니다.

- Diffusion 모드(FastDiffuser) - 원시 처리량의 주력 모드입니다. 모델은 32-token 블록을 한 번에 채우고, 이를 반복적으로 디노이즈하며, 자신감 임계값이 각 단계에서 어떤 토큰을 "충분히 좋은"지 결정합니다.

- Self-speculation(LinearSpec) - 이 모드가 저희가 가장 좋아하는 버전입니다. 동일한 모델이 블록을 양방향으로 초안을 작성한 후, 이를 인과적으로 검증합니다;whatever 접두사와 일치하는 부분이 커밋됩니다. 온도 0에서 AR 대비 손실 없이 출력이 되며, speedbench 데이터셋의 B200에서 약 865 tok/s를 달성합니다. 같은 하드웨어에서 AR 기준선의 약 4배에 해당합니다.

## 지금 바로 시작하기

Nemotron-Labs Diffusion은 diffusion 스타일의 생성을 개발자가 실제로 사용할 수 있는 형태로 제공합니다: 오픈 모델, 친숙한 AR 호환성, diffusion 디코딩, 그리고 자체-추정 기반 가속이 한 가족 안에 포함됩니다. Nemotron-Labs Diffusion을 통해 개발자는 애플리케이션을 변경하지 않고도 텍스트 생성을 초안 작성, 정제, 검증 및 가속하는 새로운 방법을 얻습니다.

시작하려면 Nemotron-Labs Diffusion [model family](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion)를 살펴보고, [technical report](http://bit.ly/Nemotron-Labs-Diffusion-Report)를 읽고, 이용 가능한 [training recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/diffusion/recipes/nemotron_labs_diffusion)를 시도해 보세요.
