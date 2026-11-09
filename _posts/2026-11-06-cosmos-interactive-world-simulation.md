---
layout: post
title: "인터랙티브 월드 시뮬레이션을 위한 새로운 NVIDIA Cosmos 3 오픈 모델"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/677c471f8b4c4e271e57eaa5/rtmGVvFdD6JqkjBc26vsp.png
image: assets/images/blog/posts/2026-11-05-cosmos-interactive-world-simulation/thumbnail.png
authors:
  - user: nvidia
slug: "cosmos-interactive-world-simulation"
source_url: "https://huggingface.co/blog/nvidia/cosmos-interactive-world-simulation"
source_published_date: "2026-11-05"
source_published_at: "2026-11-05T19:35:16+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [New NVIDIA Cosmos 3 Open Models for Interactive World Simulation](https://huggingface.co/blog/nvidia/cosmos-interactive-world-simulation)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/cosmos-interactive-world-simulation -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 인터랙티브 월드 시뮬레이션을 위한 새로운 NVIDIA Cosmos 3 오픈 모델

## TL;DR {#section-1}

[NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) 월드 모델 개발 플랫폼에 인터랙티브 월드 시뮬레이션과 다중 센서·다중 뷰 주행 장면을 생성하기 위한 [Cosmos 3](https://huggingface.co/collections/nvidia/cosmos3) 모델 제품군의 새로운 오픈 웨이트 체크포인트 4개가 추가되었습니다.

- 🌦️ [Cosmos3-Nano-Transfer-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto): 다양한 날씨, 조명, 시간대의 다중 뷰 주행 데이터를 생성합니다.

- 🚗 [Cosmos3-Nano-Sim-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto): 주행 시뮬레이션을 위해 동기화된 카메라 및 LiDAR 관측을 스트리밍합니다.

- 🎨 [Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer): 시뮬레이터 제어와 텍스트 프롬프트를 사실적인 비디오로 변환합니다.

- 🤖 [Cosmos3-Nano-Sim-Bimanual](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual): 로봇 또는 카메라 동작으로부터 비디오를 예측합니다.

네 모델 모두 Cosmos 3 Nano의 사후 학습 예제이며, 학습 가능한 파라미터는 16B입니다. 모델은 OpenMDW1.1을 사용하므로 해당 조건에 따라 상업적 및 비상업적 용도로 사용할 수 있습니다.

[체크포인트 다운로드](https://huggingface.co/collections/nvidia/cosmos3-examples) · [기술 보고서 읽기](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/technical-report.pdf) . [데모 보기](https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/)

## 이러한 체크포인트가 필요한 이유 {#section-2}

🦾 로보틱스에는 변화하는 장면과 뷰포인트에 대응할 수 있는 환경이 필요합니다. 이러한 체크포인트는 환경을 구축하는 상호 보완적인 방법을 제공합니다. Sim-Bimanual은 동작으로부터 시각적 결과를 예측하고, Sim-Transfer는 물리 시뮬레이터의 제어를 생성된 관측으로 변환합니다.

🚦 자율주행에서는 외관과 기하 구조가 일치해야 합니다. 동기화된 카메라 및 LiDAR 생성은 시각적 관측과 3D 센싱을 연결합니다. 장면 및 교통 신호 제어를 사용하면 연구자가 도로 배치와 신호 상태를 변경한 다음, 주행 정책이 어떻게 반응하는지 연구할 수 있습니다.

🔄 각 관측은 다음 동작을 형성할 수 있습니다. 스트리밍 생성은 비디오 예측을 지속적인 인터랙션에 연결하여, 월드 모델 연구자가 동작과 관측이 함께 전개되는 방식을 탐색할 수 있게 합니다.

🌐 다른 사람들이 기반으로 활용할 수 있는 오픈 기반. NVIDIA의 비전은 연구자와 개발자가 더 풍부하고 반응성이 높은 시뮬레이션 월드를 만들 수 있도록 지원하는 것입니다. 오픈 웨이트와 사후 학습 레시피는 모델을 조정하고, 방법을 비교하고, 개선 사항에 기여할 수 있는 공통의 출발점을 제공합니다. 이러한 문제는 여전히 어렵습니다. 앞으로의 과제는 모델이 정책 학습 및 평가를 얼마나 충실하게 지원하는지 측정하고, 한계를 파악하며, 부족한 부분을 개선하는 것입니다.

### Cosmos3-Nano로 자율주행 차량 월드 구축

## 🌦️Transfer-Auto: 주행 시나리오 → 다중 센서 데이터 {#section-3}

[Cosmos3-Nano-Transfer-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto)는 오프라인 합성 주행 데이터 생성을 위한 양방향 모델입니다. 피시아이를 포함해 최대 11개의 카메라 뷰에서 동기화된 비디오를 생성하며, RGB–LiDAR 공동 생성을 지원합니다. 카메라별 World Scenario Map (WSM) 제어는 장면 구조와 교통 움직임을 지정하고, 텍스트 프롬프트는 외관을 안내합니다.

개발자는 환경 조건을 변경하여 학습 및 평가 범위를 확장할 수 있으며, 선택적으로 초기 RGB 이미지를 추가해 장면 외관을 고정할 수 있습니다. LiDAR 공동 생성에는 정렬된 HD-map range-map 제어도 추가로 사용됩니다.
Transfer-Auto는 Cosmos를 자체 센서 구성과 주행 도메인에 맞게 조정하기 위한 기반으로도 사용할 수 있습니다. 먼저 자체 데이터로 사후 학습하여 카메라 배치, LiDAR 구성 및 대상 환경에 맞추세요. 조정된 모델은 오프라인 데이터 생성에 직접 사용하거나, 인과 학습 및 few-step 증류를 거쳐 빠른 스트리밍 시뮬레이션에 사용할 수 있습니다. 이는 Cosmos3-Nano-Sim-Auto를 만드는 데 사용된 개발 경로입니다.

에이전트가 오픈 스킬을 사용하도록 지정하세요: [Cosmos3-Nano-Transfer-Auto로 추론 실행](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-nano-transfer-auto/inference/SKILL.md)

노트북을 선호하시나요? [Transfer-Auto](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/simulation/auto/run_auto_with_cosmos_framework.ipynb)를 여세요.

## 🚗Sim-Auto: 주행 제어 → 스트리밍 관측 {#section-4}

[Cosmos3-Nano-Sim-Auto](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto)는 Transfer-Auto를 인과적으로 증류한 스트리밍 변형 모델입니다. 생성된 히스토리와 업데이트된 주행 제어를 사용하여 동기화된 카메라 비디오와 LiDAR의 연속적인 짧은 청크를 몇 번의 디노이징 단계만으로 생성하며, 빠른 스트리밍과 장시간의 인터랙티브 롤아웃을 지원합니다.

카메라별 캡션과 동기화된 WSM 제어를 제공하고, 선택적으로 초기 RGB 이미지를 추가하세요. LiDAR 생성에는 카메라 제어에 맞춰 정렬된 HD-Map range 제어를 포함하세요. 증류된 모델은 롤링 히스토리를 사용하여 연속적인 청크를 생성합니다. 폐루프 인터랙션에는 주행 입력에 따라 해당 제어를 업데이트하는 시뮬레이션 런타임과 WSM 렌더러가 필요합니다. 새로운 센서 구성이나 도메인에 적용하려면 먼저 Transfer-Auto를 조정한 다음 Sim-Auto로 증류하세요.

에이전트가 오픈 스킬을 사용하도록 지정하세요: [Cosmos3-Nano-Sim-Auto로 추론 실행](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-nano-sim-auto/inference/SKILL.md)

노트북을 선호하시나요? [Sim-Auto](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/simulation/auto/run_sim-auto_with_cosmos_framework.ipynb)를 여세요.

## 🎨 Sim-Transfer: 시뮬레이터 제어 → RGB {#section-5}

시뮬레이터는 움직임, 접촉 및 물리 상태를 계산합니다. Sim-Transfer는 시뮬레이터의 공간 제어와 텍스트 프롬프트를 사용하여 정책이 사용할 수 있는 색상 관측을 렌더링합니다.

동작 → 시뮬레이터 업데이트 → 공간 제어 → 생성된 RGB → 다음 정책 동작

- Depth: 카메라로부터의 상대적 거리.

- Segmentation: 영역과 객체 경계.

- Edges: 장면 윤곽.

- Blur: 대략적인 외관과 레이아웃.

텍스트는 외관을 안내하고, 물리와 ground-truth 상태는 시뮬레이터가 담당합니다.

Depth 제어에서 비디오로

Segmentation 제어에서 비디오로

에이전트가 오픈 스킬을 사용하도록 지정하세요: [Cosmos3-Nano-Sim-Transfer로 추론 실행](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-nano-sim-transfer/inference/SKILL.md)

노트북을 선호하시나요? [Transfer](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/simulation/transfer/run_transfer_with_vllm_omni.ipynb)를 사용하세요.

## 🤖 Sim-Bimanual: 동작 → 비디오 {#section-6}

입력 프레임

Sim-Bimanual 시뮬레이션 체크포인트에서 출력된 비디오

[Cosmos3-Nano-Sim-Bimanual](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual)은 양손 로봇의 동작에 따라 환경이 변화하는 방식을 시뮬레이션합니다. 정책 또는 사람이 조작하는 조이스틱으로 제어되는 YAM 및 AgiBot embodiment의 인터랙티브 양팔 시뮬레이션을 지원합니다.

이미지로 시작해 로봇이나 카메라를 움직이고 다음 관측을 생성하세요. 하나의 제어 시퀀스에서 뷰포인트 변경과 조작을 전환할 수 있습니다.

혼합 동작 예제는 59개의 제어 값을 사용하며, 학습 중에는 64개로 패딩됩니다. 해당 스키마는 좌표 프레임, 회전, 그리퍼 보정, 유효성 마스크 및 제어 도메인을 정의합니다. 각 값이 의도한 움직임을 나타내도록 이러한 규칙을 따르세요.

타이밍을 맞추세요. T개의 전이에는 T+1개의 관측이 필요합니다. 학습 변환은 원시 캡처를 모델의 동작 표현으로 변환하며, 공개된 추론 동작은 이미 정규화되어 있습니다.

여기서 Sim-Transfer는 물리 시뮬레이터가 업데이트한 상태로부터 시각 정보를 생성하고, Sim-Bimanual은 장면과 로봇 동작으로부터 다음 관측을 직접 생성합니다.

[데이터 계약](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-nano-sim-bimanual/post-training/references/data-contract.md)과 [동작 스키마](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-nano-sim-bimanual/post-training/references/unified-v1-action-schema.md)를 확인하세요.

요구 사항: Linux, NVIDIA GPU, Docker 및 NVIDIA Container Toolkit. 공개된 vLLM-Omni 설정은 하나의 GB200에서 테스트되었으며, 이는 최소 메모리 사양이 아닙니다. 서버는 가중치를 다운로드하고, 클라이언트는 샘플 입력을 가져옵니다.

에이전트가 오픈 스킬을 사용하도록 지정하세요: [Cosmos3-Nano-Sim-Bimanual로 추론 실행](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-nano-sim-bimanual/inference/SKILL.md#run-inference-with-cosmos3-nano-sim-bimanual)

노트북을 선호하시나요? [Bimanual](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/simulation/bimanual/run_bimanual_with_vllm_omni.ipynb)을 여세요.

## 스트리밍 생성의 작동 방식 {#section-7}

Cosmos 3의 Mixture-of-Transformers 백본은 어텐션을 통해 별도의 텍스트 처리 네트워크와 시각 생성 네트워크를 연결합니다. 생성기는 프롬프트, 사용 가능한 제어 및 생성된 히스토리로부터 각각의 짧은 세그먼트를 예측합니다.

제어 + 히스토리 → 청크 생성 → 이미지 디코딩 → 히스토리 업데이트

디노이징은 노이즈가 있고 압축된 비디오 표현을 세그먼트로 정제합니다. 시간적 인과 어텐션은 사용 가능한 히스토리와 제어를 읽을 수 있게 하면서 미래의 대상 관측은 차단합니다. 키/값 캐시는 다음 예측을 위해 어텐션 정보를 유지합니다.

모델은 latent라고 하는 압축된 비디오 표현으로 작동합니다. 초기 이미지 이후 각 latent 시간 단계는 4개의 RGB 프레임을 나타냅니다. Transfer는 청크마다 하나의 시간 단계를 예측하고, 보고서의 최적화된 Bimanual 설정은 2개를 예측합니다. 청크가 작을수록 새로운 제어가 생성에 더 자주 영향을 줄 수 있습니다.

### 4단계 학습 레시피

- Stage 0 — 양방향 조정: 완전히 기록된 클립에서 도메인과 제어에서 비디오로 이어지는 관계를 학습합니다.

- Stage 1 — 교사 강제: 정답과 미래 대상은 마스킹한 상태에서, 이전에 기록된 관측과 사용 가능한 제어로부터 각 청크를 예측합니다.

- Stage 2 — 일관성 증류: 동결된 교사를 사용하여 4회의 디노이징 단계로 각 청크를 생성하는 학생 모델을 학습합니다.

- Stage 3 — 셀프 포싱: 학생이 생성한 히스토리로 학습합니다. 동결된 교사와 보조 score 모델이 분포 일치 신호를 제공합니다.

최종 단계가 중요한 이유: 객체가 약간 잘못 예측되면 이후 프레임의 컨텍스트가 되므로 오류가 누적될 수 있습니다. 셀프 포싱은 학습 중 이러한 생성 히스토리에 노출되도록 합니다. 증류와 캐싱은 각 관측을 생성하는 비용을 줄입니다.

## 🛠️ 자체 데이터로 조정 {#section-8}

사후 학습 레시피는 Cosmos3-Nano 또는 호환 가능한 기본 가중치에서 시작하며 4개 단계 전체를 다룹니다. 재개할 수 있도록 전체 학습 체크포인트를 보관하세요. 내보낸 추론 가중치에는 옵티마이저 및 기타 학습 상태가 포함되지 않습니다.

- [Transfer-auto](https://github.com/NVIDIA/cosmos-framework/tree/main/.agents/skills/cosmos3-nano-transfer-auto/post-training): 주행 데이터로 사후 학습하여 센서 구성과 도메인에 맞추고 오프라인 합성 데이터를 생성합니다.

- [Sim-auto](https://github.com/NVIDIA/cosmos-framework/tree/main/.agents/skills/cosmos3-nano-sim-auto/post-training): 조정된 Transfer-Auto 모델에 인과 학습과 증류를 적용하여 빠른 스트리밍 시뮬레이션을 수행합니다.

- [Sim-Transfer](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-nano-sim-transfer/post-training/SKILL.md): 로컬 데이터 준비, 단계 구성 및 재개 기능을 사용하여 환경에 맞게 체크포인트를 사후 학습합니다.

- [Sim-Bimanual](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cosmos3-nano-sim-bimanual/post-training/SKILL.md): 데이터 검증, 단계 구성, 체크포인트 전달 및 내보내기를 사용하여 embodiment에 맞게 사후 학습합니다.

코딩 에이전트에 스킬과 데이터셋 위치를 제공하세요. 환경 검사를 따르고, 샘플을 검증하며, 장시간 실행하기 전에 학습과 재시작을 테스트하세요. 보류된 에피소드에서 품질을 평가하세요. 짧은 실행이 성공했다고 해서 실행이 검증된 것일 뿐, 수렴이 검증된 것은 아닙니다.

### 추가 리소스

- 🤗[체크포인트 다운로드](https://huggingface.co/collections/nvidia/cosmos3-examples)

- 📺 [Cosmos Labs 라이브스트림 다시 보기](https://www.youtube.com/watch?v=8lNoGg_j8S8)

- 📖 [기술 블로그 읽기](https://developer.nvidia.com/blog/build-world-models-for-interactive-simulation-with-nvidia-cosmos/)
