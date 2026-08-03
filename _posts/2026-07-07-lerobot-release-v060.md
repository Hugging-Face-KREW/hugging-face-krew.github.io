---
layout: post
title: "LeRobot v0.6.0: 상상하고, 평가하고, 개선하기"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-07-07-lerobot-release-v060/thumbnail.png
authors:
  - user: imstevenpmwork
  - user: pepijn223
  - user: CarolinePascal
  - user: lilkm
  - user: maximellerbach
  - user: nepyope
  - user: nikodembartnik
  - user: Nico-robot
  - user: thomwolf
slug: "lerobot-release-v060"
source_url: "https://huggingface.co/blog/lerobot-release-v060"
source_published_date: "2026-07-07"
source_published_at: "2026-07-07T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [LeRobot v0.6.0: Imagine, Evaluate, Improve](https://huggingface.co/blog/lerobot-release-v060)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/lerobot-release-v060 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# LeRobot v0.6.0: 상상하고, 평가하고, 개선하기

이 새 릴리스의 핵심은 로봇 학습 루프를 닫는 것입니다: 행동하기 전에 미래를 상상하는 정책, 로봇이 성공했는지 알려주는 보상 모델, 실패를 학습 데이터로 바꿔주는 배포 CLI, 그리고 이를 모두 측정하기 위한 여섯 개의 새로운 시뮬레이션 벤치마크. 또한 깊이 센싱, VLM 기반 데이터셋 주석, 맞춤 비디오 인코딩, HF Jobs에서의 클라우드 학습, 그리고 더 간소해진 설치를 제공합니다.

## 한줄 요약 {#section-1}

LeRobot v0.6.0은 미래를 상상하는 월드 모델 정책(VLA-JEPA, FastWAM, LingBot-VA)을 도입하고, 미래를 상상하는 VLAs의 물결(GR00T N1.7, MolmoAct2, EO-1, EVO1, Multitask DiT), 그리고 새로운 보상 모델 API(Robometer, TOPReward)를 포함합니다. 또한 `lerobot-eval` 아래에 여섯 개의 새로운 시뮬레이션 벤치마크를 포함하고, DAgger 스타일의 휴먼-인-더 루프 보정을 갖춘 `lerobot-rollout` CLI, FSDP 학습, HF Jobs에서의 클라우드 학습이 함께 제공됩니다. 데이터셋은 깊이 지원, 자동 언어 주석 파이프라인, 맞춤 비디오 인코딩, 그리고 최대 2배 빠른 데이터 로딩을 제공하며, 더 간소화된 설치 위에 구축됩니다.

![LeRobot 0.6.0](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/lerobot%20v0.6.0.png)

## 목차 {#section-2}

* [월드 모델: 미래를 상상하는 정책들](#section-3)
* [VLAs: 모델 모음이 계속 확장됩니다](#section-4)
* [보상 모델: 로봇이 성공했는지 아는 방법](#section-5)
* [데이터셋: 더 빠른 로딩, 더 풍부한 데이터](#section-6)
* [벤치마크: 모든 것을 평가하는 하나의 CLI](#section-7)
* [훈련 및 추론](#section-8)
* [코드베이스: 더 간결하고 깔끔하게](#section-9)
* [커뮤니티 & 생태계](#section-10)
* [Final thoughts](#section-11)

## 월드 모델: 미래를 상상하는 정책들 {#section-3}

로봇 공학 세계는 큰 질문에 직면해 있다: 월드 모델이 실제로 로봇 정책에 도움이 되는가? v0.6.0은 이 질문에 답하기 위해 LeRobot에 세 가지 정책을 도입합니다. 각 정책은 학습의 일부로 미래를 상상하는 법을 배우고, 각각은 그 상상을 합리적으로 유지하기 위해 서로 다른 경로를 택합니다.

### VLA-JEPA

VLA-JEPA는 잠재 공간에서 미래를 예측하도록 컴팩트한 VLA(Qwen3-VL-2B 기반)를 학습하는 동안, 행동하는 방법을 배우게 합니다: 학습 중에 JEPA 월드 모델은 모델의 자체 행동으로 다가오는 프레임을 예측해야 합니다는 점입니다. 비밀은 추론 시 월드 모델이 사라져 추가 추론 비용 없이 월드-모델 감독을 얻을 수 있다는 것입니다. 허브에 준비된 세 가지 바로 사용 가능한 체크포인트가 있으며, 미세 조정을 위한 DROID 사전학습 기본 모델도 포함되어 있다:

```bash
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --policy.repo_id=${HF_USER}/my_finetuned_policy
```


자세한 내용은 [VLA-JEPA documentation](https://huggingface.co/docs/lerobot/v0.6.0/vla_jepa)와 [paper](https://arxiv.org/abs/2602.10098)를 확인하십시오.

### LingBot-VA

LingBot-VA는 한 걸음 더 나아가: 미래의 비디오와 행동을 함께 예측하는 자기회귀 비디오-액션 모델로, 청크 단위로 처리하고 실제 관측값을 다시 피드백해 상상을 구체화합니다. 로봇이 상상한 것을 저장하고 (`--policy.save_predicted_video=true`) 실제로 일어난 일과 비교할 수 있다. 추론은 한 개의 24–32 GB GPU에서 실행됩니다. 기술적 세부 내용은 [documentation](https://huggingface.co/docs/lerobot/v0.6.0/lingbot_va)와 [paper](https://arxiv.org/pdf/2601.21998)를 참고하세요.

![LingBot-VA imagined rollout vs real rollout](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/lingbot_va_viz_1.gif)

### FastWAM

FastWAM은 논문 제목에서 질문을 던진다: 테스트 시점의 미래 상상이 필요할까? 약 50억 파라미터 규모의 비디오 생성 전문가와 하나의 네트워크에 연결된 컴팩트한 행동 전문가를 매치한 구조로, 모델이 실제로 자신의 롤아웃을 꿈꾸도록 학습합니다. 추론 시에는 꿈꾸는 단계 자체를 건너뛰고 직접 액션 청크를 노이즈 제거합니다. [lerobot/fastwam_base](https://huggingface.co/lerobot/fastwam_base)에서 미세조정하고, 자세한 내용은 [documentation](https://huggingface.co/docs/lerobot/v0.6.0/fastwam)를 참조하세요.

## VLAs: 모델 모음이 계속 확장됩니다 {#section-4}

### GR00T N1.7

NVIDIA의 교차 구현 기반 파운데이션 모델의 최신 공개 세대인 GR00T N1.7으로 NVIDIA GR00T 연동을 업데이트했다. N1.7은 이전 VLM을 Cosmos-Reason2-2B(Qwen3-VL 기반)로 교체하고, 흐름 매칭 액션 헤드를 공급합니다. 우리의 통합은 NVIDIA의 원래 Isaac-GR00T 구현과 동등하게 테스트되었으며, 입력은 동일하고 출력도 같다. Flash-attention은 이제 선택 가능하며, `pip install 'lerobot[groot]'`는 그대로 작동하고 [NVIDIA's published checkpoints](https://huggingface.co/nvidia/GR00T-N1.7-3B)를 바로 로드할 수 있다.

> [!NOTE]
> GR00T N1.7은 LeRobot에서 N1.5를 대체합니다. N1.5가 필요하면 `lerobot==0.5.1`를 고정(pin)하십시오.

### MolmoAct2

Allen Institute for AI의 비전-언어-액션 모델 MolmoAct2가 LeRobot에 포팅되었으며, 전체 수명주기를 다룹니다: 미세 조정(전체 또는 LoRA), 평가, 그리고 실제 로봇 배포까지. 보정이 내장된 미리 제작된 체크포인트를 사용하면 SO-100/101에서 제로샷으로 실행할 수 있다:

```bash
lerobot-rollout \
  --policy.path=lerobot/MolmoAct2-SO100_101-LeRobot \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{cam0: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, cam1: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
  --task="pick up the red cube" --duration=30
```


추론은 bf16에서 약 12 GB로 가능하고, LoRA 미세 조정은 단일 24 GB GPU에서 가능하다. 전체 배포 가이드는 [MolmoAct2 documentation](https://huggingface.co/docs/lerobot/v0.6.0/molmoact2)를 참조하세요.

![MolmoAct2 Zero-Shot in LeRobot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/molmoact2_4.gif)

### EO-1

EO-1은 간섭된 비전-텍스트-액션 데이터로 사전학습된 VLA를 LeRobot에 합류시켰다: Qwen2.5-VL-3B 백본과 흐름 매칭 액션 헤드를 포함하며, 논문 저자 중 한 명이 기여했다. 표준 `lerobot-train` 워크플로우를 사용하여 `--policy.type=eo1`로 학습합니다. 세부 내용은 [documentation](https://huggingface.co/docs/lerobot/v0.6.0/eo1)와 [paper](https://arxiv.org/abs/2508.21112)에 있다.

### Multitask DiT

다중 작업 확산 트랜스포머 정책은 TRI Large Behavior Models 레시피를 LeRobot으로 가져온다: CLIP 비전 및 언어 임베딩으로 조건화된 약 4.5억 파라미터의 확산 트랜스포머로, 자연어로 선택된 여러 작업을 하나의 모델이 학습합니다. 확산 및 흐름 매칭 목표를 모두 지원하며, 직접 학습하기에 충분할 정도로 작습니다. 자세한 내용은 [documentation](https://huggingface.co/docs/lerobot/v0.6.0/multi_task_dit)를 참고하세요.

### EVO1

VLAs가 반드시 거대할 필요는 없다. EVO1은 0.77B 파라미터의 정책을 InternVL3-1B 백본과 함께, 흐름 매칭 액션 헤드를 탑재하고 담아 두어, 보유한 modest GPUs에서도 미세 조정 및 실시간 실행이 가능하다. 두 단계의 미세 조정과 실시간 청킹 지원이 기본으로 제공됩니다. [EVO1 documentation](https://huggingface.co/docs/lerobot/v0.6.0/evo1)와 [paper](https://arxiv.org/abs/2511.04555)를 참고하세요.

## 보상 모델: 로봇이 성공했는지 아는 방법 {#section-5}

성공 탐지와 진행 추정은 로봇 학습 루프에서 빠진 조각이며, v0.6.0이 이를 지원합니다. LeRobot은 이제 정책 API를 모방하는 단일 인터페이스 뒤에 네 가지 보상 모델을 제공하는 통합 보상 모델 API(`lerobot.rewards`)를 갖추었다. 여기에 HIL-SERL 보상 분류기, SARM, 그리고 두 가지 새로운 추가 항목이 포함됩니다:

### Robometer

Robometer는 사전학습된 범용 보상 모델입니다: 어떤 LeRobot 데이터셋이든 [lerobot/Robometer-4B](https://huggingface.co/lerobot/Robometer-4B)를 지시점으로 삼아 원시 비디오와 언어 지시를 바탕으로 작업 진행도와 성공 여부를 점수화하며, 특정 작업에 대한 학습이 필요 없다. Qwen3-VL-4B 위에 구축되었고, 백만 개가 넘는 로봇 궤적 데이터셋([RSS 2026 paper](https://arxiv.org/abs/2603.02115))에 걸친 궤적 비교를 통해 학습되었다.

![LeRobot Robometer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/rm_robometer.gif)

### TOPReward

TOPReward은 완전히 제로샷으로 동작합니다: 보상 가중치가 전혀 없다. 일반 상용 VLM(Qwen3-VL)을 래핑하고, 궤적 비디오와 작업 지시에 주어진 토큰 "True"의 로그-확률을 읽는다. 능력 있는 어떤 VLM도 보상 함수가 됩니다.

두 모델 모두 각 프레임의 진행 곡선을 데이터셋에 쓰는 라벨링 스크립트를 함께 제공하므로 보상 인식 행동 복제(RA-BC), 데이터셋 품질 검사, 진행 오버레이 동영상에 대비됩니다. [Robometer](https://huggingface.co/docs/lerobot/v0.6.0/robometer)와 [TOPReward](https://huggingface.co/docs/lerobot/v0.6.0/topreward) 문서를 확인하세요.

## 데이터셋: 더 빠른 로딩, 더 풍부한 데이터 {#section-6}

### 당신의 코덱, 당신의 규칙

녹음은 더 이상 하나의 하드코드된 코덱에 갇히지 않습니다. 새로운 `--dataset.rgb_encoder.*` 옵션은 전체 인코딩 표면(codec, 품질, 픽셀 형식, GOP, 프리셋)을 노출하고, `vcodec=auto`은 NVENC, VideoToolbox, VAAPI, QSV와 같은 하드웨어 인코더를 먼저 탐지한 후 기본 소프트웨어 AV1 인코더로 전환합니다. 기존 데이터셋에 대해서는 한 번의 명령으로 모든 것을 재인코딩합니다:

```bash
lerobot-edit-dataset \
    --repo_id ${HF_USER}/my_dataset \
    --operation.type reencode_videos \
    --operation.rgb_encoder.vcodec h264 \
    --operation.rgb_encoder.crf 23
```


전체 세부 정보는 [video encoding documentation](https://huggingface.co/docs/lerobot/v0.6.0/video_encoding_parameters)에 있습니다.

### 깊이 지원, 엔드 투 엔드

Intel RealSense를 연결하고 `use_depth: true`를 설정하면 LeRobot은 깊이 맵을 엔드 투 엔드로 기록합니다: 밀리미터 단위로 캡처되며, RGB 카메라 옆에 12비트 깊이 비디오 스트림으로 압축되고 학습 시 물리적 단위로 다시 디코딩됩니다. 깊이는 녹화 중 실시간으로 렌더링되며 `lerobot-dataset-viz`에서도 확인 가능하고, SO-100/101, Koch, OpenArm, reBot, Unitree G1 등에서도 작동합니다.

![LeRobot Depth Camera](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/depth2.gif)

### 대규모 언어 주석

데이터셋은 더 이상 에피소드당 하나의 작업 문자열이 아니다. LeRobot 데이터셋은 이제 풍부한 언어 주석(타임스탬프가 달린 하위 작업, 계획, 메모리, 수정, 음성, 및 카메라별 VQA 페어)을 네이티브로 저장하며, 새로운 `lerobot-annotate` CLI가 에피소드를 관찰하는 VLM을 사용해 이를 자동으로 채운다:

```bash
lerobot-annotate \
    --repo_id=${HF_USER}/my_dataset \
    --new_repo_id=${HF_USER}/my_dataset_annotated \
    --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct \
    --push_to_hub=true
```


그 다음 YAML 레시피 레이어가 이 주석들을 샘플 시점의 채팅 스타일 학습 메시지로 렌더링합니다: 내일의 장기적 계획과 대화를 나누는 로봇 정책이 학습할 정확한 데이터다. HF Jobs로 확장하고 더 자세한 내용은 [annotation pipeline docs](https://huggingface.co/docs/lerobot/v0.6.0/annotation_pipeline)를 읽어보라.

### 데이터 로딩 최대 2배 빠르게

비디오 데이터셋으로의 학습은 이제 기본적으로 약 2배 빠르다: 다중 카메라 프레임은 병렬로 디코딩되고, 데이터로더 워커는 컴팩트한 uint8 프레임을 전송하며(프로세스 간 메모리 4배 감소), 지속형 워커가 에포크 간에 디코더 캐시를 유지합니다. 대형 데이터셋의 하위 집합(`episodes=[...]`) 로딩은 벤치마크에서 275초에서 0.06초로 감소했다. 샘플링은 이제 결정적이고 재개 가능하므로 중단된 학습도 샘플-정확하게 재개됩니다.

## 벤치마크: 모든 것을 평가하는 하나의 CLI {#section-7}

<!-- TODO: gif idea: grid of rollouts across the six new benchmarks, hosted at documentation-images/lerobot-blog/release-v0.6.0/ -->

v0.5.0은 LeRobot을 VLAs 평가 허브로 확립했고, v0.6.0은 여섯 개의 새로운 시뮬레이션 벤치마크를 추가하여 실제 기능으로 만들었다. 모두 동일한 `lerobot-eval` CLI를 통해 실행 가능하고, 각 벤치마크에는 문서 페이지, Docker 이미지, CI에서 스모크 테스트된 SmolVLA 기준 체크포인트가 있다:

- [LIBERO-plus](https://huggingface.co/docs/lerobot/v0.6.0/libero_plus)는 VLAs에 약 10,000개의 변형 버전의 LIBERO를 7개 축에 걸쳐 스트레스 테스트하며, 조명과 카메라 시점에서부터 재작성된 지시문에 이르기까지 포괄합니다. 정책이 실패하는 시점을 알려줍니다.
- [RoboTwin 2.0](https://huggingface.co/docs/lerobot/v0.6.0/robotwin)은 SAPIEN에서 50개의 양손 조작 작업을 다루고, 도메인 무작위화를 크게 적용하며, Hub의 [more than 100k ready-to-train trajectories](https://huggingface.co/datasets/lerobot/robotwin_unified)를 포함합니다.
- [RoboCasa365](https://huggingface.co/docs/lerobot/v0.6.0/robocasa)은 모바일 매니퓰레이터의 2,500개 절차적으로 생성된 주방 작업 365개를 포괄하는 가장 큰 작업 면적을 다룹니다.
- [RoboCerebra](https://huggingface.co/docs/lerobot/v0.6.0/robocerebra)는 언어 기반의 중간 지시하에 3~6개의 하위 목표를 연결한 에피소드로 장기 행동을 평가하고, 6,660-에피소드 데이터셋을 제공합니다.
- [RoboMME](https://huggingface.co/docs/lerobot/v0.6.0/robomme)는 기억력 시험입니다: 정책이 반복 수를 셀 수 있는가, 숨겨진 물체를 추적하는가, 시연된 절차를 모방하는가? 4개의 기억 구간에서 16개의 작업.
- [VLABench](https://huggingface.co/docs/lerobot/v0.6.0/vlabench)은 조작에서의 지식과 추론을 테스트합니다. 물리 문제에서 커피 양조와 같은 복합 작업까지 다룹니다.

```bash
lerobot-eval \
  --policy.path=lerobot/smolvla_robotwin \
  --env.type=robotwin \
  --env.task=beat_block_hammer \
  --eval.n_episodes=100 --eval.batch_size=1
```


시뮬레이터 백엔드는 고유 설치 단계가 필요한 특정 시스템 의존성을 요구합니다. 각 문서 페이지에는 정확한 레시피가 있고, 설정을 건너뛰고 싶다면 각 벤치마크에는 미리 만든 Docker 이미지가 함께 제공합니다.

LIBERO, Meta-World, NVIDIA IsaacLab-Arena와 함께 한 지붕 아래 9개 벤치마크 계열이 구성되며, 새 [Adding a New Benchmark guide](https://huggingface.co/docs/lerobot/v0.6.0/adding_benchmarks)은 자신만의 벤치를 연결하는 방법을 정확히 문서화합니다. 평가도 빨라졌는데, 병렬 평가가 이제 비동기 벡터화 환경으로 기본값으로 설정되어 최대 2배 빠름으로 벤치마크됩니다.

![LeRobot benchmarks](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/benchmarks2.gif)

## 훈련 및 추론 {#section-8}

### `lerobot-rollout`: 배포 전용 CLI를 제공합니다

정책 배포는 예전에 `lerobot-record` 위에 얹은 해킹에 가까웠습니다. 새로운 `lerobot-rollout` CLI는 배포를 독립적인 워크플로우로 만들고, 플러그 가능한 전략과 추론 백엔드를 제공합니다(느린 호환 VLAs에 대해 실시간 청킹 포함). `base` 전략은 단순히 정책을 실행합니다. `sentry`는 연속적으로 기록하며 에피소드를 순환하고 진행 중 허브에 업로드합니다. `highlight`는 링 버퍼를 유지하고 키를 누르면 마지막 N초를 저장해 흥미로운 순간을 놓치지 않게 합니다. `episodic`는 전형적인 에피소드/리셋 기록 워크플로를 반영합니다. 그리고 `dagger`은 배포를 데이터 수집으로 바꿉니다.

DAgger 전략을 사용하면 정책이 작동하는 모습을 지켜보다, 잘못되었을 때 순간에 키를 누르고(USB 풋 페달도 가능) 리더 팔로 제어를 넘겨 보정을 기록한 뒤 다시 제어를 넘겨줍니다. 작동하는 리더는 제어를 넘길 때까지 팔로워의 포즈로 이동해 핸드오버를 매끄럽게 만듭니다. 모든 보정 프레임은 `intervention` 플래그가 달려 있으며, 결과 데이터세트는 다음 미세 조정을 위해 준비됩니다:

```bash
lerobot-rollout \
    --strategy.type=dagger \
    --policy.path=${HF_USER}/my_policy \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --dataset.repo_id=${HF_USER}/dagger_corrections \
    --dataset.single_task="Grasp the block"
```


배포하고, 보정을 수집하고, 미세 조정하고, 반복하세요: 로봇 학습 플라이휠은 이제 CLI 플래그입니다. [deployment docs](https://huggingface.co/docs/lerobot/v0.6.0/inference)를 읽어보라.

<!-- TODO: gif idea: DAgger takeover moment (policy fails, human grabs leader arm, correction recorded), hosted at documentation-images/lerobot-blog/release-v0.6.0/ -->

### FSDP: GPU보다 큰 모델 학습

로봇 파운데이션 모델은 단일 GPU를 넘어 커지고 있다. LeRobot 학습은 이제 Accelerate를 통해 FSDP(완전 분할 데이터 병렬) 학습을 지원합니다: 파라미터, 그래디언트, 옵티마이저 상태가 GPU 간에 샤딩되고, 체크포인트는 다른 정책처럼 로드되는 단일 파일 `model.safetensors`로 다시 모아진다. 다른 GPU 수에서 FSDP 실행을 재개할 수도 있다. [multi-GPU training docs](https://huggingface.co/docs/lerobot/v0.6.0/multi_gpu_training)를 참조하세요.

### HF Jobs로 클라우드 학습

GPU가 없나요? 문제 없습니다. 동일한 `lerobot-train` 명령에 하나의 플래그를 더하면 이제 클라우드에서도 실행됩니다.

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_test \
  --policy.type=act \
  --policy.repo_id=${HF_USER}/my_policy \
  --job.target=a10g-small
```


LeRobot은 필요하다면 로컬 데이터셋을 개인 Hub 리포지토리로 푸시하고, 작업을 제출하며, 로그를 터미널로 스트리밍하고, 학습된 정책을 끝에 Hub로 푸시합니다. T4에서 8x H200까지의 리소스를 `--job.target`으로 선택할 수 있으며(계산은 사용량에 따라 청구됩니다). [Check out the documentation](https://huggingface.co/docs/lerobot/v0.6.0/hardware_guide#hugging-face-jobs)

## 코드베이스: 더 간결하고 깔끔하게 {#section-9}

- `pip install lerobot`은 이제 본질적으로 가볍고, 기본 의존성이 약 40% 정도 줄었다. 기능 범위의 확장 추가(`[training]`, `[core_scripts]`, `[evaluation]`, ...)가 나머지를 커버하며, 누락된 의존성 오류는 어떤 확장을 추가해야 하는지 정확히 알려줍니다. LeRobot 데이터셋만 사용합니다면 더 이상 하드웨어 관련 의존성을 설치할 필요가 없다 ;) 
- 지원되는 PyTorch 버전은 2.7–2.11로 옮겨졌으며, CUDA 12.8 휠은 Linux `uv` 설치를 위해 기본으로 고정되어 있다. `--policy.dtype=bfloat16`은 이제 Accelerate를 통해 실제 혼합-정밀도 학습을 구동합니다.
- 약속된 `uv.lock`은 CI, Docker, 개발에 대한 권위 있는 의존성 명세이며, 문서에는 모든 단계에 대한 `uv` 설치 경로가 포함되어 있으며, CUDA 휠을 단일 플래그로 선택하는 방법까지 자세히 설명합니다.
- `--display_mode=foxglove`은 원격 조작, 녹화, 롤아웃을 [Foxglove](https://foxglove.dev)로 스트리밍하며, 이 시각화 도구는 로봇 공학 세계의 많은 사람들에 의해 이미 사용됩니다. 원격 설정에서도 작동하며, `lerobot-dataset-viz`은 정리 가능한 데이터셋 재생을 제공합니다.
- Pip로 설치 가능한 `lerobot_env_*` 패키지들은 이제 환경 сам 등록을 수행합니다. 플러그인 시스템은 로봇, 카메라, 원격 조작기, 정책, 엔벌(Environment) 등 다섯 가지 구성 요소 유형을 다룹니다.
- 녹화 중 키보드 컨트롤은 이제 Wayland에서, SSH를 통해, 헤드리스 설정에서, macOS에서도 접근성 권한 없이 작동합니다.

> [!WARNING]
> 전체 목록과 breaking changes에 대한 마이그레이션 포인터는 [release notes](https://github.com/huggingface/lerobot/releases)를 확인하십시오.

![LeLab a graphical user interface for LeRobot](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.6.0/gifs/LeLab.gif)

## 커뮤니티 & 생태계 {#section-10}

- LeLab은 전체 LeRobot 워크플로우(보정, 원격 조작, 녹화, 로컬 혹은 HF Jobs에서의 학습, 배포)를 브라우저 UI에 담아 CLI가 필요 없다. 현재 SO-ARM101을 지원합니다. [Try it out!](https://github.com/huggingface/leLab)
- Isaac Teleop은 NVIDIA 팀과의 협업으로 CloudXR/OpenXR를 통해 VR 컨트롤러로 SO-101를 원격 조작할 수 있다. 결과는 [NVIDIA's Isaac Teleop stack](https://github.com/NVIDIA/IsaacTeleop)이며, 더 자세한 내용은 [documentation](https://huggingface.co/docs/lerobot/v0.6.0/isaac_teleop)를 참조하세요.
- 새롭게 추가된 [compute hardware guide](https://huggingface.co/docs/lerobot/v0.6.0/hardware_guide)은 모든 신규 사용자가 묻는 두 가지 질문에 답합니다: 어떤 GPU가 필요한가, 학습 시간이 얼마나 걸리는가? 정책 계열별로 측정된 VRAM 엔벨로프와 RTX 4090에서 4x H100까지의 참조 학습 시간을 제공합니다.
- 재작성된 [Adding a Policy guide](https://huggingface.co/docs/lerobot/v0.6.0/bring_your_own_policies)은 PR 없이도 자체 정책을 택배하는 방법을 보여줍니다.

## Final thoughts {#section-11}

이들 핵심 기능 외에도 v0.6.0은 코드베이스 전반에 걸친 수백 가지 버그 수정, 문서 개선, 그리고 더 나은 기본값에서 더 신뢰할 수 있는 CI까지 다양한 품질 개선을 포함합니다.

커뮤니티 모든 분들께 거대한 감사의 마음을 전합니다. 이번 릴리스에는 학계, 산업계, 취미 팀이 LeRobot을 그들의 모델과 벤치마크의 보금자리로 선택한 작업이 포함되어 있다. 모든 PR과 버그 리포트가 오픈 소스 로보틱스를 앞으로 나아가게 합니다.

더 많은 소식이 곧 다가옵니다 🤗 시작하려면 [here](https://github.com/huggingface/lerobot)를 확인하세요!
– The LeRobot 팀 ❤️
