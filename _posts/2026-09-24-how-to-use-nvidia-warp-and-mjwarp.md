---
layout: post
title: "NVIDIA Warp와 MjWarp를 사용해 로보틱스 시뮬레이션 및 학습 워크플로 가속하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6994dc99f850a10f03fd0b21/rQ6tGCJEaH16bQQ8X4M8b.png
image: assets/images/blog/posts/2026-09-23-how-to-use-nvidia-warp-and-mjwarp/thumbnail.png
authors:
  - user: nvidia
slug: "how-to-use-nvidia-warp-and-mjwarp"
source_url: "https://huggingface.co/blog/nvidia/how-to-use-nvidia-warp-and-mjwarp"
source_published_date: "2026-09-23"
source_published_at: "2026-09-23T18:41:40+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [How to Use NVIDIA Warp and MjWarp to Accelerate Robotics Simulation and Learning Workflows](https://huggingface.co/blog/nvidia/how-to-use-nvidia-warp-and-mjwarp)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/how-to-use-nvidia-warp-and-mjwarp -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# NVIDIA Warp와 MjWarp를 사용해 로보틱스 시뮬레이션 및 학습 워크플로 가속하기

기존 MuJoCo는 로봇 개발, 테스트, 제어를 위한 빠른 CPU 기반 [robot simulation](https://www.nvidia.com/en-us/use-cases/robotics-simulation/)을 제공하며 CPU 코어 전체에서 샘플링을 병렬화할 수 있습니다. 하지만 학습 워크로드가 커지면 질문은 하나의 월드가 얼마나 빠르게 실행되는가에서 한 번에 얼마나 많은 월드를 실행할 수 있는가로 바뀝니다. GPU 가속을 사용하면 시뮬레이션 및 학습 데이터를 디바이스에 가깝게 유지하면서 이러한 월드를 대규모 배치로 진행할 수 있습니다.

[MuJoCo Warp (MJWarp)](https://mujoco.readthedocs.io/en/latest/mjwarp/)은 [NVIDIA Warp](https://developer.nvidia.com/warp-python)을 기반으로 하며, 호환되는 MuJoCo 모델을 GPU 규모의 실행 영역으로 가져옵니다. 이 글에서는 SO-101 follower arm을 익숙한 MuJoCo 워크플로에서 최대 2,048개의 병렬 MJWarp 환경으로 옮기고, 이러한 전환을 가능하게 하는 기술과 검증 단계를 살펴봅니다.

[![image1](https://cdn-uploads.huggingface.co/production/uploads/6994dc99f850a10f03fd0b21/8SmNNsqLiBHC0W31zT4cf.png)](https://cdn-uploads.huggingface.co/production/uploads/6994dc99f850a10f03fd0b21/8SmNNsqLiBHC0W31zT4cf.png)
그림 1. MJWarp가 Python을 GPU 시뮬레이션에 연결하는 방식. MuJoCo는 MJCF 모델을 로드하고 컴파일하며, MJWarp는 NVIDIA Warp에서 물리 연산을 구현하고 CUDA 커널을 컴파일하여 NVIDIA GPU에서 시뮬레이션 상태를 진행합니다.

이 글은 Physical AI를 위한 시뮬레이션의 현황 시리즈의 두 번째 글입니다. 첫 번째 글에서는 로봇 시뮬레이션 생태계를 살펴보았습니다. 여기서는 시뮬레이션 환경을 준비하고 확장하며, 정책을 학습하지는 않습니다. 이후 Newton 및 Isaac Lab 글에서 다음 통합 계층을 다룹니다.

## 구성 요소 통합 {#section-1}

| 계층 | 스택에서의 역할 |
| --- | --- |
| NVIDIA Warp | Python 커널 언어: 단일 명령, 다중 스레드(SIMT), autodiff, PyTorch/JAX interop |
| MJWarp | Warp에서 실행되는 MuJoCo 물리: 동일한 MJCF, 배치 GPU 처리량 |
| 사용자의 scene (SO-101) | 익숙한 Menagerie / Robot Studio asset + task geometry |
| 다음 단계 (Newton / Isaac Lab) | 멀티 솔버 API, USD, 센서, 매니저, 학습 루프 |

선택을 위한 간단한 기준:

| 필요한 작업… | 선택할 도구… |
| --- | --- |
| 단일 로봇 MPC / teleop | MuJoCo CPU |
| 순수 MuJoCo 물리에서 최대 처리량 | MJWarp (또는 [mjlab](https://github.com/mujocolab/mjlab)) |
| JAX 학습 레시피 | [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) / MJX (impl='warp') |
| 멀티 솔버 + Isaac Lab 통합 | Newton — 이 시리즈의 다음 글 |

## 유용한 Warp Kernel 하나로 시작하기 {#section-2}

[NVIDIA Warp](https://github.com/NVIDIA/warp)은 고성능 GPU 가속 커널을 작성하기 위한 Python 프레임워크입니다. Warp를 사용하면 개발자가 Python으로 정적 타입 커널을 작성하고 CPU 또는 CUDA 실행을 위해 컴파일할 수 있습니다. 최초 실행 시 네이티브 모듈을 빌드하고 캐시하며, 이후 실행에서는 이를 재사용합니다. 커널 언어는 성능 지향적인 Python의 부분집합이고, 일반 Python은 구성, 할당, 실행 오케스트레이션을 담당합니다.

이 작은 로보틱스 지향 커널은 중력 아래에서 점의 위치를 진행합니다. 하나의 논리 스레드가 하나의 점을 처리하므로, 제어 흐름에 GPU 용어를 도입하지 않고도 동일한 코드가 두 개의 점에서 수백만 개의 점까지 확장됩니다.

Warp의 세 가지 핵심 가치는 다음과 같습니다.

| 핵심 요소 | 얻을 수 있는 것 |
| --- | --- |
| 성능 | JIT 컴파일, 커널 융합, CUDA Graphs를 통한 네이티브 CUDA 속도 |
| 사용 편의성 | 기본 제공 벡터, 행렬, 쿼터니언, BVH, 해시 그리드, 희소 행렬, 타일 프리미티브를 활용한 순수 Python 작성 |
| 기능 | 미분 가능한 커널과 DLPack 스타일 interop를 통해 시뮬레이션을 ML 학습 루프 내부에 배치 |

```
import numpy as np
import warp as wp

@wp.kernel
def integrate(
   positions: wp.array[wp.vec3],
   velocities: wp.array[wp.vec3],
   dt: float,
):
   i = wp.tid()
   velocities[i] += wp.vec3(0.0, 0.0, -9.81) * dt
   positions[i] += velocities[i] * dt

wp.init()
device = "cuda:0" if wp.is_cuda_available() else "cpu"
start = np.array([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5]], dtype=np.float32)
positions = wp.array(start, dtype=wp.vec3, device=device)
velocities = wp.zeros_like(positions)

wp.launch(
   integrate,
   dim=len(start),
   inputs=[positions, velocities, 0.01],
   device=device,
)
wp.synchronize_device(device)
print(positions.numpy())
```


### 다음 세 가지 특성이 로보틱스에서 Warp를 유용하게 만듭니다:

- 명시적인 병렬 작업. wp.tid()는 현재 논리 스레드가 담당하는 점, 접촉, 바디 또는 월드를 식별합니다.

- 명시적인 디바이스 배열. 배열은 선택된 디바이스에 존재합니다. CUDA 배열에서 .numpy()를 호출하면 동기화가 발생하고 CPU 메모리로 복사됩니다. 이는 zero-copy 경로가 아닙니다. 디바이스 상주 PyTorch 또는 JAX 파이프라인에서는 Warp의 프레임워크 어댑터나 DLPack 호환 공유를 대신 사용하세요.

- 조합 가능한 커널 실행. 프로그램은 일련의 목적별 커널을 실행하고 지원되는 CUDA 작업을 그래프로 캡처하여 반복적인 dispatch 오버헤드를 줄일 수 있습니다. 그래프 캡처는 기존 버퍼에 대한 실행을 재생하며, 임의의 커널을 융합하지는 않습니다.

### 미분 가능성과 결정성

이 글의 SO-101 워크플로에서는 사용하지 않지만, Warp의 두 가지 추가 기능도 알아둘 가치가 있습니다. Warp 커널은 미분 가능합니다. wp.Tape는 해당 컨텍스트 내부에서 실행된 정방향 커널 실행을 기록하고, backward()가 호출되면 그 adjoint를 역순으로 재생합니다. 따라서 팀들은 시뮬레이션 및 설계 최적화를 위한 [CAE workflows](https://developer.nvidia.com/topics/cae)을 포함해 Warp에서 미분 가능한 geometry, CFD, 사용자 정의 물리를 구축합니다. Warp는 Warp 1.15에서 도입된 결정적 실행도 지원합니다. GPU atomic은 기본적으로 스케줄러에 따라 실행되므로 동일한 커널을 반복 실행해도 결과가 약간 달라질 수 있으며, opt-in 결정적 모드는 시뮬레이션, 검증, 회귀 테스트에서 재현 가능한 순서를 얻는 대신 일부 성능을 양보합니다. 이는 Warp의 기능이며, 전체 MJWarp rollout의 미분 가능성이나 결정성을 보장하는 것은 아닙니다. 자세한 내용은 미분 가능성과 결정적 실행에 관한 Warp 문서를 참고하세요.

Warp를 사용해 보세요: pip install warp-lang( GPU 결정성에는 ≥ 1.15), 그 다음 python -m warp.examples.browse를 실행하거나 [tutorial notebooks](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/tutorials/warp/notebooks)을 사용하세요.

## MuJoCo Warp(MJWarp)란? {#section-3}

로봇 시뮬레이터는 다음에 일어날 일을 반복적으로 계산합니다. 즉, 현재 관절 위치, 속도, 제어 입력, 접촉을 바탕으로 작은 timestep만큼 scene을 진행합니다. 이 글에서 월드는 해당 scene과 상태를 독립적으로 복사한 하나를 의미합니다. 한 월드에는 큐브를 향해 움직이는 SO-101 arm이 있을 수 있고, 다른 월드에는 약간 다른 pose에서 시작하는 동일한 arm이 있을 수 있습니다.

MuJoCo와 MJWarp는 동일하게 호환되는 로봇과 task를 실행할 수 있지만, 작업을 구성하는 방식은 다릅니다. MuJoCo는 하나 또는 소수의 CPU 월드를 개발하고 검사하는 데 자연스럽게 적합합니다. MJWarp는 MuJoCo의 물리 파이프라인을 NVIDIA Warp로 구현한 것으로, 모델과 독립적인 상태의 배치를 NVIDIA GPU에 배치합니다. mjw.step을 한 번 호출하면 전체 배치가 진행됩니다.

MJWarp의 가치는 반드시 하나의 월드에서 더 빠른 step을 제공하는 데 있지는 않습니다. 수백 또는 수천 개의 월드를 함께 진행하여 GPU에 충분한 병렬 작업을 제공하고, 총 처리량, 즉 초당 완료되는 전체 world-step 수를 높일 수 있다는 점에 있습니다. 이는 경험 수집이 하나의 환경에서 지연 시간을 최소화하는 것보다 중요한 강화 학습 및 대규모 샘플링에 유리합니다.

이 글에서는 다음을 다룹니다.

- 하나의 MuJoCo 월드를 검증하고,

- 이를 MJWarp로 옮겨 배치를 구성하고,

- 올바르게 검증하고 측정합니다.

솔버 튜닝, Jacobian 표현, 특수한 멀티 GPU 또는 결정성 관련 주제는 이 마이그레이션에 필요하지 않으며 별도로 다룰 수 있습니다.

그러면 구분은 다음과 같이 명확합니다.

- 지연 시간은 하나의 시뮬레이션 step에 걸리는 wall-clock 시간입니다.

- 총 처리량은 측정된 wall-clock 1초 동안 완료되는 전체 world-step 수입니다.

### 기본 사용법: structs, batch sizes, minimal step

- 핵심 API 전환은 간단합니다.

| MuJoCo 호스트 워크플로 | MJWarp 워크플로 |
| --- | --- |
| mujoco.MjModel | mjw.put_model(mjm)이 디바이스 모델 생성 |
| mujoco.MjData | mjw.put_data(mjm, mjd, ...)가 기존 상태를 보존하고 배치화 |
| mujoco.mj_step(mjm, mjd) | mjw.step(m, d)이 d의 모든 월드를 진행 |
| mjd.ctrl과 같은 호스트 배열 | shape이 (nworld, nu)인 d.ctrl과 같은 배치 디바이스 배열 |

기본값/새 상태가 필요한 경우에는 mjw.make_data()를 사용하세요. 정확히 초기화된 MuJoCo 상태를 마이그레이션 경계를 넘어 전달해야 하는 경우에는 mjw.put_data()를 사용하세요.

배치 리소스를 할당하려면 다음 매개변수를 정의해야 합니다([Batch sizes](https://mujoco.readthedocs.io/en/latest/mjwarp/#batch-sizes) 참고).

| 매개변수 | 의미 |
| --- | --- |
| nworld | 병렬 환경의 총 개수 |
| nconmax | 각 월드에서 예상되는 접촉 수(전체 용량 ≈ nconmax * nworld) |
| naconmax | 대체 설정: 모든 환경을 합친 전역 최대 접촉 수(둘 다 정의된 경우 우선 적용) |
| njmax | 월드당 제약 조건의 하드 상한 |

### 성능 튜닝

1. CUDA graph capture:mjw.step은 많은 커널을 실행하므로 한 번 캡처하고 반복 재생합니다.

```
with wp.ScopedCapture() as capture:
     mjw.step(m, d)
wp.capture_launch(capture.graph)
```


2. nconmax / naconmax / njmax를 여유 없이 설정하세요. 메모리와 작업량은 이 값에 따라 증가합니다. mjwarp-testspeed: --measure_alloc으로 튜닝하고 mjwarp-viewer에서 overflow를 확인하세요.

추가 튜닝 고려 사항. 접촉 및 제약 조건 버퍼의 크기를 정한 후 task 동작을 변경하지 않고 솔버 반복 횟수 제한을 테스트하세요. Mesh와 CCD 설정은 메모리 사용량을 늘릴 수 있으며, 측정된 접촉 수가 허용하는 경우 nccdmax / naccdmax를 사용해 CCD 버퍼 할당을 줄일 수 있습니다. MJWarp의 compact solver는 별도의 Newton physics-engine framework가 아니라 MuJoCo의 Newton constraint solver와 sleeping을 사용합니다. Compact-solver 및 멀티 GPU 구성은 이 walkthrough의 범위를 벗어나므로 MJWarp 성능 튜닝 문서를 참고하세요.

MJWarp 물리에서 정책을 학습하려면 다음을 사용합니다.

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/feature/newton) via [Newton](https://github.com/newton-physics/newton)

- [mjlab](https://github.com/mujocolab/mjlab) (MJWarp + PyTorch에서 직접 manager API 사용)

- MJX (impl='warp')를 통한 [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)

설치 / 사용해 보기: pip install mujoco-warp · mjwarp-viewer path/to/scene.xml · [Colab tutorial](https://colab.research.google.com/github/google-deepmind/mujoco_warp/blob/main/notebooks/tutorial.ipynb)

## MuJoCo scene을 MjWarp로 마이그레이션하는 워크플로 {#section-4}

- MuJoCo CPU baseline 수립

Scene. 아직 MJWarp에 특화된 내용은 없습니다. 일반적인 MJCF로 작성한 SO-101 arm, table, stack할 두 개의 cube입니다.

[![image2](https://cdn-uploads.huggingface.co/production/uploads/6994dc99f850a10f03fd0b21/Tn07F7pRaVdJXRnB7jSV1.png)](https://cdn-uploads.huggingface.co/production/uploads/6994dc99f850a10f03fd0b21/Tn07F7pRaVdJXRnB7jSV1.png)

그림 2. MuJoCo CPU 시뮬레이션에서 렌더링한 SO-101 pick-and-place scene. task는 빨간색 44 mm cube를 집어 파란색 cube 위에 쌓는 것이며, 동일한 robot과 scene을 MJWarp 검증에 사용합니다.

```
<mujoco model="so101_pick_place">
  <include file="so101.xml"/>

  <worldbody>
    <light pos="0.3 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05"/>

    <geom name="table" type="box" pos="0.35 -0.04 0.012"
          size="0.16 0.26 0.012" rgba="0.32 0.32 0.32 1"
          friction="1 0.005 0.0005" condim="3"/>

    <body name="red_cube" pos="0.33 -0.13 0.046">
      <freejoint name="red_cube_joint"/>
      <geom type="box" size="0.022 0.022 0.022" mass="0.08"
            rgba="0.85 0.05 0.04 1" friction="1.2 0.005 0.0005" condim="3"/>
    </body>

    <body name="blue_cube" pos="0.33 0.06 0.046">
      <freejoint name="blue_cube_joint"/>
      <geom type="box" size="0.022 0.022 0.022" mass="0.08"
            rgba="0.05 0.20 0.90 1" friction="1.2 0.005 0.0005" condim="3"/>
    </body>
  </worldbody>
</mujoco>
```


MJCF box에서 size 값은 half-extents입니다. size=”0.022 …”는 모서리 길이가 44 mm인 cube를 정의합니다. task에서는 이 크기를 성공 조건의 threshold로 사용합니다. arm base는 원점에 있고, arm의 도달 방향은 +X이며, cube는 Y를 따라 배치됩니다.

동반 repository에서는 이 파일을 직접 작성하지 않고 생성합니다. resolve_pick_place_scene()이 Menagerie arm을 .generated/에 복사하고, robot profile에서 table과 cube 좌표를 채운 다음 scene_pick_place.xml을 작성합니다. walkthrough에서는 SO-101 profile을 사용하며, 선택 사항인 reBot variant는 아래에서 설명합니다.

로드하기. 컴파일과 step은 일반적인 MuJoCo 방식입니다.

```
import mujoco

mjm = mujoco.MjModel.from_xml_path("scene_pick_place.xml")
mjd = mujoco.MjData(mjm)

fps = 50 # controller rate
sim_substeps = 10 # physics steps per control frame
frame_dt = 1.0 / fps
mjm.opt.timestep = frame_dt / sim_substeps

controller = PickPlaceController(spec=spec) # waypoints + damped-least-squares IK

for _ in range(600): # 600 control frames
    ctrl = controller.step(mjm, mjd, frame_dt)
    for _ in range(sim_substeps):
        mjd.ctrl[: mjm.nu] = ctrl
        mujoco.mj_step(mjm, mjd)
```


이 구조를 기억해 두세요. 프레임마다 제어 입력을 한 번 계산하고, 물리 연산을 sim_substeps 횟수만큼 step합니다. Gate 2에서는 내부 루프만 변경하므로 마이그레이션을 쉽게 검토할 수 있습니다.

시뮬레이션 주기와 제어 주기를 일치시키세요. 초당 50개의 제어 프레임과 프레임당 10개의 물리 substep을 사용하는 경우 physics timestep은 0.002초로 설정합니다. CPU rollout 전에, 그리고 mjw.put_model을 사용해 모델을 업로드하기 전에 이를 설정하여 두 backend가 동일한 시뮬레이션 시간을 진행하도록 하세요.

```
mjm.opt.timestep = frame_dt / sim_substeps # 50 Hz × 10 substeps -> 0.002 s
```


이 줄이 없으면 이후의 모든 측정이 불일치를 물려받습니다. parity 비교, “simulated seconds”로 인용되는 처리량 수치, 그리고 action rate가 더 이상 deployment와 일치하지 않는 학습된 정책이 여기에 포함됩니다.

cube가 성공적으로 쌓였는지 확인하세요. 44 mm cube의 경우 성공은 두 가지 측정 가능한 조건으로 정의됩니다. 수평 중심 오차 xy_err ≤ 0.015 m(cube 중심 간 측정)와 cube 중심 간 수직 간격 0.035 m ≤ dz ≤ 0.055 m(하나의 cube edge에 해당하며 settling을 위한 여유 포함)입니다. cube가 안정된 후 두 조건을 모두 평가하세요. 프로세스가 성공적으로 종료되었다는 사실만으로 task 성공이 입증되지는 않습니다.

동반 checkout에서 CPU task를 실행하세요. 게시 전 차단 사항: 이 지침을 게시하기 전에 접근 가능한 repository URL과 고정된 dependency 및 asset 버전을 확인하세요. 아래 repository placeholder는 실행 가능한 URL이 아닙니다.

```
git clone https://github.com/NVIDIA/accelerated-computing-hub.git blogs
cd blogs/tutorials/sim2real-blogs/notebooks/mujoco
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

cd /tutorials/sim2real-blogs/notebooks/mujoco
python solutions/so101_pick_place_solution.py --headless-steps 600 --debug
```


실행이 끝나면 위의 두 수(stack check: xy_err=… dz=…)를 출력하며, 이는 글의 나머지 부분에서 비교하는 assertion입니다. 옆의 so101_pick_place.py는 동일한 프로그램이지만 physics step은 연습 과제로 남겨 둔 버전입니다.

arm은 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)에서 알려진 정상 commit으로 고정해 가져옵니다. Menagerie asset은 변경될 수 있으므로 scene을 template으로 취급하세요. 선택 사항인 reBot variant. 동반 code는 scene layout, gripper, capacity limit을 위한 별도의 profile과 함께 --robot rebot도 제공합니다(nconmax=256, njmax=500). 이 walkthrough에서는 SO-101을 사용합니다. 결과를 보고하기 전에 reBot asset과 task를 별도로 검증하세요.

- 단일 월드 MJWarp parity 검증

먼저 GPU에서 하나의 월드를 실행하되 host를 계속 루프에 포함하세요. 그러면 동일한 viewer에서 동일한 task를 확인하고 동일한 두 수를 비교할 수 있습니다. 모델을 업로드하고, 배치 상태를 할당하고, 초기화된 host 상태로 시드한 다음, step하기 전에 한 번의 forward pass를 실행합니다.

```
wp.init()
import mujoco_warp as mjw

device = wp.get_device()

m = mjw.put_model(mjm)
d = mjw.make_data(mjm, nworld=1, nconmax=spec.nconmax, njmax=spec.njmax)

wp.copy(d.qpos, wp.array(mjd.qpos[None, :], dtype=wp.float32, device=device))
wp.copy(d.qvel, wp.array(mjd.qvel[None, :], dtype=wp.float32, device=device))
wp.copy(d.ctrl, wp.array(mjd.ctrl[None, :], dtype=wp.float32, device=device))
mjw.forward(m, d)
```


모든 device array에는 선행 world 차원이 있으므로 host state는 (nq,)가 아니라 shape (1, nq)인 mjd.qpos[None, :]로 인덱싱합니다. 이후 수천 개의 world로 확장할 때 변경되는 것은 이 선행 차원뿐이며 호출 자체는 바뀌지 않습니다. mjw.put_model()은 호환성 검사 역할도 합니다. 지원되지 않는 기능을 사용하는 model이면 이를 조용히 삭제하지 않고 오류를 발생시킵니다.

세 필드를 명시적으로 시드하는 방식은 투명한 선택이며, 정확히 무엇이 device로 전달되는지 명확히 보여 줍니다. 대신 mjw.put_data(mjm, mjd, nworld=…)는 초기화된 전체 struct를 한 번의 호출로 전달합니다.

이제 frame loop는 내부 step을 GPU로 전환하고 다시 미러링하는 Gate 1 loop가 됩니다.

```
def simulate_frame() -> None:
    ctrl = controller.step(mjm, mjd, frame_dt)
    for _ in range(sim_substeps):
        mjd.ctrl[: mjm.nu] = ctrl
        wp.copy(d.ctrl, wp.array(mjd.ctrl[None, :], dtype=wp.float32, device=device))
        mjw.step(m, d)
        mjd.qpos[:] = d.qpos.numpy()[0]
        mjd.qvel[:] = d.qvel.numpy()[0]
    mujoco.mj_forward(mjm, mjd)
```


.numpy() 읽기 작업은 매 substep마다 동기화하고 host로 데이터를 복사하므로, 이는 task 검증 경로이지 처리량 벤치마크가 아닙니다. inverse kinematics, viewing, task check는 host에서 계속 수행됩니다. qpos와 qvel을 복사한 후에는 mujoco.mj_forward(mjm, mjd)를 호출하여 제어, viewing, stack check에 사용하기 전에 mjd.xpos와 같은 파생 host quantity를 갱신하세요. loop 후 이러한 필드를 읽는다고 해서 자동으로 갱신되지는 않습니다. Gate 4에서는 처리량 경로에서 이러한 step별 host 복사를 제거합니다.

- 접촉 및 제약 조건 용량 설정

MJWarp는 step하기 전에 접촉 및 제약 조건 buffer를 할당합니다. 이러한 용량을 초과하면 exception 없이 overflow warning과 함께 실행이 계속되더라도 검증 또는 벤치마킹을 위한 해당 rollout은 무효가 됩니다. 관련 limit을 늘리고 task를 다시 실행하세요. 더 큰 buffer는 더 많은 GPU 메모리를 사용하므로, 할당을 줄이기 전에 전체 task에서 용량을 검증하세요.

시뮬레이션할 robot과 task에 맞춰 접촉 및 제약 조건 limit을 설정하세요. SO-101 profile은 시작 용량으로 nconmax=128과 njmax=300을 사용합니다. task에서 접촉이 가장 많은 구간에 이 limit이 충분한지 확인하세요.

```
d = mjw.make_data(mjm, nworld=nworld, nconmax=spec.nconmax, njmax=spec.njmax)
```


task에서 접촉이 가장 많은 순간을 기준으로 크기를 정하세요. pick-and-place에서는 arm이 빈 공간에 떠 있는 순간이 아니라 양쪽 jaw와 table이 cube에 닿는 순간입니다. overflow는 발생시켜 올리는 대신 보고됩니다. Option.warn_overflow가 기본값인 경우 MJWarp는 증가시켜야 할 budget(“narrowphase overflow - please increase nconmax to …”)을 script 또는 viewer가 실행 중인 terminal에 출력하고, 영향을 받은 world를 Data.overflow에 표시하여 step 후 읽을 수 있게 합니다. mjw.put_data만은 이미 보유한 MuJoCo state와 budget을 비교할 수 있으므로 오류를 즉시 발생시킵니다. mjwarp-testspeed --measure_alloc은 scene이 실제로 사용한 contact와 constraint를 보고하며, 어느 world에서든 overflow가 발생하는 즉시 문제가 발생한 world ID와 함께 rollout을 중단합니다. 이러한 보고를 실패로 취급하세요. limit을 높이고 trajectory나 benchmark를 신뢰하기 전에 다시 실행한 다음, model, collision geometry 또는 task가 변경될 때마다 다시 조정하세요.

- 2,048개 world로 확장

단일 월드 parity가 통과하면 target size로 다시 할당하고 초기화된 상태를 batch 전체에 복제합니다. Gate 2와 비교해 바뀌는 것은 두 가지입니다. nworld와, step마다 PCIe bus를 통해 아무것도 전달되지 않는다는 점입니다.

```
nworld = 2_048
d = mjw.make_data(mjm, nworld=nworld, nconmax=spec.nconmax, njmax=spec.njmax)

wp.copy(d.qpos, wp.array(np.tile(mjd.qpos, (nworld, 1)), dtype=wp.float32, device=device))
wp.copy(d.qvel, wp.array(np.tile(mjd.qvel, (nworld, 1)), dtype=wp.float32, device=device))
wp.copy(d.ctrl, wp.array(np.tile(mjd.ctrl, (nworld, 1)), dtype=wp.float32, device=device))
mjw.forward(m, d)

with wp.ScopedCapture() as capture:
    mjw.step(m, d)
step_graph = capture.graph
```


np.tile은 모든 world에 동일한 시작 상태를 제공하며, 이는 처리량 측정에 적합한 baseline입니다. 반대로 world별 randomization을 사용하면 device에서 d.qpos의 서로 다른 row에 서로 다른 값을 기록하게 됩니다.

CUDA Graphs는 여기서 캡처한 model과 data buffer를 재사용합니다. replay 사이에는 d.ctrl을 제자리에서 업데이트하고, buffer를 교체하거나 nworld를 변경하거나 model을 다시 빌드한 후에는 새 graph를 캡처하세요. Graph capture에는 CUDA가 필요합니다.

[![image3](https://cdn-uploads.huggingface.co/production/uploads/6994dc99f850a10f03fd0b21/UMuh1gwUTiT1VKcO6E6w1.png)](https://cdn-uploads.huggingface.co/production/uploads/6994dc99f850a10f03fd0b21/UMuh1gwUTiT1VKcO6E6w1.png)

그림 3. 동일하게 호환되는 model을 사용해 SO-101 task를 하나의 CPU world에서 2,048개의 독립적인 GPU state로 확장하는 모습. 하나의 MJWarp step이 전체 batch를 진행합니다. 이 개념도는 wall-clock 1초당 world-step 수로 측정한 총 처리량을 강조합니다.

- 검증한 후 측정

GPU 실행은 비동기이므로 단순한 timer는 GPU가 작업을 완료하는 속도가 아니라 Python이 작업을 queue에 넣는 속도를 측정합니다. 먼저 warm-up을 수행하세요. 최초 실행에는 kernel compilation과 allocation 비용이 발생합니다. 그런 다음 측정 구간의 직전과 직후에 즉시 동기화합니다.

```
import time
for _ in range(10): # warm-up: compilation, allocation, caches
    wp.capture_launch(step_graph)
wp.synchronize()

t0 = time.perf_counter()
for _ in range(200):
    wp.capture_launch(step_graph)
wp.synchronize() # without this you time the queue, not the work
elapsed = time.perf_counter() - t0
total = 200 * nworld
print(f"{total / elapsed:,.0f} world-steps/second")
```


batch size와 함께 총 world-steps per second와 batched step당 밀리초를 모두 보고하세요. 측정된 curve를 사용해 추가 world가 처리량을 향상시키는 지점과 메모리 또는 연산 한계로 이점이 감소하는 지점을 식별하세요. 결과는 scene, simulation setting, hardware에 따라 달라집니다. 단일 월드의 지연 시간 비교만으로는 배치 처리량을 입증할 수 없습니다.

사용 중인 hardware에서 해당 curve를 확인하려면 scaling_study.py가 batch size를 순회하며 ms/step과 처리량 및 speedup을 함께 출력합니다.

```
cd /tutorials/sim2real-blogs/notebooks/mujoco/part2
python solutions/so101_mjwarp_solution.py --headless-steps 600     # parity, needs CUDA
python scaling_study.py --worlds 1 64 1024 2048 8192 --steps 100
```


## 시작하기 {#section-5}

Warp (kernel layer) pip install warp-lang → python -m warp.examples.browse → [docs](https://nvidia.github.io/warp/) · [GitHub](https://github.com/NVIDIA/warp)

MJWarp (GPU MuJoCo) pip install mujoco-warp → mjwarp-viewer benchmarks/humanoid/humanoid.xml → [docs](https://mujoco.readthedocs.io/en/latest/mjwarp/) · [GitHub](https://github.com/google-deepmind/mujoco_warp) · [Colab tutorial](https://colab.research.google.com/github/google-deepmind/mujoco_warp/blob/main/notebooks/tutorial.ipynb)

SO-101 context [SO-101 sim-to-real course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) · [Physical AI learning paths](https://docs.nvidia.com/learning/physical-ai/)

MJWarp 위에서 학습하기 [mjlab](https://github.com/mujocolab/mjlab) · [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) · Isaac Lab + Newton(예정된 글)

## 다음 단계 {#section-6}

이 글에서는 raw Warp → MJWarp를 다뤘습니다. GPU kernel, batched stepping, 그리고 mjw.step을 사용하는 SO-101 scene이 내용에 포함됩니다.

다음으로 동일한 MJCF environment를 Newton으로 포팅하고, MuJoCo Warp를 rigid-body solver(newton.solvers.SolverMuJoCo)로 사용합니다. Newton은 model, state, control, contact를 관리하며 MJWarp는 내부에서 실행됩니다.

또한 Newton이 추가하는 기능도 살펴봅니다. multi-format asset, 교체 가능한 solver, sensor/IK helper, Isaac Lab 경로가 여기에 포함됩니다.

마이그레이션 가이드는 동일한 SO-101 task와 선택 사항인 reBot profile을 계속 사용하며, Newton에 필요한 변경 사항과 별도의 Isaac Lab 통합을 설명합니다.

Warp 또는 MJWarp로 무언가를 구축했다면 링크된 repository에 issue를 열거나 Discord [NVIDIA Omniverse](https://discord.com/invite/nvidiaomniverse)에서 저희를 찾아보세요.

## 참고 자료 {#section-7}

- Blog 1: *[The State of Simulation for Physical AI: An Overview*](https://huggingface.co/blog/nvidia/state-of-simulation-for-physical-ai) — 이 시리즈의 첫 번째 글.

- [NVIDIA Warp — GitHub](https://github.com/NVIDIA/warp) · [Documentation](https://nvidia.github.io/warp/) · [v1.15.0 release (GPU determinism)](https://github.com/NVIDIA/warp/releases/tag/v1.15.0) · [Deterministic execution guide](https://nvidia.github.io/warp/latest/user_guide/execution_and_performance/deterministic_execution.html)

- [MuJoCo Warp — GitHub](https://github.com/google-deepmind/mujoco_warp) · [Official MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/)

- [Build Accelerated, Differentiable Computational Physics Code for AI with NVIDIA Warp](https://developer.nvidia.com/blog/build-accelerated-differentiable-computational-physics-code-for-ai-with-nvidia-warp/)

- [Introducing Tile-Based Programming in Warp 1.5.0](https://developer.nvidia.com/blog/introducing-tile-based-programming-in-warp-1-5-0/)

- [mjlab](https://github.com/mujocolab/mjlab) · [arXiv:2601.22074](https://arxiv.org/abs/2601.22074)

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)

- [NVIDIA SO-101 sim-to-real course](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)

- [Newton](https://github.com/newton-physics/newton) 다음 글: SolverMuJoCo로서의 MJWarp 및 이 environment 포팅
