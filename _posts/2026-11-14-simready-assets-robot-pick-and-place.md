---
layout: post
title: "SimReady Assets로 로봇 Pick-and-Place 시뮬레이션 구축하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/HFZWtQ2i9VN_agjeUw2HB.gif
image: assets/images/blog/posts/2026-11-13-simready-assets-robot-pick-and-place/thumbnail.gif
authors:
  - user: nvidia
slug: "simready-assets-robot-pick-and-place"
source_url: "https://huggingface.co/blog/nvidia/simready-assets-robot-pick-and-place"
source_published_date: "2026-11-13"
source_published_at: "2026-11-13T18:00:39+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Build a Robot Pick-and-Place Simulation with SimReady Assets](https://huggingface.co/blog/nvidia/simready-assets-robot-pick-and-place)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/simready-assets-robot-pick-and-place -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# SimReady Assets로 로봇 Pick-and-Place 시뮬레이션 구축하기

[![figure-01-pick-and-place-highlight](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/NlngGWIKZEQ8kuWKZtoyM.gif)](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/NlngGWIKZEQ8kuWKZtoyM.gif)

이 튜토리얼에서는 [UR17G arm](https://huggingface.co/datasets/TeradyneRobotics/UniversalRobots/blob/d5d8eb795a487a5c929909bc337ee4834d62b0da/ur17g_1300/simready_usd/ur17g_1300.usda), [Robotiq 2F-140 gripper](https://huggingface.co/datasets/Robotiq-Official/simready-assets/blob/9c6b0cedf0066efd474b3bf473e7eaf7ae7e569f/Robotiq_2F_85/simready_usd/Robotiq_2F_85.usda), [beverage can](https://huggingface.co/datasets/nvidia/simready-assets/blob/653d91e182011f357cf1644013cc94ddf194d349/food_beverage_can_n01/simready_usd/sm_food_beverage_can_n01_01.usd)을 사용해 Isaac Sim에서 장면을 구성하고, 동작을 확인한 다음, 에이전트가 생성한 제어로 pick-and-place 예제를 실행하는 음료 캔 pick-and-place 작업을 소개합니다.

로보틱스 워크플로에서 SimReady란 특정 시뮬레이션 사용 사례에 필요한 속성(예: 충돌, 질량, 관절, joint drive)이 설정된 [OpenUSD](https://www.nvidia.com/en-us/glossary/openusd/) asset을 의미합니다. Hugging Face의 오픈 asset 라이브러리인 [SimReady Central](https://huggingface.co/spaces/nvidia/simready-central)을 사용하면 시뮬레이션 속성과 프로필 검증 근거가 작성된 로봇 및 객체 asset을 재사용할 수 있습니다. 라이브러리와 출시 내용에 대한 자세한 내용은 [announcement blog](https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories)를 참조하세요.

## 로봇 asset을 SimReady로 만드는 요소는 무엇인가요? {#section-1}

[SimReady Foundation](https://nvidia.github.io/simready-foundation/)은 시뮬레이션 사용 사례에 OpenUSD asset이 갖춰야 할 기능을 정의합니다. 사양은 다음 세 가지 수준을 연결합니다.

- 프로필: 로봇 본체와 같은 특정 사용 사례를 위한 기능 그룹입니다.

- 기능: 제어 가능한 관절과 같이 사용 사례에 필요한 기능입니다.

- 요구 사항: 관절의 drive 설정과 같이 각 기능에 필요한 설정을 확인하는 검사입니다.

이 작업에서 이러한 속성에는 arm과 gripper의 관절, 그리고 beverage can asset의 충돌과 질량이 포함됩니다. 검증에서는 프로필의 요구 사항에 따라 이러한 설정을 확인합니다. 그런 다음 시뮬레이션에서 조립된 로봇이 이를 사용해 캔을 집어 이동할 수 있는지 테스트합니다.

## asset에 포함된 항목과 사용자가 추가하는 항목은 무엇인가요? {#section-2}

[UR17G](https://huggingface.co/datasets/TeradyneRobotics/UniversalRobots/blob/d5d8eb795a487a5c929909bc337ee4834d62b0da/ur17g_1300/simready_usd/ur17g_1300.usda), [Robotiq 2F-140](https://huggingface.co/datasets/Robotiq-Official/simready-assets/blob/9c6b0cedf0066efd474b3bf473e7eaf7ae7e569f/Robotiq_2F_85/simready_usd/Robotiq_2F_85.usda), [beverage can](https://huggingface.co/datasets/nvidia/simready-assets/blob/653d91e182011f357cf1644013cc94ddf194d349/food_beverage_can_n01/simready_usd/sm_food_beverage_can_n01_01.usd) 패키지에는 모델과 시뮬레이션 설정이 포함되어 있습니다. 사용자는 장면을 조립하고 작업 코드를 추가해 arm–gripper 조립체, 작업 표면, 시작 위치, arm 이동 및 gripper 열기와 닫기를 위한 제어를 정의합니다.

[![Figure 2](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/J7YL_C3mUh7MWUNJ_pVtN.png)](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/J7YL_C3mUh7MWUNJ_pVtN.png)

그림 2. SimReady Central은 asset 모델과 시뮬레이션 설정을 제공하며, 예제 장면과 제어는 pick-and-place 작업을 정의합니다.

## pick-and-place 작업을 실행하기 전에 무엇을 확인해야 하나요? {#section-3}

로봇은 Isaac Sim에서 정지해 있을 때 시각적으로는 올바르게 보이더라도, 시뮬레이션이 실행되면 잘못 움직이거나 객체와 올바르게 상호작용하지 않을 수 있습니다. pick-and-place를 실행하기 전에 관절, 이동, 접촉 동작을 테스트하세요. 에이전트에게 이러한 테스트를 수행하도록 요청하거나 Isaac Sim에서 직접 수행할 수 있습니다.

- 관절 설정: arm과 gripper의 관절 제한 및 drive 설정을 검토합니다.

- 이동: arm을 조금 움직이고 gripper를 열고 닫는 동작을 테스트합니다.

- 접촉: 캔이 작업 표면을 뚫고 떨어지지 않고 그 위에 놓이는지 확인합니다.

pick-and-place 중에는 arm이 움직일 때 beverage can이 gripper 안에 계속 유지되는지 관찰하세요. asset 구조는 서로 다르므로, 로드한 장면의 설정을 출발점으로 사용하세요. 아래 단계에서 asset을 다운로드하고 장면을 구성하는 전체 과정을 살펴보겠습니다.

## Isaac Sim에서 pick-and-place 시뮬레이션 구축하기 {#section-4}

이 예제에서는 Isaac Sim에서 [UR17G](https://huggingface.co/datasets/TeradyneRobotics/UniversalRobots/blob/d5d8eb795a487a5c929909bc337ee4834d62b0da/ur17g_1300/simready_usd/ur17g_1300.usda) arm, [Robotiq 2F-140](https://huggingface.co/datasets/Robotiq-Official/simready-assets/blob/9c6b0cedf0066efd474b3bf473e7eaf7ae7e569f/Robotiq_2F_85/simready_usd/Robotiq_2F_85.usda) gripper, [beverage can](https://huggingface.co/datasets/nvidia/simready-assets/blob/653d91e182011f357cf1644013cc94ddf194d349/food_beverage_can_n01/simready_usd/sm_food_beverage_can_n01_01.usd)을 결합합니다. 작업은 캔을 잡고, 들어 올리고, 목표 지점으로 이동한 다음, 놓는 것입니다. 또한 [SimReady Central Space](http://nvidia-simready-central.hf.space) search service endpoint를 활용해 OEM이 선별한 콘텐츠를 검색할 수도 있습니다.

또한 frontier AI model을 사용해 개방된 Isaac Sim 장면에서 작업 코드를 생성하고 실행하는 데 도움을 받습니다. 이 예제에서는 GPT-6 Sol을 사용하지만, 필요한 skill을 로드하고 Isaac Sim에 대해 Python을 실행할 수 있다면 다른 agent와 model 조합도 사용할 수 있습니다(생성된 코드와 결과는 달라질 수 있습니다). 프롬프트는 출발점을 제공합니다. 작업을 조정하기 전에 생성된 코드를 검사하고 그 결과로 나타난 동작을 확인하세요.

필요한 소프트웨어:

- 이 예제에서 사용하는 버전인 [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/quickstart_index.html) 6.2를 설치하고 시스템 요구 사항을 확인합니다.

- Isaac Sim skill을 사용할 수 있는 agent를 선택해 설정합니다. [Agent Skills documentation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/development_tools/agent_skills.html)에 따라 번들 skill을 agent에서 사용할 수 있도록 합니다.

- 명령줄 asset 다운로드 옵션을 선택한 경우에만 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)를 설치합니다.

- [USD Search endpoint](http://nvidia-simready-central.hf.space)를 사용합니다.

## 1단계: 로봇 arm, gripper, beverage can asset 다운로드하기 {#section-5}

[SimReady Central](https://huggingface.co/spaces/nvidia/simready-central)을 열고 여기서 사용한 [UR17G](https://huggingface.co/datasets/TeradyneRobotics/UniversalRobots/blob/d5d8eb795a487a5c929909bc337ee4834d62b0da/ur17g_1300/simready_usd/ur17g_1300.usda) arm, [Robotiq 2F-140](https://huggingface.co/datasets/Robotiq-Official/simready-assets/blob/9c6b0cedf0066efd474b3bf473e7eaf7ae7e569f/Robotiq_2F_85/simready_usd/Robotiq_2F_85.usda) gripper, [beverage can](https://huggingface.co/datasets/nvidia/simready-assets/blob/653d91e182011f357cf1644013cc94ddf194d349/food_beverage_can_n01/simready_usd/sm_food_beverage_can_n01_01.usd)을 찾습니다. 각 asset에서 Download를 선택한 다음 Save to folder 또는 제공된 Hugging Face 명령을 사용합니다. Isaac Sim이 지원 파일을 찾을 수 있도록 세 asset 폴더 전체를 그대로 유지하세요.

이 예제에 연결된 asset 버전을 사용하고, arm, gripper, beverage can의 로컬 USD 파일 경로를 agent에 제공합니다. 아래에는 gripper 경로 예시가 나와 있으며, 다운로드 위치는 다를 수 있습니다.

```
~/ASSETS/hf/src/simready-assets-c7fa7615/Robotiq_2F_140/simready_usd/Robotiq_2F_140.usda
```


[![figure-05-find-robotiq](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/FSec580RNJKj0DIXx9703.gif)](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/FSec580RNJKj0DIXx9703.gif)

그림 3. SimReady Central의 검색 및 대화형 미리보기를 통해 이 예제에서 사용하는 Robotiq 2F-140 gripper와 beverage can을 식별할 수 있습니다.

[![figure-04-download-assets](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/LJpqYrl3l5QkQ2T4JJzYd.gif)](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/LJpqYrl3l5QkQ2T4JJzYd.gif)

그림 4. beverage can의 다운로드 옵션에는 asset 패키지를 폴더에 저장하거나 Hugging Face CLI를 사용하는 방법이 포함됩니다.

## asset 검증은 무엇을 알려주나요? {#section-6}

[arm validation report](https://huggingface.co/datasets/TeradyneRobotics/UniversalRobots/blob/d5d8eb795a487a5c929909bc337ee4834d62b0da/ur17g_1300/.metadata/ur17g_1300_validation.json)는 Robot-Body 4.0.0 프로필을 사용하고, [gripper validation report](https://huggingface.co/datasets/Robotiq-Official/simready-assets/blob/c7fa7615a353ebac03990208cbc2aa57426f4c48/Robotiq_2F_140/.metadata/Robotiq_2F_140_validation.json)는 Robot-Gripper 3.0.0을 사용합니다. 두 보고서 모두 관절 drive 설정을 검사하는 [PhysX driven-joint checks](https://nvidia.github.io/simready-foundation/latest/features/FET_022_PHYSX.html)를 통과한 것으로 기록합니다.

SimReady 프로필 검증은 이러한 asset을 재사용하기 위한 공통 기준을 제공합니다. 그런 다음 pick-and-place 실행에서 조립된 장면의 arm, gripper, can, controller가 함께 어떻게 동작하는지 테스트합니다.

## 2단계: 로봇을 조립하고 pick-and-place를 위한 장면 구성하기 {#section-7}

asset을 다운로드한 후 Isaac Sim skill과 함께 agent를 사용해 로봇 조립, 검증 도구 실행, pick-and-place 제어 생성을 수행할 수 있습니다. 연결된 문서를 사용해 이 단계를 직접 수행할 수도 있습니다.

agent가 Python을 통해 시뮬레이션과 상호작용할 수 있도록 remote Python server를 활성화한 상태로 설치 디렉터리에서 Isaac Sim을 시작합니다.

```
./isaac-sim.sh --enable isaacsim.code_editor.python_server
```


agent를 사용하는 경우 아래 단계를 진행하면서 필요한 skill에 액세스할 수 있도록 Isaac Sim 디렉터리에서 agent를 시작해야 합니다.

세 asset이 모두 누락된 파일 오류 없이 나타나는지 확인합니다. agent에게 arm과 gripper의 관절 제한 및 drive 설정을 검사하도록 요청하거나 Isaac Sim에서 직접 검토합니다. 로봇과 gripper에는 Isaac Sim GUI에서 테스트하고 검증할 수 있는 physics joint도 포함되어 있습니다. 이제 arm과 gripper를 조립하여 하나의 articulation으로 함께 제어할 수 있도록 하겠습니다.

수동으로 조립하려면 [Robot Assembler tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/robot_setup/assemble_robots.html)을 따라 할 수 있습니다. 여기서는 agent에게 조립을 완료하고 검증하도록 요청합니다.

이 예제에서 사용한 샘플 프롬프트:

```
Assemble the robot arm <local path to Simready ur17g robot usd>
with the gripper <local path to downloded simready Robotiq_2F_140 gripper usd>
using the Robot Assembly tutorial in Isaac Sim
https://docs.isaacsim.omniverse.nvidia.com/6.2.0/robot_setup/assemble_robots.html.

Validate the assembly on completion.
```


agent는 Isaac Sim에 패키지로 포함된 ‘isaac-sim-remote’ 및 ‘usd-articulation’ skill을 사용해 조립을 완료하고 로봇의 물리를 검증합니다. 이제 pick and place를 실행하는 데 필요한 모든 asset, 즉 조립된 로봇과 soda can을 갖추었습니다. 이제 워크플로의 마지막 부분으로 진행하겠습니다.

[![figure-05-joint-settings](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/NcKj9Cbp2iBlXh65rUf0T.gif)](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/NcKj9Cbp2iBlXh65rUf0T.gif)

그림 5. Isaac Sim에서 UR17G arm과 Robotiq 2F-140 gripper의 관절 설정.

장면에 조립된 로봇과 캔이 있으면 agent에게 작업을 실행하도록 요청합니다. 다음 단계의 검사 항목을 사용해 시뮬레이션 동작을 평가하세요.

## 3단계: 제어를 테스트하고 pick-and-place 실행하기 {#section-8}

이제 asset에 시뮬레이션 속성과 프로필 검증이 포함되어 있으므로, 조립된 장면과 controller에서 테스트할 수 있습니다. controller는 UR17G arm의 움직임과 Robotiq 2F-140 gripper의 열기 및 닫기를 조정합니다. 아래 동작 검사를 사용해 agent가 생성한 작업을 평가하세요. controller를 직접 이해하거나 조정하려면 [Adding a Manipulator Robot tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/core_api_tutorials/tutorial_core_adding_manipulator.html)을 참조하세요.

이 예제에서 사용한 샘플 프롬프트:

```
Run pick and place with the assembled robot and soda can
<path to soda can USD from SimReady library> in Isaac Sim.

Ensure you do a contact based grasp for picking up the can and validate
the accuracy/placement error for the pick and place as well.
```


사용된 skill: ‘isaac-sim-remote’, ‘manipulaiton-ik’, ‘physics-simulation’

전체 작업에서는 이동 중 gripper가 캔을 잡고 있는지, 목표 지점에서 캔을 놓는지를 테스트합니다. 장면을 초기화한 다음 접근, 잡기, 들어 올리기, 유지, 이동, 놓기 순서를 실행합니다. 실행 결과를 시작 동영상 및 아래의 성공 기준과 비교하세요.

[![figure-06-pick-and-place-landscape](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/TBsKTaclDU8oJvkXYCie8.gif)](https://cdn-uploads.huggingface.co/production/uploads/6abfe9e7d7cf09a9b38ef308/TBsKTaclDU8oJvkXYCie8.gif)

그림 6. Isaac Sim에서 UR17G arm과 Robotiq 2F-140 gripper가 노란색 픽업 마커에서 초록색 목표 지점으로 beverage can을 이동합니다. 4배속으로 표시했습니다.

실행 결과가 다르면 순서가 어디에서 실패하는지 기록하고 agent에게 장면을 검사한 후 제어를 생성하도록 요청하세요. 결과를 비교할 수 있도록 다시 시도하기 전에 캔과 로봇을 동일한 시작 상태로 초기화합니다.

시뮬레이션 가정:

- 고정된 설정: 캔은 알려진 위치에서 시작하고, 로봇은 고정되어 있으며, 작업 공간은 평평하고 장애물이 없습니다. vision은 사용하지 않았고 충돌 검사는 제한적으로 수행했습니다.

- 단순화된 캔: 변형과 액체의 움직임을 모델링하지 않았으므로 이 예제에서는 찌그러짐을 테스트하지 않습니다.

- 스크립트로 정의된 grasp 및 motion: 수평 grasp에는 시뮬레이션 테스트를 통해 선택한 2 N·m drive-effort 상한을 사용하며, 손끝 힘에 맞춰 보정하거나 제조업체가 권장한 값은 아닙니다. arm이 고정된 waypoint를 따라 재계획 없이 움직일 때 접촉과 마찰로 beverage can을 유지합니다. 하드웨어 gripping-force 사양은 [Robotiq product sheet](https://robotiq.com/hubfs/Product-sheets/Adaptive%20Grippers/Product-sheet-Adaptive-Grippers-EN.pdf)를 참조하세요.

이 결과는 이 스크립트 기반 설정에만 적용됩니다. 불확실한 객체 위치, 다른 payload, 실제 하드웨어에 대한 견고성을 입증하는 결과는 아닙니다.

## SimReady Assets로 다음 시뮬레이션 구축하기 {#section-9}

[SimReady Central](https://huggingface.co/spaces/nvidia/simready-central)의 asset을 사용해 자신만의 pick-and-place 작업을 구축한 다음, 원하는 agent의 도움을 받아 장면을 조립하고 Isaac Sim에서 실행해 보세요.

asset 탐색

- 로봇과 객체 찾기: Hugging Face에서 [SimReady Central](https://huggingface.co/spaces/nvidia/simready-central)을 둘러봅니다.

- asset이 지원하는 항목 확인하기: asset의 validation report와 함께 [SimReady Foundation](https://nvidia.github.io/simready-foundation/latest/)의 프로필과 요구 사항을 검토합니다.

pick-and-place 장면 구축하기

- simulator 설정하기: 이 예제에서 사용한 버전에 해당하는 [Isaac Sim installation guide](https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/quickstart_index.html)를 따릅니다.

- agent 연결하기: [Isaac Sim Agent Skills documentation](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/development_tools/agent_skills.html)에 따라 번들 skill을 agent에서 사용할 수 있도록 합니다.

- 로봇 조립하기: arm과 gripper를 수동으로 연결하려면 [Robot Assembler tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/robot_setup/assemble_robots.html#assembling-robots)을 따릅니다.

콘텐츠 준비 및 기여하기

- CAD에서 시작하기: [5 Steps to Create SimReady Assets for Robotics](https://developer.nvidia.com/blog/5-steps-to-create-simready-assets-for-robotics-with-frontier-ai-models/)에 따라 자신만의 robotics asset을 준비합니다.

- asset 검증 및 제출하기: [SimReady Central submission guide](https://github.com/NVIDIA-Omniverse/simready-central/blob/main/.agents/skills/submit-simready-assets/SKILL.md)에 따라 다른 개발자가 asset을 사용할 수 있도록 합니다.

영감을 얻으려면 [Michael Wagner’s voice-controlled VR demo](https://lnkd.in/p/g3-5svW6) 또는 [Maziyar Panahi’s Microduck simulation](https://lnkd.in/p/gnxQ3YS8)을 살펴보세요. 두 예제 모두 개발자가 자신의 프로젝트에서 SimReady asset을 어떻게 사용하고 있는지 보여줍니다.

여러분은 무엇을 구축할 예정인가요? 사용한 asset과 Isaac Sim 버전을 포함해 Hugging Face 댓글에 장면과 결과를 공유해 주세요.

즐겁게 구축해 보세요!
