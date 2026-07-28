---
layout: post
title: AWS에서 Foundation Model 학습 및 추론을 위한 구성 요소
description: AWS 기반 Foundation Model 학습과 추론 인프라를 구성하는 컴퓨트, 네트워크, 스토리지, 운영 요소를 정리합니다.
author: dailybot
categories:
- Translation
- HuggingFace
image: assets/images/blog/posts/2026-05-12-foundation-model-building-blocks/thumbnail.png
slug: foundation-model-building-blocks
source_url: https://huggingface.co/blog/amazon/foundation-model-building-blocks
source_published_date: '2026-05-11'
source_published_at: '2026-05-11T23:18:26+00:00'
locale: ko
translation_status: draft
translator: openai
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Building Blocks for Foundation Model Training and Inference on AWS](https://huggingface.co/blog/amazon/foundation-model-building-blocks)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/amazon/foundation-model-building-blocks -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# AWS에서 Foundation Model 학습 및 추론을 위한 구성 요소

오랜 기간 동안, 파운데이션 모델의 "스케일링"은 대개 한 가지를 뜻했다: 프리-트레이닝에 더 많은 컴퓨트를 투자하면 능력이 상승한다. 이 직감은 [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) 같은 경험적 연구가 뒷받침했으며, 모델 파라미터 수, 데이터셋 크기, 학습 컴퓨트의 증가에 따라 손실이 예측 가능한 멱 법칙(power-law) 트렌드를 보인다고 보고했다. 실제로 이러한 경향은 대규모 가속기 용량과 이를 효율적으로 활용하는 분산 인프라스트럭처에 지속적인 투자를 정당화했다.
그러나 프론티어는 진화했고—스케일링은 더 이상 하나의 곡선이 아니다. NVIDIA의 "하나에서 세 가지 스케일링 법칙" 프레이밍은 프리-트레이닝을 넘어서 성능이 포스트-트레이닝(예: 감독 학습 미세 조정(fine-tuning, SFT) 및 강화학습(RL) 기반 방법) 및 테스트 타임 컴퓨트("오랜 사고", 검색/검증, 다중 샘플 전략)에서 점차 스케일링된다는 점을 유용하게 강조한다.

Figure: Adapted from ["AI's Three Scaling Laws, Explained"](https://blogs.nvidia.com/blog/ai-scaling-laws/) (NVIDIA Blog).

이러한 스케일링 규칙들이 모여 파운데이션 모델 생애주기—사전 학습(pre-training), 이후 학습(post-training), 추론(inference)—을 밀접하게 결합된 인프라 요구사항으로 밀어 넣는다: 가깝게 연결된 가속기 컴퓨트, 대역폭이 높은 저지연 네트워크, 그리고 분산 스토리지 백엔드. 또한 리소스 관리 orchestration의 중요성과 대규모로 클러스터 건강을 유지하고 성능 경로(pathologies)를 진단하기 위한 어플리케이션- 및 하드웨어 수준의 관찰성(observability)의 중요성도 제고된다.

또 다른 핵심 흐름은 파운데이션 모델 생애주기가 OSS(Open-Source Software) 생태계에 더욱 의존하게 되는 경향이다. OSS 생태계는 모델 개발 프레임워크, 클러스터 리소스 관리, 운영 도구를 포괄한다. 클러스터 계층에서 리소스 관리는 일반적으로 [Slurm](https://slurm.schedmd.com/documentation.html)과 [Kubernetes](https://kubernetes.io/docs/) 같은 시스템으로 제공된다. 모델 개발 및 분산 학습은 일반적으로 [PyTorch](https://pytorch.org/)와 [JAX](https://jax.readthedocs.io/) 같은 프레임워크에서 구현된다. 모니터링 및 시각화—즉, 관측성(observability)—은 일반적으로 메트릭 수집을 위한 [Prometheus](https://prometheus.io/docs/introduction/overview/)와 시각화 및 경보를 위한 [Grafana](https://grafana.com/docs/grafana/latest/)를 사용해 운영 계층에서 인프라 및 리소스 관리 위에 위치한다. Figure 1은 하드웨어 인프라가 리소스 오케스트레이션을 지원하고, 그 위에서 ML 프레임워크가 작동하며, 관측성이 모든 계층에 걸쳐 확산되는 계층형 아키텍처를 시각화한다.

Figure 1: The layered architecture of open-source software stacks for foundation model training and inference

본 글은 OSS 프레임워크 위에 구축된 워크플로에 특히 주목하는 파운데이션 모델 학습 및 추론에 관여하는 머신러닝 엔지니어와 연구자를 대상으로 한다. AWS 인프라—다중 노드 가속기 컴퓨트, 고대역폭 저지연 네트워킹, 분산 공유 스토리지, 관련 관리형 서비스—가 파운데이션 모델 생애주기 전반의 일반적인 OSS 스택과 어떻게 상호 작용하는지 분석한다. 주요 목표는 사전 학습, 사후 학습, 추론에 걸친 시스템 병목 현상과 스케일링 특성을 이해하기 위한 기술적 기초를 제공하는 것이다. 이 소개 글은 전체 시스템 아키텍처를 제시하고, 대규모 분산 학습 및 추론을 뒷받침하는 AWS 인프라 구성요소와 OSS 도구 간의 통합 지점을 강조한다.

## AWS 빌딩 블록

이 시리즈의 나머지 부분은 이 계층형 아키텍처가 AWS에서 어떻게 구현되는지—인프라, 리소스 오케스트레이션, ML 소프트웨어 스택, 관측성의 각 계층을 통해—를 살펴본다. 아래 섹션들은 각 계층을 예고한다.

### 인프라: 컴퓨트, 네트워크, 및 스토리지

Figure 1에 도시된 바와 같이, 인프라는 세 가지 상호 결합된 구성요소로 구성된다: 대형 디바이스 메모리를 가진 가속화된 컴퓨트, 집합적 통신을 위한 광대역 대역폭의 인터커넥트, 데이터 및 체크포인트(checkpoint)를 위한 확장 가능한 분산 스토리지.

대규모 파운데이션 모델의 프리-트레이닝, 포스트-트레이닝, 추론의 기초를 이루는 가속화된 컴퓨트는 필수적이다. AWS는 [Amazon EC2 accelerated computing instances](https://aws.amazon.com/ec2/instance-types/accelerated-computing/)의 여러 세대를 제공하며, 그 중 [P계 인스턴스 패밀리](https://aws.amazon.com/ec2/instance-types/p5/)가 포함된다. p5.48xlarge은 여덟 대의 [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) GPU를 갖추고 있으며, p5.4xlarge는 소형 규모의 작업에 적합한 단일 H100 GPU를 제공하고, p5e.48xlarge/p5en.48xlarge 변형은 [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) GPU를 탑재한다. [P6 인스턴스 패밀리](https://aws.amazon.com/ec2/instance-types/p6/)는 [NVIDIA Blackwell B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) 아키텍처를 도입하며 p6-b200.48xlarge 및 [Blackwell Ultra B300](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)의 p6-b300.48xlarge를 제공한다.
이 세대 전반에 걸쳐 지배적인 스케일링 축은 피크 Tensor 처리량(throughput), HBM 용량 및 대역폭, 그리고 노드 간 인터커넥트 대역폭이다.

일차 근사로, 피크 Tensor Core 처리량은 초당 부동소수점 연산(FLOPS)으로 측정되며, 이 값을 통해 이 가속기들을 공통 축에 위치시키는 데 도움을 준다. 아래 표는 dense BF16/FP16 및 FP8 텐서 연산의 GPU당 피크 처리량, HBM 용량 및 대역폭을 SXM/HGX급 사양과 일치하도록 요약한다. 이 표는 NVSwitch/NVLink 기반의 다중-GPU 노드에 맞춘다.

참고: NVIDIA의 제품 표는 종종 "희소성(sparsity)과 함께" 텐서 처리량을 보고하므로, 이 표는 밀집(dense) 처리량을 보고한다. 해당되는 경우, 밀집 처리량은 HGX급 플랫폼에 대한 NVIDIA의 가이드에 따라 희소 처리량의 절반으로 간주한다([NVIDIA](https://www.nvidia.com/en-us/data-center/hgx/)). DGX 수치는 시스템 수준이며, B200 HBM 용량 및 대역폭 값은 DGX 총합을 8로 나누어 GPU당 표현한다([NVIDIA](https://www.nvidia.com/en-us/data-center/dgx-b200/)).

모델이 스케일링될수록, 단계 시간은 종종 순수 계산 처리량이 아니라 집합적 통신 및 메모리 이동에 의해 좌우된다. 따라서 명시적 스케일 업/스케일 아웃 대역폭 계산이 필요하다.
다중-GPU 인스턴스의 경우, GPU 간 통신은 두 가지 규칙으로 나뉜다. 내부 스케일업(NVLink/NVSwitch)은 한 노드 내에서 고대역폭, 저지연의 GPU 간 연결을 제공하여 all-reduce 및 all-gather 같은 집합 연산이 호스트 네트워크 스택을 거치지 않고 실행되도록 한다. 외부 스케일 아웃(EFA)은 노드 간 OS 우회 네트워킹을 제공하며, AWS는 이를 [Amazon EC2 UltraClusters](https://aws.amazon.com/ec2/ultraclusters/)의 기본 빌딩 블록으로 사용한다. 이 구성에서 통신 집약적 집합 연산은 수천 대의 인스턴스에 걸쳐 확장된다. 아래 표는 이러한 인스턴스 유형들 간의 주요 사양을 요약한다:

참고: EFA 대역폭은 일관성을 위해 다른 대역폭 지표와의 비교를 위해 Gbps를 GB/s로 변환하였다(÷8). 또한 NVLink 및 EFA 대역폭 수치는 링크당 값이 아닌 인스턴스당 합계 값으로 표시된다; 내부 노드 간 인터커넥트 및 네트워킹 특성은 해당 [P5 인스턴스 계열 페이지](https://aws.amazon.com/ec2/instance-types/p5/) 및 [P6 인스턴스 계열 페이지](https://aws.amazon.com/ec2/instance-types/p6/)를 참조하자.

Elastic Fabric Adapter (EFA)는 OS-bypass 원격 직접 메모리 접근(RDMA)을 지원하는 Amazon EC2의 네트워크 인터페이스다. Scalable Reliable Datagram(SRD) 프로토콜을 사용하는 Libfabric API를 통해 네트워크 디바이스와 애플리케이션이 직접 통신하도록 하여 분산 학습의 집합 연산에서 대기 시간을 줄이고 처리량을 향상시킨다.

다양한 세대의 EFA가 서로 다른 인스턴스 계열에서 제공된다. Amazon EC2 P5 및 P5e 인스턴스는 EFA 버전 2(EFAv2)를 탑재한다. [EFA 버전 3(EFAv3)]은 P5en 인스턴스에서 제공되며 EFAv2에 비해 패킷 지연을 약 35% 감소시키는 것으로 알려져 있다. [EFA 버전 4(EFAv4)], P6 인스턴스에서 사용 가능하며 EFAv3 대비 집합 연산 성능을 추가로 약 18% 개선한다.

대규모로 갈수록, 분산 학습(다중 샘플링 코퍼스 스트리밍 및 멀티 테라바이트 체크포인트 작성)과 대규모 추론(가중치 스테이징 및 KV 캐시 성장 관리)은 지역 NVMe SSD를 핫 데이터용으로, Lustre를 공유 고처리량 접근용으로, 그리고 [Amazon S3](https://aws.amazon.com/s3/)를 내구성 있는 지속성을 위한 저장소로 사용하는 계층적 스토리지 구성을 필요로 한다.

이 시리즈의 주요 다중-GPU 인스턴스에서 로컬 NVMe는 인스턴스 스토어(휘발성)로 제공되며, 원시 용량은 30.72 TB(8 × 3.84 TB NVMe SSD)이다; [EC2 accelerated-computing instance store specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html#ac_instance-store)를 참조하라.

[Lustre](https://www.lustre.org/about/)는 고성능 컴퓨팅(HPC)에서 폭넓게 사용되는 오픈 소스, POSIX 준수 분산 파일 시스템으로, 다수의 클라이언트에 걸쳐 높은 집계 처리량으로 공유 네임스페이스를 제공한다. [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/)는 Lustre를 완전 관리형 서비스로 제공하고, 초당 테라바이트급 처리량, 수백만 IOPS, 밀리초 미만 지연을 실현하는 병렬 파일 시스템으로 노출한다. Data Repository Associations를 통해 [Amazon S3](https://aws.amazon.com/s3/)와 통합이 가능하며, 학습 데이터 세트의 느린 로딩 및 내구성을 위한 체크포인트 자동 익스포트를 지원한다.

클러스터 규모에서, 이 인스턴스들은 [Amazon EC2 UltraClusters](https://aws.amazon.com/ec2/ultraclusters/)로 배치되며, 수천 대의 가속 인스턴스를 단일, 밀집된 클러스터로 구성하고 페타비트 규모의 논블로킹 네트워크로 상호 연결한다.

Figure: 2nd-generation Amazon EC2 UltraClusters (example P5 UltraCluster).

작업당 통신 강도가 높은 워크로드(예: MoE 모델의 expert parallelism에서 모든 토큰 디스패치가 다수의 GPU에 걸쳐 이루어짐)의 경우, NVLink 도메인의 크기가 1차 제약이 될 수 있다. 내부 스케일업 축의 확장으로 NVLink 도메인이 커지면 성능에 중요한 통신이 NVLink 패브릭을 벗어나지 않도록 하는 효과가 있다.

[Amazon EC2 UltraServers](https://aws.amazon.com/ec2/ultraservers/)는 단일 EC2 인스턴스의 NVLink 도메인을 넘어서 여러 구성 요소 인스턴스를 전용 가속기 인터커넥트를 통해 연결한다. AWS는 [P6e-GB200 UltraServers](https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-p6e-gb200-ultraservers-gpu-performance-ec2/)가 [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) 플랫폼 위에 구축되며 하나의 NVLink 도메인 내에서 최대 72대의 Blackwell GPU와 총합 13.4 TB의 HBM3e를 제공한다고 밝혔다. 더 큰 규모에서 EFA는 다중 UltraServer 작업의 크로스-노드 패브릭으로 남아 있지만, 도메인 내 GPU 수를 늘리면 성능에 중요한 통신이 NVLink 패브릭을 떠날 필요가 줄어든다.

이러한 시스템은 NVIDIA Grace–Blackwell 슈퍼칩으로 구성되며, Grace CPU 메모리와 Blackwell GPU HBM을 캐시 일관성 있는 [NVLink-C2C](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/)를 통해 연결한다. 이를 통해 CPU-메모리와 GPU-메모리 간의 명시적 호스트–디바이스 복사를 피하면서 GPU 워크로드에 효과적인 메모리 용량을 확장할 수 있다. 다만 로컬 HBM에 비해 지연은 더 크고 대역폭은 낮다.

P6e-GB200 UltraServers의 컴포넌트 인스턴스 타입은 [`p6e-gb200.36xlarge`](https://aws.amazon.com/ec2/instance-types/p6/))로, 네 대의 GPU와 Elastic Fabric Adapter (EFA) v4 네트워킹을 제공한다. 아래 표들은 인스턴스당 구성과 총합 UltraServer 구성을 요약한다.

참고: `p6e-gb200.36xlarge`의 EFA 대역폭은 게시된 합계 EFA 네트워킹(4 × 400 Gbps)을 GB/s로 변환한 값이다(÷8); [EC2 accelerated computing networking specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html#ac_network)를 참조하라.

참고: UltraServer EFA 대역폭은 AWS가 보고한 Tbps 단위에서 GB/s로 변환한 값이다(÷8); [P6e-GB200 UltraServers 발표](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p6e-gb200-ultraservers-powered-by-nvidia-grace-blackwell-gpus-for-the-highest-ai-performance/) 및 [P6 인스턴스 계열 페이지](https://aws.amazon.com/ec2/instance-types/p6/)를 참조하라.

### 리소스 오케스트레이션: Slurm과 Kubernetes

수백 또는 수천 대의 가속기가 학습에 사용될 때, 수동 리소스 관리의 난이도는 컨트롤하기 어렵다. 예를 들어 512 GPU가 필요한 학습 작업은 동시에 64개의 8-GPU 노드(P-인스턴스)를 함께 스케줄링하고, 완료 혹은 실패 시 원자적으로 리소스를 해제해야 한다. Slurm(및 Kubernetes) 모두 이 도전을 컨트롤-플레인 아키텍처를 통해 해결한다. 중앙 집중식 스케줄러가 클러스터 상태를 유지하고 할당 결정을 내리며, 워커 노드가 할당된 작업을 실행한다.

Figure 2: AWS의 Slurm 기반 및 Kubernetes 기반 리소스 오케스트레이션의 고수준 아키텍처

[Slurm](https://slurm.schedmd.com/)은 Simple Linux Utility for Resource Management의 약자로, 대규모 HPC 분야에서 지배적인 워크로드 매니저이다. 모듈식 플러그인 아키텍처를 통해 스케줄링 알고리즘, 토폴로지 모델, 리소스 유형, 회계 백엔드를 독립적으로 구성할 수 있다. 스케줄링 모델은 자원을 파티션(partitions)로 구성하고, `sbatch`를 통해 작업 제출을 받으며, 할당된 노드에서 동기화된 시작으로 병렬 태스크를 `srun`으로 실행한다. 분산 학습에 있어 중요한 점은 Slurm이 작업 단위로 스케줄링한다는 사실이다—다중 노드 작업 전체를 원자적으로 할당하고, 태스크가 시작되기 전에 먼저 배치한다. 낮은 우선순위를 가진 작업을 재배치하는 백필(backfill) 스케줄러와 공정 점유를 위한 다요인 우선순위 시스템, 다중 임차인 간의 큐를 관리하는 큐 운영 정책이 있다. 또한 Slurm은 [토폴로지 인식 배치(topology-aware placement)](https://slurm.schedmd.com/topology.html) 플러그인을 통해 네트워크 스위치 계층 구조를 모델링하고, AWS에서 EFA 패브릭 토폴로지를 이용해 최소 스위치 홉으로 노드에 작업을 배치하도록 한다. 그리고 GPU 유형을 추적하고 기기 친화성을 강제하는 Generic Resource(GRES) 인터페이스를 통해 기본 스케줄링도 지원한다.

AWS는 Slurm 기반 오케스트레이션에 대해 여러 배포 옵션을 제공한다. [AWS ParallelCluster](https://github.com/aws/aws-parallelcluster)은 EC2에서 Slurm 클러스터 배포를 자동화하는 오픈 소스 클러스터 관리 도구로, 헤드 노드 프로비저닝, 컴퓨트 플릿 확장 및 공유 스토리지와의 통합을 처리한다. [AWS Parallel Computing Service (PCS)](https://aws.amazon.com/pcs/)는 관리형 컨트롤 플레인을 제공하는 대안을 제시한다. 분산 학습 워크로드의 경우, [Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)는 Slurm 모드를 지원하며 대규모 학습에 특화된 기능(예: [지속적 노드 건강 모니터링](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-cluster-health-check.html) 및 [작업 자동 재개 기능](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-resiliency-slurm-auto-resume.html))을 제공한다.

[Kubernetes](https://kubernetes.io/)는 선언적이고 API 주도적 접근을 취한다: 사용자는 리소스 매니페스트를 통해 원하는 상태를 지정하고, 컨트롤러는 실제 상태를 이에 맞춰 조정한다. Kubernetes는 모델 배포에 탁월하지만, 밀접하게 결합된 분산 학습의 경우 기본 스케줄링 모델에는 여러 간극이 있다. Kubernetes는 파드 단위로 스케줄링하며, 작업 단위의 원자성이 없으면 다중 노드 학습 작업이 부분적으로 시작되어 일부 랭크가 Pending 상태로 남아 GPU가 낭비되거나 교착 상태가 발생할 수 있다. 일반적인 Kubernetes는 우선순위 기반 백필, 네트워크 패브릭 토폴로지(NVLink 도메인, EFA 인터커넥트) 인식 및 커뮤니케이션 집약적 집합 연산에 대한 배치를 기본으로 하는 배치 큐 의미를 기본적으로 제공하지 않는다.

다수의 Kubernetes 네이티브 프로젝트들이 서로 다른 계층에서 이러한 간극을 해결한다. [Kueue](https://kueue.sigs.k8s.io/)는 기본 스케줄러 위에 어드미션 컨트롤러로 작동해 작업 단위의 Gang 어드미션, 계층적 공정 공유를 위한 다 tenant 쿼타, 우선순위 기반 선점 등을 관리하고, admitted 작업은 기본 스케줄러에 넘긴다. [Volcano](https://volcano.sh/)와 [NVIDIA KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler)는 기본 스케줄러를 대체하거나 보강하여 토폴로지 인식된 파드 배치를 통해 Gang 스케줄링을 직접 통합한다. 이러한 계층은 상호 보완적이다: Kueue가 어드미션 및 쿼타 정책을 관리하고, admitted 작업을 토폴로지 인식 스케줄러에 넘겨 배치를 수행한다.

AWS에서 Kubernetes 기반 오케스트레이션을 사용하려면, [Amazon Elastic Kubernetes Service (EKS)](https://aws.amazon.com/eks/)가 GPU 스케줄링을 위해 [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin)을 제공하는 관리형 Kubernetes를 제공한다. Amazon SageMaker HyperPod도 [EKS 모드](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks.html)를 지원하며, HyperPod의 학습 특화 기능과 함께 Kubernetes 오케스트레이션을 결합한다. HyperPod EKS는 기저 AWS 인프라에서의 확장 가능한 파운데이션 모델 학습을 위한 기능을 추가한다. [Task governance](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-operate-console-ui-governance-policies.html)는 팀 간의 컴퓨트 할당과 정책 적용을 제공하고, 관리형 [Kueue](https://kueue.sigs.k8s.io/)를 통한 어드미션 컨트롤과 [Karpenter](https://karpenter.sh/)를 통한 즉시 노드 프로비저닝을 통합한다. [Checkpointless training](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-checkpointless.html)은 전통적인 체크포인트 기반 장애 허용 방식에서의 회복 지연을 해결한다. 대신 체크포인트 작성 없이도 GPU 간 피어 투 피어 상태 복제를 유지한다. 장애가 발생하면 생존 노드가 EFA 기반 통신으로 손실된 상태를 재구성한다. [Elastic training](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-eks-elastic-training.html)은 리소스 가용성에 따라 작업이 자동으로 확장되도록 한다. 추가 가속기가 가능해지는 경우(예: 완료된 작업으로부터, 혹은 새로 프로비저닝된 용량으로) 탄력적 작업은 확장될 수 있으며, 더 높은 우선순위의 작업이 리소스를 요구하면 축소되더라도 학습 진행은 유지된다.

### ML 소프트웨어 스택

분산 학습과 추론은 올바르게 구성되고 조정되어야 하는 여러 소프트웨어 계층을 포함한다. 실행 스택을 하드웨어에 인접한 구성요소에서 프레임워크 수준의 추상화까지 다섯 계층으로 보는 유용한 모델이 있다: 하드웨어 활성화, 가속기 런타임 및 수학 라이브러리, 통신 기반, ML 프레임워크, 분산 학습/추론 프레임워크.

Figure 3: The ML software stack for distributed training and inference on EC2 instances

#### 하드웨어 활성화: 커널 드라이버

하단에는 Linux 커널 드라이버가 직접 하드웨어에 접근할 수 있게 한다. NVIDIA GPU 드라이버는 컴퓨트 기능을 노출하고 [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)를 통해 GPU와 네트워크 어댑터 간의 직접 데이터 전송을 지원한다. [GDRCopy](https://github.com/NVIDIA/gdrcopy) 드라이버(`gdrdrv`)는 GPU 메모리로의 저지연 CPU-주도 복사를 가능하게 하며, NCCL에서 소형 메시지 전송에 사용된다. [EFA 드라이버](https://github.com/amzn/amzn-drivers)는 [libfabric](https://github.com/ofiwg/libfabric) API를 통해 OS-bypass 네트워킹을 제공하고, [Lustre 클라이언트](https://www.lustre.org/) 드라이버는 FSx for Lustre 병렬 파일 시스템에 대한 POSIX 접근을 가능하게 한다.

#### 가속기 런타임, 컴파일러, 커널 라이브러리

[CUDA](https://developer.nvidia.com/cuda-toolkit) 플랫폼은 GPU 컴퓨트용 프로그래밍 모델과 런타임을 제공한다. CUDA에 대해 컴파일된 애플리케이션은 NVIDIA GPU에서 커널을 실행하고, 디바이스 메모리를 관리하며, 다수 디바이스 간의 실행을 조정한다. 현재 릴리스는 CUDA Toolkit 13.x이며, Blackwell 아키텍처(컴퓨트 기능 10.x)를 지원한다.

현대의 학습 및 추론 성능은 일반적인 벤더 PRIMITIVES 보다 특수 최적화 라이브러리와 커스텀 커널에 의해 더 좌우된다. 예를 들어 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 같은 커널은 주의(attention)를 하나의 메모리-효율적 패스로 융합해 HBM 트래픽을 줄이고 처리량을 향상시킨다. 또한 각 팀은 모델에 맞춘 형태(shape)와 정밀도(precision)에 특화된 융합 커널(layernorm/residual/activation, quantized GEMMs, MoE dispatch, KV-cache ops 등)를 작성한다. 이는 [Triton](https://triton-lang.org/) 같은 프로그래머블 도구체인과 NVIDIA의 [CuTe](https://github.com/NVIDIA/cutlass/blob/1d9e1f6d7a1e820a884765cf3dcfda05c95f39d1/media/docs/cpp/cute/00_quickstart.md) 같은 텐서 레이아웃 및 와프-레벨 DSL, [CUTLASS](https://github.com/NVIDIA/cutlass) 같은 고성능 GEMM 및 퓨전 빌딩 블록의 지원이 필요하다. 실제로 이 커널 및 컴파일러 계층은 ML 프레임워크 못지않게 엔드-투-엔드 성능을 좌우한다.

#### 통신 기반: NCCL 및 트랜스포트 플러그인

다중-GPU 학습은 효율적인 집합 연산에 의존한다. [NVIDIA Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl)는 모든-감소(all-reduce), 모두 모으기(all-gather), 감소-분배(reduce-scatter), 모든-모두(all-to-all), 브로드캐스트, 포인트-투-포인트 송수신 등의 집합 연산을 구현하며, 토폴로지 인식 알고리즘으로 노드 내 NVLink를 활용하고 노드 간 트래픽은 네트워크 전송으로 처리한다. NCCL은 통신 토폴로지를 동적으로 감지하고, 메시지 크기와 가용 대역폭에 따라 링(Ring) 또는 트리(Tree) 알고리즘을 선택한다. 데이터 병렬 및 텐서 병렬 전략은 주로 all-reduce 및 all-gather에 의존하고, Mixture-of-Experts(MoE) 모델의 expert parallelism은 토큰을 GPU 간에 라우팅하기 위해 all-to-all 집합 연산에 의존한다. 예를 들어 dispatch all-to-all은 토큰을 해당 전문가를 보유한 GPU로 보내고, combine all-to-all은 전문가 출력을 시작 GPU로 되돌려 보낸다([NVIDIA Developer Blog]). 모든 GPU가 전문가 파라미터를 교환하므로, expert-parallel리스크가 증가하면 모든-to-all 통신량이 증가하고 성능 병목이 커질 수 있다.

AWS에서는 NCCL의 노드 간 통신이 [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) 플러그인을 통해 가능하도록 하여 NCCL의 트랜스포트 API를 [libfabric](https://github.com/ofiwg/libfabric) 인터페이스로 매핑한다. 이를 통해 NCCL은 OS-bypass와 SRD 프로토콜을 애플리케이션 수정 없이도 활용할 수 있다.

추론 워크로드의 경우, 집합 연산만으로 모든 통신 패턴을 포착하지 못한다. Prefill과 Decode를 서로 다른 GPU 풀로 분리하는 분리된 추론 아키텍처(disaggregated inference architecture)는 KV 캐시 상태를 인스턴스 간에 전송하는 등 포인트-투-포인트 데이터 이동이 효과적이어야 한다. [NVIDIA Inference Xfer Library (NIXL)](https://github.com/ai-dynamo/nixl)은 메모리 계층(HBM, DRAM, NVMe, 분산 스토리지) 및 인터커넥트(NVLink, InfiniBand, Ethernet) 간의 포인트-투-포인트 전송을 위한 통합 API를 제공함으로써 이 요구를 다룬다. NIXL은 NVIDIA Dynamo와 같은 추론 프레임워크와 통합되며 UCX 및 GPUDirect Storage와 같은 백엔드를 지원한다.

#### ML 프레임워크: PyTorch

Foundation 모델 개발에서 지배적인 두 프레임워크는 [PyTorch](https://pytorch.org/)와 [JAX](https://jax.readthedocs.io/)다. JAX는 XLA를 통해 SPMD(Single Program Multiple Data) 접근을 취하며, 동일한 프로그램이 디바이스상에서 자동으로 데이터 분산 및 집계 축소를 통해 실행된다. 이 시리즈는 OSS 생태계에서 널리 채택되고 분산 학습 및 추론 프레임워크의 기초가 되는 PyTorch에 초점을 맞춘다.

PyTorch는 GPU 가속을 이용한 텐서 연산, 자동 미분, 그리고 유연한 eager 실행 모델을 제공한다. 분산 워크로드의 경우, PyTorch의 `torch.distributed` 모듈이 핵심 원시 기능을 제공한다: 집단 통신을 위한 프로세스 그룹, 그리고 [Distributed Data Parallel (DDP)](https://docs.pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) 및 [Fully Sharded Data Parallel (FSDP2)](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)와 같은 분산 데이터 병렬 추상화를 포함한다. DDP는 모델을 GPU 간에 복제하고 모든 축소(all-reduce)를 통해 그래디언트를 동기화하는 반면, FSDP2는 파라미터, 그래디언트 및 옵티마이저 상태를 워커 간에 샤딩(shard)하여, 단일 GPU 메모리 용량을 초과하는 모델의 학습을 가능하게 한다.

#### 분산 학습 및 추론 프레임워크

최상위 계층은 PyTorch를 기반으로 분산 학습과 대규모 추론에 대한 고수준 추상화를 제공하는 프레임워크들로 구성된다. 학습을 위해서는 복잡성-성과 트레이드오프의 서로 다른 지점을 다루는 세 가지 범주의 프레임워크가 있다. 아래는 몇 가지 예이다.

[Hugging Face Transformers](https://huggingface.co/docs/transformers/)는 분산 학습을 위한 내장 지원을 갖춘 `Trainer` 클래스를 제공하는데, 이는 [Accelerate](https://huggingface.co/docs/accelerate/)를 통해 DDP, FSDP, DeepSpeed를 추상화한다. 이 경로는 사용 편의성과 폭넓은 모델 호환성에 중점을 두며, 구성의 단순성이 최대 처리량보다 더 중요할 때 미세 조정 및 중간 규모 학습에 적합하다. Hugging Face Hub, Datasets, Spaces, Inference Endpoints, Inference Providers 같은 제품명은 검색성과 문서 호환성을 위해 원문 표기를 유지한다.

[NVIDIA Megatron Core](https://developer.nvidia.com/megatron-core)는 대규모에서의 최대 효율을 목표로 3D 병렬성(텐서, 파이프라인, 전문가 병렬성)을 구현하고, Transformer Engine을 통한 FP8 혼합 정밀도 등 최적화를 포함한다. [NeMo Framework](https://docs.nvidia.com/nemo-framework/user-guide/latest/)는 Megatron Core를 기반으로 사전 학습(pre-training) 및 미세 조정 파이프라인을 제공한다.

RLHF(또는 관련 포스트-트레이닝 방법)에서는 [veRL](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning)이 PPO, GRPO, REINFORCE++ 등의 알고리즘을 구현하는 유연한 프레임워크를 제공한다. veRL의 HybridFlow 아키텍처는 학습 백엔드(FSDP2, Megatron)와 추론 엔진(vLLM, SGLang)을 동일 작업에서 혼합하는 것을 가능하게 하며, actor와 rollout 컴포넌트 간 모델 가중치를 메모리에 공유해 가중치 동기화 오버헤드를 줄인다.

추론 서빙(serving)을 위한 프레임워크로는 [vLLM](https://github.com/vllm-project/vllm)이 PagedAttention를 구현하고 KV 캐시를 페이지화된 가상 메모리로 관리해 단편화를 줄이고 더 큰 배치 크기를 가능하게 한다. [SGLang](https://github.com/sgl-project/sglang)은 이러한 기능을 RadixAttention으로 확장해 요청 간 프리픽스 재사용을 자동화하고, CPU 스케줄링과 GPU 연산을 중첩하는 제로-오버헤드 배치 스케줄러 및 예측된 캐시 적중률 기반의 로드 밸런서를 제공한다. 두 프레임워크 모두 단일 GPU 메모리 초과를 위한 텐서 병렬성을 지원하고, [NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)와의 통합으로 데이터 분리(serving의 prefill/decode를 분리하는 아키텍처)에 대응한다.

### 관측성

관측성은 대규모 분산 학습 시스템의 디버깅 및 운영에 선행된다. 학습 작업이 정체되거나 처리량이 저하될 때, 원인이 하드웨어 장애인지, 네트워크 혼잡인지, 스토리지 병목인지, 애플리케이션 레벨의 비효율인지에 대한 가시성이 필요하다. 이 시리즈에서 다루는 인프라 규모(수천 개의 GPU, 페타비트급 인터커넥트 대역폭, 테라바이트 단위의 체크포인트 데이터)에서는 모니터링만으로는 충분하지 않고 체계적인 원격 측정 수집, 저장 및 분석으로의 확장이 필요하다. 관측성은 인프라 메트릭(GPU, 네트워크, 스토리지), 워크로드 메트릭(학습 처리량, 큐 지연), 그리고 사전 예방적 장애 탐지를 위한 경보의 세 가지 텔레메트리 카테고리를 포함한다.

#### 코어 스택: Prometheus와 Grafana

쿠버네티스 및 HPC 환경에서 관측성의 사실상 표준은 메트릭 수집을 위한 [Prometheus](https://prometheus.io/docs/introduction/overview/)와 시각화 및 경보를 위한 [Grafana](https://grafana.com/docs/grafana/latest/)의 조합이다. Prometheus는 폴링 기반 모델로 작동하며, 메트릭 익스포터가 노출하는 HTTP 엔드포인트를 주기적으로 긁어 수집된 메트릭을 시계열 데이터베이스(TSDB)에 저장하고, PromQL을 통해 집계, 필터링 및 경보 규칙 평가를 수행한다. Grafana는 Prometheus를 데이터 소스로 사용하고, PromQL 표현식에 따라 대시보드를 렌더링하고 경보를 트리거한다.

생산 배포의 경우, [Amazon Managed Service for Prometheus (AMP)](https://docs.aws.amazon.com/prometheus/latest/userguide/what-is-Amazon-Managed-Service-Prometheus.html)는 운영자가 스토리지, 복제, 고가용성을 관리하지 않아도 초당 수백만 샘플을 수집할 수 있는 완전 관리형 Prometheus-호환 시계열 데이터베이스를 제공한다. [Amazon Managed Grafana (AMG)](https://docs.aws.amazon.com/grafana/latest/userguide/what-is-Amazon-Managed-Service-Grafana.html)는 AMP와의 원활한 통합 및 IAM Identity Center를 통한 AWS 인증이 내장된 관리형 Grafana 작업 공간을 제공한다. 이 두 서비스는 운영 오버헤드를 줄이고 기존 Prometheus 익스포터와 Grafana 대시보드와의 호환성을 유지한다.

#### GPU, 네트워크, 및 애플리케이션 관찰성

[DCGM-Exporter](https://github.com/NVIDIA/dcgm-exporter)는 활용도, 메모리 사용량, 전력, 온도, ECC 오류 및 XID 이벤트 같은 하드웨어 상태 지표를 Prometheus 형식으로 노출한다. 학습 워크로드의 경우 SM 활동 지표(`DCGM_FI_PROF_SM_ACTIVE`)가 기본 활용도 메트릭보다 계산 효율의 더 정확한 척도를 제공할 때가 많다.

EFA는 드라이버 수준의 통계를 노출한다(바이트/패킷/재전송/타임아웃). 이를 통해 분산 학습의 집합 연산 병목 현상을 진단할 수 있다. [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) 플러그인은 NCCL을 libfabric 인터페이스에 매핑하여 MDOS의 OS-bypass 및 SRD 프로토콜을 활용할 수 있게 한다. 운영자는 EFA 카운터를 NCCL 진단(NCCL_DEBUG=INFO)과 결합해 네트워크 계층의 이슈를 파악할 수 있다.

[Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/)은 처리량 및 메타데이터 지연 시간(latency) 등 클라이언트 측 메트릭을 노출하고, 애플리케이션 차원의 메트릭(학습의 경우 단계 시간, 초당 토큰 수, 손실 값; 추론의 경우 TTFT, 토큰 간 지연)도 Prometheus 클라이언트 라이브러리를 통해 수출할 수 있다.

#### GPU 건강 상태 모니터링 및 경보

사전 장애 탐지는 하드웨어 이슈가 장기간 학습 중단으로 번지지 않도록 방지한다. 일반적인 워크플로우는 DCGM 건강 메트릭을 모니터링하고, 오류 수가 임계값을 초과하면 경보를 트리거한다. ECC 단일 비트 오류(SBE)는 소수의 수에서는 허용될 수 있지만, SBE 비율이 높아지면 이중 비트 오류(DBE) 또는 다른 실패로 이어질 수 있다. XID 63(행 매핑 실패), XID 64(GPU 버스에서 이탈), XID 94/95(포함/비포함 오류) 등은 일반적으로 즉시 노드 교체를 필요로 한다.

[GPU Health - Cluster 대시보드](https://grafana.com/grafana/dashboards/21645-gpu-health-cluster/) (Grafana 대시보드 ID 21645)는 일반적인 GPU 오류 패턴에 대한 참고 시각화를 제공한다. 이 대시보드는 모든 클러스터 노드의 ECC 오류, XID 이벤트, 열 관리 위반, 행 매핑 상태를 집계하여 운영자가 학습 작업에 영향을 주기 전에 하드웨어의 문제를 식별하도록 돕는다.

Figure 4: GPU Health - Cluster 대시보드에서 GPU 오류 패턴 및 인스턴스 보고

이 시리즈의 Part 5에서는 관측성 아키텍처에 대한 포괄적 다룰 예정으로, 메트릭 수집 전략, 대시보드 구성, 대규모 클러스터 건강 유지를 위한 경보 패턴 등을 다룬다.

## 결론

사전 학습 스케일링 법칙 하나에서 세 가지 보완적 규칙으로의 변화—사전 학습, 사후 학습, 테스트 타임 컴퓨트—은 인프라 요구사항을 분절시키지 않았다. 오히려 이를 강화했다. 세 가지 축 모두 가깝게 연결된 가속기 컴퓨트, 고대역폭 저지연 네트워킹, 확장 가능한 분산 스토리지가 필요하며, workload 구성과 리소스 스케줄링 패턴에 따라 차이가 난다.

이 글은 AWS에서 그 요구사항에 부합하는 네 가지 계층 구조를 제시한다: 인프라 빌딩 블록(EC2 P-인스턴스, EFA 네트워킹, 계층형 스토리지), 리소스 오케스트레이션(Slurm 및 SageMaker HyperPod와 함께하는 Kubernetes), ML 소프트웨어 스택(커널 드라이버에서 CUDA를 거쳐 NCCL 및 PyTorch까지), 그리고 관측성(Prometheus, Grafana, GPU 건강 모니터링). 각 계층은 그 위의 계층을 제약하고 가능하게 한다. 잘못 구성된 드라이버나 포화된 네트워크 링크는 최적화된 학습 실행을 병목시킬 수 있으며, 잘못된 병렬화 전략도 마찬가지다.

이러한 통합 지점을 이해하는 것은 Foundation model 생애주기 전반에 걸쳐 성능 병목을 진단하고 합리적인 스케일링 결정을 내리기 위한 기초다.

## 저자

[Aman Shanbhag](https://www.linkedin.com/in/aman-shanbhag/)은 NVIDIA의 MARS MLOps 팀에서 AI 성능 및 인프라 엔지니어로 재직 중이며, 연구 팀이 확장 가능한 고성능 ML 학습 및 추론 시스템을 구축하도록 돕는다. 그는 이전에 AWS에서 Specialist Solutions Architect로 일하며 전세계 고객의 ML 학습 및 추론 최적화를 지원했다. Aman은 Rice University에서 컴퓨터과학, 수학, 창업학 학위를 보유하고 있으며, AI 인프라, 성능 최적화, 분산 학습 및 추론에 집중한다.

[Pavel Belevich](https://www.linkedin.com/in/pbelevich/)은 Amazon Web Services의 GenAI ML Frameworks 팀의 선임 응용과학자로, 분산 학습 및 대형 모델 추론에 대한 연구를 생산적 규모의 고객 워크로드에 적용한다. AWS에 합류하기 전에는 PyTorch 분산 팀에서 FSDP 및 Pipeline Parallelism과 같은 핵심 분산 학습 기술에 기여했다. AWS에서 그는 MoE 통신 패턴 및 대규모 서빙/학습 워크플로우를 다룬다. 또한 전문 병렬성 및 대형 모델 시스템에 관한 기술 심층 분석을 통해 모범 사례를 정기적으로 공유한다.

[Keita Watanabe](https://www.linkedin.com/in/keitawatanabe/)은 Amazon Web Services의 GenAI ML Frameworks 팀의 수석 솔루션 아키텍트로, ML 시스템 성능 엔지니어링과 AWS에서의 ML 학습 및 추론 최적화를 전 세계 고객에게 지원한다. 그의 배경은 ML 연구 및 개발에 있다. AWS에 합류하기 전에는 Rakuten에서 연구 과학자로 일하며 이미지 기반 상품 검색 시스템을 개발했다. Keita는 도쿄 대학에서 과학 박사 학위를 보유하고 있다.
