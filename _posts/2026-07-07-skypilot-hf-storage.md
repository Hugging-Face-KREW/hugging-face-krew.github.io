---
layout: post
title: "모든 클라우드에서 AI 워크로드 실행, 허깅페이스에 저장: SkyPilot로 제로 이그레스 스토리지"
author: dailybot
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-07-07-skypilot-hf-storage/thumbnail.png
authors:
  - user: huggingface
  - user: njha
slug: "skypilot-hf-storage"
source_url: "https://huggingface.co/blog/skypilot-hf-storage"
source_published_date: "2026-07-07"
source_published_at: "2026-07-07T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Run AI workloads on any cloud, store on Hugging Face: zero-egress storage with SkyPilot](https://huggingface.co/blog/skypilot-hf-storage)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/skypilot-hf-storage -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 모든 클라우드에서 AI 워크로드 실행, 허깅페이스에 저장: SkyPilot로 제로 이그레스 스토리지

대부분의 팀에서 모델과 데이터셋은 하나의 클라우드의 하나의 리전에 있는 버킷에 저장됩니다. 개발, 학습, 서비스 용도로 사용할 수 있는 GPU는 점점 데이터가 있는 클라우드와 다른 클라우드에 위치합니다. 두 위치가 분리되는 순간, 자신의 데이터를 자신의 GPU로 읽어들이려면 교차 클라우드 전송 비용을 지불해야 합니다.

허깅페이스와 함께, 두 절반을 하나로 합쳤습니다: 모델과 데이터셋은 허깅페이스 허브에 남아 있고, SkyPilot은 GPU가 있는 어떤 클러스터에서 개발, 학습 또는 서비스 작업을 실행합니다. 하나의 `hf://` URL과 이미 가진 `HF_TOKEN`를 사용하여 허깅페이스 Bucket 또는 허깅페이스 허브 저장소를 SkyPilot 작업에 마운트한 다음, 용량이 있는 곳에서 시작합니다. 허깅페이스는 데이터 전송 비용을 청구하지 않으므로, 해당 GPU로 데이터를 읽는 데 비용이 들지 않습니다. 어떤 클라우드에서든 말이죠.

다음은 새로 도입된 내용입니다:

- **허깅페이스 허브 데이터는 모든 작업에서 사용 가능합니다.** `store: hf`은 허깅페이스 Bucket(읽기-쓰기) 또는 어떤 **모델 / 데이터셋 / Space 저장소**(읽기 전용)를 SkyPilot 작업에 `hf://` URL과 이미 가진 `HF_TOKEN`를 통해 마운트하고, `MOUNT` 또는 `COPY`를 통해 연결합니다.
- **어떤 GPU에서든, 어떤 클라우드에서든 실행합니다.** [SkyPilot](https://docs.skypilot.co/)는 20개 이상 클라우드, Kubernetes, Slurm, 온프렘에서 작업 계산 리소스를 찾아, 예약 GPU나 온디맨드 GPU 중 사용 가능한 것을 활용하고, 어떤 벤더에서든 실행됩니다.
- **데이터를 읽기 위한 egress가 없습니다.** 허깅페이스 Storage 요금은 [no egress or CDN fees](https://huggingface.co/pricing)이므로 SkyPilot이 작업을 배치하는 곳과 상관없이 모델과 데이터셋은 같은 버킷에서 직접 읽히고, 클라우드 간 복사 없이 egress 비용도 청구되지 않습니다.
- **Xet-backed 중복 제거.** 버킷은 [Xet](https://huggingface.co/docs/hub/xet/overview)를 기반으로 구축되므로, 증분 체크포인트와 모델 변형은 변경된 청크만 업로드됩니다.
- **함께 구축되었습니다.** [Hugging Face](https://huggingface.co/)와 [SkyPilot](https://docs.skypilot.co/)가 이를 함께 출시했고, 허깅페이스 팀은 권한이 없는 컨테이너에서도 작동하도록 `hf-mount` FUSE 수정 사항을 상류에 반영했습니다.

## 허깅페이스 Storage가 이제 SkyPilot의 1급 백엔드가 되었습니다 {#section-1}

![SkyPilot mounts Hugging Face models, datasets, and checkpoints into jobs running on reserved GPU clusters across CoreWeave, Nebius, GCP, and 20+ more, with zero-egress reads.](/blog/assets/skypilot-hf-storage/architecture.png)

SkyPilot 작업은 이미 로컬 경로에 마운트하여 S3, GCS, Azure, R2 등 다양한 클라우드 객체 스토리지를 읽고 쓸 수 있습니다. 허깅페이스 Storage가 이제 `store: hf`로 이 목록에 합류했고, `hf://` 체계를 통해 도달합니다:

```yaml
file_mounts:
  # A Hugging Face Bucket, read-write, for checkpoints, logs, processed data.
  /checkpoints:
    source: hf://buckets/my-org/qwen-sft
    store: hf
    mode: MOUNT # or COPY
  # A model repo, mounted read-only.
  /base-model:
    source: hf://Qwen/Qwen3.5-4B
    store: hf
    mode: MOUNT
  # A dataset repo, pinned to a revision, read-only.
  /data:
    source: hf://datasets/my-org/my-dataset@main
    store: hf
    mode: MOUNT
```


그 하나의 `hf://` 체계는 전체 수명주기를 다룹니다: 저장소에서 **모델**과 **데이터셋**을 읽고, 학습하는 동안 **체크포인트**를 버킷에 쓰고, 완성된 모델을 다시 저장소에 게시하며, 서비스를 제공할 때 추론 서버로 끌어옵니다. 대부분의 팀은 이미 모델과 데이터셋을 허깅페이스 허브에 보관하고 있어 이관 단계나 새 스토리지 계정을 만들 필요도 없습니다.

`MOUNT`은 허깅페이스의 [`hf-mount`](https://github.com/huggingface/hf-mount) FUSE 백엔드를 사용하므로 버킷이나 저장소가 SkyPilot의 다른 FUSE 마운트( `gcsfuse`, `blobfuse2`, `rclone`, `goofys` ) 옆의 로컬 경로로 표시됩니다. 이 패치는 파일 시스템 계층에서 일어나며, 코드가 `read()`를 호출하면 드라이버가 Xet 백엔드에서 필요한 바이트만 가져오므로 실제로 다루는 데이터만 네트워크를 통해 전송되고, `hf-mount`은 디스크 캐시를 유지해 반복 읽기를 로컬로 유지합니다. 이 디스크 기반 캐시는 SkyPilot이 다른 백엔드에서 제공하는 동작으로, [`MOUNT_CACHED`](https://docs.skypilot.co/en/latest/reference/storage.html) 아래에서, 일반적인 `MOUNT`는 버킷에서 매번 읽기를 스트리밍하고 로컬에 보관하는 데이터가 없습니다. `hf` 저장소의 경우, `MOUNT` 및 `MOUNT_CACHED`는 동일하게 동작하여 어느 모드에서든 캐시를 유지합니다.

읽기가 지연되므로, 전체 파일이 다운로드되기 전에 큰 파일을 처리하기 시작할 수 있습니다. 이렇게 하면 GPU를 거의 즉시 바쁘게 유지하면서 데이터가 스트리밍되며 학습이 진행됩니다(데이터셋이나 체크포인트를 복사하는 동안 유휴 상태로 두지 않음). 이것은 보통 첫 에폭에서 큰 이점을 제공합니다. `COPY`은 반대 경로를 택해 `huggingface_hub`를 통해 미리 다운로드합니다. 특별한 요건은 없습니다.

인증은 이미 가지고 있는 토큰입니다. 환경 변수에 `HF_TOKEN`를 설정하고 런에 [`--secret HF_TOKEN`](https://docs.skypilot.co/en/latest/running-jobs/environment-variables.html)를 전달하십시오; SkyPilot은 작업이 배치되는 어떤 클라우드에서든 마운트에 이를 사용합니다. 이 토큰은 작업이 AWS, GCP, Azure, Nebius, Lambda, 또는 귀하의 Kubernetes 클러스터에 배치되든 하나의 토큰으로 작동하므로, 클라우드별 버킷 키를 따로 관리할 필요가 없습니다.

## 데이터 전송 비용 없음: 스토리지가 실행 위치를 결정하지 않음 {#section-2}

GPU 용량은 더 이상 한 곳에서 나오지 않습니다. 충분한 H100과 H200을 확보하려면 팀은 여러 벤더에 걸쳐 예약된 용량을 한꺼번에 확보하고(하이퍼스케일러의 블록, 네오클라우드의 클러스터, 어쩌면 온프렘 랙), 할당된 곳에서 실행합니다. SkyPilot은 이를 위해 만들어졌습니다: 하나의 작업 명세로 20개 이상 클라우드(Kubernetes, 온프렘 포함)에 걸쳐 스케줄링되며, 예약된 클러스터 중 비어 있는 곳에 도달합니다.

오브젝트 스토리지는 여전히 문제의 원인입니다. 오브젝트 스토어는 지역별이고 클라우드별이므로, 다른 벤더의 데이터 센터에 위치한 GPU나 추론 서버에 데이터를 공급하려면 데이터를 각 벤더의 버킷에 복제하거나 데이터를 전송해야 합니다. 대부분의 클라우드는 데이터가 네트워크를 떠날 때 egress를 청구하고( AWS에서 약 $0.09/GB 외부 전송), 같은 클라우드 내에서도 지역 간 전송 비용이 발생합니다. 기본 모델을 모든 추론 노드에 올리거나 다른 클라우드의 클러스터에서 여러 에폭에 걸쳐 데이터셋을 반복 학습하는 경우에도, 이미 예약한 GPU 비용 위에 추가 요금이 붙습니다. 팀은 데이터를 가진 벤더의 실행에 각 런을 고정시키고 나머지 용량은 비활성 상태로 남겨두는 경향이 있습니다.

읽기는 여전히 무료입니다: [no egress or CDN fees](https://huggingface.co/pricing)와 Storage의 월 비용이 12-18달러/TB로 책정되고(AWS S3의 약 23달러/TB에 비해), 같은 버킷은 이 클러스터들 가운데 어디에서든 접근 가능하며 GPUs가 실행되는 곳에 상관없이 읽기는 무료입니다. 쓰기는 여전히 계산 클라우드의 일반적인 egress 비용이 들며, 이는 오프클라우드 저장소에 대해 지불하는 것과 동일하지만, 대부분의 AI 작업에서 읽기가 지배적이므로 수많은 에폭에 걸친 데이터셋 스트리밍이나 모든 새로운 학습/추론 노드에 모델 가중치를 가져오는 경우가 많습니다. 따라서 각 실행을 데이터의 사본을 가진 벤더에 고정하지 않게 됩니다.

## 간단한 벤치마크 {#section-3}

벤치마크 수치를 얻기 위해, 우리는 작은 파인-튜닝을 실행했습니다: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)를 [`HuggingFaceH4/Multilingual-Thinking`](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) 데이터세트에 대해 TRL의 [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer)와 함께, 허깅페이스 허브 저장소에서 모델을 읽기 전용으로 마운트하고 각 체크포인트를 허깅페이스 Bucket에 기록했습니다. 동일한 SkyPilot YAML이 AWS, GCP, Lambda에서 실행되었고, `--infra`만 다릅니다. SkyPilot은 GPU가 비어 있는 곳에 각 작업을 배치했고, 세 가지 모두 같은 버킷에서 읽고 썼습니다.

```yaml
# qwen-sft.yaml. Launch anywhere: sky launch qwen-sft.yaml --infra aws|gcp|...
resources:
  accelerators: H100:1 # or whatever the cloud has

file_mounts:
  /base-model:
    source: hf://Qwen/Qwen3.5-4B # read-only, lazy-mounted from the Hub
    store: hf
    mode: MOUNT
  /checkpoints:
    source: hf://buckets/my-org/qwen-sft # read-write Bucket
    store: hf
    mode: MOUNT

run: |
  python train.py --model /base-model --output_dir /checkpoints
```


다음은 측정한 내용:

- **모델이 모든 클라우드에서 무료로 로드되었습니다.** 지연 읽기 덕분에 `from_pretrained`가 건드리는 부분만 읽어 와 약 30초 안에 학습 시작이 가능했습니다(최대 약 500 MB/s). 허깅페이스는 egress를 청구하지 않으므로 그 읽기는 비용이 들지 않습니다. 만약 모델이 S3에 저장되어 있었다면, 다른 클라우드의 GPU로의 모든 읽기에 egress가 청구되었을 것입니다( AWS에서 약 $0.09/GB).
- **체크포인트를 버킷으로 직접 스트리밍하여 최대 약 170 MB/s 속도(가중치당 약 8.43 GB)로 전달되었고, GPU 인스턴스가 제거된 뒤에도 유지되었습니다.**

클라우드별로, 체크포인트가 버킷으로 기록된 속도는:

| 클라우드              | GPU  | 체크포인트 기록 속도 |
| :----------------- | :--- | :--------------- |
| AWS (us-east-2)    | L40S | ~168 MB/s        |
| GCP (us-central1)  | L4   | ~123 MB/s        |
| Lambda (us-west-3) | H100 | ~112 MB/s        |

## Xet-backed 저장소: 체크포인트 및 모델 변형의 중복 제거 {#section-4}

허깅페이스 Bucket은 [Xet](https://huggingface.co/docs/hub/xet/overview) 위에 구축되며, 이는 [content-defined chunking](https://huggingface.co/docs/hub/xet/deduplication)를 사용해 파일을 약 64 KB 청크로 분할하고 각 고유 청크를 한 번만 저장합니다. 경계가 콘텐츠를 따라가기 때문에 편집하면 닿는 청크만 변경되고 나머지는 이미 저장된 것으로 간주됩니다. 이는 몇 가지 상황에서 이점이 있습니다:

- **점진적 업데이트 및 어댑터 체크포인트.** 계층을 고정하고 어댑터를 학습시키거나 저장 사이에 대부분의 가중치를 건드리지 않는 경우, 변경된 청크만 업로드되고 전체 체크포인트는 업로드되지 않습니다.
- **기본 모델을 공유하는 모델 변형.** 기본 모델의 미세 조정과 양자화는 겹치는 부분이 많으므로 공유 청크가 모든 버전에 한 번만 저장됩니다.
- **덧붙이는 데이터셋.** 대화 추적이나 추론 출력 같은 로그가 Parquet 파일에 행을 추가하며 커지게 됩니다. 기존의 행 그룹은 바이트 단위로 동일하게 유지되므로 새로운 행만 전송됩니다: 허깅페이스의 [test](https://huggingface.co/blog/parquet-cdc)에서 100K 행의 테이블에 10K 행을 추가하면 전체 약 106 MB 대신 약 10 MB가 이동합니다. (행을 제자리에서 편집하거나 삭제하는 경우, 변경 사항을 로컬로 유지하려면 `use_content_defined_chunking=True`로 작성하십시오.)
- **다시 업로드 시 이미 저장된 것은 건너뜁니다.** 우리의 테스트에서 버킷에 이미 있는 8.43 GB 블롭을 다시 업로드하는 데 약 8초가 걸렸고, 최초 업로드는 24초였는데, 이는 청크 해시만 이동하기 때문입니다. 같은 메커니즘은 저장소 간 서버 측 `hf buckets cp`가 바이트 재업로드 없이 참조로 복사되도록 해줍니다.

당신의 아티팩트가 얼마나 겹치는지에 따라 절약 효과가 달라지지만, 중복 제거는 자동으로 작동합니다: 체크포인트를 일반적으로 기록하면, 새로운 청크만 기계에서 벗어나게 됩니다.

## 시작하기 {#section-5}

```bash
pip install "skypilot[huggingface]"
hf auth login  # or: export HF_TOKEN=<your-token>
```


어떤 SkyPilot 작업에도 `hf://` 마운트를 추가하고 실행합니다. `MOUNT`는 glibc 2.34+를 가진 기본 이미지와 `/dev/fuse`가 필요합니다.

## 함께 구축되었습니다: 허깅페이스와 SkyPilot {#section-6}

초기 `store: hf` 지원은 Nikhil Jha로부터 나왔습니다. 허깅페이스 팀은 [carried it forward](https://github.com/skypilot-org/skypilot/pull/9698)를 진행했고, 무권한 컨테이너에서도 마운트할 수 있도록 `hf-mount` FUSE 수정을 상류에 반영했습니다. 이는 많은 Kubernetes 클러스터의 기본값입니다. SkyPilot 팀은 이를 저장소 백엔드에 연결했습니다. 전체 경로는 오픈 소스입니다: SkyPilot, 허깅페이스의 `hf-mount`, 그리고 `huggingface_hub` 클라이언트.

## Resources {#section-7}

- [SkyPilot storage docs](https://docs.skypilot.co/en/latest/reference/storage.html)
- [Hugging Face Storage Buckets guide](https://huggingface.co/docs/hub/storage-buckets)
- [`hf-mount`](https://github.com/huggingface/hf-mount)
- [Xet: content-defined chunking and deduplication](https://huggingface.co/docs/hub/xet/deduplication)
- [SkyPilot Slack community](https://slack.skypilot.co/)
