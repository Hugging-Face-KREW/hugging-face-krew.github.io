---
layout: post
title: "Trainium3을 트랙에 올리다: LTX-2.5로 동영상 생성하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/video-trainium3/thumbnail.jpg
image: assets/images/blog/posts/2026-11-27-video-trainium3/thumbnail.jpg
authors:
  - user: dacorvo
  - user: Jingya
  - user: sayakpaul
slug: "video-trainium3"
source_url: "https://huggingface.co/blog/video-trainium3"
source_published_date: "2026-11-27"
source_published_at: "2026-11-27T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Taking Trainium3 to the Track: Generating Videos with LTX-2.5](https://huggingface.co/blog/video-trainium3)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/video-trainium3 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Trainium3을 트랙에 올리다: LTX-2.5로 동영상 생성하기

AWS와의 파트너십 덕분에 한 달 동안 Trainium3 머신([trn3.48xlarge](https://aws.amazon.com/ec2/instance-types/trn3/), 64 NeuronCores)을 사용할 수 있었습니다. 그 한 달의 대부분을 그 위에서 Hugging Face 모델을 테스트하고, 더 많은 모델이 작동하도록 만드는 데 썼습니다.

5일이 남았을 때, 저는 잠시 재미를 즐겨 보기로 했습니다. 페라리를 갖고 있다면 가끔은 트랙에 몰고 나가고 싶어지기 마련이니까요. 그래서 스스로에게 물었습니다. 동영상을 생성해 보면 어떨까?

동영상과 사운드를 함께 생성하는 Lightricks의 [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)를 골랐습니다. 결국 모델을 로드한 뒤에는 사운드가 포함된 10초 클립을 머신의 절반만 사용해 22~23초 만에 생성할 수 있었습니다. 결과는 다음과 같습니다.

<figure>
  <video
    controls
    playsinline
    preload="metadata"
    style="display: block; width: 100%; height: auto;"
    aria-label="The seal film, made with LTX-2.5 on Trainium3"
    src="https://huggingface.co/buckets/dacorvo/video-generation/resolve/entertainment/seal.mp4"
  ></video>
  <figcaption style="text-align: center;">
    <em>A seal dreams of warm sand. Generated entirely on Trainium3.</em>
  </figcaption>
</figure>

## 두 가지 목표 {#section-1}

목표는 두 가지였습니다. 첫 번째는 Trainium3에서 [diffusers](https://github.com/huggingface/diffusers)로 동영상을 생성하는 일이 얼마나 쉬운지 확인하는 것이었습니다. 예제에 나오는 단순한 클립이 아니라, 실제 사용 사례에 가까운 방식으로 대본에서 단편 영화를 만들고 싶었습니다. 영화를 초안으로 제작하는 것처럼 말이죠. 두 번째는 성능이었습니다. CUDA나 LTX 자체 API를 사용할 때 얻는 성능에 가까워지고 싶었습니다.

CUDA와 비교할 만한 실제 수치는 찾지 못했습니다. 유일한 수치는 B200 GPU 4개에서 10초 분량의 720p 클립을 6.8초에 생성한다는 LTX의 주장뿐인데, 어떻게 측정했는지에 대한 세부 정보가 없었습니다. 그래서 CUDA는 제쳐 두고 LTX의 온라인 API와 비교했습니다. [LTX에 따르면](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights) 이 API는 10초 분량의 1080p 클립을 23.7초에 생성합니다. 결국 이 설정은 그 수치와 맞먹었고, 약 1초 더 빨랐습니다. 모델을 로드한 뒤 머신의 절반만 사용해 2048×1024 해상도의 10초 클립을 22~23초에 생성했습니다. 파이프라인을 최적화할 시간이 거의 없었다는 점을 고려하면 나쁘지 않은 결과입니다.

## 작동 방식 {#section-2}

입력은 영화입니다. 각 쇼트에 이름과 프롬프트가 포함된 JSON 목록으로, 스토리보드와 비슷합니다. LTX-2.5는 긴 프롬프트에서 더 좋은 성능을 내므로 각 프롬프트는 스타일, 피사체, 움직임, 카메라, 조명, 사운드를 설명하는 한 단락으로 작성됩니다. 다음은 물개 영화의 파일이며, 프롬프트는 짧게 줄였습니다.

```json
[
  {"name": "dream", "prompt": "Style: cinematic, realistic, shallow depth of field. A plump seal ... is fast asleep on warm golden sand under palm trees ...",
   "seed": 3},
  {"name": "nap", "prompt": "Style: cinematic, realistic, wide-angle lens. ... a plump grey seal is fast asleep on a small ice floe ...",
   "seed": 1, "to": {"shot": "iceberg", "frames": 9}},
  {"name": "iceberg", "prompt": "Style: cinematic, realistic, wide-angle lens. ... the whole iceberg slowly rolls over ... and washes over the seal. ..."},
  {"name": "shake", "prompt": "Style: cinematic, realistic, wide-angle lens. ... It slowly shakes its head and body ... and flops down on its belly on the floe, ...",
   "from": {"shot": "iceberg", "frames": 25}}
]
```


각 쇼트에는 `name`와 `prompt`이 있으며, 여러 테이크 중 선택한 테이크를 유지하기 위해 `seed`를 지정할 수도 있습니다. 쇼트는 두 개의 키를 사용해 연결합니다. 프롬프트만으로는 한 쇼트에서 다음 쇼트까지 피사체의 일관성을 유지할 수 없기 때문에 이 연결이 중요합니다.

- `from`: 쇼트가 이전 쇼트의 마지막 프레임에서 시작합니다. 이전 쇼트의 마지막 프레임에서 시작하면 컷을 넘어 피사체의 모습을 유지할 수 있고, 마지막 25개 프레임(1초)에서 시작하면 움직임도 이어집니다.
- `to`: 쇼트가 이후 쇼트의 첫 프레임에서 끝나므로, 다음 쇼트로 자연스럽게 이어집니다.

쇼트를 잘라 내거나 그 위에 캡션을 표시할 수도 있습니다.

쇼트로 영화를 만드는 방식은 애니메이션과 비슷하며, 진행 방식은 두 가지입니다. 스트레이트 어헤드 방식에서는 각 쇼트가 바로 앞의 쇼트를 이어 갑니다. 포즈 투 포즈 방식에서는 먼저 핵심 쇼트를 만들고 마음에 드는 결과를 얻을 때까지 다듬습니다. 그런 다음 그 앞의 쇼트는 핵심 쇼트의 첫 프레임(`to`)에서 끝나고, 뒤의 쇼트는 핵심 쇼트의 마지막 프레임(`from`)에서 시작합니다. 영화가 페이오프(모든 전개가 향해 가는 순간)로 이어지는 구조라면, 페이오프가 핵심 쇼트입니다. 저는 시행착오를 거쳐 이 순서를 찾았고, 물개 영화는 `iceberg`에서 시작해 포즈 투 포즈 방식으로 만들었습니다.

페이오프는 `iceberg`입니다. 빙산이 뒤집히고, 그 파도가 물개를 흠뻑 적십니다. 잠든 물개인 `nap`은 `iceberg`의 첫 9개 프레임(`to`)에서 끝나므로, 페이오프까지 물개와 빙원, 빙산이 같은 모습으로 유지됩니다. `shake`는 `iceberg`의 마지막 25개 프레임(`from`)에서 시작합니다. 물을 뚝뚝 흘리는 물개가 몸을 털어 물을 털어 내고 빙원 위에 털썩 주저앉습니다. `dream`은 별도로 만들고, 컷으로 나머지 부분에 연결했습니다.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/video-trainium3/shot-chain-diagram.svg" alt="Diagram of the seal film's four shots: dream stands alone, nap ends on the first 9 frames of iceberg, and shake starts from the last 25 frames of iceberg." width="90%">
  <figcaption>The seal film's shot chain. <code>iceberg</code>, the payoff, is made first; <code>nap</code> ends on its first 9 frames and <code>shake</code> starts from its last 25, so the seal, the floe and the iceberg stay the same.</figcaption>
</figure>

diffusers 측에서는 세 개의 파이프라인이 [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)의 모델을 공유합니다. `LTX2Pipeline`는 프롬프트만으로 쇼트를 만들고, `LTX2ConditionPipeline`는 다른 쇼트의 프레임에서 시작하거나 끝나는 쇼트를 만듭니다. 이 두 파이프라인은 모델 카드에 설명된 2단계 증류 레시피를 따릅니다. 먼저 8단계를 수행하고, 세 번째 파이프라인인 `LTX2LatentUpsamplePipeline`가 최종 3단계를 수행하기 전에 latent를 ×2로 업스케일합니다.

모든 생성 작업은 diffusers 파이프라인이 처리합니다. 영화 제작 부분은 그 위에 얹은 약 160줄의 Python 코드입니다. 영화 파일을 읽고, 각 키프레임이 이를 필요로 하는 쇼트보다 먼저 존재하도록 쇼트 순서를 정하고, 프레임을 잘라 전달합니다. 그런 다음 트리밍과 캡션 삽입을 수행하고, 사운드 레벨을 고르게 만든 뒤 영화를 조립합니다. 코드는 [dacorvo/video-generation](https://github.com/dacorvo/video-generation)에 있습니다. 제가 보관한 영화와 JSON 파일은 [dacorvo/video-generation](https://huggingface.co/buckets/dacorvo/video-generation/tree/entertainment) 버킷에 있습니다. 물개, 여우, 염소 영화가 있으며, 염소 영화는 두 쇼트를 잘라 내고 페이오프 위에 캡션을 표시합니다.

<figure>
  <video
    controls
    playsinline
    preload="metadata"
    style="display: block; width: 100%; height: auto;"
    aria-label="The fox film, made with LTX-2.5 on Trainium3"
    src="https://huggingface.co/buckets/dacorvo/video-generation/resolve/entertainment/fox.mp4"
  ></video>
  <figcaption style="text-align: center;">
    <em>The fox film: a walk in the snow, made "straight ahead": each shot continues the one before.</em>
  </figcaption>
</figure>

<figure>
  <video
    controls
    playsinline
    preload="metadata"
    style="display: block; width: 100%; height: auto;"
    aria-label="The goat film, made with LTX-2.5 on Trainium3"
    src="https://huggingface.co/buckets/dacorvo/video-generation/resolve/entertainment/goat.mp4"
  ></video>
  <figcaption style="text-align: center;">
    <em>The goat film: two trimmed shots, and a caption drawn over the payoff.</em>
  </figcaption>
</figure>

## Trainium3에서 실행하기 {#section-3}

기본 상태에서는 모델이 NeuronCores에서 실행되지 않습니다. transformer가 하나의 코어에 들어가지 않고, diffusers에는 이를 분할할 계획이 없기 때문입니다. 실행에 성공한 첫 시도는 모델 카드의 파이프라인에 transformer를 4개 코어에 분할하는 작업과 몇 가지 패치를 적용한 것이었는데, 10초 클립 하나에 약 **5분 20초**가 걸렸습니다. 그중 NeuronCores가 실제로 계산한 시간은 약 45초뿐이었습니다. 게다가 목표 해상도에는 도달할 수 없었습니다.

5일 후:

| 사운드가 포함된 10초 클립 | NeuronCores | 총 소요 시간 | NeuronCores에서의 시간 | 첫 시도 대비 |
|---|---|---|---|---|
| 첫 시도, 960×544 | 4 | 5분 20초 | 45초 | 1배 |
| 최적화 후, 2048×1024(픽셀 약 4배) | 32 | 22~23초 | 15초 | 14배 |

여기까지 도달하려면 세 종류의 작업이 필요했습니다. 모델을 NeuronCores에서 실행되게 만들고(텐서 병렬 처리 필요), 더 많은 코어에 분할하고(시퀀스 병렬 처리), 호스트가 코어의 속도를 따라가도록 만드는 일이었습니다.

<figure class="image text-center">
  <img class="mx-auto" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/video-trainium3/parallelism-layout.svg" alt="Layout of the 32 NeuronCores that make one clip: the clip split in 8 stretches of time, one per chip (sequence parallelism); each core also keeps one band of every layer's weight table (tensor parallelism). Each core holds a quarter of the weights and an eighth of the tokens." width="90%">
  <figcaption>TP-4 × SP-8: 32 cores on one clip. The weights split 4 ways, one band of every layer per core; the video tokens split 8 ways, one stretch of the clip per chip.</figcaption>
</figure>

아래 수정 사항의 대부분은 디바이스에서 이루어졌고, 대부분의 노력도 그쪽에 들어갔습니다. 엔드투엔드 기준으로 가장 크게 시간을 줄인 부분은 호스트였습니다. 낭비가 발생한 곳이 분명히 호스트였기 때문에, 가장 쉽게 개선할 수 있는 부분이기도 했습니다.

### 작동하게 만들기까지 필요한 작업

| 문제 | 해결 방법 | 결과 |
|---|---|---|
| transformer가 하나의 NeuronCore에 들어가지 않고, diffusers에는 이를 위한 텐서 병렬 계획이 없음 | [계획](https://github.com/dacorvo/video-generation/blob/main/parallel.py#L50)과 한 rank의 헤드에서 실행되는 [attention processor](https://github.com/dacorvo/video-generation/blob/main/parallel.py#L65)를 구현 | transformer가 4개 코어에서 실행됨 |
| transformer 옆에 텍스트 인코더를 NeuronCore에 함께 배치할 수 없어 CPU에서 실행됨. diffusers는 이런 방식의 파이프라인 분할을 지원하지만, LTX-2 파이프라인은 텍스트 인코더의 입력을 NeuronCores에서 전달함 | 파이프라인을 호스트에 고정하고, transformer의 입력을 코어로 이동한 뒤 출력을 다시 가져옴 | transformer를 코어에서 실행하는 파이프라인이 작동함 |
| 하나의 그래프로 컴파일하면 transformer가 특정 크기에서 NaN을 반환하고, 그래프가 모두 캐시에 있어도 새 프로세스마다 다시 트레이싱함 | `compile_repeated_blocks`를 사용해 블록 하나를 컴파일 | 유한한 출력과 함께 트레이싱 시간이 388초에서 18초로 감소 |
| torch-neuronx는 상수를 포함해 bf16 GELU를 완전히 bf16으로 계산함([#1049](https://github.com/aws-neuron/torch-neuronx/issues/1049)) | GELU를 fp32로 계산 | fp32 대비 transformer 오차: 0.297 → 0.158(CPU bf16: 0.160) |
| 2048×1024에서 4방향으로 분할된 transformer의 블록 그래프를 로드할 수 없음 | 비디오 토큰을 상위 단계에서 8방향으로 분할(시퀀스 병렬 처리): diffusers의 `set_attn_processor` 훅에 두 개의 custom attention processor를 추가하고, 각각 collective 호출 전후에 몇 가지 텐서 reshape를 수행([코드](https://github.com/dacorvo/video-generation/blob/main/parallel.py#L141)) | Stage 2가 4 × 8 = 32개 코어에서 스텝당 3.0초로 실행됨 |
| 프레임에서 시작하는 쇼트에는 각 비디오 토큰마다 timestep이 있어 블록 그래프를 로드할 수 없음 | 두 timestep만 임베딩하고, 각 토큰에 하나를 선택([코드](https://github.com/dacorvo/video-generation/blob/main/timesteps.py)) | 프레임에서 시작하는 쇼트가 프롬프트에서 시작하는 쇼트와 거의 같은 시간에 생성됨 |

### 빠르게 만들기까지 필요한 작업

| 문제 | 해결 방법 | 개선 폭 |
|---|---|---|
| 각 rank(NeuronCore당 하나의 프로세스)가 전체 파이프라인을 실행했고, 모든 rank가 한 소켓의 CPU에서 프롬프트를 인코딩함 | rank 0에서만 파이프라인을 실행하고, 다른 rank에는 각 호출의 인자를 전달 | 프롬프트 인코딩: 약 205초 → 7.5초 |
| torch-neuronx의 allocator가 대부분의 메모리 세그먼트를 사용하지 않음([#1052](https://github.com/aws-neuron/torch-neuronx/issues/1052)) | 파라미터를 최대 1GiB 크기의 버퍼에 패킹 | 코어당 메모리: 22.96GiB → 10.2GiB |
| 시퀀스 길이가 128토큰의 배수가 아니면 attention이 일반 연산으로 실행됨 | `NEURON_LOGICAL_NC_CONFIG=2`을 사용해 1024×512(프레임당 512토큰)로 생성 | transformer 스텝: 4.3초 → 3.0초 |
| CPU에서 비디오 디코딩이 느리고, CPU 코어를 늘려도 도움이 되지 않음 | NeuronCores에서 프레임 단위로 나누어 디코딩하고, 공유 메모리를 통해 프레임을 rank 0에 전달 | 2048×1024 디코딩: 357초 → 1.6초 |
| 디바이스로 복사되는 strided slice가 디바이스에서 느린 복사 방식으로 채워짐([#1050](https://github.com/aws-neuron/torch-neuronx/issues/1050)) | 복사 전에 [`.contiguous()`](https://github.com/dacorvo/video-generation/blob/main/parallel.py#L231-L235) | 16개 코어에서의 스텝: 5.6초 → 1.0초 |
| rank 0만 파이프라인의 CPU 단계(text encoder, upsampler, vocoder)를 실행하지만, 모든 rank가 노드 전체에 걸친 torch 스레드 풀을 할당받아 31개 rank가 이를 고갈시킴 | rank 0의 스레드를 다른 rank가 사용하지 않는 CPU에 고정하고, 나머지 rank의 스레드 수를 공정한 몫으로 제한([코드](https://github.com/dacorvo/video-generation/blob/main/generate.py#L330)) | 프롬프트 인코딩: 7.4초 → 3.3초. Stage 1 스텝: 0.9초 → 0.6초 |
| 파이프라인이 프롬프트를 1024토큰으로 패딩하고, CPU에서 실행되는 text encoder도 패딩을 인코딩함 | 프롬프트의 실제 토큰만 text encoder에서 실행 | 프롬프트 인코딩: 3.4초 → 1.4초 |

세 가지 문제는 끝내 설명하지 못했습니다. transformer를 하나의 그래프로 컴파일하면 특정 크기에서 NaN을 반환했고, 새 프로세스에서 첫 번째 키프레임 종료 쇼트가 느렸으며(모든 그래프가 캐시에 있어도 마찬가지였습니다), latent upsampler는 transformer 옆에서 로드되지 않아 CPU에 남아 있습니다.

이 문제 중 일부는 upstream에서 수정되고 있습니다. [diffusers#15420](https://github.com/huggingface/diffusers/pull/15420)(open)은 LTX-2 파이프라인을 수정하며, 이를 통해 위의 두 번째 해결 방법은 필요 없어집니다. [diffusers#14725](https://github.com/huggingface/diffusers/pull/14725)(still open)은 텐서 병렬 처리와 컨텍스트 병렬 처리를 하나의 `ParallelConfig`로 통합하며, LTX-2.5를 위한 `_tp_plan`도 계획되어 있습니다.

## 배운 점 {#section-4}

- **호스트와 디바이스 모두 크게 개선됐다.** 일단 실행된 뒤에는 클립의 호스트 시간이 약 280초에서 약 8초로 줄었고(약 35배), NeuronCores에서의 시간은 45초에서 15초로 줄었습니다(3배). 픽셀 수는 4배였고, 첫 시도에서는 로드조차 할 수 없었던 두 번째 단계도 포함된 결과입니다.
- **시퀀스 병렬 처리가 하나의 클립으로 머신을 사용하는 방법이다.** 텐서 병렬 처리는 메모리 제약(transformer가 하나의 코어에 들어가지 않음)으로 강제되고, 32개의 attention 헤드에 의해 제한됩니다. 시퀀스를 더 많은 코어에 분할해야 64개 NeuronCores 중 32개를 하나의 클립에 사용할 수 있습니다. 또한 각 코어의 그래프 크기를 로드 가능한 수준으로 줄여 2048×1024를 가능하게 만든 것도 시퀀스 병렬 처리입니다.
- **Trainium에서는 그래프 크기를 기준으로 계획해야 한다.** shape에 따라 attention이 빠른 커널을 사용할 수 있는지가 결정되고, 그래프가 지나치게 크면 로드에 실패합니다.
- **먼저 CPU와 비교하라.** 모델을 분할할 때마다 디바이스에서 실행하기 전에 CPU에서 확인했고, CPU와 비교하는 과정에서 GELU 문제를 발견했습니다.
- **트레이싱된 코드를 안정적으로 유지하라.** 제가 사용한 빌드에서는 torch-neuronx의 캐시 키에 각 traced op의 파일과 줄 번호가 포함되어 있었습니다. 따라서 수정 한 번으로 그래프당 최대 30분이 걸리는 재컴파일이 발생할 수 있었습니다. 이미 이 문제를 보고했고([aws-neuron/torch-neuronx#1027](https://github.com/aws-neuron/torch-neuronx/issues/1027)), 현재는 종료되었습니다.
- **영화에는 프롬프트 이상의 것이 필요하다.** 프롬프트만으로는 한 쇼트에서 다음 쇼트까지 캐릭터나 산을 유지할 수 없으므로, 쇼트는 다른 쇼트의 프레임에서 시작하거나 끝납니다. 각 쇼트는 장소도 유지해야 합니다. 여우 영화의 초기 버전에서 한 쇼트가 여우를 숲에서 호수로 이동시켰을 때, 모델은 다른 여우로의 디졸브 전환을 만들어 냈습니다.
- **각 쇼트를 여러 번 생성하라.** 모델은 세부 사항을 자주 틀립니다. 같은 프롬프트를 사용해도 염소에게 우스꽝스러운 뿔이 생긴 결과가 4번 중 3번 나왔습니다. 저는 새 쇼트마다 4개의 테이크를 만들고 그중 하나를 골랐습니다.

## 전문적인 사용 사례 {#section-5}

성능 측면에서는 모델을 로드한 뒤 사운드가 포함된 2048×1024 해상도의 10초 클립을 22~23초에 생성합니다. 인스턴스의 절반만 사용해 LTX의 유료 API에 필적하는 성능입니다. 서비스에서는 프로세스를 계속 실행해 둘 수 있습니다. 모델을 로드하는 데 약 3분이 걸리고, 첫 클립을 생성하는 데 80초가 걸리기 때문입니다.

기능 측면에서 diffusers는 영화 초안을 만들 수 있는 요소를 갖추고 있습니다. 쇼트는 프롬프트에서 생성되며, 다른 쇼트의 프레임에서 시작하거나 끝나도록 해 쇼트 간 피사체를 유지합니다. 테이크는 seed에 따라 달라지고, 영화 파일에는 제가 선택한 테이크가 기록됩니다. 같은 코드와 영화 파일로 다시 실행하면 바이트 단위로 동일한 영화를 얻습니다. 쇼트는 스토리보드 프레임이나 캐릭터 시트 같은 이미지에서 시작할 수도 있습니다. condition pipeline은 이를 프레임과 같은 방식으로 처리합니다(영화에서는 시도하지 않았습니다).

아직 초안 단계에 머무르는 이유는 다음과 같습니다.

- 모델이 여전히 세부 사항을 틀리므로 새 쇼트마다 여러 테이크가 필요합니다.
- 각 쇼트가 자체 사운드를 생성합니다. 영화 전체에서 배경 음량은 고르게 조정하지만, 사운드가 한 쇼트에서 다음 쇼트로 이어지지는 않습니다.
- 쇼트의 길이는 모델의 제한인 최대 20초입니다.

## 영화에서 월드 모델로 {#section-6}

LTX는 LTX-2.5를 월드 모델이라고 부르며, 로보틱스를 새로운 활용 분야로 제시합니다. 로봇, 자동차, 드론은 카메라에 보이는 장면의 동영상으로부터 학습하고, 월드 모델은 그보다 더 많은 동영상을 만들 수 있습니다. 촬영하기에는 너무 드물거나 위험한 사건도 포함해서 말입니다. 그래서 같은 코드로 실제 촬영 영상에도 시도해 보았습니다. 실제 카메라 클립의 첫 1초에서 쇼트를 시작하고 그다음에 일어날 일을 만들어 낼 수 있습니다. 한 프레임만으로는 충분하지 않았습니다. 모델은 한 프레임만 보고는 사물이 얼마나 빠르게 움직이는지 알 수 없기 때문입니다.

다음은 Fourier GR1-T2 휴머노이드 로봇의 헤드 카메라 영상입니다(NVIDIA의 [GR1-100](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1)에서 가져온 클립). 카메라는 테이블을 내려다보고 있습니다. 실제 6초 클립에서 로봇의 왼손은 접시에서 오렌지를 집어 트레이에 떨어뜨립니다. 모델은 그 클립의 첫 1초만 봅니다. 손이 올라가 있지만 아직 아무것도 움직이지 않은 시점입니다. 그리고 나머지 장면을 만들어 냅니다. 실제로 일어난 일과 모델이 만들어 낸 장면을 나란히 비교해 보겠습니다.

<figure>
  <video
    controls
    playsinline
    preload="metadata"
    style="display: block; width: 100%; height: auto;"
    aria-label="Real footage from the head camera of a GR1-T2 humanoid next to the continuation invented from its first second"
    src="https://huggingface.co/buckets/dacorvo/video-generation/resolve/robotics/gr1-orange/happened-side-by-side.mp4"
  ></video>
  <figcaption style="text-align: center;">
    <em>Real footage from the head camera of a GR1-T2 humanoid, next to the continuation LTX-2.5 invented from its first second.</em>
  </figcaption>
</figure>

같은 장면, 같은 집기 동작입니다. 모델은 스스로 집는 동작을 만들어 내면서도 테이블과 손, 과일을 실제 영상과 일관되게 유지합니다. 더 흥미로운 것은 실제로 일어나지 않은 사건입니다. 같은 1초에서 프롬프트만 바꾸자 사람이 몸을 숙여 오렌지를 함께 집고, 사람의 손과 로봇의 손이 오렌지를 함께 트레이로 옮깁니다. 실제 영상을 먼저 보여 주고 다섯 번의 테이크를 이어 붙였습니다.

<figure>
  <video
    controls
    playsinline
    preload="metadata"
    style="display: block; width: 100%; height: auto;"
    aria-label="The real clip followed by five generated takes of a person helping the robot carry the orange"
    src="https://huggingface.co/buckets/dacorvo/video-generation/resolve/robotics/gr1-orange/human-side-by-side.mp4"
  ></video>
  <figcaption style="text-align: center;">
    <em>The real clip, then five generated takes of a person helping the robot carry the orange.</em>
  </figcaption>
</figure>

각 테이크에서 사람의 동작과 타이밍이 달라지고, 로봇의 팔은 그에 맞춰 움직입니다. 이것이 학습에 유용한 이유입니다. 클립을 한 번 이어 붙인 결과가 아니라, 각각의 영화 파일과 seed로 재현할 수 있는 여러 결과의 분포를 얻을 수 있습니다.

버킷의 [robotics](https://huggingface.co/buckets/dacorvo/video-generation/tree/robotics) 폴더에는 JSON 파일과 함께 다섯 개의 테이크가 들어 있으며, 눈보라 속 자동차의 전방 카메라로 촬영한 두 번째 예시([Boreas](https://www.boreas.utias.utoronto.ca/))도 있습니다. 이 예시에서는 지나가는 트럭이 렌즈에 슬러시를 튀기는 것이 대체 사건입니다. 대신 자동차를 가속하라고 지시했지만 모델은 속도를 유지했습니다. 프롬프트가 영상의 첫 1초에 이미 암시된 내용을 덮어쓸 수는 없기 때문입니다.

이 동영상은 로봇이나 자동차의 명령이 아니라 프롬프트를 따릅니다. 따라서 학습 설정에서는 해당 동작을 조건으로 한 생성이 필요합니다. 이를 위해 한 팀은 자체 로봇의 프레임과 동작을 사용해 [LTX-2.5의 사전 학습 체크포인트](https://huggingface.co/Lightricks/LTX-2.5-Pre-Trained)를 파인튜닝합니다.

페라리는 실린더의 절반만 사용해 트랙을 한 바퀴 돌았습니다.
