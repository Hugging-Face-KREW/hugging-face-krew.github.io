---
layout: post
title: "TRL과 OpenEnv로 수채화를 그리는 코딩 모델 학습하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/train-to-paint-with-code/thumbnail.png
image: assets/images/blog/posts/2026-09-03-train-to-paint-with-code/thumbnail.png
authors:
  - user: sergiopaniego
slug: "train-to-paint-with-code"
source_url: "https://huggingface.co/blog/train-to-paint-with-code"
source_published_date: "2026-09-03"
source_published_at: "2026-09-03T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Training a coding model to paint watercolours with TRL and OpenEnv](https://huggingface.co/blog/train-to-paint-with-code)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/train-to-paint-with-code -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# TRL과 OpenEnv로 수채화를 그리는 코딩 모델 학습하기

![Training a coding model to paint watercolours](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/thumbnail.png)

8월 23일, [Surya
Narreddi](https://x.com/kickingkeys/status/2091570990048276897)이(가) 언어 모델이 그린 수채화의 아름다운 영상을 게시했습니다. 이 모델은 [p5.brush](https://github.com/acamposuribe/p5.brush)을(를) 통해 JavaScript를 작성합니다. [p5.brush](https://github.com/acamposuribe/p5.brush)은(는) "p5.js에 자연스러운 드로잉 도구를 추가하는" 라이브러리입니다. 영상은 빠르게 바이럴되었고, 이 글을 쓰는 시점에 조회 수가 150만 회를 넘었습니다.

영상에는 프로젝트의 더 이르고 범위가 좁은 단계, 즉 안타깝게도 아직 오픈 artifact는 없는 영상 속 전체 구성 대신 클로즈업한 꽃을 학습한 과정을 설명하는 [a blog
post](https://surya.website/rling-qwen-to-paint-with-code)도 함께 올라왔습니다. 그의 사이트에 따르면 전체 기술 보고서가 공개될 예정이니, 그를 팔로우해 두세요. 원래 아이디어는 예술과 디자인 측면에서 출발한 그의 아이디어이며, [his skills are way beyond mine](https://x.com/kickingkeys/status/2094901433149612118). 제가 시도한 것은 엔지니어링 측면에서의 접근으로, 모든 구성 요소를 공개한 채 레시피를 오픈 방식으로 재현하는 것입니다.

> **참고:** Surya 본인이 직접 설명하는 프로젝트의 배경이 궁금하다면 [this
> video of his thesis](https://vimeo.com/1190839818)을(를) 시청하세요.

이 글에서는 [TRL](https://huggingface.co/docs/trl) 및 [OpenEnv](https://github.com/huggingface/OpenEnv)을(를) 사용해 그의 아이디어를 재현해 보겠습니다. 참조 pool 데이터셋, RL 환경, 학습 스크립트와 학습된 모델을 모두 공개합니다.

전체 파이프라인은 처음부터 끝까지 Hugging Face에서 실행됩니다.

* [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)에서 학습
* RL 환경과 scorer 모델은 [Spaces](https://huggingface.co/docs/hub/spaces)(으)로
* pairwise judge는 [Inference Providers](https://huggingface.co/docs/inference-providers)을(를) 통해
* 모든 artifact는 [one collection](https://huggingface.co/collections/HuggingEnvs/paint-with-code-6a955b79d63f67f1631d9be6)에 모아 Hub에 공개

두 Spaces가 올라오면 레시피 실행은 명령어 하나로 끝납니다. [environment](https://huggingface.co/spaces/HuggingEnvs/watercolour-env)과(와) [scorer model](https://huggingface.co/spaces/HuggingEnvs/watercolour-hpsv3)을(를) 복제하고, reward mix를 위한 환경 변수 두 개를 설정한 뒤 실행하세요.

```bash
hf jobs uv run train/watercolour_grpo.py --flavor h200 --timeout 48h --secrets HF_TOKEN -- \
  --env-url https://<you>-watercolour-env.hf.space \
  --model Qwen/Qwen3.5-35B-A3B --lora --all-linear --bf16 --gradient-checkpointing \
  --subject 'a peach hibiscus' --references 4 \
  --top-p 0.95 --top-k 20 \
  --lr 5e-5 --lr-scheduler constant_with_warmup --warmup-steps 5 \
  --scale-rewards none \
  --steps 110 --n-episodes 240 --num-generations 8 \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 \
  --max-completion-length 8192 \
  --run-tag my-run --out <you>/watercolour-grpo --push-to-hub
```


이 글의 나머지 부분에서는 그 과정에 도달하기까지의 이야기와 [every piece is in the
repo](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour)을(를) 다룹니다.

저는 원문 블로그를 단계별로 따라갔고, 꼭 필요한 경우에만 무언가를 변경했습니다. 제 아이디어는 모두 실험에 반영하지 않고 목록에 적어 두었으며, 그 목록은 마지막의 전체 공개 artifact 목록과 함께 "다음에 시도해 볼 것"이라는 섹션이 되었습니다. 그의 글을 이미 읽었다면 문제 설정과 reward 설계는 익숙할 것입니다. 새롭게 추가된 내용은 오픈 구현, 직접 평가한 pool, 학습하고 비교한 세 가지 reward mix이며, [The RL
environment you need to build](#the-rl-environment-you-need-to-build)에서 시작합니다.

<figure class="image text-center">
  <video controls autoplay muted loop playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-runs-evolution.mp4" title="Three runs evolving in parallel"></video>
  <figcaption>Three runs, one per reward mix, evolving in parallel. Each frame shows the median painting of a step. No need to tell them apart yet, the article explains which run is which.</figcaption>
</figure>

## 사람들이 이 작업을 좋아한 이유 {#section-1}

그림은 느슨하고, 불완전하며, 손으로 만든 듯한 느낌을 줍니다. 이미지 모델이 완벽하고 (통계적으로 평균적인) 이미지를 생성하는 시대에 말이죠. 제 생각에는 이 대비가 영상이 바이럴된 큰 이유입니다. 생성형 AI 아트의 초창기, 매체를 탐구하는 것이 목적이었던 시기가 떠올랐습니다. [DeepDream](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/)(2015)은 사람들이 예술로 바꿔 놓은 디버깅 도구였고, [Edmond de
Belamy](https://en.wikipedia.org/wiki/Edmond_de_Belamy)(2018) 같은 작품은 GAN으로 무엇을 할 수 있는지 탐구한 예술가들에게서 나왔으며, [Mario Klingemann](https://quasimondo.com) 같은 예술가들은 그 시절 [dreamy portraits with neural networks](https://artsandculture.google.com/asset/memories-of-passerby-i-mario-klingemann/aAHG7iV3aXme8g)을(를) 만들고 있었습니다.

이 프로젝트는 그 초창기의 작업들과 더 가까운 느낌입니다. Surya는 자신의 논문에서 여기까지 오게 된 과정을 설명합니다. 그는 텍스트-이미지 모델에 프롬프트를 입력하는 것에서 시작했습니다. 프롬프트가 조작할 수 있는 유일한 레버이고, 세부 정보를 늘려 제어력을 높이는 데도 한계가 있습니다. 모델 자체를 학습하면 더 나아갈 수 있습니다. 아이디어의 다른 절반은 매체입니다. 모델은 이미지를 그리는 약 150줄의 JavaScript 프로그램을 작성합니다. 모델의 출력은 코드입니다. 이를 읽고, 수정하고, 다시 실행할 수 있으며, 각 붓질 뒤의 결정이 드러납니다. 스타일은 모델이 *라이브러리의 메서드 10개만* 사용할 수 있도록 제한한 데서 나옵니다. 이에 대해서는 아래에서 더 설명하겠습니다.

같은 시기에 [Anna Ridler](https://annaridler.com/works/myriad-tulips)은(는) 수천 송이의 튤립을 촬영하고, 하나하나 손으로 라벨링했으며, 데이터셋 자체를 작품으로 전시한 뒤 모델을 학습시키기도 했습니다. 이 프로젝트를 만드는 동안 AI agent가 가져온 참고 문헌을 통해 그녀의 작업을 알게 되었고, 이미지를 손으로 선별한 다음 그 이미지들을 기준으로 학습한다는 점에서 이 프로젝트와 매우 비슷해서 마음에 들었습니다.

## 취향에 대한 RL {#section-2}

최근 언어 모델 RL 작업의 대부분은 검증할 수 있는 reward를 사용합니다. 예를 들어 정답이 알려진 수학 문제, 테스트를 통과하는 코드, 또는 실행 비용이 저렴하고 맞거나 틀렸다고 판별할 수 있는 grader가 있습니다. 이 프로젝트는 인간의 선호로부터 보상 모델을 학습하는 과거의 예외적인 방식인 RLHF에 더 가깝습니다.

여기서 reward는 미적 선호입니다. *정답*은 없습니다. 이 프로젝트의 진짜 질문은 취향에 대해 RL을 수행할 수 있는가입니다.

그의 블로그에서 정의하고 제가 구축한 RL 환경에서 구현한 reward는 다음과 같습니다.

| 항목 | 가중치 | 측정 대상 |
| --- | --- | --- |
| `gate` | 0.05 | 스케치가 컴파일되고, 무언가를 그리며, 치팅하지 않는지 |
| `length` | 0.05 | 더 긴 코드 스니펫을 향한 약한 유도 |
| pairwise judge | 0.60 | pool에서 가져온 reference와 비교한 스타일 |
| [HPSv3](https://huggingface.co/MizzenAI/HPSv3) | 0.30 | 렌더링 결과의 미적 선호 |

[HPSv3](https://huggingface.co/MizzenAI/HPSv3)은(는) 공개된 7B preference model입니다. 이미지와 텍스트 설명을 입력하면 사람이 해당 이미지를 얼마나 선호할지에 대한 점수를 반환합니다. 이미지 쌍 사이의 인간 선택으로 이루어진 대규모 데이터셋으로 학습되었으므로, 그 점수는 많은 사람의 취향을 평균낸 것입니다. pairwise judge는 [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)이며, HF Inference Providers를 통해 호출하는 일반 vision model입니다. pairwise judge는 후보 그림을 pool에서 무작위로 선택한 네 개의 reference와 나란히 보고, 무엇을 중점적으로 평가할지(번짐, 반투명한 워시, 부드러운 가장자리)를 설명한 지침을 따릅니다. 각 비교는 두 가지 제시 순서로 모두 수행하며, 점수는 후보가 이긴 비교의 비율입니다. 유일한 기준은 pool이므로, 그 점수는 해당 평가에 반영된 제 취향입니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/reward-diagram.png" alt="The reward pipeline, piece by piece">
  <figcaption>Two of the four terms in the reward function are models. Both are proxies for someone's taste.</figcaption>
</figure>

이것이 Narreddi가 수렴한 가중치입니다. 여기서 취향을 정의하는 것은 pool입니다. 따라서 작업의 초점은 hyperparameter 조정에서 무엇이 아름다운지를 결정하는 집합을 구축하는 일로 이동합니다.

이 reward로 세 번의 run을 학습했습니다. 차이는 두 model judge 사이에서 가중치를 어떻게 나누었는지뿐입니다.

| run | pairwise judge | HPSv3 | 역할 |
| --- | --- | --- | --- |
| `judge-led` | 0.60 | 0.30 | 원래 mix, step 110에서 중단 |
| `hps-led` | 0.30 | 0.60 | 중간 지점, step 110에서 중단 |
| `hps-only` | 0.00 | 0.90 | validation run, step 60에서 중단 |

먼저 `hps-only`부터 시작해 파이프라인이 실제로 학습할 수 있는지 검증했습니다. reward가 상승하고 metric이 정상적이라는 것을 확인한 뒤에는 더 오래 실행할 이유가 없었으므로, 대신 두 개의 장기 run을 시작했습니다. 장기 run이 던지는 질문은 HPSv3의 능력 중 얼마나 많은 부분을 pairwise judge에 넘길 수 있는가입니다. judge의 가중치가 클수록 reward는 모두의 취향이 아니라 *제* 취향을 더 많이 의미하고, 상승하기도 더 어려워집니다. 더 밀어붙이거나 스타일이 평균에서 너무 멀어지면 모델이 아예 멈출 수도 있습니다.

다행히 멈추지는 않았고, pairwise judge를 사용한 두 run 모두 학습되었습니다. 직접 평가한 pool은 적어도 metric과 최종 그림이 보여 주는 범위에서는 policy를 유도할 수 있습니다. 수치는 아래와 같습니다.

> **면책 조항.** frontier model을 사용하면 이미 프롬프트로 수채화를 그리는 JavaScript 코드를 생성할 수 있습니다. 그것이 출발점입니다. 여기서의 작업은 사람의 예술적 선호와 결합된 방식으로 더 작은 모델을 가르치는 것입니다.

## 구축해야 하는 RL 환경 {#section-3}

이 환경은 모델과 reward 사이에 있는 모든 것을 감쌉니다. 모델이 그림을 그리는 데 사용하는 JavaScript 라이브러리, 사용을 제한하는 system prompt, 각 스케치를 렌더링하는 headless Chromium, 그리고 치팅을 거부하는 gate가 여기에 포함됩니다.

이 라이브러리는 보기보다 훨씬 많은 일을 합니다.
[p5.brush](https://github.com/acamposuribe/p5.brush)은(는) [@acamposuribe](https://x.com/acamposuribe)이(가) 만든 것으로, 단순히 도형을 그리는 대신 매체를 시뮬레이션합니다. 안료는 채운 영역의 가장자리 밖으로 번지고, 종이에는 질감이 있으며, 획에는 질량이 있고, flow field는 붓질을 끌어 움직입니다. 모델이 `brush.fillBleed(0.25)`을(를) 호출할 때는 잉크가 얼마나 멀리 번질지를 결정하는 것입니다.

> **참고.** p5.brush의 저자는 이 모든 일이 있기 훨씬 전부터 기계에게 그림을 그리도록 가르치려 했습니다. 2022년에는 p5.js가 아이처럼 그리도록 가르치는 과정을 담은 일기를 숨긴 generative art 시리즈를 만들었습니다. *"크레용을 간신히 사용할 수 있을 뿐이다 [...] 간단한 명령도 따르지 못한다. 오늘은 여기까지 하겠다. 정말 짜증 난다."* 이 시리즈는 세 작품으로 구성될 예정이었지만 두 작품만 만들어졌습니다. Surya의 영상이 바이럴되자 [he
> quoted it](https://x.com/acamposuribe/status/2091668313449316651)은(는) 그 일기를 공유하며 이 작업이 저절로 도착한 세 번째 작품이라고 말했습니다.

p5.brush는 47개의 메서드를 노출합니다. 프롬프트에서는 `scaleBrushes`, `noStroke`, `fill`, `noFill`, `fillBleed`, `fillTexture`, `beginShape`, `vertex`, `endShape` 및 `circle`의 10개만 허용합니다. 나머지 37개가 추가하는 선, 해칭, custom brush는 수채화 느낌을 깨뜨립니다. 이 10개만으로 모델은 채워진 도형만 그릴 수 있고, 라이브러리가 각각에 번짐 효과를 추가합니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/sketch-and-render.png" alt="A generated sketch and the painting it produces">
  <figcaption>Part of one rollout's <code>draw()</code> and what it renders to. The comments are the model's own. Reward 0.864, 129 lines, step 22. The full source of every painting is in the rollouts dataset.</figcaption>
</figure>

그의 블로그 글 덕분에 프롬프트를 반복해서 수정하느라 낭비했을 시간을 크게 줄일 수 있었습니다. 긴 API reference를 제공하면 모델은 존재하지 않는 메서드를 만들어 내고, 그의 200회 GEPA iteration은 documentation 없이 엄격한 allowlist로 수렴했습니다. 저도 같은 실패를 확인했고 allowlist를 직접 작성했습니다. 이 레시피에 추가한 유일한 내용은 꽃잎 하나를 두세 번 칠하라는 한 문장입니다. 먼저 크게 칠하고, 그 안에 더 작고 불투명한 획을 추가합니다. 이 작은 변경으로 제 출력은 훨씬 더 다채로워졌습니다.

> **참고.** [GEPA](https://huggingface.co/papers/2507.19457)을(를) 처음 들어 본다면, 이는 자동 prompt optimizer입니다.
> 언어 모델이 현재 프롬프트가 실패한 지점을 평이한 말로 성찰하고 더 나은 프롬프트를 제안하면, 이 루프가 반복됩니다.

gate는 마지막 구성 요소입니다. 스케치는 컴파일되어야 하고, 직접 p5 호출 대신 라이브러리를 사용해야 하며, 캔버스에 실제 안료를 칠해야 하고, 예를 들어 캔버스에 텍스트를 쓰는 식으로 scorer를 속이려 해서는 안 됩니다.

## Pool이 reward function이다 {#section-4}

pool은 [178 paintings](https://huggingface.co/datasets/HuggingEnvs/watercolour-reference-pool)으로 구성되며, 제 개인적인 선호에 따라 `love`과(와) `okay`의 두 tier로 나뉩니다. 이 이미지는 모두 실제로 모델이 생성한 것입니다. Inference Providers를 통해 호출한 네 개의 open-weight model이 p5.brush 스케치를 작성했고, 각 모델은 iNaturalist의 실제 공개 라이선스 히비스커스 사진을 바탕으로 작업했습니다. vision model은 각 스케치에 대해 세 번의 refinement iteration 동안 텍스트 feedback을 제공했습니다. 이후 모든 최종 렌더링을 제가 하나씩 평가했고, 178개가 선정되었습니다.

| generator | 그림 수 |
| --- | --- |
| [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 64 |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 57 |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 35 |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 22 |

여기서는 서로 다른 스타일을 테스트하기 위해 네 가지 서로 다른 model family를 선택했습니다. 이 네 가지는 간단한 reliability check에서 매번 유효한 스케치를 생성한 open model이며, 다른 두 후보는 이를 통과하지 못해 제외했습니다. 직접 pool을 만들고 싶다면 다른 모델을 선택해도 됩니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/love-and-okay.png" alt="A love reference beside an okay one">
  <figcaption>The two tiers, as they actually look. Disagreeing with the rating is reasonable, somebody's judgement is now the reward function.</figcaption>
</figure>

tier는 reward에서 실제로 중요한 역할을 합니다. pairwise judge가 네 개의 reference를 선택할 때 절반은 `love`, 나머지 절반은 `okay`에서 가져옵니다. 따라서 policy는 때때로 이길 수 있는 rival과 항상 맞붙게 되고, 어느 tier를 상대로 이겨도 같은 보상을 받습니다. 이것은 제가 의도적으로 변경한 몇 안 되는 부분 중 하나입니다. 원문에서는 최상위 tier하고만 비교하지만, 초기의 약한 policy도 signal을 받을 수 있도록 더 쉬운 tier를 추첨에 포함했습니다.

사람이 만든 그림은 하나도 포함되어 있지 않으며, 이는 실질적인 한계입니다. p5.brush는 틈새 라이브러리이고, 접근 가능한 코드와 함께 존재하는 인간의 작업도 몇 작품에 불과합니다. 그의 블로그에서도 언급했듯이 training corpus에 필요한 규모와는 비교할 수 없습니다.

앞서 이미 이야기했듯이 여기서 흥미로운 아이디어는 모델이 pool에 포함된 내용을 모방하도록 학습한다는 것입니다. 환경이 다른 데이터셋을 가리키도록 하면 코드 한 줄도 건드리지 않고 reward가 자동으로 바뀝니다. 제가 생성해 공개한 데이터셋에는 source sketch도 함께 포함되어 있습니다.

두 judge를 자세히 살펴보면 서로 다른 질문에 답한다는 것을 알 수 있습니다. HPSv3는 이것이 꽃인지 판단하고, pairwise judge는 제가 선택한 스타일로 잘 그려졌는지를 판단합니다.

## YOLO run 한 번만 더 {#section-5}

무언가가 작동하기 전에는 reward curve가 평평하게 이어지는 긴 시간이 있었습니다. 공개 artifact 없이 research paper나 blog를 재현해 보았다면 공감할 것입니다. 각 run에서는 무엇이 문제인지에 대해 제가 합리적이라고 생각한 가설을 시험했습니다. 한 번의 run에는 오랜 시간이 걸리므로, 이전 run의 결과를 살펴보는 동안에도 다음 run을 대기시켜 두었습니다. 늘 그렇듯, 먼저 더 쉬운 문제에서 작동하는 것을 시작한 다음 그 위에 쌓아 올리는 것이 답이었습니다. browser도 judge도 없는 단순한 control task가 처음으로 학습된 시도였고, 이유는 제 learning rate가 그저 너무 낮았기 때문입니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/four-curves.png" alt="Three reward experiments flat against the run that worked">
  <figcaption>Three experiments on the reward, three flat lines. I swapped the pool, removed renderer noise and turned the pairwise judge off, and none of it moved the curve. The run that moves changed the trainer configuration, so the difference is the trainer, not the reward mix.</figcaption>
</figure>

시간을 들여 찾아낸 또 다른 변경 사항은 LoRA parameter를 올바르게 조정하는 일이었습니다. 일반적인 `target_modules` 목록은 dense model을 가정하지만, [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)은(는) 대부분의 projection을 다른 이름으로 지정하는 mixture of experts이므로 adapter가 40개 layer 중 10개만 학습하고 있었습니다. 이를 `all-linear`(으)로 변경해 모든 linear layer에 도달하도록 하여 해결했습니다. 이 architecture의 routed expert는 `all-linear`조차 frozen 상태로 두는 fused tensor지만, 나머지에는 모두 adapter가 적용되었고, 그것만으로도 학습하기에 충분했습니다.

수정 사항은 TRL의 [`GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer)에서 네 가지였습니다.

| 설정 | 이전 | 변경 후 | 이유 |
| --- | --- | --- | --- |
| learning rate | 2e-5 | **5e-5** | GRPO에 대해 *LoRA Without Regret*이 사용하는 상한 |
| scheduler | `linear` | **`constant_with_warmup`** | linear decay가 run 중반까지 learning rate 대부분을 소진해 reward가 상승하지 않았음 |
| `scale_rewards` | `group` | **`none`** | gate rejection 하나가 group 내 다른 모든 advantage를 줄이고 있었음 |
| `target_modules` | 수동 목록 | **`all-linear`** | 모든 linear layer에 도달 |

이 네 가지 변경으로 reward가 뚜렷하게 개선되는 첫 성공적인 run인 `hps-only`을(를) 실행할 수 있었습니다.

이 설정에서는 세 run 모두 학습되었습니다. 두 judge run은 모두 200 step으로 시작했지만 110에서 중단했으며, reward는 여전히 천천히 상승하고 있었습니다. 한 step에 15~18분이 걸리고 mix 간 비교가 이미 안정적이었으므로, compute를 절약하기 위해 둘 다 중단했습니다. 각 run의 첫 1/3과 마지막 1/3에서의 평균 group reward는 다음과 같습니다.

| run | steps | 첫 1/3 | 마지막 1/3 | Δ |
| --- | --- | --- | --- | --- |
| `hps-only` | 60 | 0.58 | 0.71 | +0.13 |
| `judge-led` | 110 | 0.45 | 0.72 | +0.27 |
| `hps-led` | 110 | 0.57 | 0.82 | +0.24 |

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-mixes.png" alt="The three runs, one curve each">
</figure>

세 curve는 제 취향이 얼마나 큰 가중치를 차지하는지와 일치합니다. judge의 가중치가 클수록 시작점은 낮고 상승은 더 noisy합니다. `judge-led`은(는) 처음 30 step 동안 거의 평평하게 머문 뒤 움직였습니다. debugging을 해결한 것과 같은 방식입니다. 무언가가 학습될 때까지 문제를 축소한 다음, 어려운 부분을 한 번에 하나씩 다시 추가하는 것입니다.

pairwise judge 항 자체도 이를 사용한 두 run에서 상승했습니다. 학습이 진행될수록 모델은 pool을 상대로 더 많은 비교에서 이겼으며, 이는 `hps-only`이(가) 주장할 수 없었던 내용입니다. 어느 run에서도 어떤 group도 동일한 reward로 붕괴하지 않았습니다. 이는 gradient를 없애는 GRPO의 failure mode입니다. 궁금한 분들을 위해 metric별 curve(HPSv3, paint coverage, entropy)는 [in the
repository as CSV](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour/results)에 있습니다.

전체 launch command, hardware, 그리고 이것을 다른 두 run으로 바꾸는 두 환경 변수는 [the
recipe](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour)에 있습니다.

## 실제로 무엇을 학습했는가 {#section-6}

**모든 run에서 모델이 가장 먼저 학습한 것은 나쁜 그림을 그리지 않는 것이었습니다.** 즉 total reward가 0.3 미만인 거의 빈 캔버스와 형태 없는 wash를 중단하는 것입니다. `hps-only`에서는 group mean 상승의 4분의 3이 나쁜 그림이 드물어지는 데서 비롯됩니다. judge run에서는 붕괴가 더욱 가파릅니다. `judge-led`의 1/3 구간 비교에서 0.3 미만 rollout은 99개에서 16개로, `hps-led`에서는 37개에서 4개로 감소했습니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/median-vs-best.png" alt="Median against best, per step">
</figure>

이것이 각 step의 가장 좋은 그림이라는 뻔한 시각적 결과가 `hps-only`에서 거의 차이를 보이지 않는 이유입니다. run 전체에서 **+0.034**만큼 움직인 반면 median은 **+0.155**만큼 움직였습니다. 학습은 분포의 중간에서 드러납니다.

**pairwise judge가 바꾸는 것은 상위 부분입니다.** `hps-only`에서는 그림이 더 안정적이 되었을 뿐 더 좋아지지는 않았습니다. 좋은 그림의 품질은 group mean에 +0.03만 더했고, HPSv3가 중심과 줄기를 둘러싼 꽃잎을 확인한 뒤에는 더 많은 안료를 요구하지 않았습니다. judge를 켜면 이야기의 나머지 절반이 나타납니다. 여기서 *더 좋다*는 것은 pool에 더 가까워진다는 뜻이며, 제가 "좋다"고 평가했거나 더 마음에 들어 한 것에 가까워진다는 뜻입니다. 그 결과 `judge-led`에서는 +0.12, `hps-led`에서는 +0.16이 추가되었습니다. 각 step의 최고 그림도 상승했고, 두 run에서 paint coverage가 모두 두 배가 되었습니다(0.11에서 0.23으로, 0.13에서 0.30으로). 반면 `hps-only`에서는 거의 변하지 않았습니다. 이길 reference가 있으면 좋은 그림도 계속 더 좋아질 수 있고, 모델은 더 많은 안료를 사용하는 데 대한 보상을 받기 시작합니다.

한 가지 더 발견한 점이 있습니다. 모델은 명시적인 지시를 무시하며, 그렇게 하는 것이 옳습니다. system prompt는 15~30개의 채워진 도형을 요구합니다. 실제 mean을 보면 7~9개 사이이고, `n_shapes`은(는) 어떤 run에서도 reward와 거의 상관이 없습니다(+0.000, −0.14, +0.07). policy는 해당 문장을 따르는 데 대해 보상을 받지 않으므로, 그 문장을 따르지 않습니다.

`hps-only` 경로에는 상한도 있습니다. 모든 rollout이 좋은 rollout과 같아진다면 해당 run의 mean은 0.771이 됩니다. 더 많은 step이 이 한계를 깨뜨릴지는 아직 열린 질문입니다.

그림은 표에서 놓치는 부분도 보여 줍니다. 각 run 내부의 그림은 모두 비슷해 보입니다. 학습이 진행될수록 각 group 안의 reward가 서로 가까워지고, 도입 영상의 median 그림은 같은 꽃을 여러 번 그린 것처럼 보입니다. 이는 한 가지 대상을 바탕으로 만든 pool에서 GRPO가 설계된 대로 작동한 결과입니다. 무엇을 다양성으로 볼지 결정하는 것도, 무엇을 품질로 볼지 결정하는 것도 pool입니다. reward가 꽃 하나를 닮는 데만 보상을 준다면 모델은 그 꽃 하나를 그리는 법을 학습합니다. 더 다양한 출력에는 더 다양한 pool이 필요하고, 이를 만드는 데는 더 많은 curation 작업이 필요합니다. Surya의 새로운 구성은 이를 보여 주는 예입니다. [Alex Yango's animal
paintings](https://x.com/alexyango/status/2091696296931574217)은(는) pool에서 다른 선택을 한 동일한 레시피입니다. 이것이 미적 reward와 수학 grader의 가장 큰 차이입니다. 숫자 뒤에는 reward set에 무엇을 포함할지 결정하는 매우 인간적인 작업이 있습니다. [Jason Liu's essay on
taste](https://x.com/jxnlco/status/2073819508729684462)은(는) 이를 한 줄로 일반화해 말합니다. AI는 병목을 만드는 일에서 알아차리는 일로 옮겼습니다.

Surya는 자신이 가장 좋아하는 작품 몇 점으로 블로그를 마무리합니다. 제 작품을 고르는 대신, 아래에는 두 judge run에서 reward 점수가 가장 높았던 178개의 그림을 특별한 순서 없이 벽처럼 펼쳐 놓았습니다. reference pool과 같은 수입니다. 열어 보고 직접 골라 보세요.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/pick-your-favourites.png" alt="178 paintings from the two judge runs, unlabelled">
  <figcaption>The reward's 178 favourites from the two judge runs, shuffled. Now pick yours.</figcaption>
</figure>

고르기 위해 많은 그림을 보고 몇 개를 남겼을 텐데, 이것이 바로 이 프로젝트의 reward를 만든 작업입니다. 모든 run의 모든 그림과 sketch, reward는 rollout 데이터셋에 있으며 [this gallery](https://huggingface.co/spaces/HuggingEnvs/watercolour-gallery)에서 둘러볼 수 있습니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/three-styles.png" alt="The median painting of the last step of each run">
  <figcaption>The last step's median painting of each run, in the same order as the opening video. Same base model, same pool, three reward mixes, three styles.</figcaption>
</figure>

reward가 부분적으로 제 취향에 기반했으므로, 보는 사람으로서 제 판단으로 마무리하는 것이 공정할 것 같습니다. 제 눈에는 `judge-led`이(가) 가장 다양하고 예술적으로 흥미롭게 끝나는 run입니다. `hps-led`은(는) 설득력 있는 수채화를 그리지만, 최고의 그림들도 부드러운 wet-on-wet 느낌을 공유하며 거의 독자적인 스타일처럼 보입니다. `hps-only`은(는) 가장 강하게 수렴하고, 대부분의 그림이 같은 색에 안착합니다. 모든 run의 모든 그림을 step과 reward별로 정렬할 수 있는 [the
gallery](https://huggingface.co/spaces/HuggingEnvs/watercolour-gallery)에서 직접 판단해 보세요.

## Infra는 어렵다 {#section-7}

이 프로젝트는 대부분 infra 작업입니다. 하나의 run을 위해서는 trainer, 두 개의 Spaces, inference router와 websocket이 여러 시간 동안 계속 정상 상태를 유지해야 합니다. 어느 한 구성 요소라도 조용히 실패하면 다른 곳에서 잘못된 숫자로 나타납니다. 작업의 절반은 읽은 숫자가 실제로 일어난 일을 반영하는지 확인하는 것입니다.

**Infra의 실패가 reward에 0으로 들어가고 있었습니다.** 렌더링이 timeout되거나 scorer가 응답하지 않으면 나쁜 그림과 똑같이 group 안에서 0.0을 받았습니다. 제 모든 run에서 이는 rollout의 약 1.5%였고, 최악의 run에서는 5.2%에 달했습니다. 이렇게 하면 모델이 noise를 학습하므로, 이제 해당 경로는 `None`을(를) 반환하고 rollout은 group에서 제외됩니다.

**OpenEnv에서 bug도 발견해 upstream에 fix를 보냈습니다.** client는 persistent websocket 하나를 유지하는데, 원격에서 닫힌 socket이 cache에 남아 이후의 모든 호출이 실패했습니다. 환경 자체는 정상인데도 말입니다. 이를 찾는 데 반쯤 끝난 run 두 개를 소모했습니다. fix는 [submitted
upstream](https://github.com/huggingface/OpenEnv/pull/1103)이며, 이를 사용해 시작한 run은 이후 깨끗하게 실행되고 있습니다.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/step-11-vs-step-12.png" alt="Steps 11 and 12 of the judge-led run, best four of each">
  <figcaption>Two consecutive steps of the <code>judge-led</code> run, best four of each. Whether step 12's paintings look half a point worse is for you to decide.</figcaption>
</figure>

**한 step의 reward는 어떤 reference를 뽑았는지에 따라 달라집니다.** pairwise judge는 step마다 네 개의 reference를 sample하므로 각 step은 서로 다른 rival 집합을 상대하고, 어떤 draw는 단순히 더 어렵습니다. GRPO 자체는 group 내부에서 advantage를 계산하므로 대부분 안전하며, 어려운 draw가 group 전체를 함께 움직입니다. 제가 읽고 있던 curve는 안전하지 않았고, 나쁜 step처럼 보였던 것 중 일부는 그저 어려운 draw였습니다. 위 이미지는 한 예입니다. Step 12가 step 11보다 0.5점 낮았던 주된 이유는 해당 run에서 가장 어려운 reference를 뽑았기 때문이며, 그림 자체는 비슷해 보입니다.

## 비용은 얼마나 드는가 {#section-8}

반올림한 수치이며, 완료된 run에 대해서만 계산했습니다.

| 구성 요소 | 필요한 것 |
| --- | --- |
| trainer | H200 1개. 60 step에 **18시간**, 110 step에 약 **34시간** |
| HPSv3 | 전체 run 동안 켜 두는 `a100-large` Space |
| environment | 충분한 시간 안에 렌더링하는 `cpu-upgrade` Space |
| pairwise judge | `Qwen/Qwen3-VL-30B-A3B-Instruct`에 필요한 Inference Providers quota |
| pool, 일회성 | iNaturalist의 공개 라이선스 사진, 네 generator를 위한 Inference Providers quota, 그리고 직접 평가하는 시간 |

한 step은 8개의 rollout으로 구성되며 15~18분이 걸립니다. 그중 **70~80%가 렌더링**입니다. 단일 렌더링은 90초 deadline에 대해 69~96초가 걸립니다. 일부는 예상된 결과입니다. Space에 GPU가 없으므로 Chromium이 WEBGL canvas를 software로 렌더링하고, p5.brush의 번짐과 질감은 무거운 pixel 작업이기 때문입니다. 그래도 더 빠를 것으로 예상했으며, 전체 원인은 아직 찾지 못했습니다.

scorer를 유지하는 비용이 이를 사용하는 학습보다 더 클 수 있습니다. HPSv3는 전체 run 동안 켜져 있어야 하므로, run이 끝나면 Space를 pause하거나 sleep timer를 설정하세요.

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train-to-paint-with-code/infra-diagram.png" alt="The infrastructure: what is billed during a run, and what outlives it">
  <figcaption>Four paid services have to stay healthy at once. Only the Hub and trackio outlive the run.</figcaption>
</figure>

모든 것은 [HF
Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs)에서 실행되며, environment는 [Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker), metric은 [trackio](https://huggingface.co/docs/trackio)에 있습니다.

## 다음에 시도해 볼 것 {#section-9}

이 프로젝트의 원칙은 모든 resource를 공개한 채 레시피를 재현하는 것이었지, 이를 개선하는 것이 아니었습니다. 그래서 시도하지 않은 아이디어가 진행 중에 계속 쌓였습니다. 다음은 근거가 얼마나 많은지를 기준으로 실제로 시도해 보고 싶은 것들입니다.

**Multi-step, 그리고 모델이 자신이 그린 것을 보도록 하기.** 이것이 제가 가장 먼저 시도할 일입니다. 원문 블로그는 single turn으로 학습하므로 저도 single turn으로 학습했지만, 이 설정에서 모델은 눈을 감고 그림을 그립니다. 이미지가 입력되는 일은 없고, 받는 feedback도 숫자 하나뿐입니다. feedback loop가 작동한다는 증거는 pool 자체입니다. reference 그림은 vision critic 아래에서 세 라운드를 반복한 모델로부터 나왔고, 뒤의 라운드가 더 좋았습니다. reward를 정의하는 자료는 policy가 사용할 수 없는 loop로 만들어졌습니다.

**더 작은 모델.** 35B가 필요한 크기보다 크다는 증거가 있습니다. 별도의 실험에서 4B도 gate를 통과하는 유효한 sketch를 이미 작성했습니다. 4B가 이를 학습할 수 있다면 실험 비용은 한 자릿수 배만큼 줄어듭니다.

목록에 있는 다른 아이디어로는 RL을 시작하기 전에 pool source에 SFT를 적용하는 것, 안료에 명시적으로 보상하는 것, run이 진행될수록 judge의 reference mix를 easy에서 hard로 옮겨 `love`만 남기는 것, 더 넓은 시각적 범위를 위해 10개 method allowlist를 확장하는 것(제가 시도한 결과는 더 많은 sketch를 망가뜨렸고 수채화 느낌도 깨졌습니다), 그리고 같은 이미지를 두 번 채점해 pairwise judge가 실제로 얼마나 일관적인지 확인하는 것이 있습니다.

그리고 이 방법은 꽃에만 한정되지 않습니다. [Alex Yango painted animals with the same
mechanism](https://x.com/alexyango/status/2091696296931574217), 그리고 [Brendan Hogan
trained canvas animations](https://x.com/brendanh0gan/status/2092650655789855222)은(는) 직접 평가한 clip pool을 대상으로 할 수 있습니다. 이전에도 [Simon
Willison's pelican
benchmark](https://huggingface.co/blog/sergiopaniego/pelican-env-openenv)을(를) 사용해 비슷한 작업을 해 본 적이 있습니다. 이 경우 code를 image로 렌더링하고 점수를 매깁니다.

그리고 그 모든 것 아래에는 이 프로젝트가 답을 내리지 못하는 질문이 놓여 있습니다. **모델이 만든 178개의 그림이 이 학습된 모델이 아름답다고 여기는 것을 정의합니다.** 병목은 pool이며, pipeline에서 원칙적인 답이 없는 부분도 바로 pool입니다.

## 원문과 달라진 점 {#section-10}

이를 재현하려는 분들을 위해, Narreddi의 레시피와 의도적으로 달라진 두 가지를 소개합니다. 둘 다 이 글 앞부분에서 설명했습니다.

- pairwise judge는 reference의 절반을 `love`에서, 나머지 절반을 `okay`에서 가져옵니다. 최상위 tier하고만 비교하는 대신, 초기의 약한 policy도 signal을 받을 수 있도록 했습니다.
- 프롬프트에 craft에 관한 짧은 문장 하나를 추가했습니다. 꽃잎 하나를 두세 번 칠하되, 먼저 크게 칠하고 그 안에 더 작고 불투명한 획을 추가합니다.

나머지, 즉 LoRA의 `all-linear`, infra failure가 0.0 대신 `None`을(를) 반환하도록 한 것, judge run을 step 110에서 중단한 것은 그의 블로그에 명시되어 있지 않아 진행 중 제가 결정해야 했던 사항입니다. 그의 구현은 공개되지 않았으므로 이것이 그의 선택과 일치하는지, 아니면 다른지 알 수 없습니다.

## 모든 것을 공개했습니다 {#section-11}

| artifact | 위치 |
| --- | --- |
| 레시피와 재현 방법 | [`02-watercolour/`](https://github.com/adithya-s-k/HuggingEnvs/tree/main/02-watercolour) |
| 모든 source sketch가 포함된 reference pool | [`watercolour-reference-pool`](https://huggingface.co/datasets/HuggingEnvs/watercolour-reference-pool) |
| 복제 가능한 environment | [`watercolour-env`](https://huggingface.co/spaces/HuggingEnvs/watercolour-env) |
| 복제 가능한 HPSv3 scorer | [`watercolour-hpsv3`](https://huggingface.co/spaces/HuggingEnvs/watercolour-hpsv3) |
| hps-only adapter와 rollout | [`watercolour-grpo-hps-only`](https://huggingface.co/HuggingEnvs/watercolour-grpo-hps-only) · [`watercolour-rollouts-hps-only`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-hps-only) |
| judge-led adapter와 rollout | [`watercolour-grpo-judge-led`](https://huggingface.co/HuggingEnvs/watercolour-grpo-judge-led) · [`watercolour-rollouts-judge-led`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-judge-led) |
| hps-led adapter와 rollout | [`watercolour-grpo-hps-led`](https://huggingface.co/HuggingEnvs/watercolour-grpo-hps-led) · [`watercolour-rollouts-hps-led`](https://huggingface.co/datasets/HuggingEnvs/watercolour-rollouts-hps-led) |
| 모든 그림을 둘러볼 수 있는 gallery | [`watercolour-gallery`](https://huggingface.co/spaces/HuggingEnvs/watercolour-gallery) |
| training curve | live: [`judge-led`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-judge-led) · [`hps-led`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-hps-led) · [`hps-only`](https://huggingface.co/spaces/HuggingEnvs/watercolour-trackio-hps-only), CSV 파일은 `results/` |
| 전체 | [Paint with Code](https://huggingface.co/collections/HuggingEnvs/paint-with-code-6a955b79d63f67f1631d9be6) |

이 글의 rollout별 수치는 공개된 데이터셋으로 재계산할 수 있습니다. 여기의 어떤 내용도 Space가 계속 켜져 있어야 한다는 전제에 의존하지 않습니다.

방법과 원래 아이디어는 Surya Narreddi의 것입니다. 라이브러리는 Alejandro Campos Uribe의 것입니다.
