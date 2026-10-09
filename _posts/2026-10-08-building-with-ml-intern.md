---
layout: post
title: "존재하지 않던 모델을 직접 만들다"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/building-with-ml-intern/thumbnail.png
image: assets/images/blog/posts/2026-10-08-building-with-ml-intern/thumbnail.png
authors:
  - user: ysharma
  - user: abidlabs
slug: "building-with-ml-intern"
source_url: "https://huggingface.co/blog/building-with-ml-intern"
source_published_date: "2026-10-08"
source_published_at: "2026-10-08T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [The model that didn't exist, so you made it yourself](https://huggingface.co/blog/building-with-ml-intern)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/building-with-ml-intern -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 존재하지 않던 모델을 직접 만들다

<p align="center">
<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/ml_intern_intro_noaudio_cropped.mp4" width="400" ></video>
</p>

지난주, Qwen-Image 2.1에 포함된 [prompt rewriter](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)의 작은 버전을 원했다. 공식 모델은 9B 모델로, 약 20 GB의 메모리가 필요하고 단락 하나를 쓰기 전에 수천 개의 토큰을 사용해 사고한다. Hub에서 찾을 수 있었던 것은 같은 9B 모델의 압축 버전뿐이었다. 그래서 원하는 것을 [ML Intern](https://huggingface.co/chat/?mode=ml-intern)에 설명했고, 다음 날에는 CPU에서 실행되는 [0.8B version](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B)를 갖게 되었다. 이 모델은 99.7%의 확률로 유효한 출력을 반환하며 teacher 모델 토큰의 약 4분의 1을 사용한다. 9B 모델이 예시 요청 8,797개에 라벨을 지정한 작업을 포함해 프로젝트 전체에 사용한 컴퓨팅 비용은 USD 16이었다.

그 후 며칠 동안 같은 방식으로 모델을 다섯 개 더 만들었다. 각각은 ML-intern을 켠 HuggingChat의 메시지로 시작했고, 모델 카드에 평가가 포함된 공개 모델로 Hub에 게시되며 끝났다. ML-intern은 작업을 계획하고, 비용을 사용하기 전에 예산을 요청하며, 실제 작업 전에 소규모 테스트를 실행한 다음 Hugging Face 하드웨어에서 학습하고 평가한 뒤 게시한다.

## ML Intern에 프롬프트하는 방법 {#section-1}

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/prompt_example.mp4"></video>

내가 가장 많은 노력을 들이는 부분은 첫 번째 메시지다. 아래에서 공유한 *citrus* 모델을 위한 첫 프롬프트는 약 450단어였다. 6번째 프로젝트에 이르러서는 다음 프로젝트에 반영하고 싶은 것을 각 프로젝트에서 배웠기 때문에 2,000단어에 가까워졌다. 일곱 개의 프롬프트는 모두 내가 작성한 그대로 [yvrjsharma/ml-intern-prompts](https://github.com/yvrjsharma/ml-intern-prompts)의 GitHub에 올라와 있다.

프롬프트는 아이디어를 한 줄로 설명하고 내가 그것을 원하는 이유를 밝히는 것으로 시작한다. 그런 다음 데이터셋, base model, training script 등 정확한 구성 요소를 지정한다. 이미 확인한 내용은 무엇이든 "Verified facts, do not re-derive"라고 적힌 제목 아래에 둔다. 그러면 에이전트가 내가 이미 알고 있는 사실을 다시 발견하는 대신 예산을 실제 작업에 사용할 수 있다. 카메라 앵글 LoRA에서는 이 섹션에 투명 이미지 지원을 막 추가한 trainer와, 대체 trainer를 위험하게 만드는 공개 GitHub 이슈를 명시했다.

**프롬프트에서 중요한 두 줄**이 있다. **첫 번째**는 학습 전에 baseline을 요청하는 내용이다. 예를 들어 _citrus prompt_에는 다음과 같이 적혀 있다. "또한 학습 전에 동일한 metric으로 base model의 zero-shot score를 보고하여 성능 향상을 확인할 수 있게 해 주세요." 이 내용이 없으면 학습된 모델은 얻을 수 있지만, 시작점보다 나아졌는지 알 수 없다. **두 번째**는 검사 항목이 포함된 smoke test다. 이미지 LoRA에서는 50 training steps를 실행한 다음, 전체 작업 비용을 지불하기 전에 저장된 weights가 실제로 변경되었는지 확인하도록 요청했다.

프롬프트의 마지막에는 예상 deliverables를 정리하고 비용을 제한한다. 모델 카드에 무엇을 포함할지 정의하고 다음과 같은 지시를 넣는다. "총 지출을 USD 12로 제한하고, 이를 초과하기 전에 나에게 물어보세요." ML-intern은 모든 작업을 0달러 예산으로 시작하고 비용이 발생하는 작업을 실행하기 전에 승인을 받아야 하므로 이 지출 한도는 엄격하게 적용된다. 예산을 지정하지 않으면 에이전트는 프로젝트 규모에 따라 몇 가지 경로를 제안하고 어느 경로를 선호하는지 묻는다.

첫 시도부터 이 모든 내용을 반드시 포함할 필요는 없다. 예를 들어 _citrus_ brief에는 *verified-facts* 섹션이 없었지만, ML Intern은 [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 모델보다 [tripled the accuracy](https://huggingface.co/ML-Intern-lab/citrus-disease-vlm#results) 이상인 모델을 그래도 만들어냈다. 며칠 만에 Ml-Intern과 함께 만든 6가지 결과를 소개하겠다.

## 1. 여러분의 분야를 아는 모델 {#section-2}

일반 vision model은 누렇게 변한 감귤 잎을 설명할 수 있다. 하지만 그것이 응애 문제인지 마그네슘 결핍인지, 그리고 식물을 치료하기 위한 유기·비유기 처방이 무엇인지 알려주는 일은 매우 어렵다. Claude를 사용해 Hub의 [Project-AgML](https://huggingface.co/Project-AgML) 조직이 호스팅하는 세 가지 소스에서 가져온 training dataset을 하나로 합쳤다. 그 결과 만들어진 [citrus-disease-vlm-instruct](https://huggingface.co/datasets/ML-Intern-lab/citrus-disease-vlm-instruct)는 21개의 서로 다른 해충, 질병, 영양 결핍 및 치료 방법에 걸쳐 주석이 달린 이미지 3,017개를 포함하는 데이터셋이다. ML-intern은 이 예시를 사용해 Qwen3.5-2B의 fine-tuning을 수행했으며, 그 전에 foundation model의 benchmark를 반드시 측정했다.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/citrus-doctor.mp4"></video>

335장의 테스트 사진에서 base model이 올바른 문제를 식별한 비율은 14.9%였다. 하나의 A10G에서 두 epoch를 학습한 뒤 fine-tuned model은 52.8%를 기록했다. 컴퓨팅 비용은 약 USD 1.90이었다.

살펴보기: [Model](https://huggingface.co/ML-Intern-lab/citrus-disease-vlm) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/citrus-disease-vlm-instruct) · [Citrus Doctor App](https://huggingface.co/spaces/ML-Intern-lab/citrus-doctor)

## 2. 여러분의 캐릭터를 그리는 모델 {#section-3}

이미지 모델은 수많은 캐릭터를 알고 있다. Hugging Face 브랜드 asset의 평면적인 스타일로 그린 Huggy는 그중 하나가 아니었다. ML-Intern에 [FLUX.2 klein base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)용 LoRA를 요청했고, [Chunte/huggy_for_training](https://huggingface.co/datasets/Chunte/huggy_for_training) 데이터셋에서 가져온 캡션이 있는 그림 84개로 학습했다.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/huggy-model-samples.mp4"></video>

에이전트는 100 steps마다 checkpoint를 저장하고 각 checkpoint로 동일한 prompt 세트를 그렸기 때문에 쉽게 선택할 수 있었다. Step 200에서 Huggy가 완전히 *on-model*이 된 것이 처음으로 확인되었다. Step 500부터는 Huggy와 아무 관련이 없는 prompt에도 Huggy의 스타일이 번지기 시작했다! 학습된 LoRA는 4 steps의 distilled klein model에서도 작동한다. 컴퓨팅 비용은 약 USD 7.60이었다.

살펴보기: [Model](https://huggingface.co/ML-Intern-lab/huggy-flux2-klein-lora) · [Dataset](https://huggingface.co/datasets/Chunte/huggy_for_training) · [Huggy Generator App](https://huggingface.co/spaces/ML-Intern-lab/huggy-generator)

## 3. 새로운 기술을 수행하는 모델 {#section-4}

1. **카메라 앵글 LoRA**는 이전 Qwen-Image 모델에서 커뮤니티가 가장 많이 좋아요를 누른 add-on 중 하나다. 모델에 물체 사진을 보여주고 왼쪽에서 45도 떨어진 각도에서 보도록 요청할 수 있다. Qwen-Image 2.1 모델이 출시된 지 며칠 후 확인했을 때 아무도 이를 만들지 않았기에 ML-intern에 제작을 맡겼다.

<p align="center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/viewpoint-orbit-lora.gif" alt="Qwen camera angle LoRA" width="400" >
</p>

ML-intern은 [Google Scanned Objects](https://huggingface.co/datasets/suvadityamuk/google-scanned-objects)에서 스캔한 가정용 물체 1,030개를 각각 24개의 각도에서 렌더링했다. 총 24,722개의 투명 이미지가 생성되었으며, CPU 작업에 든 비용은 몇 센트에 불과했다. 이후 학습용 물체 461개와 테스트용으로 제외한 물체 40개를 확정했고, 23개의 카메라 instruction에 걸쳐 before-and-after training pair 1,844개를 균등하게 구성했다.

학습은 하나의 A100에서 약 90분 동안 2,000 steps를 실행했으며 비용은 약 USD 3.75였다. 누락된 package나 잘못된 path 때문에 실패해 ML-Intern이 다시 제출해야 했던 작업까지 포함해, 전체 프로젝트에는 약 반나절과 48개의 작업이 걸렸다. 총 컴퓨팅 비용은 약 USD 16이었다.

살펴보기: [Model](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/gso-orbit-rgba) · [Viewpoint Orbit App](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-viewpoint-orbit-LoRA)

2. **Doodle-in LoRA**도 또 하나의 멋진 아이디어다. 자홍색 낙서가 그려진 사진을 업로드하고 물체 이름을 지정하는 짧은 prompt를 추가한다. LoRA는 원래의 조명과 구도를 일관되게 유지하면서 낙서를 해당 물체로 바꾼다.

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/qwen-image-2.1-doodle-in-lora-sample2.mp4"></video>

이를 위한 데이터셋은 존재하지 않았기 때문에 프롬프트에 데이터셋을 만드는 방법을 설명했다. Open Images의 실제 사진에서 시작해 LaMa inpainting model로 물체 하나를 제거하고, 물체가 있던 자리에 낙서를 그린다. 수정하지 않은 사진이 target이다. ML-intern은 CPU sandbox에서 pair-building script를 작성하고 테스트한 다음 GPU 작업으로 실행했으며, 그 과정에서 모든 원본 사진의 작성자와 license를 기록했다. 학습용 pair 6,042개와 160쌍의 테스트 세트를 만들었고, 테스트 pair 중 40개는 학습에서 완전히 제외된 23개 물체 class에서 가져왔다.

학습 전에 base model을 단독으로 실행했을 때와 Viggle turbo LoRA를 사용했을 때 각각 측정했고, 편집 작업을 batch로 실행해도 동일한 이미지가 생성되는지 확인하여 evaluation 비용을 낮췄다. 학습은 하나의 A100에서 1시간 38분 동안 2,000 steps를 실행했으며 비용은 약 USD 4였다. 저장된 checkpoint를 48개의 테스트 pair에서 비교한 결과 step 500을 선택했다.

[Viggle turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)와 함께 6 steps로 실행했을 때 LoRA는 매우 좋은 성능을 보였다. 물체가 그려진 위치에서 감지된 비율은 [67.5%](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA#main-comparison--160-test-pairs)였으며, 편집 하나에 4.7초가 걸렸다. 보지 못했던 23개 class의 물체도 나머지와 마찬가지로 안정적으로 배치되었다(65.0% 대 64.2%). 프로젝트에는 하루를 조금 넘는 시간과 59개의 작업이 걸렸다. 총 컴퓨팅 비용은 약 USD 24였다.

살펴보기: [Model](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/doodle-in-pairs) · [Doodle-in App](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-doodle-in-LoRA)

## 4. 여러분의 기기에 맞는 모델 {#section-5}

1. 이 글의 앞부분에서 소개한 **Pocket Rewriter**가 첫 번째다. ML-intern은 Inference Providers를 통해 작은 instruct model로 짧은 이미지 요청 8,797개를 생성하는 일부터 시작했다. 사진, 포스터, 로고, 인포그래픽 등을 섞은 데이터 구성을 프롬프트에서 지정했고, 그중 약 3분의 1은 따옴표 안의 정확한 텍스트를 요청했으며 영어 이외의 언어로 작성된 것도 많았다. 그런 다음 9B teacher가 하나의 A100에서 2시간 37분 동안(~USD 6.50) 모든 요청을 다시 작성했다. 품질을 기준으로 필터링한 뒤 1,840개의 예시를 training dataset에 선택했다.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/qwen-image-2.1-pocket-studio.png" alt="Qwen Image 2.1 Pocket Studio">

0.8B와 2B student를 학습하는 데 A10G에서 각각 12분과 18분이 걸렸으며, 두 모델을 합친 비용은 USD 0.75였다. 0.8B 모델은 CPU에서 실행할 수 있도록 812 MB GGUF 파일로도 제공된다. 프로젝트에는 약 11시간과 24개의 작업이 걸렸다. 총 컴퓨팅 비용은 약 USD 16이었다.

살펴보기: [Pocket rewriter 0.8B Student](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B) · [2B Student](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-2B) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/Qwen-Image-2.1-rewriter-distill) · [Pocket Studio App](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-pocket-studio) · [Compare the teacher-student in Rewriter Arena](https://huggingface.co/spaces/ML-Intern-lab/Qwen-Image-2.1-rewriter-arena)

2. **Agate-Preview-002-4step**이 두 번째다. [Logolabs' Agate Preview 002](https://huggingface.co/Logolabs/agate-preview-002)는 260M-parameter text-to-image model로 브라우저에서 실행할 수 있을 만큼 작지만, guidance를 사용하면 50 steps가 필요해 이미지 하나당 network pass가 100회 발생한다. ML-intern에 이를 단 4회의 pass로 distill해 달라고 요청했다!

<video controls autoplay loop muted playsinline src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/building-with-ml-intern/agate-preview-002-4step.mp4"></video>

두 번의 실행이 필요했다. 첫 번째 실행에서는 155,000개의 training image를 latent로 cache하고 guidance를 model에 반영한 다음, 모두 A100에서 step 수를 16에서 8, 다시 4로 단계적으로 줄였다. 4-step student는 GenEval과 FID에서 동일한 4 steps로 실행한 teacher보다 높은 성능을 보였고, 이후 ML-intern은 이를 브라우저용 ONNX로 export했다. 이 실행에는 약 13시간과 USD 22가 들었다.

4-step student를 조금 더 개선해 달라고 ML-Intern에 요청하여 두 번째 training run을 진행했다. ML-Intern은 16 steps의 teacher로 image pair를 24,000개 더 만든 뒤, 약 한 시간 동안 이를 사용해 4-step student를 fine-tune했다. GenEval은 0.509에서 0.536으로 상승했으며, teacher는 50 steps에서 0.563을 기록했다. 이는 compute를 4분의 1만 사용하면서도 단 4 steps로 얻은 결과다. ML-intern은 브라우저 버전을 다시 export했다. 두 번째 실행에는 약 8시간이 걸렸다. 두 실행을 합친 총 컴퓨팅 비용은 약 USD 37이었다.

살펴보기: [Agate 4-step model](https://huggingface.co/ML-Intern-lab/agate-preview-002-4step) · [Dataset](https://huggingface.co/datasets/ML-Intern-lab/agate-preview-002-4step-latents) · [Run Agate in your browser](https://huggingface.co/spaces/ML-Intern-lab/agate-preview-002-4step-webgpu) · [Agate 4-step LIVE](https://huggingface.co/spaces/ML-Intern-lab/agate-preview-002-4step-live)

## 비용 {#section-6}

| 모델 | Base model | 컴퓨팅 비용 |
|---|---|---|
| Citrus Doctor | Qwen3.5-2B | USD 1.90 |
| Huggy LoRA | FLUX.2 klein base 4B | USD 7.60 |
| Pocket rewriter (0.8B and 2B) | Qwen3.5-0.8B and 2B | USD 16.05 |
| Viewpoint Orbit LoRA | Qwen-Image 2.1 | USD 16 |
| Doodle-in LoRA | Qwen-Image 2.1 | USD 24.30 |
| Agate 4-step (two runs) | Agate Preview 002 | USD 37 |
| **합계** | | **약 USD 103** |

각 세션에 대해 보고된 GPU 및 CPU 작업 비용이다.

## 직접 만들어 보세요 {#section-7}

ML-intern은 [HuggingChat](https://huggingface.co/chat/)에 있다. ML-intern 모드를 켜고 프롬프트를 붙여 넣는다. 시작점이 필요하다면 [my example prompts](https://github.com/yvrjsharma/ml-intern-prompts)을 무료로 복사해 사용할 수 있다. 존재했으면 하는 모델과 이미 가지고 있는 데이터셋 또는 설명할 수 있는 데이터셋에서 시작한다. 작은 예산을 지정하고, baseline과 smoke test를 요청한 다음, 한도를 높이기 전에 결과를 읽어 본다.

이를 사용해 무언가를 만들었다면 X에서 공유하고 [@Gradio](https://x.com/Gradio) 및 [@HuggingFace](https://x.com/huggingface)을 tag해 주세요. 다른 사람이라면 여러분을 위해 만들 생각조차 하지 못했을 모델을 여러분이 만드는 모습을 보고 싶습니다.
