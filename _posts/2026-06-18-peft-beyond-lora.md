---
layout: post
title: "Beyond LoRA: 가장 인기 있는 미세조정 기법을 이길 수 있을까?"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-06-18-peft-beyond-lora/thumbnail.png
authors:
  - user: BenjaminB
  - user: sayakpaul
  - user: hubnemo
  - user: kashif
slug: "peft-beyond-lora"
source_url: "https://huggingface.co/blog/peft-beyond-lora"
source_published_date: "2026-06-18"
source_published_at: "2026-06-18T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Beyond LoRA: Can you beat the most popular fine-tuning technique?](https://huggingface.co/blog/peft-beyond-lora)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/peft-beyond-lora -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Beyond LoRA: 가장 인기 있는 미세조정 기법을 이길 수 있을까?

<p align="center">
    <img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/lora-celebration.png" alt="Is LoRA the best PEFT technique?" width="600"/>
</p>

# 파라미터 효율적으로 모델을 미세조정하려고 계획할 때는 LoRA를 넘어서 생각해 보세요

오픈 모델을 자신의 데이터로 미세조정하고자 한다면, 아마도 소위 매개변수 효율적 미세조정, 간단히 *PEFT*에 관심이 있을 것입니다. 이 용어는 모델을 미세조정하는 데 필요한 메모리 요구량을 크게 줄여주는 기술들을 설명합니다. 이러한 기법은 수십 가지가 있지만, 거의 모두 "LoRA"라는 것을 선택합니다. 이 블로그 글에서 LoRA가 정말 최선의 선택인지, 정보에 기반한 의사결정을 내리기 위해 어떤 도구들이 있는지, 그리고 LoRA를 넘어 시야를 넓힘으로써 어떻게 이익을 얻을 수 있는지 살펴봅니다.

# What is PEFT and when do you need it

수많은 오픈 모델이 있지만, 그것들이 자주 당신의 사용 사례에 충분하지는 않습니다. 프롬프트 엔지니어링이 도움이 될 수는 있지만 보통은 충분하지 않습니다. 처음부터 새 모델을 학습시키기보다는 기존 모델을 미세조정하는 것을 고려해야 합니다.

하지만 미세조정은 메모리를 많이 요구합니다: 일반적으로 전체 모델을 여러 번 적합시키려면 충분한 메모리가 필요합니다. 양자화는 모델의 메모리 footprint를 줄여주지만 양자화된 모델은 직접 미세조정할 수 없습니다. 그래서 미세조정에 필요한 메모리를 줄이기 위해 여러 기법이 등장했고 이를 "매개변수 효율적 미세조정"(PEFT)이라고 부릅니다.

PEFT를 사용하면 메모리의 일부만으로도 모델을 미세조정하고 심지어 양자화된 모델도 미세조정할 수 있습니다. 또한 아주 작은 체크포인트 크기, 재앙적 망각에 대한 더 큰 저항성, 같은 기본 모델에서 여러 개의 미세조정을 제공할 수 있는 등의 이점이 있습니다.

허깅페이스에서 우리는 [`PEFT` library](https://github.com/huggingface/peft)를 개발합니다. 이 라이브러리는 하나의 통합 API 아래에서 많은 PEFT 기법을 구현하고 생태계와 잘 통합되며, 예로 [`Transformers`](https://huggingface.co/docs/transformers/main/en/peft)와 [`Diffusers`](https://huggingface.co/docs/diffusers/main/en/api/loaders/peft)가 있습니다. 또한 [multiple quantization methods](https://huggingface.co/docs/peft/developer_guides/quantization)도 지원하여 매개변수 효율적 미세조정에 대한 접근성을 높입니다. `PEFT`는 자체 데이터로 미세조정하든 새로운 PEFT 방법을 연구하든 좋은 시작점을 제공합니다.

# LoRA: 미세조정 기술의 여왕 👑

초기에 등장해 꽤 효과적임이 입증된 하나의 매개변수 효율적 미세조정 기법은 “Low Rank Adaptation”, 줄여서 [“LoRA”](https://huggingface.co/papers/2106.09685)라고 불립니다. 이는 기본 모델의 가중치를 고정하고 상위에 몇 개의 매개변수를 추가한 뒤, 그 소수의 매개변수들만 학습시키는 방식으로 작동합니다.

모든 PEFT 기법 중 LoRA가 단연 가장 대중적입니다. 아래는 몇 가지 추정치입니다:

* 정확히 하나의 PEFT 기법을 언급한 20,834 [model cards on Hugging Face Hub](https://huggingface.co/datasets/librarian-bots/model_cards_with_metadata) 표본 중 20,509가 LoRA를 언급했습니다(98.4%).
* 외부 사이트에서도 이미지 생성에 인기 있는 PEFT 기법들을 확인했습니다. 10,000개의 체크포인트 샘플을 사용한 결과, 그 중 7,111이 LoRA였습니다. 나머지로 확인된 PEFT 기법은 LoCon(363)와 DoRA(11)로, 이는 LoRA의 변형으로 간주됩니다. 이는 PEFT 체크포인트의 95.0%가 LoRA임을 뜻합니다.
* GitHub에서 코드 스니펫 `from peft import <PEFT CONFIG>`를 검색한 결과([example GH query](https://github.com/search?q=%22from+peft+import+LoraConfig%22&type=code)), 결과의 71.3%가 LoRA에 해당합니다. 그다음으로 상위권은 LoHa(3.7%)와 AdaLoRA(3.5%)입니다.

이 추정치들이 완벽하지는 않지만, LoRA가 단연 가장 일반적으로 사용되는 PEFT 기법이라는 결론은 여전히 타당합니다.

이것은 단지 LoRA가 모두에게 가장 잘 작동한다는 것을 의미할 수 있으며, 이 사실이 사용 통계에 반영됩니다. 그러나 또 다른 가능성도 있습니다: LoRA는 초기의 인기 있는 PEFT 기법 중 하나였으므로 그 사용이 자기 강화적으로 확산되었을 수 있습니다. 즉, LoRA가 가장 높은 가시성, 가장 많은 튜토리얼/예제 수, 그리고 다운스트림 패키지에서 가장 잘 지원되는 결과로 인해 LoRA의 인기가 스스로를 강화했다는 이야기입니다.

이 모든 것은 다음과 같은 질문으로 이어집니다: *더 나은 기법을 배제함으로써 성능을 놓치고 있는 걸까요?* 결국 LoRA를 이겼다고 주장하는 논문들이 무수히 있습니다. 그것이 LoRA를 넘어 새로운 기술로 가야 한다는 충분한 증거가 아닐까요?

# 논문 결과를 바탕으로 올바른 PEFT 기법을 선택하는 것은 문제점이 있다

LoRA 외의 미세조정 기법을 연구하는 논문이 수십 편 있습니다. `PEFT` 라이브러리 안에서도 아직 작성 시점에 40개가 넘는 서로 다른 PEFT 기법이 존재합니다(PEFT 기법의 변형을 포함하면 더 많습니다). 거의 모든 기법에 대해, 연구자들이 그들의 벤치마크에 따라 자신의 기법이 LoRA를 능가한다고 주장하는 것을 볼 수 있습니다.

이러한 주장들의 문제점은 연구자들이 기존 벤치마크를 능가하는 결과를 제시하도록 압박을 받고 있다는 점입니다. 의도하지 않더라도, 이는 결과에 편향을 불러일으킬 수 있습니다. 예를 들어 연구자들이 제안한 기법보다 대안 기법의 튜닝에 적은 시간을 할애함으로써 말이죠. 한 연구는 예를 들어 LoRA가 학습률을 조정함으로써 소위 더 나은 PEFT 기법들과 맞먹을 수 있다고 밝혔습니다( [https://arxiv.org/abs/2602.04998](https://arxiv.org/abs/2602.04998) ).

또 다른 문제는 각 논문이 비교하는 PEFT 기법의 집합과 실행하는 벤치마크의 집합이 서로 다르다는 점입니다. 같은 기법이 같은 벤치마크에서 비교되더라도 코드가 공개되지 않았거나 실행하기 어려운 경우가 많아 결과 재현이 어렵습니다.

전반적으로, 논문 결과만으로 어떤 PEFT 기법이 당신에게 가장 잘 맞는지 판단하기는 어렵습니다. 따라서 기본값인 LoRA를 그냥 선택하고 싶은 유혹에 빠질 수 있습니다.

# PEFT에서 벤치마킹에 어떻게 접근합니다

허깅페이스에서 사용자가 어떤 PEFT 기법을 사용할지에 대해 정보에 입각한 결정을 내릴 수 있도록 어떻게 도울 수 있을지 고민했습니다. `PEFT` 라이브러리를 통해 이미 많은 PEFT 기법을 구현하고 동일한 API로 노출하는 패키지를 제공하고 있습니다. 다음 단계는 논의된 이슈에 대해 더 많은 정보를 제공하는 벤치마크를 제공하는 것입니다.

이미 [checks fine-tuning of LLMs on a math dataset](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA)를 한동안 보유하고 있었습니다. 이 벤치마크는 LLM을 가져와 사고 체인 추론(chain-of-thought reasoning)으로 미세조정하여 수학 문제에 대한 결과를 생성하도록 하며, 지시문으로 미세조정되지 않은 기본 모델을 사용합니다. 벤치마크는 따라서 모델이 수학적 추론을 학습하고 생성된 출력을 기대 형식에 맞추어 조정할 수 있는지 여부를 검사합니다.

다른 모달리티에 대한 우리의 발견을 확장하기 위해 또 하나의 [image generation benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/image-gen)를 추가했습니다. 이 벤치마크는 모델이 새로운 개념인 [cat plushy](https://huggingface.co/datasets/peft-internal-testing/cat-image-dataset)을 학습하도록 미세조정할 수 있는지, 그리고 기존 개념을 잊지 않고 새로운 맥락에서 이를 생성할 수 있는지 테스트합니다.

<table align="center">
  <tr>
    <td><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/metamath-question.png" style="width: 400px; border: 1px solid #ccc; padding: 4px;"></td>
    <td><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/cat-plushy-train-image.jpg" style="width: 400px; border: 1px solid #ccc; padding: 4px;"></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><em>Left: Sample question and answer from the MetaMathQA dataset. Right: Sample image from the cat plushy dataset.</em></td>
  </tr>
</table>

모든 PEFT 기법은 정확히 같은 조건으로 평가됩니다: 동일한 기본 모델, 동일한 데이터세트, 동일한 학습 및 평가 코드, 동일한 하드웨어. 서로 다른 사용자의 요구가 다르기 때문에, 우리는 테스트 성능뿐만 아니라 더 많은 지표를 추적합니다. VRAM 사용량 외에도 망각/드리프트, 런타임, 체크포인트 크기와 같은 지표를 추적합니다. 결과는 일반 소비자용 하드웨어에서 실행되도록 설계되었으며, 새로운 실험을 추가하려면 새로운 `PEFT` 구성을 추가하고 스크립트를 실행하면 됩니다.

모든 PEFT 기법을 동등한 조건에서 비교하고 특정 기법을 편들지 않기 때문에, 이 벤치마크가 서로 다른 PEFT 기법들이 얼마나 잘 작동하는지에 대한 객관적인 그림을 제시한다고 믿습니다. 귀하의 데이터세트가 있다면 비슷한 접근법을 취하고 `PEFT` 라이브러리를 활용해 여러 PEFT 기법을 평가해 보시길 권합니다.

# Our findings: LoRA works well but is not necessarily the best choice

벤치마크를 마친 뒤, LoRA가 잘 작동하더라도 다른 PEFT 기법들이 한 가지 혹은 여러 축에서 이를 능가할 수 있으며 따라서 고려되어야 한다는 것을 발견했습니다. 아래 이미지는 LoRA와 다섯 가지 다른 PEFT 기법의 성능을 비교합니다.

<table align="center">
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/benchmark-highlights.png" width="900"/></td>
  </tr>
  <tr>
    <td align="center"><em>Some results from the benchmark. When it comes to test performance and memory usage, LoRA is not necessarily the best choice. Left: MetaMathQA benchmark; right: image generation benchmark. Consult this <a href="https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison">Space</a> for the most up-to-date results.</em></td>
  </tr>
</table>

위의 결과를 해석하는 한 가지 방법은 트레이드오프 관점에서 생각하는 것입니다. 예를 들어: 모델이 테스트 데이터에서 얼마나 잘 작동하는지와 이를 학습하는 데 필요한 메모리가 얼마나 되는지 사이의 trade-off는 어떤가요? 어떤 PEFT 기법도 이 두 지표를 동시에 다른 어떤 기법보다 더 잘 이길 수 없다면, 그것은 *파레토 프런티어*에 속합니다. 다시 말해: 더 나은 테스트 정확도를 원한다면 더 많은 메모리가 필요하고, 메모리 효율성을 더 원한다면 정확도를 포기해야 한다는 뜻입니다.

LLM 수학 데이터셋 벤치마크의 결과를 좀 더 자세히 살펴보겠습니다. 테스트 정확도와 메모리 측면에서 보자면 LoRA는 실제로 파레토 프런티어에 속합니다. 최고점에서 53.2%의 테스트 정확도와 22.6 GB의 VRAM이 필요합니다. 그러나 파레토 프런티어에 속하는 다른 PEFT 기법도 있습니다. 예를 들어, [BEFT](https://huggingface.co/docs/peft/main/en/package_reference/beft)은 32.9%의 테스트 정확도를 달성하고 최대 메모리는 20.2 GB에 불과합니다. 반대편 끝에는 [Lily](https://huggingface.co/docs/peft/main/en/package_reference/lily)이 있는데, 54.9%의 테스트 정확도를 달성하지만 25.6 GB의 메모리가 필요합니다. 무엇이 더 중요한지에 따라 LoRA가 최적의 트레이드오프를 제시하지 않는다고 결론지을 수도 있습니다.

<table align="center">
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/metamath-pareto.png" width="900"/></td>
  </tr>
  <tr>
    <td align="center"><em>Test accuracy vs memory usage tradeoff of fine-tuning <code>meta-llama/Llama-3.2-3B</code> and evaluating it on GSM8K. LoRA does well but so do other PEFT techniques.</em></td>
  </tr>
</table>

또한 이 작업에서 LoRA가 잘 수행하더라도 이것은 일반 LoRA를 뜻하는 것은 아닙니다. 한쪽은 LoRA가 [rank stabilized initialization](https://huggingface.co/papers/2312.03732)와 함께하는 경우로, 이는 LoRA 기여를 기본 초기화와 다르게 스케일링하는 기법으로 매우 높은 테스트 정확도(53.2%)를 제공합니다. 반대편에는 LoRA 가중치의 일부를 고정하고 더 메모리 효율적인 [LoRA-FA](https://huggingface.co/papers/2308.03303)을 사용하는 경우가 있습니다(20.2 GB). 일반 LoRA는 22.5 GB 메모리에서 정확도 48.1%에 그쳐 대안들보다 피하는 것이 좋습니다.

다음으로 이미지 생성 벤치마크를 살펴보겠습니다. [Hugging Face Space](https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison)에서 “Select Task” 드롭다운에서 “image-gen”을 선택하면 결과가 표시됩니다. 이 작업의 목표는 새 개념인 고양이 봉제인형을 학습하고 이를 새로운 프롬프트로 일반화하는 것입니다.

<table align="center">
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/cat-plushy-lora.png" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>Cat plushy image created with LoRA fine-tuned on <code>FLUX.2-klein-base-4B</code>.</em></td>
  </tr>
</table>

이 작업의 주요 지표는 '디노 유사도'(dino similarity)로, 생성된 이미지가 보류 테스트 데이터셋의 이미지와 얼마나 닮았는지를 측정하며, 값이 높을수록 좋습니다. 항상 그렇듯 메모리 사용량도 주시합니다. 이 두 지표의 파레토 프런티어를 시각화하면 LoRA가 그 프런티어 아래에 위치하는 것을 볼 수 있습니다. 구체적 수치를 보자면: LoRA의 유사도 점수는 0.697이고, [OFT](https://huggingface.co/docs/peft/package_reference/oft)는 0.708를 달성합니다; 메모리 측면에서 LoRA는 9.97 GB가 필요하고 OFT는 9.01 GB가 필요합니다. 따라서 이 지표들에 대해 OFT가 LoRA를 엄격하게 능가합니다.

<table align="center">
  <tr>
    <td align="center"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/image-gen-pareto.png" width="900"/></td>
  </tr>
  <tr>
    <td align="center"><em>Test accuracy vs memory usage tradeoff of fine-tuning <code>FLUX.2-klein-base-4B</code> and evaluating it on the test set. Other PEFT techniques like OFT beat LoRA in terms of test score and lower memory usage.</em></td>
  </tr>
</table>

물론 파레토 프런티어에 근접한 다른 PEFT 방법들도 확인해야 합니다. 지표는 난수성으로 인해 작은 차이가 있을 수 있습니다. 또한 다른 지표를 탐색해 보아야 합니다: 런타임 성능이 중요한가, 체크포인트의 크기가 중요한가? 드롭다운에서 관련 지표를 선택하면 그림이 크게 달라질 수 있습니다. 이미지 생성 벤치마크의 경우, 미세조정된 모델의 능력을 가늠할 수 있도록 생성 샘플 이미지를 확인해 보십시오.

# 제한점

> 반대 의견: 그러나 벤치마크가 한 방법을 다른 방법보다 우월하게 만듭니다!

`PEFT` 벤치마크에 대해 제기될 수 있는 한 가지 비판은 하이퍼파라미터의 선택이 한 기술을 다른 기술보다 유리하게 만들 수 있다는 점입니다. 이것은 사실이며, 이만큼 많은 기술에 대해 완전하고 공정한 하이퍼파라미터 스윕을 수행하는 것은 어렵습니다. 다만 모든 사람이 자신의 실험을 `PEFT`에 기여하는 것은 매우 쉽습니다: 특정 PEFT 기법이 서로 다른 하이퍼파라미터를 선택함으로써 개선될 수 있다고 믿는다면 PR을 만들어 주세요! [instructions on how to do that](https://github.com/huggingface/peft/tree/main/method_comparison#creating-new-experiments)를 추가했습니다. 비슷한 맥락으로, 완전히 새로운 벤치마크를 기여하고 싶다면 아이디어를 논의하기 위해 저희에게 연락해 주세요.

벤치마크의 또 다른 문제는 특정 PEFT 기법의 능력을 완전히 반영하지 못할 수 있다는 점입니다. 우리는 다양한 차원에서 기법들을 비교하고 이러한 트레이드오프에 따라 최상위를 발견할 수 있도록 했습니다. 그러나 이렇게 해서 모든 면모를 포착하는 것은 불가능합니다. 예를 들어, Cartridges ([https://huggingface.co/docs/peft/package_reference/cartridges](https://huggingface.co/docs/peft/package_reference/cartridges))라는 PEFT 기법은 긴 프롬프트를 압축하기 위해 개발되었지만 벤치마크에는 측정되지 않았습니다. 선택에 영향을 줄 수 있는 다른 요인들도 있습니다. 예를 들면:

* PEFT 기법에 따라 수정할 수 있는 특정 레이어 유형만 있습니다.
* 모든 PEFT 기법이 양자화된 기본 모델을 지원하는 것은 아니지만(하지만 `PEFT`에서 지원을 활발히 확장하고 있습니다).
* 일부 PEFT 기법은 [merging of the adapter](https://huggingface.co/docs/peft/main/en/developer_guides/model_merging)을 통해 런타임 오버헤드를 줄일 수 있지만, 다른 기법은 그렇지 않습니다.

벤치마크가 연구를 완전히 대신해 주지는 못하지만, 합리적인 방향성을 제공합니다.

<table align="center">
  <tr>
    <td align="center"><a href="https://huggingface.co/spaces/peft-internal-testing/PEFT-shop"><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/peft-shop.png" width="100%"/></a></td>
  </tr>
  <tr>
    <td align="center"><em>Click on the image to peruse the PEFT shop to find the best PEFT technique for you. It allows you to browse not only by benchmark metrics but also by capabilities, like quantization support.</em></td>
  </tr>
</table>

> 반대 의견: llama.cpp/vLLM/...은 LoRA만 지원합니다

LoRA가 아닌 다른 PEFT 기법은 LoRA가 보는 다운스트림 패키지에서의 광범위한 지원을 받지 못하는 한계가 있습니다. 예를 들어 vLLM으로 모델을 서비스하고 싶다면 LoRA 체크포인트만 로드 가능합니다. 다행히도 `PEFT`는 이제 [converting other adapters into LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora_conversion)를 지원합니다. 따라서 비-LoRA 체크포인트를 LoRA로 변환하여 vLLM이나 다른 다운스트림 패키지에서 사용할 수 있습니다.

이를 테스트하기 위해 GraLoRA 기법을 사용한 이미지 어댑터를 LoRA 체크포인트로 변환했습니다. 변환 후 테스트 점수는 사실상 동일했습니다(유사도 0.702 → 0.694, 0.260 → 0.269). 아래는 프롬프트 “sks 해변의 고양이”에 대한 테스트 이미지입니다.

<table align="center">
  <tr>
    <td><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/lora-image-gen.png" style="width: 400px; border: 1px solid #ccc; padding: 4px;"></td>
    <td><img src="https://huggingface.co/datasets/peft-internal-testing/peft-blog-assets/resolve/main/peft-beyond-lora/gralora-image-gen-converted.png" style="width: 400px; border: 1px solid #ccc; padding: 4px;"></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><em>Left: Image generated by GraLoRA. Right: Image generated by the same GraLoRA checkpoint converted to a LoRA checkpoint. The images quality is comparable.</em></td>
  </tr>
</table>

현재 모든 PEFT 기법에 대한 변환을 구현한 것은 아니지만, 수요가 있다면 지원 범위를 확장하겠습니다.

# Conclusion and what *you* can do

`PEFT` 패키지 작업을 하는 동안 LoRA가 많은 모멘텀을 갖고 있다는 것을 확인했습니다. 반면 다른 PEFT 기법들이 잠재적으로 더 나을 수 있습니다. 따라서 다양한 지표에서 다양한 PEFT 기법이 얼마나 잘 작동하는지 보다 객관적인 그림을 그릴 수 있도록 PEFT에 벤치마크를 추가하기로 했습니다.

발견한 결과를 바탕으로 LoRA가 나쁜 선택은 아니지만 더 나은 선택이 있을 수 있다고 자신 있게 결론지을 수 있습니다. 특히 이미지 생성 벤치마크를 확인할 때 LoRA는 다른 기법들에 의해 능가됩니다. 지표 외에도 올바른 PEFT 기법을 선택할 때 고려해야 할 다른 요인이 있다고 논의했습니다. 그러나 그럼에도 불구하고 `PEFT`를 더 확장하여 LoRA와 다른 기법들 간의 기능적 동등성을 달성하려고 합니다.

저희의 여정은 아직 끝나지 않았습니다. 기존 벤치마크를 확장하고 개선하며 앞으로 더 많은 벤치마크를 추가할 계획도 있습니다. 커뮤니티의 기여를 쉽게 할 수 있도록 보장했으며, 이것에 기여하고 싶다면 [issue on the `PEFT` repository](https://github.com/huggingface/peft/issues)를 열고 기여 방법을 알려 주세요.

이 글에서 한 가지라도 꼭 기억해야 한다면, LoRA가 귀하의 사용 사례에 맞는 PEFT 기법의 자동 기본값이 되어서는 안 된다는 점입니다. `PEFT`가 제공하는 통합 API 덕분에 한 PEFT 기법에서 다른 기법으로 바꾸는 것은 코드의 구성 하나만 바꾸는 것만큼 쉽습니다. 그리고 LoRA를 계속 고수하더라도 `PEFT`에서 지원하는 모든 변형들을 확인해 보세요: DoRA, rs-LoRA, LoRA-FA 등. 이 다른 기법들을 시도해 보면 놀라울 정도로 마음에 들 수 있습니다.

<p align="left">
  <em>Example: Changing from LoRA to OFT using `PEFT`</em>
</p>

```diff
from transformers import AutoModelForCausalLM
-from peft import LoraConfig, get_peft_model
+from peft import OFTConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", dtype="bfloat16")
-config = LoraConfig(target_modules=["q_proj", "v_proj"])
+config = OFTConfig(target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
```
