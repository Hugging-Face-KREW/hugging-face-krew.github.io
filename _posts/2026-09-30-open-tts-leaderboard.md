---
layout: post
title: "Open TTS Leaderboard: 다국어 Text-to-Speech 및 Voice Cloning을 위한 확장 가능한 평가"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/open-tts-leaderboard/thumbnail.png
image: assets/images/blog/posts/2026-09-30-open-tts-leaderboard/thumbnail.png
authors:
  - user: bezzam
  - user: Steveeeeeeen
  - user: eustlb
  - user: mrfakename
slug: "open-tts-leaderboard"
source_url: "https://huggingface.co/blog/open-tts-leaderboard"
source_published_date: "2026-09-30"
source_published_at: "2026-09-30T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Open TTS Leaderboard: Scalable Evaluation for Multilingual Text-to-Speech and Voice Cloning](https://huggingface.co/blog/open-tts-leaderboard)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/open-tts-leaderboard -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Open TTS Leaderboard: 다국어 Text-to-Speech 및 Voice Cloning을 위한 확장 가능한 평가

### TLDR 👉 오픈 소스 및 다국어에 초점을 맞춘 새로운 [TTS leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard)

오픈 소스 Text-to-Speech(TTS) 모델 출시 속도는 놀라울 정도입니다. Hugging Face Hub에는 2026년 9월 30일 기준으로 [8K TTS models](https://huggingface.co/models?pipeline_tag=text-to-speech)개가 넘는 모델이 제공되고 있습니다 🚀

<figure class="image text-center">
  <iframe src="https://eustlb-tts-models-on-the-hub.static.hf.space" width="100%" height="450" frameborder="0" scrolling="no"></iframe>
</figure>

**하지만 평가 방식은 그 속도를 따라가지 못했으며, 여전히 파편화되고 표준화되지 않았습니다.** 표준으로 여겨지는 방식은 MOS나 MUSHRA와 같은 사람의 선호도 점수입니다([metrics](https://picovoice.ai/blog/measuring-tts-quality/)에서 자세히 설명). 이를 위해 여러 아레나 기반 리더보드가 커뮤니티에 유용한 참고 지점으로 자리 잡았습니다.

1. [TTS Arena v2](https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2)
2. [Artificial Analysis](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice)
3. [Voice Arena](https://voicearena.com/tts-leaderboard)

이러한 아레나는 두 모델의 TTS 출력을 사용자에게 제시하고, 어느 쪽을 선택하는지 묻는 방식으로 모델을 비교합니다. 충분한 수의 투표를 수집한 후 [Elo score](https://en.wikipedia.org/wiki/Elo_rating_system)를 계산해 모델의 순위를 정하며, 일반적으로 Bradley–Terry 모델을 사용합니다([Voice Arena methodology](https://voicearena.com/tts-methodology) 참조).

사람의 선호도가 궁극적인 판단 기준이기는 하지만, **아레나는 TTS 출시 속도를 따라갈 만큼 확장될 수 없습니다.** 이는 오픈 소스 모델이 아레나 스타일 리더보드에 적게 포함되는 이유를 부분적으로 설명할 수 있습니다. 2026년 9월 30일 기준으로 [Artificial Analysis](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice)의 모델 92개 중 오픈 웨이트 모델은 16개뿐이며, [Voice Arena](https://voicearena.com/tts-leaderboard)에서도 비슷한 편향이 나타납니다. 이는 실질적인 요인을 반영하는 것으로 보입니다. API 모델을 추가하는 데는 API 키 외에 거의 필요한 것이 없지만, 오픈 모델은 아레나 운영자가 호스팅하고 서빙해야 하며, 상용 제공업체는 오픈 소스 개발자보다 배치되기를 원할 동기가 더 크기 때문입니다. 아레나 스타일 평가의 또 다른 한계는 투표자의 일관성입니다. 어떤 아레나도 동일한 투표자들이 동일한 “더 나음”의 기준으로 시간에 따라 모델을 일관되게 평가하도록 보장할 수 없습니다. 한 사람의 선호도조차 시간에 따라 변합니다(헤라클레이토스가 말한 유명한 문구처럼 “같은 강에 두 번 발을 담글 수는 없다”).

이를 위해 저희는 [Open TTS Leaderboard](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard)를 구축했습니다. 이 리더보드는 객관적인 메트릭을 사용해 서로 보완적인 성능 측면에서 모델을 평가합니다.

1. **명료성**: [Qwen3 ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)를 사용해 프롬프트와 생성된 오디오의 전사 결과 사이의 단어/문자 오류율(WER 및 CER)을 측정합니다([Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)의 최고 순위 오픈 소스 모델).
2. **속도**: H200 GPU에서 배치 오프라인 추론의 역실시간 계수(RTFx)와, H200 GPU 및 CPU에서 배치 크기 1의 스트리밍 지연 시간을 정량화하는 첫 오디오까지의 시간(TTFA)을 측정합니다.
3. 생성된 오디오와 참조 클립의 [WavLM speaker embeddings](https://huggingface.co/bezzam/wavlm_large_finetune_seed_tts_eval) 사이 코사인 유사도(SIM)를 계산해 **화자 유사도**를 측정합니다.

객관적인 메트릭을 사용하면 **모델 평가에 걸리는 시간이 몇 주(투표 수집에 필요한 시간)에서 몇 시간으로 단축됩니다.** ⚡

중요한 점은 Open TTS Leaderboard가 사람의 선호도 순위를 대체하지 않는다는 것입니다. ASR 기반 WER은 명료성을 나타내는 대리 지표이고, 화자 유사도는 음성 정체성 보존 정도를 추정합니다. 어느 쪽도 자연스러움, 표현력 또는 청취자의 선호도를 직접 측정하지는 않습니다. 그럼에도 이러한 메트릭은 투표 기반 리더보드에서 어떤 모델을 평가에 포함할지 결정하는 데 도움을 줄 수 있습니다.

이 리더보드가 **커뮤니티에 의해 만들어지기를** 바랍니다. 평가가 계속 관련성 있고 유익하게 유지될 수 있도록 여러분의 피드백을 듣고 싶습니다. 다음 몇 개의 섹션에서는 Open TTS Leaderboard의 주요 기능을 살펴봅니다.

## 다국어 + voice cloning 평가 {#section-1}

리더보드의 기본 보기에서는 [Seed TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval)([paper](https://huggingface.co/papers/2406.02430))와 [CV3 Eval](https://github.com/QwenAudio/CV3-Eval)(zero shot)의 영어 분할 데이터에 대한 macro-average WER([paper](https://huggingface.co/papers/2505.17589))로 모델 순위를 정합니다.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/english_table.png" width="1024px" alt="thumbnail" />
</div>

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/english_pareto.png" width="1024px" alt="thumbnail" />
</div>

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/english_bars.png" width="1024px" alt="thumbnail" />
</div>

[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3), [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)는 이 두 분할 데이터의 평균 영어 WER에서 선두를 차지합니다. Pareto 플롯은 WER, 배치 추론(RTFx), 모델 크기 사이에서 균형을 잘 맞추는 모델을 시각화합니다.

영어 성능이 반드시 다른 언어로 이어지는 것은 아닙니다. 여러 언어를 선택해 다국어 성능을 기준으로 모델 순위를 정할 수 있습니다. Seed TTS Eval에는 영어와 중국어 오디오만 있으므로, 다른 언어의 점수는 단순히 CV3 Eval(zero shot)의 점수입니다. 중국어, 일본어, 한국어는 문자 기반 언어이므로 문자 오류율(CER)이 보고되며, 언어별 “Average WER”은 언어 전체에 대한 macro-average라는 점에 유의하세요.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/multilingual.png" width="1024px" alt="thumbnail" />
</div>

[k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro), [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)는 강력한 다국어 모델입니다.

“Voice cloning”을 선택하면 이 기능을 지원하는 모델을 선택한 언어에서 비교할 수 있습니다.

또한 이제 표에 화자 유사도를 나타내는 SIM 열이 표시되며, SIM, 배치 추론, 모델 크기 간의 트레이드오프를 시각화하는 Pareto 플롯 두 개도 추가로 제공됩니다.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/voice_clone_table.png" width="1024px" alt="thumbnail" />
</div>

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/voice_clone_pareto.png" width="1024px" alt="thumbnail" />
</div>

[bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) 및 [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)와 같은 일부 모델은 voice cloning을 사용할 때, 즉 참조 오디오가 제공될 때 평균 WER이 개선됩니다.

## TTS 출력 비교 및 투표 {#section-2}

수치만으로는 전체 이야기를 알 수 없으며, 앞서 언급했듯이 **사람의 선호도가 궁극적인 판단 기준입니다.** “Listen” 탭에서 메트릭의 기반이 되는 생성 출력을 비교하고, 어떤 모델을 선호하는지 확인할 수 있습니다!

관심 있는 **언어/데이터셋**을 선택하고, **voice cloning**을 비교할지 정한 다음, 원하는 모델을 선택하거나 무작위로 선택된 모델의 출력을 들어볼 수 있습니다.

**“Listen” 탭은 기존 TTS 리더보드의 중요한 공백을 메웁니다. 다양한 모델의 출력을 탐색할 수 있는 공간입니다.**

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/listen_tab.png" width="1024px" alt="thumbnail" />
</div>

생성된 출력에 대한 피드백을 남길 수도 있습니다. 커뮤니티에서 더 많은 투표를 수집하면 이 데이터를 리더보드에 포함할 수 있습니다. **그러니 투표해 주세요! 단, 스팸/봇을 걸러낼 수 있도록 HF 계정으로 로그인해 주세요.**

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/listen_outputs.png" width="1024px" alt="thumbnail" />
</div>

## 스트리밍 성능 {#section-3}

“Streaming” 탭에서는 스트리밍 기능을 비교합니다. 모델은 TTFA(첫 오디오까지의 시간)를 기준으로 순위가 정해집니다. TTFA는 모델에 요청한 후 재생할 수 있는 오디오를 얻기까지 사용자가 기다리는 시간을 정량화합니다. 이는 음성 에이전트와 기타 대화형 앱에서 중요합니다.

스트리밍 모델(“Streaming API” 아래에 ✅ 표시)의 경우 첫 번째 오디오 청크가 도착할 때까지의 시간입니다. 스트리밍을 지원하지 않는 모델의 경우에는 전체 발화가 생성될 때까지의 시간입니다. 그보다 일찍 재생을 시작할 수 없기 때문입니다. 모든 모델은 동일한 CV3-Eval의 영어 프롬프트 50개를 사용해, 동일한 하드웨어와 기본 음성으로 한 번에 하나의 오디오(배치 크기 1)를 실행합니다. 처음 3회 실행은 워밍업으로 제외하고 나머지 실행에서 TTFA의 중앙값을 보고합니다.

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/streaming_table.png" width="1024px" alt="thumbnail" />
</div>

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/streaming_bars.png" width="1024px" alt="thumbnail" />
</div>

기본 보기에서는 H200 GPU에서의 성능을 비교합니다. CPU 결과도 일부 모델에서 제공되며, 그 수는 계속 늘어나고 있습니다!

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-tts-leaderboard/streaming_cpu.png" width="1024px" alt="thumbnail" />
</div>

[kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)는 GPU와 CPU 모두에서 스트리밍에 뛰어난 모델입니다!

## 결론 {#section-4}

Open TTS Leaderboard의 목표는 놀라운 속도로 출시되는 TTS 모델을 따라가는 것뿐만 아니라, 커뮤니티에 의해 만들어지는 것입니다. 평가가 계속 관련성 있고 유익하게 유지될 수 있도록 여러분의 피드백을 듣고 싶습니다. 어떤 데이터셋, 모델, 메트릭을 보고 싶은지 알려주세요!

현재는 다음에 초점을 맞추고 있습니다.

1. **오픈 소스 모델**: 아레나 스타일 평가에서 소외되어 온 훌륭한 모델을 많이 소개하기 위해서입니다.
2. **다국어**: 영어 성능은 다른 언어를 적절히 대변하는 지표가 아니기 때문입니다.

Open ASR Leaderboard [repo](https://github.com/huggingface/open_asr_leaderboard)와 마찬가지로 곧 평가 스크립트를 오픈 소스로 공개할 예정입니다. 이를 통해 GitHub Issues와 PRs를 통해 직접 피드백과 제안을 보내실 수 있습니다! 함께 TTS 평가를 만들어 갑시다 🤗
