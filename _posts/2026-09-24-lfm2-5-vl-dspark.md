---
layout: post
title: "LFM2.5-VL-DSpark로 비전-언어 모델 가속하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/WgfT8N8Xyb6lBxSif7XLO.gif
image: assets/images/blog/posts/2026-09-24-lfm2-5-vl-dspark/thumbnail.gif
authors:
  - user: LiquidAI
slug: "lfm2-5-vl-dspark"
source_url: "https://huggingface.co/blog/LiquidAI/lfm2-5-vl-dspark"
source_published_date: "2026-09-24"
source_published_at: "2026-09-24T14:08:57+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Accelerating vision-language models with LFM2.5-VL-DSpark](https://huggingface.co/blog/LiquidAI/lfm2-5-vl-dspark)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/LiquidAI/lfm2-5-vl-dspark -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# LFM2.5-VL-DSpark로 비전-언어 모델 가속하기

오늘 저희는 비전-언어 모델(VLM) [LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B)을 위한 실험적 [DSpark](https://huggingface.co/papers/2607.05147) draft 모델을 출시합니다. 저희의 [recently released LFM2.5-DSpark drafter models](https://huggingface.co/blog/LiquidAI/lfm25-dspark)와 마찬가지로, 출력 품질을 변경하지 않으면서 메모리 사용량을 최소한으로 늘려 더 큰 속도 향상을 제공하는 speculative decoding 경로를 추가합니다.

- 더 빠른 추론: 디바이스에서 최대 3.13배, H100에서 2.66배의 디코딩 속도 향상, end-to-end 성능은 최대 2.62배 및 2.27배 향상

- 적은 메모리 비용: drafter가 280M개의 파라미터를 추가하며, 3B target 대비 8.9% 증가

- 출시 첫날부터 지원: llama.cpp, MLX-VLM, SGLang용 LFM 호환 DSpark 통합

## VLM에서 speculative decoding은 어떻게 작동하나요 {#section-1}

vision drafter는 텍스트 LFM2.5-DSpark drafter와 동일한 아키텍처를 사용합니다. 고정된 tapped layer 집합에서 target model의 hidden state를 추출하고, 이를 조건으로 사용해 k개의 candidate token 블록을 draft합니다. 이미지 패치와 텍스트 토큰은 해당 레이어에 도달하기 전에 공유 표현으로 투영되므로, 입력 modality와 관계없이 drafter는 동일한 차원의 hidden-state 벡터를 대상으로 작동합니다. 따라서 추론 알고리즘은 텍스트 모델과 달라지지 않습니다.
[![DSpark-Vision](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/P7UX63U74cbjMWDapPjFm.png)](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/P7UX63U74cbjMWDapPjFm.png)

## 학습 및 아키텍처 {#section-2}

저희는 vision-language SFT 데이터 혼합물을 사용하는 DSpark 레시피를 따르며, 모델이 처리할 것으로 예상되는 workload에 더 큰 가중치를 둡니다. 3개, 4개, 5개 레이어에 걸친 ablation을 바탕으로 draft model은 4개 레이어와 블록 크기 9를 사용하는 단순화된 attention-only drafter입니다. 최종 혼합물로 10 epoch를 실행하고 각 epoch 후 acceptance를 측정했으며, 추가 학습 토큰에 따라 acceptance가 향상되다가 점차 수익이 감소하는 지점에 도달했습니다. 추론 시에는 하드웨어에 따라 블록 크기 8 또는 9를 권장합니다.

그 결과 생성된 drafter는 약 280M개의 파라미터를 가지며, 배포된 모델의 파라미터 수를 단 8.9% 증가시킵니다.

| 구성 요소 | LFM2.5-VL-3B |
| --- | --- |
| 디코더 스택 (4개 레이어) | 193.0M |
| Hidden-state projection | 21.0M |
| Markov head | 65.5M |
| Norms + confidence head | 6.4k |
| 합계 | 279.5M |

## CPU 및 GPU에서의 추론 속도 향상 {#section-3}

LFM2.5-VL-3B용 DSpark draft model은 [llama.cpp](https://github.com/ggml-org/llama.cpp), [MLX-VLM](https://github.com/Blaizzy/mlx-vlm), [SGLang](https://github.com/sgl-project/sglang)를 출시 첫날부터 지원합니다.

디바이스 내 추론과 GPU 추론을 모두 측정합니다. 두 구성 모두 DSpark 블록 크기 8을 사용하며, [MMSpec benchmark](https://huggingface.co/papers/2603.14989)에 따라 일반 VQA, 텍스트 VQA, 이미지 캡셔닝, 차트 VQA, 복잡한 추론, 멀티턴 대화를 포함한 6개의 다양한 비전 기반 task에서 평가합니다.

디바이스 내 추론. M5 Max에서 MLX를 사용하면 task에 따라 디코딩이 2.30배에서 3.13배 더 빨라집니다. end-to-end 지연 시간은 1.56배에서 2.62배 향상됩니다. M3 Ultra에서 llama.cpp를 사용하면 디코딩은 1.57배에서 2.14배, end-to-end 성능은 1.30배에서 1.77배 향상됩니다.

[![Screenshot 2026-09-24 at 15.49.03](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/n638hOT2C6Yoflniz2ffs.png)](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/n638hOT2C6Yoflniz2ffs.png)

GPU 추론. H100에서 동일한 drafter는 20.4배에서 2.66배 더 빠른 디코딩을 제공하며, end-to-end 성능은 1.64배에서 2.27배 향상됩니다.

[![Screenshot 2026-09-24 at 15.49.27](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/QUPP6CX21dma5OiIYcEIL.png)](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/QUPP6CX21dma5OiIYcEIL.png)

## vision workload에서 speculative decoding의 한계 {#section-4}

LLM에서 prefill은 대부분 compute-bound이며, 비용은 prompt 길이에 따라 (sub)quadratic하게 증가합니다. VLM에서는 이미지가 먼저 vision encoder를 통과한 다음 language backbone이 텍스트 prompt와 함께 수백 개의 visual token을 처리하므로 이 문제가 더 커집니다. 엣지 디바이스는 datacenter GPU보다 연산 성능이 훨씬 낮기 때문에 prefill이 end-to-end 지연 시간에서 더 큰 비중을 차지합니다. 이는 Apple silicon과 H100에서의 time-to-first-token 및 decode 측정 결과에서도 확인됩니다. (M5의 코어별 GPU neural accelerator가 이 격차를 줄입니다.)

Speculative decoding은 decode만 가속하며 vision encoding이나 prefill은 가속하지 않습니다. 이러한 단계가 이미 전체 실행 시간의 상당 부분을 차지하는 경우, decode가 크게 빨라져도 end-to-end 향상은 제한적입니다. 이는 가속되지 않는 workload 부분에 의해 전체 속도 향상이 제한되는 Amdahl의 법칙에 해당합니다.

## LFM2.5-VL-DSpark 사용 방법 {#section-5}

SGLang에서 DSpark draft model을 실행하려면 LFM2 target을 위한 DSpark 지원이 포함된 SGLang 빌드([PR #40651](https://github.com/sgl-project/sglang/pull/40651))가 필요합니다. drafter를 연결해 target을 실행합니다.

```
python -m sglang.launch_server \
  --model-path LiquidAI/LFM2.5-VL-3B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path LiquidAI/LFM2.5-VL-3B-DSpark \
  --speculative-draft-attention-backend flashinfer \
  --speculative-dspark-block-size 9 \
  --disable-radix-cache
```


그런 다음 `http://localhost:30000/v1`의 OpenAI 호환 endpoint를 조회합니다. 블록 크기는 draft의 `config.json`에서 읽습니다. baseline은 세 가지 `--speculative-*` 플래그를 제외한 동일한 명령입니다.

llama.cpp에서 실행하려면 해당 llama.cpp 빌드([PR#29339](https://github.com/ggml-org/llama.cpp/pull/29339))가 필요합니다.

```
llama-server -m models/LFM2.5-VL-3B-F16.gguf \
  --mmproj models/mmproj-LFM2.5-VL-3B-F16.gguf \
  -md LFM2.5-2.6B-DSpark-F16.gguf \
  --spec-type draft-dspark --spec-draft-n-max 8 --spec-draft-n-min 0 \
  -fa on -ngl 99 -c 8192
```


MLX-VLM에서 실행하려면 해당 빌드([PR#2280](https://github.com/Blaizzy/mlx-vlm/pull/2280))가 필요합니다.

```
mlx_vlm.server --model LiquidAI/LFM2.5-VL-3B --draft-model LiquidAI/LFM2.5-VL-3B-DSpark
```


블록 크기는 sidecar metadata에서 읽습니다(n-max는 해당 값으로 제한됩니다). Speculative decoding은 정확합니다. target이 제안된 모든 토큰을 검증하므로 greedy 출력은 target만 사용했을 때와 동일합니다. 응답별 `timings`은 `draft_n` / `draft_n_accepted`를 보고합니다.

## 시작하기 {#section-6}

저희의 vision DSpark draft model은 Hugging Face의 [Safetensors](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark) 및 [GGUF formats](https://huggingface.co/LiquidAI/LFM2.5-VL-3B-DSpark-GGUF)에서 사용할 수 있습니다.

LFM2.5를 통해 어디서나 실행되는 AI라는 저희의 비전을 실현하고 있습니다. 이 모델은 다음과 같습니다.

- Open-weight — 제한 없이 다운로드하고, fine-tune하고, 배포할 수 있습니다.

- 첫날부터 빠른 성능 — llama.cpp, MLX, SGLang을 출시 첫날부터 지원합니다.

- 완전한 제품군 — 커스터마이즈를 위한 base model부터 특화된 audio 및 vision 변형 모델까지, 하나의 아키텍처로 다양한 사용 사례를 지원합니다.

여러분이 무엇을 만들어낼지 기대됩니다.

## 인용 {#section-7}

인용 시 다음 reference 또는 BibTeX를 사용해 주세요: Liquid AI, "LFM2.5-VL-DSpark: 엣지와 그 너머에서 vision-language model 가속하기", Liquid AI Blog, 2026년 9월.

```
@article{liquidAI2026vldspark,
  author = {Liquid AI},
  title = {LFM2.5-VL-DSpark: Accelerating vision-language models on edge and beyond},
  journal = {Liquid AI Blog},
  year = {2026},
  note = {www.liquid.ai/blog/lfm2-5-vl-dspark},
}
```
