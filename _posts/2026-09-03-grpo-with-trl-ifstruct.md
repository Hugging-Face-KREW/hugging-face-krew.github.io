---
layout: post
title: "Fine-tuning a 350M Model for Better Structured Outputs in 100 GRPO Steps"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/grpo-with-trl-ifstruct/thumbnail.png
image: assets/images/blog/posts/2026-09-03-grpo-with-trl-ifstruct/thumbnail.png
authors:
  - user: iamleonie
slug: "grpo-with-trl-ifstruct"
source_url: "https://huggingface.co/blog/grpo-with-trl-ifstruct"
source_published_date: "2026-09-03"
source_published_at: "2026-09-03T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Fine-tuning a 350M Model for Better Structured Outputs in 100 GRPO Steps](https://huggingface.co/blog/grpo-with-trl-ifstruct)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/grpo-with-trl-ifstruct -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

이 가이드는 구조화된 출력 준수 능력을 소형 모델에서 크게 향상시키기 위한 완전 공개형 저비용 레시피입니다. [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)을(를) [TRL library](https://huggingface.co/docs/trl/en/index)을(를) 사용해 Group Relative Policy Optimization (GRPO)으로 미세 조정하고, [IFStruct benchmark](https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0)에서 평가합니다. 전체 실행에는 약 500개의 샘플과 100회의 학습 단계만 필요하므로 무료 Colab 또는 Kaggle GPU에서도 충분히 실행할 수 있으며, [GitHub](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/grpo_with_trl_ifstruct.ipynb)에서 확인할 수 있습니다. 결과에 따르면 가벼운 미세 조정 절차만으로도 IFStruct 벤치마크 성능이 **22.6%에서 29.7%로** 향상됩니다.

구조화된 출력은 LLM의 실제 사용 사례에서 가장 흔한 작업 중 하나이지만, 대부분의 벤치마크는 이를 별도로 측정하기보다 더 광범위한 추론 또는 추출 점수에 포함합니다. 모델이 요청된 형식과 구조로 유효하고 파싱 가능한 출력을 안정적으로 반환하는지, 즉 스키마를 준수하는지는 해당 모델을 다운스트림 시스템에 연결할 수 있는지를 결정하는 경우가 많습니다.

*여기서 설명하는 학습 파이프라인은 [IFStruct blog](https://www.liquid.ai/blog/ifstruct-v1.0)에 설명된 RL 모델을 학습하는 데 사용된 파이프라인과 다르다는 점에 유의하세요. 이 notebook은 IFStruct 벤치마크 점수를 재현하는 것이 아니라, 소형 모델에 대한 작업별 미세 조정으로 성능을 향상시키고 훨씬 큰 모델과 비슷한 수준에 도달할 수 있음을 보여주는 것을 목표로 합니다.*

## 사전 요구 사항 {#section-1}

이 가이드는 서로 다른 환경에서 실행되는 두 부분으로 구성됩니다:

- **미세 조정**은 GPU에서 실행됩니다. 함께 제공되는 notebook은 무료 Colab 또는 Kaggle GPU에 맞춰져 있습니다.  
- **평가**는 `llama.cpp`을(를) 통해 MacBook에서 로컬로 실행할 수 있습니다. 여기서는 Apple M5 Max와 36 GB 통합 메모리를 탑재한 MacBook Pro를 사용하며, `llama.cpp`은 IFStruct 평가기가 통신하는 OpenAI 호환 서버를 제공합니다.

Python 도구에는 [`uv`](https://docs.astral.sh/uv/)이(가), 서빙에는 `llama.cpp`이(가) 필요합니다. [Liquid AI llama.cpp deployment docs](https://docs.liquid.ai/deployment/on-device/llama-cpp)에 따라 Homebrew로 `llama.cpp`을(를) 설치하고 `llama-server`을(를) 사용할 수 있는지 확인합니다:

```shell
brew install llama.cpp
llama-server --version
```


## LFM2.5-350M (Base model)의 IFStruct 평가 {#section-2}

시작하기 전에 [IFStruct benchmark](https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0)에서 LFM2.5-350M을 평가하고 **보고된 21.1% 점수를 재현할 수 있는지** 확인해 보겠습니다.

**IFStruct**는 LLM 출력의 유효성과 스키마 준수 여부를 테스트하는 벤치마크입니다. 이 벤치마크는 [Liquid4All/ifstruct](https://github.com/Liquid4All/ifstruct)에서 오픈 소스로 제공되며, 공개 벤치마크 데이터셋은 Hugging Face의 [LiquidAI/ifstruct-v1.0](https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0)에서 사용할 수 있습니다.

```shell
git clone https://github.com/Liquid4All/ifstruct.git
```


평가 비교를 위해 `llama.cpp`을(를) 사용해 MacBook에서 모델을 로컬로 서빙합니다. `BF16` GGUF ([LiquidAI/LFM2.5-350M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF))를 사용합니다.

그런 다음 다음 명령어로 base model 서버를 시작합니다:

```shell
llama-server \
  -hf LiquidAI/LFM2.5-350M-GGUF:BF16 \
  -c 32768 \
  -np 4 \
  -ngl 99 \
  --alias LiquidAI/LFM2.5-350M \
  --host 127.0.0.1 \
  --port 8080
```


- `--alias`: OpenAI 호환 엔드포인트로 IFStruct가 전송하는 모델 이름  
- `-ngl 99`: 사용 가능한 경우 `llama.cpp`에 모든 레이어를 GPU로 오프로드하도록 요청  
- `-np 4`: 네 개의 요청을 병렬로 처리  
- `-c 32768`: 프롬프트 컨텍스트의 크기

서버가 실행되면 2000개의 샘플로 전체 벤치마크를 실행할 수 있습니다:

```shell
uv run ifstruct-eval \
  --model LiquidAI/LFM2.5-350M \
  --base-url http://localhost:8080/v1 \
  --api-key dummy \
  --dataset data/test.jsonl \
  --results-file results/lfm2.5-350m-llamacpp-base.json \
  --n-threads 4 \
  --max-tokens 2048 \
  -v
```


```
============================================================
Model: LiquidAI/LFM2.5-350M
============================================================
Overall: 452/2000 passed (22.6%)
Average latency: 1453ms

By format:
  JSON: 180/1000 passed (18.0%)
  YAML: 272/1000 passed (27.2%)

By top-level structure:
  Wrapper key 288/1011 passed (28.5%)
  Bare list   164/989 passed (16.6%)

By entity type:
  test__camera_review                 6/83 passed (7.2%)
  test__clinical_trial                20/104 passed (19.2%)
  test__conference_schedule           7/87 passed (8.0%)
  test__escaping__bug_report_batch    24/89 passed (27.0%)
  test__escaping__config_snippet_audit 15/85 passed (17.6%)
  test__escaping__customer_email_thread 5/73 passed (6.8%)
  test__escaping__dialogue_sample     14/95 passed (14.7%)
  test__escaping__interview_transcript_segment 21/80 passed (26.2%)
  test__escaping__log_parser_examples 21/72 passed (29.2%)
  test__escaping__pr_discussion       22/87 passed (25.3%)
  test__escaping__repro_steps_batch   16/73 passed (21.9%)
  test__escaping__screenplay_scene    16/92 passed (17.4%)
  test__escaping__short_story_chapter 15/84 passed (17.9%)
  test__escaping__support_ticket_batch 27/73 passed (37.0%)
  test__escaping__terminal_session_notes 20/70 passed (28.6%)
  test__event_ticket_booking          49/107 passed (45.8%)
  test__gpu_review                    6/94 passed (6.4%)
  test__invoice                       28/86 passed (32.6%)
  test__job_posting                   25/85 passed (29.4%)
  test__real_estate_listing           31/82 passed (37.8%)
  test__recipe                        3/70 passed (4.3%)
  test__rental_car_booking            27/79 passed (34.2%)
  test__scientific_experiment         13/69 passed (18.8%)
  test__travel_itinerary              21/81 passed (25.9%)

Common errors:
  7228x required field missing
  738x wrong item count
  540x type mismatch
  317x Unclosed code block
  190x extraneous field 'notes'
  181x extraneous field 'path'
  175x extraneous field 'constraints'
  170x extraneous field 'type'
  170x missing code block
  100x expected bare list, got wrapper
```


[IFStruct release blog reports 21.1% for LFM2.5-350M](https://www.liquid.ai/blog/ifstruct-v1.0). 로컬 llama.cpp/BF16 설정에서는 22.6%가 측정되었으며, 이는 IFStruct 블로그에 보고된 21.1%에 가깝습니다. 동일한 서빙 스택 비교를 위한 기준선으로 이 로컬 결과를 사용합니다.

## 구조화된 출력에 대한 TRL 기반 GRPO 미세 조정 {#section-3}

실행 가능한 전체 파이프라인은 [accompanying notebook](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/grpo_with_trl_ifstruct.ipynb)에 있습니다. 이 절에서는 관련된 부분만 다룹니다.

### 학습 데이터

[`nvidia/Nemotron-RL-instruction_following-structured_outputs`](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs)을(를) 사용합니다. 이 데이터는 각 프롬프트를 대상 JSON Schema 및 예상 필드 수와 연결합니다. 학습에는 약 500개의 샘플을 사용합니다.

Nemotron 데이터 분포가 IFStruct 평가와 다르기 때문에, 두 데이터 사이의 다음 두 가지 차이를 줄이도록 프롬프트를 보강합니다:

- **40%**에는 "fenced code block 안에 출력을 반환하라"는 지시를 추가하여, 모델이 항상 raw JSON을 출력하는 대신 형식 지시를 *따르도록* 학습합니다.  
- 서로 겹치지 않는 **20%**는 최상위 배열 작업으로 변환합니다. 스키마를 필수 항목 수가 포함된 `array`으로 감싸며, 이를 통해 bare-list 출력과 항목 수 준수를 학습합니다.

### 모델 및 LoRA

`LiquidAI/LFM2.5-350M`을(를) 로드하고 LoRA adapter를 연결합니다. LFM2.5는 attention/convolution 하이브리드 아키텍처를 사용하므로 LFM 전용 모듈 이름을 대상으로 지정합니다:

```py
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    bias="none", 
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "out_proj", "in_proj",
        "w1", "w2", "w3",
    ],
)
```


약 \~6M개의 파라미터를 학습하며, 이는 모델의 약 1.66%입니다.

### 보상 함수

그런 다음 세 가지 보상 함수를 정의합니다. 각 함수는 `[0, 1]` 스케일로 작동하며, 추출된 *구조*가 올바른지를 기준으로 모든 completion을 평가합니다:

- `json_format_reward`: 출력이 파싱 가능하고 요청된 형식인가? 요청된 형식(fenced 또는 raw)이면 만점(`1.0`), 형식은 다르지만 파싱 가능하면 `0.2`, 파싱할 수 없는 출력이면 `0.0`을 부여합니다.  
- `field_count_reward`: 객체가 예상된 최상위 필드 수를 가지고 있는가? 정확히 일치하면 `1.0`을 얻고, 차이가 클수록 점수가 선형적으로 감소합니다.  
- `schema_validation_reward`: 출력이 해당 행의 JSON Schema에 따라 검증되는가? 모든 제약 조건 위반을 계산하며, 필수 키의 충족 범위에 따라 부분 점수 부여 여부를 결정합니다.

세 함수를 `reward_weights=[1.0, 0.5, 2.0]`을(를) 사용한 가중합으로 결합합니다.

### 학습

무료 등급의 16 GB GPU에 맞춰 프롬프트 그룹당 8개의 generation으로 100 step 동안 학습합니다:

```py
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="./outputs/lfm25-350m-nemotron-schema-grpo",
    learning_rate=5e-5,
    max_steps=100,
    warmup_steps=10,
    num_generations=8,              # completions sampled per prompt group
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # 4 prompt groups per optimizer step
    steps_per_generation=2,
    max_completion_length=1024,     # room for nested JSON
    mask_truncated_completions=False,
    temperature=1.1,                # hotter sampling keeps groups varied
    beta=0.01,                      # KL penalty toward the reference model
    reward_weights=[1.0, 0.5, 2.0], # json_format, field_count, schema_validation
    logging_steps=1,
    save_steps=100,
)
```


notebook에서 볼 수 있듯이 실행이 진행되는 동안 세 가지 보상 구성 요소가 모두 상승하고, warmup 이후 reference model의 KL이 0에서 벗어나 상승하며, 잘린 completion의 비율은 0에 가깝게 유지됩니다.

### 모델 병합 및 저장

마지막으로 LoRA adapter를 base weights에 다시 병합하고, 서빙을 위해 GGUF로 변환할 수 있는 단일 독립 checkpoint로 저장합니다:

```py
MERGED_DIR = f"{training_args.output_dir}-merged"

merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
```


## GRPO로 튜닝한 LFM2.5-350M의 IFStruct 평가 {#section-4}

GRPO 미세 조정 후 IFStruct 평가를 다시 실행합니다. 이를 위해 병합된 모델 checkpoint를 BF16 GGUF로 변환해야 합니다. 변환기 스크립트는 llama.cpp 소스와 함께 제공되므로 repo를 한 번 clone하고 변환기의 `gguf` 패키지를 설치합니다.

```shell
git clone --depth 1 https://github.com/ggml-org/llama.cpp
pip install ./llama.cpp/gguf-py

mkdir -p models
python llama.cpp/convert_hf_to_gguf.py \
  PATH_TO_YOUR_MERGED_MODEL \
  --outfile ./models/lfm25-350m-grpo-bf16.gguf \
  --outtype bf16
```


그런 다음 다음 명령어로 병합된 모델을 서빙합니다:

```shell
llama-server \
  -m ./models/lfm25-350m-grpo-bf16.gguf \
  --alias lfm25-350m-grpo-structured-output \
  -c 32768 \
  -np 4 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8081
```


그런 다음 미세 조정된 모델로 전체 IFStruct 평가를 다시 실행합니다:

```shell
uv run ifstruct-eval \
  --model lfm25-350m-grpo-structured-output \
  --base-url http://localhost:8081/v1 \
  --api-key dummy \
  --dataset data/test.jsonl \
  --results-file results/lfm25-350m-grpo.json \
  --n-threads 4 \
  --max-tokens 2048 \
  -v
```


```
============================================================
Model: lfm25-350m-grpo-structured-output
============================================================
Overall: 594/2000 passed (29.7%)
Average latency: 1518ms

By format:
  JSON: 319/1000 passed (31.9%)
  YAML: 275/1000 passed (27.5%)

By top-level structure:
  Wrapper key 300/1011 passed (29.7%)
  Bare list   294/989 passed (29.7%)

By entity type:
  test__camera_review                 5/83 passed (6.0%)
  test__clinical_trial                31/104 passed (29.8%)
  test__conference_schedule           11/87 passed (12.6%)
  test__escaping__bug_report_batch    32/89 passed (36.0%)
  test__escaping__config_snippet_audit 24/85 passed (28.2%)
  test__escaping__customer_email_thread 9/73 passed (12.3%)
  test__escaping__dialogue_sample     17/95 passed (17.9%)
  test__escaping__interview_transcript_segment 13/80 passed (16.2%)
  test__escaping__log_parser_examples 33/72 passed (45.8%)
  test__escaping__pr_discussion       26/87 passed (29.9%)
  test__escaping__repro_steps_batch   23/73 passed (31.5%)
  test__escaping__screenplay_scene    34/92 passed (37.0%)
  test__escaping__short_story_chapter 24/84 passed (28.6%)
  test__escaping__support_ticket_batch 36/73 passed (49.3%)
  test__escaping__terminal_session_notes 23/70 passed (32.9%)
  test__event_ticket_booking          62/107 passed (57.9%)
  test__gpu_review                    7/94 passed (7.4%)
  test__invoice                       36/86 passed (41.9%)
  test__job_posting                   33/85 passed (38.8%)
  test__real_estate_listing           32/82 passed (39.0%)
  test__recipe                        7/70 passed (10.0%)
  test__rental_car_booking            37/79 passed (46.8%)
  test__scientific_experiment         14/69 passed (20.3%)
  test__travel_itinerary              25/81 passed (30.9%)

Common errors:
  7331x required field missing
  890x wrong item count
  555x type mismatch
  102x expected bare list, got wrapper
   62x extraneous field 'metadata.tone'
   55x 6 is greater than maximum 5
   49x extraneous field 'speaker_labels'
   47x extraneous field 'tone'
   44x 'cups' not in allowed values ['mg', 'g', 'kg', 'oz', 'lb', 'ml', 'l', 'cl', 'dl'
   44x extraneous field 'notes'
```


동일한 서빙 스택에서 두 실행 결과를 비교하면 다음과 같습니다:

| IFStruct 그룹 | base | GRPO-tuned | Δ |
| :---- | :---- | :---- | :---- |
| **전체** | 22.6% | **29.7%** | **\+7.1** |
| JSON | 18.0% | 31.9% | \+13.9 |
| YAML | 27.2% | 27.5% | \+0.3 |
| Wrapper key | 28.5% | 29.7% | \+1.2 |
| Bare list | 16.6% | 29.7% | \+13.1 |

향상된 부분은 학습이 목표로 한 지점과 정확히 일치합니다. JSON 통과율은 거의 14포인트 상승한 반면(18.0% → 31.9%), YAML은 대부분 동일하게 유지됩니다. 이는 여전히 [Qwen3.5-2B score of 33.15%](https://www.liquid.ai/blog/ifstruct-v1.0)보다 낮지만, 가벼운 작업별 미세 조정만으로도 소형 모델을 더 큰 모델에 근접시킬 수 있음을 보여줍니다.

## 결론 {#section-5}

약 500개의 샘플과 100 step으로 짧게 GRPO를 실행하면, 350M 파라미터의 소형 모델 성능을 IFStruct에서 22.6%에서 29.7%로 끌어올릴 수 있습니다. 핵심은 저렴한 작업별 보상 신호가 소형 모델의 *형식* 관련 신뢰성을 크게 높여, 몇 배 더 큰 모델과의 격차를 상당 부분 줄일 수 있다는 점입니다.

이 작업을 재현하거나 확장하려면 원본 [IFStruct v1.0 blog post](https://www.liquid.ai/blog/ifstruct-v1.0), [Liquid4All/ifstruct](https://github.com/Liquid4All/ifstruct) 벤치마크 repo, [LiquidAI/ifstruct-v1.0](https://huggingface.co/datasets/LiquidAI/ifstruct-v1.0) 데이터셋을 참고하세요.
