---
layout: post
title: 모델이 AWS Trainium에서 실행되나요? 확인해 보세요
author: dailybot
categories:
- Translation
- HuggingFace
thumbnail: /blog/assets/torch-neuronx-model-support/thumbnail.jpg
authors:
- user: dacorvo
slug: experiment-smoke-trainium-model-support
source_url: https://huggingface.co/blog/torch-neuronx-model-support
source_published_date: '2026-10-23'
source_published_at: '2026-10-23T00:00:00+00:00'
locale: ko
translation_status: draft
translator: openai
description: '* TOC {:toc} <!--toc--> _이 글은 Hugging Face 블로그의 [Does Your Model Run
  on AWS Trainium?'
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Does Your Model Run on AWS Trainium? Look It Up](https://huggingface.co/blog/torch-neuronx-model-support)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/torch-neuronx-model-support -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 모델이 AWS Trainium에서 실행되나요? 확인해 보세요

[이전 게시물](https://huggingface.co/blog/hf-torchneuron)에서는 AWS Trainium용 네이티브 PyTorch 백엔드인 TorchNeuron(`torch-neuronx` 패키지)을 소개했습니다. 이를 사용하면 Transformers는 다른 모든 디바이스에서와 마찬가지로 Trainium에서 자체 모델링 코드를 실행하므로, Hub의 모든 Transformers 모델이 실행될 수 있습니다. 실제로 실행되는 모델은 무엇일까요?

[torch-neuronx model support](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support) Space는 Hub에서 가장 인기 있는 모델을 대상으로 이 질문에 답합니다. 모델이 Trainium에서 수정 없이 실행되는지 알려 주고, 직접 실행할 수 있는 스크립트도 제공합니다.

[![torch-neuronx model support](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support/resolve/main/thumbnail.jpg)](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support)

## 모델 조회 {#section-1}

Space는 **Selected** 보기로 열립니다. 이는 다운로드 수와 트렌딩 점수를 기준으로 Hub의 상위 모델 중에서 선정한 모델 목록입니다. 각 행에는 결과, 모델이 분할된 Trainium 코어 수, 실행 시간(wall time, 컴파일 포함), 최대 디바이스 메모리가 표시됩니다.

특정 모델을 조회하려면 **Trending** 또는 **Downloads** 보기로 전환한 뒤 모델 이름의 일부를 입력하세요. 이 보기에서는 모델의 테스트 여부와 관계없이 작업별로 Hub 자체 순위를 따릅니다. 모델을 의도적으로 제외한 경우에는 그 이유를 Space에서 알려 줍니다(대부분은 우선순위가 낮았기 때문이며, 때로는 실제로 호환되지 않기 때문입니다). 확인할 Neuron 컴파일러 버전(`neuronx-cc`)과 Transformers 버전도 선택할 수 있습니다.

> [!NOTE]
> 이 글을 작성하는 시점에는 Transformers 모델만 테스트되며, 선정 범위도 `text-generation` 및 `image-text-to-text` 모델로 제한되어 있습니다. 앞으로 diffusers, sentence transformers 및 기타 transformers 작업으로 범위를 확장할 계획입니다.

## 직접 실행하기 {#section-2}

모델을 선택하면 실행 방법을 확인할 수 있습니다. Space에는 해당 모델을 테스트한 Neuron 스택, 그 위에 설치해야 하는 패키지, 그리고 실행된 `run.py` 스크립트가 표시됩니다. 이 스크립트는 [TorchNeuron announcement](https://huggingface.co/blog/hf-torchneuron)의 레시피를 따릅니다. 필요할 경우 모델을 `from_pretrained`로 로드해 분할한 다음, 정적 캐시와 `torch.compile(backend="neuron")`를 사용해 생성합니다.

해당 스택이 설치된 Trainium2(`trn2`) 인스턴스에서, 예를 들어 [AWS Neuron Deep Learning Container](https://github.com/aws-neuron/deep-learning-containers#torch-neuronx)를 사용해 패키지를 설치한 다음 다음 명령을 실행합니다.

```bash
python3 run.py
```


`torchrun`는 필요하지 않습니다. 모델이 분할된 경우 스크립트가 자체적으로 `torch.distributed` rank를 시작합니다.

처음 실행하면 모델의 그래프 컴파일부터 시작합니다. 이 단계를 건너뛰려면 공개 [NEFF cache](https://huggingface.co/buckets/aws-neuron/torch-neuronx-neff-cache) 버킷에서 테스트 중 동일한 컴파일러가 빌드한 그래프를 복사하세요. Space에서 올바른 컴파일러 버전이 포함된 명령을 제공합니다.

```bash
hf buckets sync hf://buckets/aws-neuron/torch-neuronx-neff-cache/neuronxcc-<version> /tmp/neff_cache
```


## "실행된다"는 의미 {#section-3}

각 테스트에서는 `trn2` 인스턴스에 모델을 로드하고, 가중치를 저장할 수 있는 가장 적은 수의 Trainium 코어에 모델을 분할합니다. 대부분의 모델은 텐서 병렬 처리를 사용해 분할됩니다. 텐서 병렬 처리 계획이 없는 mixture-of-experts 모델은 expert parallelism만 사용합니다. 그런 다음 테스트에서 32개의 토큰을 생성하고 이를 확인합니다. 단일 코어에서는 토큰을 동일한 체크포인트를 CPU에서 실행해 얻은 토큰과 비교합니다. 모델이 분할된 경우에는 소형 judge 모델이 텍스트가 유창한 영어인지 확인합니다.

이 글을 작성하는 시점에는 4개의 Trainium2 디바이스에 들어가는 모델만 테스트되며, 이는 최대 16코어를 의미합니다. `PASS`는 처리량이나 더 긴 생성에 대해서는 아무것도 말해 주지 않습니다.

## 결과는 데이터셋입니다 {#section-4}

Space에 표시되는 모든 내용은 공개 [`aws-neuron/torch-neuronx-model-support`](https://huggingface.co/datasets/aws-neuron/torch-neuronx-model-support) 데이터셋에서 가져오며, `neuronx-cc` 및 라이브러리 버전의 각 조합마다 하나의 Parquet 파일이 제공됩니다.

각 행에는 모델의 결과, 실패 시 오류, 전체 스택의 버전, 그리고 실행된 `run.py`가 포함됩니다. 다음과 같이 직접 쿼리할 수 있습니다.

```python
import pandas as pd

df = pd.read_parquet("hf://datasets/aws-neuron/torch-neuronx-model-support/data")
df[df.model_id == "Qwen/Qwen3-0.6B"][["neuronx_cc", "library", "library_version", "neuron_cores", "result"]]
```

각 결과의 정확한 의미는 [dataset card](https://huggingface.co/datasets/aws-neuron/torch-neuronx-model-support)에서 확인할 수 있습니다.

다음에 모델이 Trainium에서 실행되는지 궁금하다면 [확인해 보세요](https://huggingface.co/spaces/aws-neuron/torch-neuronx-model-support).
