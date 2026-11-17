---
layout: post
title: "오픈 웨이트 모델을 쉽게 실행할 수 있도록 Hugging Face와 OpenRouter가 협력합니다"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/openrouter/thumbnail.jpg
image: assets/images/blog/posts/2026-11-16-openrouter/thumbnail.jpg
authors:
  - user: celinah
  - user: victor
  - user: kramp
  - user: Wauplin
  - user: pierric
  - user: coyotte508
  - user: julien-c
slug: "openrouter"
source_url: "https://huggingface.co/blog/openrouter"
source_published_date: "2026-11-16"
source_published_at: "2026-11-16T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Hugging Face and OpenRouter join forces to make open-weight models easy to run](https://huggingface.co/blog/openrouter)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/openrouter -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 오픈 웨이트 모델을 쉽게 실행할 수 있도록 Hugging Face와 OpenRouter가 협력합니다

Hugging Face와 OpenRouter가 오픈 웨이트 모델을 쉽게 실행할 수 있도록 협력합니다. 모든 주요 오픈 웨이트 릴리스는 이미 Hugging Face Hub에 게시되고 있으며, [OpenRouter](https://openrouter.ai)는 이러한 모델을 대규모로 제공하고 각 provider의 실시간 측정값을 바탕으로 80개가 넘는 provider에 요청을 분산합니다. 이제 양사의 협력을 통해 Hub의 모든 오픈 웨이트 모델 중 provider가 제공하는 모델을 Hugging Face 토큰으로 동일한 API를 사용해 호출할 수 있으며, 각 요청에는 가장 적합한 provider가 선택됩니다.

오늘부터 [Inference Providers](https://huggingface.co/docs/inference-providers)를 자체 custom router에서 OpenRouter로 전환합니다. 이제 언어 모델에 대한 요청은 OpenRouter를 거치며, OpenRouter는 실시간 가격, 속도 및 신뢰성을 기준으로 각 호출에 가장 적합한 provider를 선택하고 첫 번째 provider가 실패하면 다른 provider로 재시도합니다. 코드는 그대로 유지됩니다.

## 이미 사용 중인 도구로 실행하기 {#section-1}

router는 provider에 관계없이 모든 chat model에 대해 OpenAI의 Chat Completions 및 Responses API와 Anthropic의 Messages API를 지원합니다. 코딩 agent 또는 SDK와 모델을 선택한 다음 단계를 복사하거나, 전체 설정을 agent에 전달할 prompt로 복사하면 됩니다:

<iframe src="https://huggingface-inference-providers-setup.static.hf.space" width="100%" height="560" frameborder="0"></iframe>

[Claude Code](https://huggingface.co/docs/inference-providers/integrations/claude-code), [Codex](https://huggingface.co/docs/inference-providers/integrations/codex), [OpenCode](https://huggingface.co/docs/inference-providers/integrations/opencode), [Pi](https://huggingface.co/docs/inference-providers/integrations/pi) 및 [Hermes Agent](https://huggingface.co/docs/inference-providers/integrations/hermes-agent) 설정 가이드

동일한 Hugging Face 토큰으로 [Storage Buckets](https://huggingface.co/storage)를 통한 스토리지, [Jobs](https://huggingface.co/docs/hub/jobs-overview)를 통한 컴퓨팅, [Inference Endpoints](https://huggingface.co/docs/inference-endpoints)를 통한 전용 추론도 이용할 수 있습니다. PRO, Team 및 Enterprise 플랜에는 컴퓨팅과 추론에 사용할 수 있는 월간 크레딧이 포함됩니다.

## 오픈 웨이트 모델은 어떻게 사용되는가 {#section-2}

OpenRouter에서 제공된 토큰 중 오픈 웨이트 모델이 차지하는 비중은 2025년 10월 26.8%에서 2026년 9월 말 71.7%로 증가했으며, 7월에는 77.5%로 정점을 기록했습니다([OpenRouter, October 2026](https://openrouter.ai/blog/insights/open-weight-vs-closed-weight-models/)). 같은 기간 주간 오픈 웨이트 사용량은 60배 이상 증가한 반면, closed model은 거의 10배 증가했습니다. 이번 주 OpenRouter에서 가장 많이 사용된 모델은 59.2조 개의 토큰을 기록한 오픈 웨이트 모델 DeepSeek V4.1 Flash 1017입니다([rankings](https://openrouter.ai/rankings)).

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/open-weight-share@2x.png" alt="Open-weight models' share of tokens on OpenRouter, from 26.8% in October 2025 to 71.7% in September 2026"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/open-weight-share-dark@2x.png" alt="Open-weight models' share of tokens on OpenRouter, from 26.8% in October 2025 to 71.7% in September 2026"/>
</div>

오픈 웨이트 모델의 평균 비용은 약 4배 낮습니다. 입력 토큰 100만 개당 $0.28, 출력 토큰 100만 개당 $1.62로, closed model의 $1.27 및 $6.36과 비교됩니다.

최고 수준의 오픈 웨이트 모델도 여전히 가장 어려운 benchmark에서는 최고 수준의 proprietary model에 뒤처집니다. [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/leaderboards/models)에서 최고의 오픈 웨이트 모델인 Kimi K3.1 Pro는 46점을 기록한 반면, Claude Fable 5.2는 65점을 기록했습니다. 작업당 비용은 약 7분의 1입니다($3.68 대 $26.54). 많은 workload에서는 이러한 절충이 충분히 가치 있습니다.

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/intelligence-vs-cost@2x.png" alt="Artificial Analysis Intelligence Index against cost per task, for the top proprietary and open-weight models"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/intelligence-vs-cost-dark@2x.png" alt="Artificial Analysis Intelligence Index against cost per task, for the top proprietary and open-weight models"/>
</div>

## OpenRouter 기반 라우팅 {#section-3}

OpenRouter는 80개가 넘는 provider의 600개가 넘는 모델을 통해 1,000만 명이 넘는 사용자에게 매월 800조 개가 넘는 토큰을 제공합니다. 라우팅은 해당 트래픽에서 모든 provider를 실시간으로 측정한 결과를 기반으로 합니다.

OpenRouter는 최근 [provider 라우팅을 재구축했습니다](https://openrouter.ai/blog/announcements/how-we-rebuilt-provider-routing/). 이제 단순히 가장 낮은 표시 가격만 확인하는 대신, 가격, 속도 및 신뢰성의 조합이 가장 우수한 provider로 각 요청을 보냅니다.

provider의 속도가 느려지거나 오류가 발생하기 시작하면 복구될 때까지 해당 provider에서 트래픽을 이동시키며, 실패한 시도는 동일한 호출 내에서 다음으로 적합한 provider에 재시도합니다. OpenRouter의 측정에 따르면 새로운 라우팅으로 capacity error가 46% 감소했으며, 첫 토큰까지 걸리는 시간은 16.9%, 처리량은 16.3% 개선되었습니다.

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/routing@2x.gif" alt="Default routing retries on another provider when the first one is rate limited, :cheapest picks the lowest price and :fastest the highest throughput"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/routing-dark@2x.gif" alt="Default routing retries on another provider when the first one is rate limited, :cheapest picks the lowest price and :fastest the highest throughput"/>
</div>

이전에는 Inference Providers가 요청에 대해 하나의 provider를 선택하고 해당 provider가 실패하면 오류를 반환했습니다. 이제 요청은 다른 provider에서 재시도됩니다. 모델 id에 suffix를 추가해 policy 또는 provider를 계속 선택할 수 있습니다:

```text
zai-org/GLM-5.3            # default: best mix of price, speed and reliability
zai-org/GLM-5.3:fastest    # highest throughput
zai-org/GLM-5.3:cheapest   # lowest price
zai-org/GLM-5.3:novita     # only Novita
```


라우팅은 cache도 인식합니다. conversation은 prompt cache를 보유한 provider에 유지됩니다. 이를 명시하려면 `X-HF-Session-Id` header에 conversation마다 고정된 UUID를 보내세요([prompt caching guide](https://huggingface.co/docs/inference-providers/prompt-caching) 참조).

## 가격 {#section-4}

가격은 변경되지 않습니다. Hugging Face 청구서에는 provider의 가격이 청구되며, Hugging Face는 여기에 markup을 추가하지 않습니다. PRO 사용자는 매월 $2의 크레딧을 받고, Team 및 Enterprise 조직은 seat당 $2를 받습니다. 자세한 내용은 [Pricing and Billing](https://huggingface.co/docs/inference-providers/pricing)을 참조하세요.

## 이미지, 비디오 및 오디오 생성 {#section-5}

이미지, 비디오 및 오디오 생성은 OpenRouter를 거치지 않고 이전과 정확히 동일하게 Hugging Face Inference Providers를 통해 계속 작동합니다. 동일한 token과 API를 계속 사용할 수 있으며, 수만 개의 [LoRAs](https://huggingface.co/models?inference_provider=all&pipeline_tag=text-to-image&other=lora)에 액세스할 수 있습니다.

## 요청이 전송되는 위치 확인하기 {#section-6}

모델 페이지에서는 widget에서 바로 모델과 chat할 수 있습니다. 새로 추가된 두 탭에서는 그 이면의 정보를 확인할 수 있습니다. Providers에는 OpenRouter가 해당 모델로 라우팅할 수 있는 모든 provider와 각 provider의 입력 및 출력 가격, 최근 24시간 동안의 속도와 uptime이 표시되며, Metrics에는 provider 중 최저 가격, 처리량, 첫 토큰까지 걸리는 시간, context length 및 uptime이 표시됩니다:

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/widget-playground-3@2x.gif" alt="Chatting with Inkling-2-Small in the model page widget, opening its code snippets, then its Metrics tab"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/widget-playground-3-dark@2x.gif" alt="Chatting with Inkling-2-Small in the model page widget, opening its code snippets, then its Metrics tab"/>
</div>

[usage overview](https://huggingface.co/settings/inference-providers/overview)에서는 각 요청이 전송된 provider별로 지출 내역을 확인할 수 있습니다:

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/usage-by-provider@2x.png" alt="Daily Inference Providers spend, broken down by provider"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/usage-by-provider-dark@2x.png" alt="Daily Inference Providers spend, broken down by provider"/>
</div>

[settings](https://huggingface.co/settings/inference-providers)에서 원하지 않는 provider를 끌 수 있습니다. router는 해당 provider로 요청을 절대 보내지 않습니다. 조직에도 동일한 설정이 적용되며, 조직에 청구되는 모든 요청에 적용됩니다.

<div>
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/settings-providers@2x.png" alt="Inference Providers settings, with a toggle for each provider"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/inference-providers-openrouter/settings-providers-dark@2x.png" alt="Inference Providers settings, with a toggle for each provider"/>
</div>

## 호환성을 깨는 변경 사항: custom provider keys {#section-7}

이제 언어 모델에 대한 요청은 OpenRouter를 거치며 Hugging Face 계정 또는 조직에 청구되므로, 이러한 모델에서 provider에 대한 자체 API key를 더 이상 사용할 수 없습니다. 모델 id에 suffix를 추가해 provider를 계속 선택하거나(`model:provider`), settings에서 provider를 끌 수 있으며 코드는 그대로 유지됩니다. 이미지 및 비디오 provider에 대한 custom key는 계속 작동합니다.

## 다음 단계 {#section-8}

이는 OpenRouter와의 partnership을 위한 첫 단계입니다. 다음과 같은 작업을 진행할 예정입니다:

- OpenRouter 계정을 Hugging Face에 연결합니다.
- 더 많은 task를 OpenRouter를 통해 라우팅합니다.

[model](https://huggingface.co/models?inference_provider=all&sort=trending&other=conversational)을 사용해 보거나, 시작하려면 [Inference Providers docs](https://huggingface.co/docs/inference-providers)를 읽어보세요.
