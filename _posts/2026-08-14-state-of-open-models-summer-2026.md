---
layout: post
title: "오픈 모델 현황: 2026년 여름 동향"
description: "2026년 1~8월 Hugging Face Hub 데이터를 바탕으로 오픈 모델 생태계를 살펴본 보고서입니다. 중국 연구소의 프런티어 규모 확대, 관심과 채택의 괴리, 라이선스, Qwen 파생 생태계, 소형 모델의 실용 계층, 에이전트라는 새로운 사용자를 다룹니다."
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/state-of-open-models-summer-2026/thumbnail.png
image: assets/images/blog/posts/2026-08-14-state-of-open-models-summer-2026/thumbnail.png
authors:
  - user: AdinaY
  - user: multimodalart
  - user: irenesolaiman
slug: "state-of-open-models-summer-2026"
source_url: "https://huggingface.co/blog/state-of-open-models-summer-2026"
source_published_date: "2026-08-14"
source_published_at: "2026-08-14T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [State of Open Models: Summer 2026 Observations](https://huggingface.co/blog/state-of-open-models-summer-2026)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/state-of-open-models-summer-2026 -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 오픈 모델 현황: 2026년 여름 동향

AI 세계에서 시간은 빠르게 흘러가는 듯합니다. 생태계 전반을 살펴본 연 2회 분석의 봄 보고서를 발표한 지 불과 몇 달이 지났지만, 이번 여름까지 관찰한 결과에서도 여러 가지 새로운 사실을 확인했습니다. 이 보고서는 2026년 1월부터 8월까지의 관찰을 정리하고, 각각을 뒷받침하는 데이터를 제시합니다.

![작업 카테고리별 Hugging Face 데이터셋 누적 증가, 2026년 100만 개 도달](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/dataset-growth.png)

Hugging Face Hub의 모델과 데이터셋은 매일 증가하고 있습니다. 공개 모델 저장소는 해당 기간에 243만 개에서 296만 개로 증가했고, 데이터셋은 71만 1천 개에서 100만 개로, Spaces는 100만 개에서 144만 개로 증가했습니다. 아래의 분포는 여전히 극단적이며, 대략 85.6%의 모델은 누적 다운로드 수가 200회 미만이고, 저장소의 1.5%가 전체 다운로드의 99.2%를 차지합니다. 이하의 모든 내용은 이러한 분포 안에서 일어납니다.

**1\. 프런티어는 빠르게 움직이고 있다**

**과거에는 명확한 발전 경로가 있었습니다. 연구소들은 작은 모델부터 공개하고 점차 규모가 큰 모델로 나아갔습니다. 2026년에는 다수의 중국 연구소가 이 발전 과정을 완전히 건너뛰었습니다.**

![2026년 월별 중국·미국 연구소의 최대 규모 오픈 모델 공개](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/frontier-ceiling-by-country.png)

**2026년 거의 매달, 중국 연구소가 공개한 가장 크고 성능이 뛰어난 오픈 모델은 미국 연구소가 공개한 어떤 모델보다도 컸습니다.** 중국의 월별 최대 규모는 7,540억 개에서 2조 7,800억 개 파라미터 사이였고, 미국 모델은 7개월 중 5개월 동안 1,300억 개 미만에 머물렀습니다. 예외는 5월과 6월의 NVIDIA [Nemotron 3 Ultra](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) 5,610억 개와 Thinking Machines Lab의 [Inkling](https://huggingface.co/collections/thinkingmachines/inkling)이었습니다.

![연구소마다 다른 규모 전략](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/lab-size-strategy.png)

이 차트는 연구소들을 두 진영으로 나눕니다. [Moonshot](https://huggingface.co/moonshotai), [MiniMax](https://huggingface.co/MiniMaxAI), [Xiaomi](https://huggingface.co/XiaomiMiMo)와 [Z.ai](http://Z.ai)는 700억 개 미만의 모델은 거의 공개하지 않으므로, 개발자가 이 연구소들을 처음 접할 때는 자신이 가진 어떤 장비에서도 실행할 수 없을 만큼 큰 모델을 만나게 됩니다. 반면 [Tencent](https://huggingface.co/tencent)와 [Alibaba Qwen](https://huggingface.co/Qwen)은 10억 개 미만부터 시작해 전 범위를 아우릅니다.

첫 번째 진영이 가능해진 데에는 두 가지 이유가 있습니다. 대규모 모델을 만드는 일은 더 이상 차별화 요소가 아닙니다. [Xiaomi](https://huggingface.co/XiaomiMiMo)와 [Ant Group](https://huggingface.co/inclusionAI), [Meituan](https://huggingface.co/meituan-longcat)은 올해 모두 1조 개 파라미터를 넘어섰고, 이들 중 어느 곳도 12개월 전에는 오픈 웨이트 분야에서 널리 알려진 이름이 아니었습니다. 그리고 연구소는 더 이상 도달 가능성을 위해 작은 모델을 내놓을 필요가 없습니다. 커뮤니티의 양자화(quantization) 계층이 며칠 안에 큰 모델도 실행 가능하게 만들어 주기 때문입니다. 이 의존 관계는 아래에서 다시 다룹니다.

결국 규모 구성은 능력보다는 의도를 보여주는 지표가 됩니다. 프런티어 전용 포트폴리오는 벤치마크(benchmark) 순위와 API 수요에 모든 것을 거는 반면, 전 스펙트럼 포트폴리오는 개발자들이 표준으로 삼는 모델 계열이 되겠다는 승부수입니다. 둘 다 합리적입니다. 다만 노리는 것이 다를 뿐입니다.

미국이 오픈 소스에 빠져 있는 것은 아닙니다.

![새로운 자체 개발 모델](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/new-homegrown-models.png)

올해 가장 많은 새로운 오픈 모델을 공개한 두 조직은 하드웨어를 만드는 회사이기도 합니다. 바로 [AMD](https://huggingface.co/amd)와 [NVIDIA](https://huggingface.co/nvidia)입니다. 두 회사는 각각 200개가 넘는 새로운 모델 저장소를 공개해 다른 경쟁자들을 크게 앞질렀고, [LiquidAI](https://huggingface.co/LiquidAI)가 약 100개로 3위를 차지했습니다. 하드웨어 업체들은 오픈 모델이 칩을 파는 수단이라는 것을 깨달았습니다. 자사 하드웨어에 최적화되어 무료로 제공되는 모델은 하드웨어가 제대로 동작한다는 가장 명확한 증거이기 때문입니다.

작은 모델과 임베딩 모델까지 포함하면, Google, Microsoft, IBM Granite, OpenAI의 구형 비전·음성 모델이 매년 수억 회의 다운로드를 기록하고 있는 가운데, **미국의 오픈 소스 AI는 성장하고 있습니다.**
NVIDIA와 같은 하드웨어·인프라 조직이 경쟁력 있는 모델을 학습시켜 오픈 웨이트로 공개하는 사례도 늘고 있습니다. NVIDIA의 Nemotron 모델 계열은 높은 성능을 자랑합니다. Meta와 같은 오랜 선두 주자들도 Meta의 [Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B)로 오픈의 뿌리를 되살리고 있습니다.


하지만 무게중심은 이동했습니다. Google과 Meta는 이제 새로운 모델 공개 수에서 NVIDIA보다 훨씬 뒤처지고 있는데, 이는 이전 몇 년간 오픈 모델 공개를 정의했던 회사들이라는 점을 생각하면 눈에 띄는 변화입니다. Meta가 플래그십 모델을 비공개로 전환하는 움직임도 이러한 변화를 더욱 부각합니다. 오픈 소스는 모델 연구소에서 **하드웨어·인프라 기업**으로 이동했습니다.

프런티어 규모에서는 상황이 크게 다릅니다. 올해 1,000억 개를 넘는 미국 모델 공개 중 일부는 새로운 모델이 아니라 중국 모델을 기반으로 만들었거나 중국 연구소의 결과물을 활용한 것으로, Thinking Machines의 [Inkling](https://huggingface.co/collections/thinkingmachines/inkling) (9,520억 개)이 대표적입니다. 주요 독자적인 미국 모델로는 NVIDIA의 [Nemotron 3 Ultra](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) (5,610억 개), [Nemotron 3 Super](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) (1,240억 개), Arcee AI의 [Trinity-Large](https://huggingface.co/collections/arcee-ai/trinity-large-thinking) (3,990억 개)가 있습니다.

AMD는 많은 변환 모델을 기여했습니다. 이 작업은 중요합니다. 1조 개 규모의 중국 모델을 미국 하드웨어에서 효율적으로 실행할 수 있게 해주기 때문입니다. 이는 모델을 만드는 일이 아니라 **유통·최적화 계층**에 해당합니다.

한편, 중국의 오픈 모델은 중국 내 자국 칩에 최적화되는 경우가 늘고 있습니다. 모델이 특정 하드웨어 생태계에 맞춰 설계되는, 정반대 방향의 경쟁이 벌어지고 있는 것입니다.

## **2\. 관심 ≠ 채택** {#section-1}

올해 누적 다운로드 기준 상위 25개 모델 저장소와 좋아요 기준 상위 25개를 뽑아봤습니다. **두 목록에 모두 등장하는 저장소는 단 하나뿐이었습니다.**

**![관심과 사용량은 서로 다른 경제](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/attention-vs-usage.png)**

다운로드는 누적치가 아니라 해당 기간 안의 수치로 집계했기 때문에, 오래 존재했다는 이유만으로 가산점을 받는 일은 없습니다. 게다가 출시 시기를 통제하면 그 차이는 더 뚜렷해집니다. 2026년에 공개된 모델 중 다운로드 상위 25위에 든 모델은 하나도 없고, 25개 중 13개는 2022년 출시작입니다. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)는 7개월 동안 좋아요 5,156개에 대해 15억 5천만 회 다운로드됐고, [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)는 좋아요 1개당 약 60회 다운로드됐습니다.

두 숫자는 서로 다른 행위를 기록합니다. 좋아요는 어떤 공개가 중요하다는 신호로, 출시 후 몇 주 안에 프런티어 모델에 몰립니다. 다운로드는 어떤 모델이 정해진 일정에 따라 돌아가는 파이프라인에 연결되어 있다는 신호로, 수년에 걸쳐 작고 안정적인 모델에 쌓입니다. 좋아요는 해당 분야가 무엇에 열광하는지를 읽는 데 적합한 지표이고, 다운로드는 현재 무엇에 의존하고 있는지를 읽는 데 적합합니다. 둘 중 하나를 다른 하나의 대리 지표로 삼는 것은 Hub에 대한 보도에서 가장 흔히 보는 실수이며, 저희의 과거 작업도 예외는 아닙니다. 같은 갈림길은 퍼블리셔 차원에서도 나타납니다.

![누가 무엇을 다운로드하는가](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/downloads-by-lab-and-model-size.png)

중국 프런티어 연구소는 Hub에서 대형 모델이 다운로드의 대부분을 차지하는 유일한 계정들입니다. 사실상 MiniMax의 2026년 다운로드는 모두 700억 개 이상 모델에서 나왔고, Moonshot 88%, DeepSeek 55%, Z.ai 39%가 뒤를 잇습니다. 미국의 대형 계정은 어느 곳도 이런 모습을 보이지 않습니다. Google, Microsoft, IBM Granite는 2026년 다운로드 중 700억 개 이상 모델에서 나온 것이 사실상 없으며, NVIDIA와 Meta도 각각 14%와 9%에 그칩니다.

총 다운로드 수로 보면 차이는 더 분명해집니다. Moonshot의 프런티어 전용 포트폴리오는 올해 3,700만 회의 다운로드를 기록한 반면, Qwen은 모델 크기 전반에 걸친 폭넓은 공개 전략으로 20억 4,500만 회(파라미터 수가 명시된 저장소 기준, 모든 저장소 포함 시 20억 6,100만 회)에 도달해 약 55배의 격차를 보였습니다. 2조 4,000억 개 파라미터의 Qwen 3.8 Max부터 270억 개와 같은 작은 변형까지 이어지는 계열의 지속적인 확장은 다양한 사용 사례를 아우르려는 같은 방향성을 보여줍니다.

시간도 중요한 역할을 합니다. 대부분의 모델은 출시 후 사용량이 급격히 감소했다가, 이후 오랫동안 안정적인 활동을 이어가는 긴 꼬리를 보입니다. 모델의 채택 여부는 대체로 출시 후 처음 몇 달 안에 결정됩니다.

이 점은 오늘날의 다운로드 규모가 최신 공개작이 아니라, 오랜 시간에 걸쳐 확립된 인프라가 된 소수의 모델에 의해 좌우되는 경우가 많은 이유를 설명해 줍니다.

## **3\. 오픈 웨이트가 가치를 쌓는 위치를 바꾼다** {#section-2}

프런티어 모델이 라이선스 사업이라면, 가장 큰 공개작에 가장 엄격한 조건이 붙을 것이라 예상할 수 있습니다. 하지만 아래 데이터는 다른 이야기를 보여줍니다.

![라이선스는 비즈니스 모델이 아니다](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/model-licenses-by-region-and-size.png)

올해 200억 개를 넘는 중국 공개 모델 178건 중 59%는 Apache 2.0, 22%는 MIT 라이선스를 달고 있으며, **비상업적 제한이 있는 경우는 거의 없습니다**. 하지만 최근 몇 주 사이, 정말 큰 모델들에서는 이러한 흐름에 변화가 나타나기 시작했습니다. [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)와 [Qwen 3.8 2.4T](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)가 라이선스에 비상업적 제한과 수익 배분 요구 조건을 포함하기 시작한 것입니다.

DeepSeek과 Z.ai는 7,000억 개에서 1조 6,500억 개 규모의 모델을 순수 MIT 라이선스로 공개합니다. 중국 연구소는 가장 큰 모델에도 가장 작은 모델과 거의 같은 수준으로, 미국 연구소가 자사 모델에 적용하는 것보다 더 관대하게 라이선스를 부여합니다. 같은 크기 구간에서 미국 쪽은 29%가 Apache 또는 MIT, 41%가 맞춤형 조건, 30%는 아무것도 명시하지 않았습니다.

이러한 공개가 무엇을 위한 것이든, 라이선스 수익은 아닙니다. 가중치는 가장 관대한 조건으로 무료 제공됩니다. 그 대가는 다른 곳에서 나와야 합니다. **API·클라우드 사업, 하드웨어·플랫폼 포지셔닝, 또는 생태계 내 위치 그 자체입니다.** 예를 들어 Z.ai와 Kimi의 기업 가치는 효과적인 오픈 소스 전략이 커뮤니티에서 주목과 성장 기회를 얻는다는 점을 보여줍니다. 다만 앞으로 업계는 오픈 소스 채택에서 더 명확한 수익화 경로로 이동할 가능성이 높습니다.

## **4\. Qwen이 커뮤니티의 기반 모델이 되다** {#section-3}

모델의 생태계 내 위치는 그 자체의 공개작만으로 정해지지 않습니다. 커뮤니티가 그 위에 얼마나 많이 쌓아 올리는지도 중요합니다. 앞서 언급했듯, Qwen은 주목과 채택을 동시에 얻고 있는 예외적인 사례입니다.

![조직별 Hugging Face 파생 모델](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/derivatives-by-organization.png)

자료: Hugging Face

이러한 기준으로 보면 Qwen은 오픈 모델 생태계에서 가장 큰 기반 중 하나가 되었습니다. Qwen 기반 모델은 현재 Hub에서 **151,448개의 파생 모델(derivative)**을 차지하고 있으며, 이는 Meta 전체 규모의 2.6배, Llama 저장소만 놓고 보면 4.7배에 달합니다. Google이 82,506개의 파생 모델로 뒤를 잇습니다. 세 번째로 큰 출처는 Unsloth로, 양자화 모델과 미세 조정(fine-tuning)용 빌드를 공개하는 커뮤니티 계정이며, 그중 다수가 Qwen 생태계를 더욱 확장하고 있습니다.

Qwen 파생 모델은 2026년 첫 7개월 동안 하루 약 **180~210개의 새로운 저장소**씩 증가했는데, 이는 채택이 개별 공개작만으로 이뤄지는 것이 아님을 보여줍니다. Qwen은 개발자가 어떤 모델을 미세 조정하고 배포할지 결정할 때 기본 워크플로의 일부가 되었습니다.

이러한 위치를 만든 요인은 여러 가지입니다. **첫째, 일관성입니다.** Qwen은 가끔 플래그십 모델을 공개하는 데 의존하지 않고, 정기적인 공개 주기를 유지하며 모델 계열을 계속 업데이트해 왔습니다. **둘째, 적용 범위입니다.** 다양한 크기와 사용 사례에 걸쳐 모델을 공개하므로, 개발자는 작은 로컬 모델이 필요하든 더 큰 배포용 모델이 필요하든 같은 생태계 안에 머무를 수 있습니다. **셋째, 개방성입니다.** Apache 2.0 라이선스는 수정, 재배포, 상업적 이용의 마찰을 줄여줍니다.

이러한 요인들은 서로를 강화합니다. 폭넓은 모델 계열은 더 많은 개발자를 끌어들이고, 더 많은 개발자는 더 많은 파생 모델을 만들며, 그 파생 모델들은 생태계를 미래 사용자에게 더욱 매력적으로 만듭니다.

이러한 위치는 대부분 커뮤니티가 만들었습니다. **151,448개의 파생 모델**은 Qwen 자체가 공개한 것이 아니라 다른 개발자들이 만든 하위 작업물입니다. Hub에 있는 Qwen 모델의 **28,531개 GGUF 변환** 중에서도 Qwen이 직접 공개한 것은 54개에 불과합니다.

## **5\. 소형 모델이 여전히 실용 계층이다** {#section-4}

파라미터 수를 명시한 모델 중 10억 개 미만 모델이 역대 전체 다운로드의 83%를 차지하고, 1,000억 개 초과 모델은 1%에 불과합니다. 2026년 누적 다운로드로 범위를 좁혀도 달라지지 않습니다. 전체 다운로드의 3%만이 700억 개 초과 모델에서 나옵니다. 이는 3월 보고서의 발견 중 가장 깔끔하게 유지된 것으로, 이유는 이전과 같습니다. 작은 모델만이 대부분의 개발자가 실제로 가진 하드웨어에서 실행되기 때문입니다.

![다운로드는 여전히 소형 모델의 몫](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/downloads-by-model-size.png)

그렇다면 1조 개 규모의 모델은 어떻게 사람들에게 도달할까요? llama.cpp를 통해서입니다.

2월에는 ggml 팀이 [Hugging Face에 합류했습니다](https://huggingface.co/blog/ggml-joins-hf). 프로젝트는 완전한 오픈 소스, 커뮤니티 주도, 동일한 기술 방향을 유지하고 있습니다. 달라진 점은 로컬 추론에서 가장 중요한 프로젝트가 이제 든든한 자원을 등에 업게 되었다는 것입니다.

![2026년 Hub에서의 llama.cpp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/llama-cpp-hub-growth.png)

llama.cpp와 함께 상한선도 이동했습니다. 7월 스냅샷에는 DeepSeek-V4-Flash의 GGUF 빌드(약 2,840억 개 파라미터)와 Kimi-K3(약 2조 8,000억 개)가 포함되어 있습니다. 로컬 추론은 예전에는 노트북에서 실행하는 80억 개 모델을 의미했습니다. 이제는 몇 대의 일반 소비자용 컴퓨터에 분산된 1조 개 규모의 전문가 혼합(MoE) 모델을 의미합니다. 이는 프런티어가 1년 전에는 갖지 못했던 대안 경로이며, 프런티어 우선 공개 전략이 애초에 가능한 이유입니다.

![사람들이 로컬에서 실제로 실행하는 것](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/local-downloads-by-model-family.png)

그리고 그 경로는 Qwen 위에서 이어집니다. 월 3,960만 회의 GGUF 다운로드를 기록해 Gemma의 2,080만 회보다 거의 두 배, Llama의 750만 회보다 다섯 배 이상 많습니다. Llama의 격차는 공급의 문제가 아닙니다. Llama 파생 GGUF 저장소 수는 Qwen보다 오히려 조금 더 많습니다. 같은 진열 공간, 5분의 1의 트래픽입니다.

모델 저장소는 이 7개월 동안 21.5% 성장했습니다. 그 주변의 여러 요소는 그보다 몇 배 빠르게 성장했습니다.

![런타임 계층이 가장 빠르게 성장 중](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/runtime-layer-growth.png)

gguf 라이브러리를 명시한 저장소는 464%, lerobot 194%, Apple의 mlx는 148% 증가한 반면, transformers와 peft는 16%, diffusers는 21% 증가했습니다. 모델링 코어는 대략 플랫폼 평균 수준으로 성장하고 있습니다. 모델이 물리적으로 어디에서 실행될 수 있는지를 결정하는 계층, 즉 로컬 추론 포맷, Apple 실리콘, 로봇 제어 스택은 그보다 3~7배 빠르게 성장하고 있습니다.

10대 모델 계열을 살펴보면, 해당 모델 뒤에 있는 연구소들은 공식 GGUF 변환을 거의 공개하지 않습니다. 하지만 GGUF 버전은 로컬에서 모델을 실행하는 개발자들이 실제로 사용하는 버전인 경우가 많습니다. 출시 시 공식 변환을 제공하고, 양자화 선택을 문서화하고, 아티팩트에 서명하는 일은 그리 큰 추가 작업이 들지 않을 것입니다. 이러한 워크플로를 내부에서 유지하기보다, 연구소는 Unsloth와 같은 기존 생태계 기여자와 협력할 수 있습니다. 그렇게 하면 모델 제작자가 테스트한 가중치와 더 넓은 커뮤니티가 채택한 버전 사이의 격차를 좁힐 수 있습니다.

## **6\. 에이전트가 새로운 사용자다** {#section-5}

이 섹션은 3월에는 쓸 수 없었을 것입니다. 측정 도구가 존재하지 않았기 때문입니다. 7월에 공개된 [agent-usage](https://huggingface.co/datasets/huggingface/agent-usage) 데이터셋은 코딩 에이전트가 huggingface\\_hub나 hf CLI를 통해 Hub를 호출할 때 보내는 agent/\<name\> 토큰을 기록합니다. 모델 검색, 데이터셋 푸시, Jobs 실행, Spaces 생성이 여기에 해당합니다. 이를 통해 Hub가 받는 에이전트 트래픽의 규모와 그 출처인 하네스를 처음으로 확인할 수 있게 되었습니다.

![Hugging Face Hub를 호출하는 에이전트](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/state-of-open-models-summer-2026/agent-hub-traffic.png)
Claude Code는 7월에 44.4%로 선두를 차지했지만, 한 달 수치만으로는 진짜 발견을 알 수 없습니다. 4월에는 67.8%, 5월에는 64%를 차지했고, Codex는 10.4%에서 20.8%로 꾸준히 상승했습니다. 한 번의 공개나 기본값 변경 한 번으로 한 달 만에 트래픽의 절반이 이동할 수 있는, 확고한 선두 주자가 없는 시장입니다.

두 번째 발견은 미등록 행입니다. 7월 에이전트 태그 트래픽의 거의 4분의 1은 데이터셋에 아직 이름이 등록되지 않은 하네스에서 나왔고, 5월에는 그 비율이 59.8%였습니다. 4월에서 7월 사이 12개가 넘는 새로운 클라이언트 식별자가 등장했습니다. 새로운 진입자가 어떤 등록부보다 빠르게 등장하고 있다는 것, 그 자체가 발견입니다.

저희는 올해 많은 시간을 이 독자, 즉 사람의 브라우저만을 위한 것이 아니라 에이전트를 위해 만드는 데 썼습니다. 3월에는 Papers가 기계가 읽을 수 있는 Markdown을 제공하기 시작했습니다. 4월에는 [일급 데이터셋 유형으로서의 에이전트 트레이스](https://huggingface.co/changelog/agent-trace-viewer)와 모든 Gradio Space의 agents.md 엔드포인트가 추가되어, 에이전트가 Space의 API를 읽고 직접 호출할 수 있게 되었습니다. 7월에는 MCP 서버에 [hf\_fs 도구](https://huggingface.co/changelog/mcp-improvements-jul-26)가 추가되어, 저장소, 스토리지, 문서, 페이퍼를 1,000여 개 토큰의 단일 인터페이스로 노출하고, 안전한 실행을 위한 연결 가능한 샌드박스도 함께 제공합니다. 프로토콜 계층에서도 같은 통합이 일어났고, MCP는 Linux Foundation 산하의 Agentic AI Foundation으로 이동했습니다.

그러다 7월, 에이전트는 읽는 존재에서 그치는 것을 멈추고 침입자가 되었습니다. 자율 에이전트가 스스로의 판단으로 지속적인 침입을 수행한 것으로 보이는 첫 번째 문서화된 사례가 저희에게 일어났습니다. 저희 팀이 확보한 공격 코드를 분석하기 위해 최첨단 비공개 모델을 사용하려 했을 때, 해당 모델들의 안전 가드레일이 작업을 거부했습니다. 분석은 결국 저희 인프라에서 실행된 양자화된 오픈 모델 GLM-5.2로 완료되었습니다. 저희는 [공개문](https://huggingface.co/blog/security-incident-july-2026)과 [전체 기술 타임라인](https://huggingface.co/blog/agent-intrusion-technical-timeline)을 게시했습니다.

## **전망** {#section-6}

봄 보고서와 비교하면, 권력의 지리적 재편은 계속 가속화되고 있습니다. 미국의 오픈 소스 모델이 여전히 경쟁력을 유지하는 가운데, 여러 중국 프런티어 모델 연구소 간의 경쟁이 큰 주목을 받고 있습니다. 이러한 프런티어 모델에 달린 많은 좋아요는 커뮤니티를 가장 흥분시키는 것이 무엇인지, 그리고 주목을 기업 가치 상승의 기회로 활용하는 회사들의 성장 가능성을 가리킵니다.

하지만 AI 경쟁은 단거리 경주만이 아니라 마라톤이기도 합니다. llama.cpp 같은 도구는 큰 모델을 로컬에 배포하는 데 도움이 되지만, 폭넓은 모델 계열과 그 채택이야말로 개발자, 퍼블리셔, 미래 사용자 사이의 선순환을 만드는 핵심입니다. 인프라에 내장되고 생태계의 일부가 되는 모델은 터널 끝에서 상업적으로 건실한 결실로 이어질 수 있습니다.

결국, 에이전트가 처음으로 HF Hub의 1위 사용자가 되면서, 다음 보고서는 매우 달라 보일 수 있습니다.

AI에서는 몇 달 만에 생태계가 완전히 바뀔 수 있습니다.

---

### **방법론 노트**

이 분석은 2026년 첫 7개월 동안 Hugging Face Hub에서 관찰된 활동을 바탕으로 합니다.

다운로드, 좋아요, 파생 모델, 모델 공개를 포함한 이 보고서의 지표는 생태계 활동의 서로 다른 측면을 나타냅니다. 이를 모델 품질, 상업적 채택, 전체 시장 점유율의 직접적인 척도로 해석해서는 안 됩니다.

다운로드는 Hub 생태계 내의 사용량을 보여주지만, API 사용량, 비공개 배포, 다른 채널을 통한 배포 모델은 포착하지 못합니다.

좋아요는 커뮤니티의 관심과 주목을 반영하고, 파생 모델은 개발자가 기존 모델 위에 얼마나 많이 구축하는지를 보여주는 신호입니다.

오픈 소스 AI의 채택은 다양한 채널에서 일어나므로, Hub 활동은 AI 시장 전체를 측정하는 완전한 지표라기보다 생태계 발전을 바라보는 하나의 관점으로 봐야 합니다.

---

### **편집자 주**

이 글은 8월 초 최신 공개 내용을 반영하여 편집되었습니다.
