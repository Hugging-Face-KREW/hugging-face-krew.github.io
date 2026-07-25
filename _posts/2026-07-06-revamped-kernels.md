---
layout: post
title: "🤗 Kernels: 주요 업데이트"
description: "Hugging Face Kernels의 새 저장소 유형, 보안 강화, CLI 개편, 프레임워크 지원 확대를 소개합니다."
author: dailybot
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-07-06-revamped-kernels/thumbnail.png
authors:
  - user: sayakpaul
  - user: danieldk
  - user: drbh
slug: "revamped-kernels"
source_url: "https://huggingface.co/blog/revamped-kernels"
source_published_date: "2026-07-06"
source_published_at: "2026-07-06T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [🤗 Kernels: Major Updates](https://huggingface.co/blog/revamped-kernels)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/revamped-kernels -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# 🤗 Kernels: 주요 업데이트

[이전 글(From Zero to GPU)](https://huggingface.co/blog/kernel-builder)에서는 custom kernels를 패키징하고, 배포하고, 사용하는 방식을 표준화하기 위한 🤗 Kernels 프로젝트를 소개했습니다. 이 프로젝트의 목표는 Hub와 최대한 잘 맞으면서도 사용 흐름은 매끄럽고 보안은 견고하게 만드는 것입니다.

지난 몇 달 동안 우리는 이 목표를 향해 작업해 왔습니다. 그 과정에서 프로젝트도 거의 전면적으로 다시 설계했습니다. 이 글에서는 지금까지 배포한 주요 업데이트와 앞으로의 방향을 정리합니다.

**목차**

* [Kernels — 새로운 저장소 유형](#kernels--a-new-repository-type)
* [보안 개선](#improved-security)
* [CLI 개편](#revamped-clis)
* [프레임워크와 백엔드 지원 확대](#more-coverage-of-frameworks-and-backends)
* [agentic kernel 개발을 위한 기반](#foundation-for-agentic-kernel-development)
* [기타 업데이트](#misc)
* [마무리](#conclusion)

## Kernels — 새로운 저장소 유형 {#section-1}

Hub에 ["kernel"](https://huggingface.co/kernels)이라는 새로운 repository type을 도입했습니다. 이를 통해 compute 관련 요구가 구체적인 사용자들을 더 잘 지원할 수 있습니다. 예를 들어 사용자는 특정 kernel이 어떤 accelerator, operating system, backend version을 지원하는지 확인할 수 있습니다.

<figure align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/revamped-kernels/flash-attn3.png" alt="Flash Attention 3 Kernel Page" width="600"/>
    <figcaption>Kernel page: <a href="https://huggingface.co/kernels/kernels-community/flash-attn3">kernels-community/flash-attn3</a></figcaption>
</figure>

Hub에서 사용 가능한 모든 kernel은 [https://huggingface.co/kernels.](https://huggingface.co/kernels)에서 둘러볼 수 있습니다.

이 kernel들을 Hub의 first-class citizen으로 만들면 AI 생태계에도 도움이 됩니다. 이제 사용자는 kernel, model, 그리고 이를 사용하는 application 전반의 흐름을 볼 수 있습니다. 또한 kernel은 사용자에게 더 쉽게 발견될 수 있습니다.

## 보안 개선 {#section-2}

Kernel은 이를 불러오는 Python process와 같은 권한으로 native code를 실행합니다. 따라서 악성 kernel은 실제 피해를 줄 수 있습니다. 그래서 보안은 Kernels 프로젝트에서 항상 가장 중요한 요소였습니다.

이 때문에 우리는 초기부터 reproducibility에 집중했습니다. 사용자가 kernel을 직접 다시 compile하고, 그 결과가 공개된 source와 일치하는지 검증할 수 있어야 합니다. 이를 가능하게 하기 위해 Nix를 사용합니다. Nix는 build recipe를 hermetic하게 평가하고 강하게 격리된 sandbox를 사용해 build를 순수하게 유지합니다. 또한 source Git SHA1을 kernel 자체에 embedding해 provenance를 더 강화했습니다.

최근 몇 달 동안 trusted kernel publisher와 code signing이라는 추가 방어 계층도 도입했습니다.

### Trusted kernel publisher

새로운 repo type과 함께 “trusted publisher”도 도입했습니다. Kernel은 사용되는 machine에서 Python process와 같은 권한으로 code를 실행하므로, 공격자는 악성 kernel을 업로드하고 사용자가 그 kernel을 쓰도록 유도해 machine을 compromise할 수 있습니다. 이런 악성 kernel을 피할 수 있도록, 이제 `kernels` package는 기본적으로 *trusted publisher*의 kernel만 load합니다. trusted publisher는 community가 선의로 행동한다고 신뢰하는 organization입니다.

물론 trusted publisher가 아닌 organization이나 user의 kernel도 load할 수 있어야 합니다. 다만 Hub에서 kernel을 load할 때 `trust_remote_code` argument를 사용해 명시적으로 opt in해야 합니다.

```py 
from kernels import get_kernel

kernel_module = get_kernel(  
   “Atlas-Inference/gdn”, version=1, trust_remote_code=True  
)  
```


기본적으로 사용자는 Hub에 kernel repository를 publish할 수 없습니다. kernel publisher가 되려면 요청해야 합니다. user와 organization은 account settings에서 access를 요청할 수 있으며, 이를 통해 우리는 요청을 case-by-case로 검토할 시간을 확보합니다.

### Kernel signing

추가하고 있는 또 다른 보안 계층은 code signing입니다. code signing은 trusted publisher의 Hub credential이 compromise되어 공격자가 해당 publisher의 kernel repo에 악성 kernel을 업로드하는 상황을 방어합니다. code signing에서는 kernel developer만 알고 있는 private key로 kernel에 서명하고, 일반적으로 공개된 public key로 이를 검증합니다. Hub가 compromise된 상황에서도 공격자는 signing에 필요한 private key를 갖고 있지 않으므로 악성 kernel에 서명할 수 없습니다.

보안을 더 강화하기 위해 Sigstore의 cosign을 사용해 ephemeral private key로 서명합니다. 이 signing key는 제한된 시간 동안만 유효하므로, 유출되더라도 공격자가 private key를 사용하기 어렵습니다. 또한 kernel이 trusted GitHub repository의 trusted GitHub workflow에서 서명되었는지도 검증합니다.

Kernel signing은 이미 kernel-builder에서 지원되며, kernel을 검증할 수 있도록 `kernels verify-signature`도 제공합니다. 다만 Kernels는 아직 kernel load 시점에 signature를 검증하지 않습니다. 이 기능을 완전히 rollout하기 전에 더 테스트하고 싶기 때문입니다. 자체 kernel에 code signing을 설정하는 예비 안내는 kernels 0.16.0 release note에서 볼 수 있습니다. [https://github.com/huggingface/kernels/releases/tag/v0.16.0.](https://github.com/huggingface/kernels/releases/tag/v0.16.0)

## CLI 개편 {#section-3}

이전에는 여러 utility가 kernels와 kernel-builder 사이에 뒤섞여 있었습니다. 이제 kernels CLI와 kernel-builder CLI 사이의 concern을 더 명확히 분리했습니다. 여기서의 mental model은 kernels가 kernel을 load하고 사용할 준비를 하는 library라는 것입니다. 따라서 “building” kernel과 관련된 기능은 포함하지 않는 것이 맞습니다.

그 결과 `kernels`와 `kernel-builder`는 모두 훨씬 더 가볍고 목적이 분명해졌습니다. 자세한 내용은 documentation을 참고하세요.

* [kernels CLI](https://huggingface.co/docs/kernels/en/cli)  
* [kernel-builder CLI](https://huggingface.co/docs/kernels/en/builder-cli)

개선된 CLI 경험은 agentic kernel development가 부상하는 흐름에도 더 잘 대응하게 해 줍니다. 이에 대해서는 [뒤에서](#foundation-for-agentic-kernel-development) 더 설명합니다.

## 프레임워크와 백엔드 지원 확대 {#section-4}

framework 지원도 확장했습니다. 가장 눈에 띄는 변화는 다음과 같습니다.

* `kernels`와 `kernel-builder`에 Torch Stable ABI 지원을 추가했습니다. Torch Stable ABI를 사용하면 kernel developer가 특정 Torch version이나 그 이후 약 2년 동안 release되는 version을 target할 수 있습니다. 예를 들어 Torch 2.9 Stable ABI를 target하는 kernel은 Torch \>= 2.9를 지원합니다.
* Apache TVM FFI는 Torch 외에 처음으로 지원되는 framework입니다. TVM FFI는 PyTorch, JAX, CuPy 같은 다른 framework와 상호 운용되는 kernel용 standardized ABI입니다. 이를 통해 kernel developer는 여러 framework에서 동작하는 kernel을 만들 수 있습니다.

## agentic kernel 개발을 위한 기반 {#section-5}

`kernel-builder`와 `kernels`는 agent가 처음부터 최적화된 kernel을 만들어 내는 agentic kernel development의 부상을 보완합니다. 두 도구를 함께 사용하면 agent가 kernel을 scaffold하고, build하고, benchmark하고, 반복적으로 optimize하는 workflow를 지원할 수 있습니다.

Agentic kernel development는 아직 초기 단계이며, 적절한 development loop도 계속 진화할 것입니다. 그렇기 때문에 단순하고 명확한 기본기가 특히 중요합니다. 도구는 사람들이 선택하는 어떤 agent workflow나 framework에도 쉽게 조합될 수 있어야 합니다.

`kernel-builder`는 kernel source code를 어떻게 scaffold하고 reproducible build에 사용할지에 대한 구조를 강제하는 데 도움을 줍니다. 이를 통해 agent는 예측 가능한 project layout과 반복 가능한 workflow 안에서 작업할 수 있습니다. CLI 역시 [agent-optimized](https://huggingface.co/blog/is-it-agentic-enough)되도록 설계되었습니다. 예를 들어 non-interactive command와 agent가 programmatic하게 해석하기 쉬운 output을 의미할 수 있습니다. 이를 위해 서로 다른 backend의 특성을 agent가 다룰 수 있도록 [backend-specific skills](https://huggingface.co/docs/kernels/en/cli-skills)도 제공합니다. 이러한 skill은 backend별 toolchain, compilation path, performance consideration을 포착할 수 있습니다.

kernel을 성공적으로 build하는 것만이 목표는 아닙니다. target hardware에서 baseline 대비 실제 speedup을 제공하는지도 확인해야 합니다. 따라서 성공적인 build는 첫 번째 validation step일 뿐입니다. 일반적으로 target hardware에는 여러 accelerator가 포함될 수 있고, 같은 accelerator의 서로 다른 family가 포함될 수도 있습니다.

따라서 관련이 있는 경우 hardware vendor와 generation 전반에서 결과를 평가하는 것이 중요합니다. [HF Jobs와의 긴밀한 integration](https://huggingface.co/docs/kernels/en/builder/github-actions)은 이 benchmarking 과정을 쉽게 만들어 줍니다. Agent는 이 integration을 사용해 benchmark suite를 실행하고, performance result를 수집하며, 정의된 baseline과 비교할 수 있습니다.

이 방식으로 agent는 서로 다른 hardware configuration 전반에서 test를 실행해 생성된 kernel의 performance에 대한 신뢰할 수 있는 feedback을 얻고, 무엇을 해야 하는지 식별할 수 있습니다. 그 feedback은 다음 optimization iteration에 반영됩니다.

아래는 agent-augmented kernel의 몇 가지 예시입니다. 이 예시들은 이 workflow를 통해 어떤 종류의 kernel을 개발하고 평가할 수 있는지 보여줍니다.

* [https://huggingface.co/kernels/drbh/yamoe](https://huggingface.co/kernels/drbh/yamoe)   
* [https://huggingface.co/kernels/sayakpaul/qk-norm-rope](https://huggingface.co/kernels/sayakpaul/qk-norm-rope)

## 기타 업데이트 {#section-6}

### 환경 설정

`kernel-builder`로 kernel을 build하기 위한 환경 설정은 부담스러울 수 있습니다. 사용자가 더 쉽게 시작할 수 있도록, 이제 한 번의 click으로 환경을 설정할 수 있는 [installation script](https://huggingface.co/docs/kernels/en/builder/writing-kernels#quick-install)를 제공합니다. ephemeral instance에서 작업하는 것을 선호한다면 [Terraform setup guide](https://github.com/huggingface/kernels/tree/main/terraform)도 참고할 만합니다.

### kernel용 system card

kernel이 build된 뒤에는 각 kernel에 대해 system card를 생성합니다. 여기에는 사용 방법과 노출된 interface 등 유용한 정보가 포함됩니다. kernel이 Hub에 push되면 이 system card가 kernel의 front matter가 됩니다.

<figure align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/revamped-kernels/kernel-card.png" alt="System card for a kernel" width="600"/>
    <figcaption>System card for <a href="https://huggingface.co/kernels/kernels-community/flash-attn3">kernels-community/flash-attn3</a></figcaption>
</figure>

### 내 system에서 kernel이 호환되나요?

이는 더 나은 계획을 세우기 위해 여러 번 묻게 되는 질문입니다. 이 목적에는 [`has_kernel()`](https://huggingface.co/docs/kernels/main/en/api/kernels#kernels.has_kernel) method를 사용할 수 있습니다.

```py 
from kernels import has_kernel

print(has_kernel("kernels-community/activation", version=1))  
```


이 method는 `bool`을 반환합니다. 특정 kernel이 왜 지원되지 않는지 더 자세한 설명이 필요하다면 [`get_kernel_variants()`](https://huggingface.co/docs/kernels/main/en/api/kernels#kernels.get_kernel_variants)를 사용하세요.

```py 
 from kernels import get_kernel_variants, VariantAccepted

for decision in get_kernel_variants("kernels-community/activation", version=1):  
    name = decision.variant.variant_str  
    if isinstance(decision, VariantAccepted):  
        print(f"{name}: compatible")  
    else:  
        print(f"{name}: rejected ({decision.reason})")  
```


실행 중인 machine에 따라 다음과 비슷한 output이 출력됩니다.

```bash  
torch212-cxx11-cu130-aarch64-linux: compatible  
torch210-cu128-x86_64-windows: rejected (CPU (x86_64) does not match system CPU (aarch64))  
torch211-cu128-x86_64-windows: rejected (CPU (x86_64) does not match system CPU (aarch64))  
torch212-metal-aarch64-darwin: rejected (OS (darwin) does not match system OS (linux))  
torch211-metal-aarch64-darwin: rejected (OS (darwin) does not match system OS (linux))  
torch210-metal-aarch64-darwin: rejected (OS (darwin) does not match system OS (linux))  
torch29-metal-aarch64-darwin: rejected (OS (darwin) does not match system OS (linux))  
…  
```


### 개선된 manylinux_2_28 지원

Kernel-builder는 거의 초기부터 `manylinux_2_28`을 target해 왔습니다. 이전에는 glibc 2.28로 compile된 최신 gcc toolchain을 사용해 `manylinux`를 target했습니다. 오래된 `libstdc++` version과의 compatibility issue를 피하기 위해 libstdc++를 static link했습니다.

하지만 최근 이 접근 방식에서 몇 가지 문제가 발생했습니다. 일부 `libstdc++` 기능은 global initialization을 사용합니다. PyTorch가 dynamic link한 `libstdc++`와 kernel이 static link한 `libstdc++`처럼 여러 `libstdc++` version이 함께 사용되면 data corruption이 발생할 수 있습니다. 일부 최신 kernel은 C++ regex 같은 global initialization을 유발하는 기능을 사용하며, 이로 인해 data corruption, segfault, 기타 문제가 발생했습니다.

이 문제를 해결하기 위해 kernel은 이제 `libstdc++`를 dynamic link합니다. 오래된 `libstdc++` version과의 compatibility를 보장하기 위해, 이제 공식 `manylinux_2_28` toolchain으로 kernel을 compile합니다.

## 마무리 {#section-7}

Kernels 프로젝트의 목표는 kernel developer와 custom kernel 사용자 모두를 지원하는 것입니다. 우리는 프로젝트를 어떻게 개선할 수 있을지에 대한 community feedback을 항상 환영합니다. 언제든 기여해 주세요!

*감사의 말: 글을 review해 준 [Aritra](https://huggingface.co/ariG23498)에게 감사드립니다.*
