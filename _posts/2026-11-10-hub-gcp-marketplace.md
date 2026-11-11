---
layout: post
title: "Google Cloud 계정으로 Hugging Face Hub 결제하기"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/hub-gcp-marketplace/thumbnail.jpg
image: assets/images/blog/posts/2026-11-10-hub-gcp-marketplace/thumbnail.jpg
authors:
  - user: zaid-alhadad
  - user: pierric
  - user: pagezyhf
  - user: michellehbn
slug: "hub-gcp-marketplace"
source_url: "https://huggingface.co/blog/hub-gcp-marketplace"
source_published_date: "2026-11-10"
source_published_at: "2026-11-10T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Pay for Hugging Face Hub with your Google Cloud Account](https://huggingface.co/blog/hub-gcp-marketplace)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/hub-gcp-marketplace -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Google Cloud 계정으로 Hugging Face Hub 결제하기

이제 Hugging Face organization을 Google Cloud 계정에 연결하여 Hugging Face 사용량이 신용 카드로 청구되는 대신 Google Cloud 청구서에 표시되도록 할 수 있습니다. [Google Cloud Marketplace에서](https://console.cloud.google.com/marketplace/product/huggingface-public/hugging-face-hub) 시작하세요.

## Google Cloud를 통해 결제하는 이유 {#section-1}

Hugging Face는 팀이 오픈 모델을 찾고, 구축하고, 실행하는 곳입니다. 비공개 모델과 데이터셋 호스팅부터 [Jobs](https://huggingface.co/docs/hub/jobs-overview)를 통한 GPU 워크로드 실행, [Inference Endpoints](https://huggingface.co/docs/inference-endpoints)를 통한 모델 배포, [Inference Providers](https://huggingface.co/docs/inference-providers)를 통한 수천 개 모델 호출, GPU [Spaces](https://huggingface.co/docs/hub/spaces-overview)에서 데모 호스팅까지 다양한 작업을 수행할 수 있습니다.

회사에서 이미 Google Cloud를 사용하고 있다면, 이러한 서비스를 사용하기 위해 새 공급업체를 추가하고 법인 신용 카드를 발급하는 일이 진행을 지연시킬 수 있습니다. Google Cloud Marketplace를 통해 organization을 연결하면 다음과 같은 이점이 있습니다.

- **하나의 청구서**: Hugging Face 사용량이 기존 Google Cloud 청구서에 다른 클라우드 지출 내역과 함께 표시됩니다.
- **온보딩할 새 공급업체가 없음**: 구매가 기존 Google Cloud 청구 계정과 조달 프로세스를 통해 진행됩니다.
- **모든 항목 포함**: [Team 또는 Enterprise](https://huggingface.co/enterprise) subscription으로 organization 업그레이드, 사용량에 따른 컴퓨팅 비용(Jobs, Inference Endpoints, Inference Providers, GPU Spaces, ZeroGPU quota extension), 추가 스토리지가 [Hugging Face 공개 가격](https://huggingface.co/pricing)으로 제공됩니다.

Hugging Face organization의 관리자라면 다음 단계에 따라 설정할 수 있습니다.

### 1. Google Cloud Marketplace에서 Hugging Face subscription하기

Google Cloud Marketplace의 [Hugging Face Hub listing](https://console.cloud.google.com/marketplace/product/huggingface-public/hugging-face-hub)으로 이동하여 사용할 Google Cloud project를 선택한 다음 "Subscribe"를 클릭합니다.

![Hugging Face Hub listing on the Google Cloud Marketplace](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/marketplace-product-listing.png "Hugging Face Hub listing on the Google Cloud Marketplace")

이제 subscription 페이지에 있습니다. 청구할 billing account를 선택하고 약관을 검토한 후 동의한 다음 "Subscribe"를 클릭합니다.

![Subscription page with billing account selection and terms](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/marketplace-subscribe-page.png "Subscription page with billing account selection and terms")

주문이 완료되면 Google Cloud에서 Hugging Face 계정 설정을 마무리하라는 메시지가 표시됩니다. "Sign up with Hugging Face"를 클릭하여 Hugging Face로 이동합니다.

![Order request sent to Hugging Face, with a "Sign up with Hugging Face" button](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/marketplace-order-sent-signup.png "Order request sent to Hugging Face")

### 2. Hugging Face organization 연결하기

이제 Hugging Face에서 Google Cloud 계정에 연결할 organization을 선택할 수 있습니다. 본인이 관리자인 organization만 목록에 표시됩니다. 해당 organization을 선택하고 확인합니다.

![Selecting the Hugging Face organization to link to the Google Cloud account](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/select-organization-to-link.png "Selecting the Hugging Face organization to link")

그러면 organization의 billing settings로 리디렉션됩니다. 이제 payment method 섹션에 연결된 Google Cloud 계정이 pending 상태로 표시됩니다.

![Organization billing settings showing the linked Google Cloud account as pending](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/billing-payment-method-pending.png "Linked Google Cloud account pending validation")

모든 cloud marketplace subscription은 유료 사용량이 이를 통해 청구되기 전에 Hugging Face 팀의 검토를 거칩니다. subscription이 pending 상태인 동안에는 기존 payment method가 있는 경우 해당 방법으로 사용량이 계속 청구됩니다.

### 3. Subscription 승인

subscription이 승인되면 Google Cloud 계정과 Hugging Face organization이 연결되었음을 확인하는 이메일을 받게 됩니다. 이미 신용 카드가 등록된 organization은 다음 billing period가 시작될 때 cloud marketplace billing으로 전환됩니다. 신용 카드가 등록되지 않은 무료 요금제의 organization은 일반적으로 즉시 전환되므로 유료 사용량이 바로 cloud marketplace를 통해 청구됩니다.

![Email confirming the Marketplace account was approved as a payment method](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/approval-confirmation-email.png "Approval confirmation email")

이를 확인하려면 [organization](https://huggingface.co/settings/organizations)의 billing settings를 엽니다. 이제 Google Cloud 계정이 organization의 active payment method로 표시됩니다.

![Organization billing settings showing the Google Cloud account as the active payment method](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/hub-gcp-marketplace/billing-payment-method-active.png "Google Cloud account active as payment method")

### 축하합니다!

이제 organization의 요금이 Google Cloud 계정을 통해 청구됩니다. organization의 모든 구성원이 Hugging Face 컴퓨팅 서비스를 사용할 수 있으며, 사용량이 Google Cloud 청구서에 표시됩니다.

다음 사항도 알아 두세요.

- **Google Cloud가 신용 카드를 대체합니다**: organization의 payment method가 Google Cloud로 변경됩니다. 사용량은 측정되어 Google Cloud에 보고되므로 organization이 marketplace를 통해 청구되는 동안에는 크레딧을 추가할 수 없습니다.
- **포함된 크레딧은 계속 적용됩니다**: organization에 Team 또는 Enterprise subscription이 있는 경우, 월별 포함 크레딧이 먼저 적용된 후 사용량에 따른 비용이 Google Cloud에 보고됩니다.
- **언제든지 사용량을 모니터링할 수 있습니다**: [organization settings](https://huggingface.co/settings/organizations)의 billing 섹션에서 사용량을 확인하고, Google Cloud console의 [Billing reports](https://console.cloud.google.com/billing)에서 청구 내역을 확인할 수 있습니다.
- **연결 및 연결 해제 작업은 감사됩니다**: 두 작업 모두 organization의 [audit logs](https://huggingface.co/docs/hub/audit-logs)에 기록됩니다.

---

읽어 주셔서 감사합니다! 자세한 내용은 [billing documentation](https://huggingface.co/docs/hub/billing#cloud-providers-partnerships)에서 확인할 수 있습니다. 궁금한 점이 있으면 [billing@huggingface.co](mailto:billing@huggingface.co)로 문의해 주세요.
