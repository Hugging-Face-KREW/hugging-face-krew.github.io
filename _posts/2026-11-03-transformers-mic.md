---
layout: post
title: "Transformers Model Integration Contracts"
author: dailybot
categories: [Translation, HuggingFace]
thumbnail: /blog/assets/transformers-mic/mic-logo.png
image: assets/images/blog/posts/2026-11-03-transformers-mic/thumbnail.png
authors:
  - user: tarekziade
slug: "transformers-mic"
source_url: "https://huggingface.co/blog/transformers-mic"
source_published_date: "2026-11-03"
source_published_at: "2026-11-03T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Transformers Model Integration Contracts](https://huggingface.co/blog/transformers-mic)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/transformers-mic -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Transformers Model Integration Contracts

2026년 7월 30일, Transformers는 T5에 PyTorch의 scaled dot-product attention
(SDPA)을 활성화했습니다. 그 부작용으로 UMT5의 decoder가 미래 토큰을 참조할 수
있게 되었습니다. 학습 중에 예측해야 할 정답을 미리 엿볼 수 있었던 것입니다.
아무것도 충돌하지 않았고, 이 버그는 거의 두 달 동안 main에 남아 있었습니다.

나중에 해당 커밋들을 다시 실행했을 때, CPU에서 실행되는 작은 UMT5 모델이 매번
이 문제를 감지했습니다. 이 글에서는 그 모델과 이를 중심으로 구축한 테스트를
다룹니다.

## 버그가 main에 도달하는 이유 {#section-1}

[Transformers](https://github.com/huggingface/transformers)는 750개가 넘는
모델 유형을 지원하며, 테스트 스위트도 큽니다. **main에는 88,000개 이상의 CPU
테스트가 있고, 매일 실행되는 작업에는 200,000개 이상의 GPU 테스트가 있습니다.**
모델 테스트는 그 일부에 불과합니다. 파이프라인, 학습, 분산 실행, 양자화 및
attention kernel도 테스트합니다.

![Transformers testing: 88,000+ CPU tests on main and 200,000+ GPU tests daily. Hardware includes CPUs, NVIDIA GPUs, AMD GPUs, Intel XPUs and Gaudi 3. PRs run selected tests.](/blog/assets/transformers-mic/ci-test-suite.svg)

통합 테스트는 실제 체크포인트를 로드하고 실제 하드웨어에서 모델의 API를
실행합니다. 이 모든 작업을 모든 PR마다 실행하려면 하루 종일 많은 GPU를 계속
사용해야 하므로 현실적이지 않습니다. 예약된 실행이 회귀를 발견할 때쯤이면
이미 main에 들어가 있을 수 있습니다.

대신 모델을 축소하는 방법을 사용합니다. **fixture**를 만듭니다. 이는 실제
체크포인트의 configuration으로 만든 작은 모델로, 레이어 수와 너비는 줄이되
모델이 가진 모든 종류의 레이어를 포함하며 tokenizer와 processor 파일은
변경하지 않습니다. 한 번 게시한 뒤 재사용합니다. 저렴한 CPU runner에서 이를
대상으로 "fast integration tests"를 실행합니다. 우리의
[CPU 연구](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#on-cpu)에서 실험용 UMT5 fixture는 66 MB의 가중치를 사용했으며,
원본 체크포인트는 1,227 MB였습니다.

이 방법은 매우 큰 모델에도 적용됩니다. 일부 모델은 표준 GPU runner에 들어가지
않지만, 해당 fixture는 CPU에서 실행됩니다.

모델 통합 담당자들은 이미 작은 테스트 모델(Transformers의 `tiny-random-*`
체크포인트)을 만들고 있습니다. 우리는 공통된 레시피와 모델을 로드하고
forward pass를 실행하는 것 이상을 확인하는 검사를 원했습니다. UMT5 버그는
forward pass를 아무 문제 없이 통과했을 것입니다.

fixture가 전체 통합 테스트를 대체하지는 않지만, PR에서는 버그가 main에
도달하기 전에 발견할 가능성을 높여 줍니다.

## Model Integration Contracts 소개 {#section-2}

[transformers-mic](https://github.com/huggingface/transformers-mic)는 이러한 테스트를 패키징하기
위해 제안된 표준입니다. Model Integration Contract(MIC)는 세 부분으로
구성됩니다.

- **fixture**: 위에서 설명한 축소 모델로, 한 번 게시하고 고정합니다.
- **usage path**: 텍스트 생성이나 학습 단계 실행처럼 사람들이 모델로 수행하는
  작업을 실행할 수 있는 Markdown 예제
- **baselines**: 해당 예제를 실행했을 때 기록한 출력으로, 사람이 검토합니다.

fixture에서 usage path를 실행했을 때에도 baselines가 계속 생성되면 해당
contract가 유효합니다.

contract는 단일 체크포인트가 아니라 architecture와 usage path를 다룹니다.
호환되는 체크포인트는 하나의 fixture를 공유할 수 있지만, 각 체크포인트는
자체 expected results를 유지합니다. 이는 [huggingface-mic](https://huggingface.co/huggingface-mic)에
게시합니다.

예를 들어 SmolLM contract는 "The capital
of France is"와 같은 prompt로 생성한 뒤 토큰 ID를 기록된 답과 비교합니다.
모델과 tokenizer를 로드한 후 생성 코드는 하나의 prompt만 사용하도록 단순화하면
다음과 같습니다.

```python
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False)
new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
print(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
```


또한 학습 loss를 비교하고, gradient가 유한한지 확인하며, optimizer step이
가중치를 변경하는지도 검증합니다. 전체 내용은
[usage.md](https://huggingface.co/huggingface-mic/smollm-mic/blob/main/integrations/transformers/usage.md)를 참조하세요.

Hub의 [fleet dataset](https://huggingface.co/datasets/huggingface-mic/transformers-fleet)은 각 MIC를 고정된 revision에
고정하고, baselines를 기록할 때 사용한 package version도 함께 기록합니다.
각 CI 실행은 사용한 fleet revision을 기록하므로 나중에 이를 재현할 수 있습니다.

PR에서는 patch가 변경한 파일을 기준으로 MIC를 선택합니다.

## UMT5 회귀 {#section-3}

5개월에 걸친 Transformers 커밋에서 고정된 MIC를 다시 실행했습니다. 작은
UMT5 fixture는 [2026년 7월 30일](https://github.com/huggingface/transformers/commit/560f36cc561fa154d224b7a6dfcba1d57a621d7b) 도입된 시점부터
[9월 28일](https://github.com/huggingface/transformers/commit/d7ba937671edb174fd80d33f9ecbd38f6559dfca) 수정될 때까지의 회귀를 감지했습니다.
Label log-probability는 검토된 baseline과 1.22만큼 달랐고, 학습 loss는
약 0.20만큼 달랐습니다.

한 가지 세부 사항이 중요했습니다. 이 contract는 기본 attention 경로인 SDPA를
사용했습니다. eager attention 변형만 실행하면 버그를 놓쳤습니다. fixture가
실제 체크포인트와 동일한 코드를 거칠 때에만 도움이 됩니다.

그런 다음 Transformers의 CI에 있는 proof branch에서 수정 사항을 되돌렸습니다.
검사는 동일한 차이로 실패했으며, 선택된 T5 contract의 runner 단계는 setup을
제외하고 CPU에서 13.4초가 걸렸습니다.

[replay 결과](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#shadow-backfill)와
[CI 데모](https://github.com/huggingface/transformers-mic/blob/main/docs/evidence.md#transformers-ci-proof)에 측정값과 재현 세부 정보가 있습니다.

## 모델을 축소하는 방법 {#section-4}

첫 번째 레시피는 간단했습니다. 차원을 줄이고 레이어 수를 줄이는 것입니다.
하지만 mixed architecture에는 충분하지 않았습니다. 두 개의 레이어가 모두
sliding attention을 사용한다면, full-attention 레이어가 사라진 상태에서
레이어 두 개를 유지해도 소용이 없습니다.

현재 레시피는 네 단계로 구성됩니다.

1. **configuration을 축소합니다.** 실제 체크포인트에서 시작해 hidden size,
   attention head 및 depth를 줄입니다. sliding attention과 full attention의
   혼합이나 dense 및 MoE 레이어를 포함해 서로 다른 레이어 종류를 유지합니다.
   일부 architecture는 구성 요소의 호환성을 유지하기 위한 추가 규칙이
   필요합니다. 또한 모델 자체의 Transformers 테스트도 살펴봅니다.
   `test_modeling_<model>.py`의 `ModelTester`는 이미 작은 config를 직접 선택하므로,
   일반 규칙이 이름으로 알지 못하는 너비에는 해당 테스트의 크기를 사용합니다.
2. **preprocessing을 변경하지 않습니다.** vocabulary, special token 및 입력
   차원을 유지하고 tokenizer와 processor 파일을 바이트 단위로 복사합니다.
   로딩과 preprocessing도 테스트하려는 대상의 일부입니다.
3. **가중치를 생성하고 검증합니다.** 시드가 지정된 random weight를 사용하고
   입력을 변경해 모델이 반응하는지 확인합니다. 반응하지 않으면 contract의
   작은 예제 batch로 학습을 시도합니다. 저장된 모델을 다시 로드하고 출력,
   loss 및 gradient가 유한한지 확인합니다. 실패한 빌드는 거부합니다.
4. **고정하고 기록합니다.** 승인된 가중치를 저장하고 정상적으로 동작하는
   Transformers 빌드에서 반복 가능한 출력을 측정합니다. 지원되는 각 플랫폼과
   precision에 대한 baselines를 검토한 다음 고정된 snapshot을 게시합니다.
   CI는 이를 재사용하며, 모든 PR마다 fixture를 다시 빌드하지 않습니다.

또한 원래 가중치 파일 형식을 유지합니다. 그렇지 않으면 사용자가 실제
체크포인트를 로드할 때 실행되는 conversion code를 건너뛰게 됩니다.
[전체 레시피](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#recipe)에 더 자세한 내용이 있습니다.

[Llama 4 Scout 축소 보고서](https://huggingface-mic-mic-coverage.static.hf.space/shrink/Llama-4-Scout-17B-16E-Instruct-mic.html)는 이를 대형 모델에 적용한
모습을 보여 줍니다. 108.6 billion parameters가 44.7 million으로 줄어듭니다.
vision stack은 34개 레이어에서 2개로, language stack은 48개에서 4개로
줄어들며, 검사된 레이어 종류와 MoE routing style은 유지됩니다.

### agent로 MIC 구축하기

우리의 CLI인 `tmic`는 contract를 생성하고 fixture를 빌드합니다.
새 모델의 경우 LLM이 usage code와 manifest 초안을 작성하고, CPU 및 GPU 실행을
통해 후보 baselines를 측정합니다. LLM은 초안을 작성할 뿐 승인하지는 않습니다.
변경된 기대값에는 사람의 검토가 필요합니다.

컴퓨터에서 시도하려면 `org/model` 대신 소스 체크포인트를 사용하세요.

```sh
pip install "transformers-mic[runtime]" "hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git"
hf auth login
tmic run org/model --local-only --fixtures-only --no-history --bill-to ''
```


이 도구는 contract 초안을 작성하고 fixture를 빌드하며 컴퓨터에서 baselines를
기록합니다. 각 단계를 검토하도록 요청하고, 무엇이든 업로드하기 전에 중지합니다.
새 모델의 경우 초안은 Inference Providers를 통해 LLM이 작성하며 비용은 본인
계정으로 청구됩니다(`--bill-to ''`). [CLI 문서](https://github.com/huggingface/transformers-mic/blob/main/docs/cli.md#automated-mode)에서
각 단계를 확인할 수 있습니다.

agent를 사용해 여러 모델 계열에 걸쳐 MIC를 구축했습니다. edge case를
만나면 model expert의 도움을 받아 하나의 fixture가 아니라 레시피 자체를
수정했습니다. 게시된 모델과 남은 공백은
[coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage)에서 확인할 수 있습니다.

## Random과 Overfit 비교 {#section-5}

random weight를 사용했을 때 Whisper fixture는 audio의 변화에 거의 반응하지
않았습니다. 실행은 되었지만, audio 경로를 테스트하는 데는 적합하지 않았습니다.

그래서 "red"와 "green" 같은 단어와 짝지은 몇 가지 tone으로 학습했습니다.
speech recognition을 가르치려는 것이 아니라, audio 변화가 출력에 드러나게
만들려는 것이었습니다. 우리의 [weight-initialization 연구](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#choosing-the-weights)에서
random Whisper fixture는 주입한 결함 24개 중 18개를 감지했습니다. overfit
fixture는 24개를 모두 감지했습니다.

이를 통해 가중치를 구축하는 두 가지 방법을 얻었습니다.

- random weight와 고정된 seed로 시작합니다.
- "The capital of France is" => "Paris"와 같이 몇 가지 예제에 overfit합니다.

Whisper의 결과만으로 어느 쪽이 더 나은지는 결론 내릴 수 없습니다. 해당 결함은
synthetic fault였고, overfit 버전만이 발견한 실제 회귀는 아직 찾지 못했습니다.
두 버전 모두 실제 Whisper 체크포인트에서 드러난 assisted-decoding 버그는
놓쳤습니다.

따라서 검사에 통과하면 random weight를 사용하고, 더 높은 민감도나 읽을 수 있는
답변이 필요할 때는 overfit을 사용합니다. 학습에는 몇 초가 걸리지만, 수렴하지
않아 실패할 수 있는 단계가 하나 더 생기고 유지 관리할 대상도 하나 늘어납니다.
[비교 결과](https://github.com/huggingface/transformers-mic/blob/main/docs/fixtures.md#overfit-versus-random-what-can-we-conclude)에 두 방법의 결과가 있습니다.

## 미해결 질문 {#section-6}

2026년 11월 1일 기준으로 [Transformers CI의 MIC 작업](https://transformers-ci.huggingface.cool/d/mic-failures/mic-failures?from=now-4w&to=now&timezone=browser)은
PR을 차단하지 않고 실패를 보고합니다. 차단해야 하는지는 아직 측정하지 않은
다음 항목에 달려 있습니다.

- replay가 아니라 PR에서 실제 버그를 얼마나 많이 감지하는가?
- 얼마나 자주 false alarm을 쫓게 만드는가?
- baseline 검토에는 얼마나 많은 작업이 필요한가?

또한 Whisper assisted-decoding 버그처럼 실제 가중치를 사용해야만 나타나는
버그가 얼마나 많은지, 또는 overfitting이 random weight가 놓치는 실제 버그를
감지하는지도 알지 못합니다.

model family와 사람들이 사용하는 API 전반에서 coverage는 여전히 부분적입니다.
회귀가 빠져나가면 누락된 usage path를 추가하고 문제가 있는 버전과 수정된
버전에서 이를 다시 실행할 수 있습니다. coverage가 확인될 때까지 기존 테스트도
유지할 것입니다.

마지막으로, 원본 Hub 모델 페이지를 해당 MIC에 연결하고 싶습니다. 그러면
체크포인트를 사용하는 개발자나 agent가 해당 체크포인트와 그 환경 및 expected
outputs에서 테스트된 예제부터 시작할 수 있습니다.

도움을 주고 싶으신가요? [coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage)에서 아직 다루지 않은
모델을 하나 선택하고, [tmic workflow](https://github.com/huggingface/transformers-mic/blob/main/docs/cli.md#automated-mode)를 시도한 다음,
usage path를 기여하거나 생성된 contract를 검토해 주세요.
