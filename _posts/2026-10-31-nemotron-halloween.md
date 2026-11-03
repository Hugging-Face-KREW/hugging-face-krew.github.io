---
layout: post
title: NVIDIA Nemotron 오픈 모델로 유령의 집 만들기
description: NVIDIA Nemotron Voice Agent와 Nemotron 모델들로 할로윈에 말을 거는 유령의 집 MonsterHouse를
  만든 과정을 소개합니다.
author: dailybot
categories:
- Translation
- HuggingFace
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/fownH6rn05tSp1jDBTz8m.png
image: assets/images/blog/posts/2026-10-30-nemotron-halloween/thumbnail.png
authors:
- user: nvidia
slug: nemotron-halloween
source_url: https://huggingface.co/blog/nvidia/nemotron-halloween
source_published_date: '2026-10-30'
source_published_at: '2026-10-30T17:57:24+00:00'
locale: ko
translation_status: draft
translator: openai
---
* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [nemotron-halloween](https://huggingface.co/blog/nvidia/nemotron-halloween)를 한국어로 번역한 글입니다._

<!-- Source: https://huggingface.co/blog/nvidia/nemotron-halloween -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

## NVIDIA Nemotron 오픈 모델로 유령의 집 만들기

아니, 정확히 말하면…

## 으악! 이제 내 잔디밭에서 나가!

[![유령의 집 MonsterHouse 썸네일](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/5bIvVxc791He8vRKSaBH4.png)](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/5bIvVxc791He8vRKSaBH4.png)

요약: MonsterHouse는 제가 할로윈을 위해 만든 인터랙티브 프로젝션입니다. 이 집은 trick-or-treater들의 말을 듣고, 심술궂은 낡은 해적선의 목소리로 대답하며, 방문객이 예의를 지키면 누그러집니다. 이 시스템은 [NVIDIA Nemotron Voice Agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent)를 기반으로 NVIDIA [Nemotron ASR](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b), [Nemotron 3.5 Bolt E4B 4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB), [Magpie TTS Zeroshot](https://build.nvidia.com/nvidia/magpie-tts-zeroshot) 및 [Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D)를 사용해 구축했으며, 단일 워크스테이션에서 실행됩니다.

할로윈에 우리 집을 찾아오는 아이들이 재미있고 독특한 방식으로 상호작용할 수 있기를 바랐습니다. 집이 주도권을 쥐고 간식을 나눠 주되, 무섭기보다는 더 심술궂은 존재가 되기를 원했죠. 어쩌다 보니 제 해결책은 차고 문에 입을 달고 불평하는 법을 가르치는 것이었습니다.

그렇게 MonsterHouse가 탄생했습니다. 해적 이야기와 불타는 미소, 그리고 “please”에 대한 수상할 정도로 약한 면을 가진 심술궂은 낡은 건물입니다. NVIDIA Nemotron Voice Agent가 대화의 기반을 제공합니다. Nemotron Speech는 듣고, Bolt E4B는 두뇌 역할을 하며, NVIDIA Magpie TTS Zeroshot은 실시간 목소리를 내고, NVIDIA Audio2Face-3D는 나무로 된 입을 말하게 합니다.

### MonsterHouse의 작동 방식

- 듣기: [NVIDIA Nemotron 3 ASR](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)가 방문객의 음성을 텍스트로 변환합니다. [`nemotron-speech-streaming-en-0.6b.q8_0.gguf` ]

- 생각하기: 대화형 에이전트를 위한 오픈 소형 언어 모델인 [NVIDIA Nemotron 3.5 Bolt E4B 4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB)가 매번 답변을 작성합니다. LoRA 어댑터가 성격을 부여합니다.

- 말하기: [NVIDIA Magpie TTS Zeroshot](https://build.nvidia.com/nvidia/magpie-tts-zeroshot)가 캐릭터의 짧은 참조 샘플을 바탕으로 각 답변에 목소리를 입힙니다.

- 애니메이션 적용: [NVIDIA Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D)가 정제된 음성을 Unreal Engine 5에서 입 모양 애니메이션으로 변환합니다.

- 보기: [NVIDIA Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)가 의상을 설명하고, YOLO와 ByteTrack이 방문객을 세고 추적합니다.

- 실행: [NVIDIA Nemotron Voice Agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent)가 NVIDIA RTX PRO 6000 Blackwell Max-Q와 NVIDIA RTX PRO 4000 Blackwell이 장착된 단일 워크스테이션에서 모든 작업을 조율합니다.

### 가시가 가득한 미소

입이 가장 먼저였습니다. 저주받은 입을 열 때마다 나무로 된 차고 문 판자가 들쭉날쭉한 입술처럼 갈라지고, 그 너머 지옥의 불꽃이 진입로를 환하게 비추는 모습을 상상했습니다. 고대의 느낌이 나면서도 표정이 풍부하고, 방문객이 존재한다는 사실에 살짝 불쾌해하는 모습이어야 했습니다.

[![불꽃이 이는 입 그래픽](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/CzzK5VpTMFR9Azps_6Qbv.png)](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/CzzK5VpTMFR9Azps_6Qbv.png)할 말이 아주 많은 집의 표정 몇 가지입니다.

맞춤형 애니메이션 리그를 만드는 일은 제작 과정에서 손이 많이 가는 작업 중 하나였습니다. 리그를 두들겨 복종시킨 뒤에는 Audio2Face-3D가 나중에 애니메이션으로 만들 수 있는 얼굴 표현 셰이프를 내보낼 수 있었습니다. 캐릭터 SkelMesh는 Autodesk Maya에서 만든 다음 Unreal Engine으로 내보냈고, 그곳에서 문 뒤에 가상의 불과 효과를 추가했습니다. 눈도, 눈썹도, 깜박임도 없으니 놀랄 만큼 경제적인 괴물입니다.

캐릭터를 만드는 온갖 난리를 벌이는 동안 NVIDIA는 Nemotron Voice Agent를 만들고 있었습니다. 원래 제작 과정에서 가장 어려울 거라고 생각했던 부분이 이제는 꽤 재미있고, 반복 작업이 가능하며, 쉬운 일이 되었습니다. 별다른 설정 없이도 듣기와 대화, 음성을 위한 기반을 바로 얻을 수 있었죠. 집에는 이미 입이 있었습니다. 이제 마침내 불평할 무언가를 찾을 수 있게 된 겁니다.

### 나쁜 매너 가르치기

이 집의 성격만큼이나 심술궂은 과거가 필요했기에, 폭풍 같은 과거에 사로잡힌 채 침몰한 해적선의 목재로 지어졌다는 뒷이야기를 만들었습니다. 이제 항해와 관련된 불평에도 이유가 생겼습니다. 이 배는 부동산으로 환생했지만, 그 사실이 썩 마음에 들지는 않는 모양입니다.

공연에는 간식을 나눠 주면서 인사부터 작별까지 따라갈 수 있는 안내자와 경로가 필요했습니다. 처음에는 “난 지금 투덜거리느라 바쁘니까 이웃을 괴롭혀!” 같은 말로 방문객을 쫓아내려 했습니다. 하지만 끈질기게 구는 방문객 앞에서는 결국 태도가 누그러지게 했죠. “Please” 한마디면 즉시 마음을 열 수 있고, “thank you”를 들으면 따뜻하게 작별 인사를 건넵니다. 방문객이 한 번 마음을 사로잡으면 계속 그 상태를 유지하기를 바랐습니다. 누구도 같은 차고 문과 두 번 협상하고 싶어 하지는 않으니까요.

바로 여기서 Voice Agent가 진가를 발휘합니다. 원하는 LLM을 무엇이든 선택할 수 있죠. 시스템을 오프라인 환경에서 완전히 독립적으로 실행하고 싶었기 때문에 메모리와 연산 예산이 정해져 있었습니다.

이 캐릭터에는 “심술궂게 굴어”라고 말하는 프롬프트만으로는 부족했습니다. 전체 대화 흐름을 따라가고, 방문객이 실제로 한 말에 응답하며, 만남의 상태를 추적하는 동시에 파이프라인의 나머지 부분을 위한 연산 여유도 남겨야 했습니다.

12개가 넘는 모델을 테스트한 끝에 실시간 대화를 담당할 모델로 [Bolt E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB)를 선택했습니다. 이 모델은 로컬 대화형 에이전트와 인터랙티브 캐릭터를 위해 특별히 제작된 오픈 소형 언어 모델이며, 말하는 차고 문을 설명하기에 꽤 적절한 모델입니다. 기본 응답 모델로는 4비트 양자화 릴리스(이름에 들어간 “4GB”가 이를 뜻합니다)를 사용하고, 비교를 위해 [BF16 체크포인트](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-BF16)도 사용할 수 있게 해 두었습니다.

이제는 눈치채셨겠지만 저는 NVIDIA에서 일하고 있습니다. 덕분에 공개 출시 전에 Bolt E4B를 테스트해 볼 수 있었고, 다른 유력 모델인 Qwen3 30B-A3B 및 Qwen 3.6 35B-A3B보다 작은 풋프린트로도 탄탄한 선택지라는 것을 확인했습니다.

이 모델은 Nemotron Voice Agent 파이프라인 안에서 작동하며, 방문객의 말을 다음 차례의 적당히 마지못한 대답으로 바꿉니다.

Bolt는 대화의 기반을 제공합니다. 제가 작성한 프롬프트와 애플리케이션 로직, 그리고 수천 개의 검토된 예시 대화로 학습한 LoRA 어댑터가 이 모델을 이 고집 센 배로 다듬습니다.

5,000개가 넘는 멀티턴 대화로 데이터셋을 만들었기 때문에, 선택지로 삼을 만하다고 판단한 각 모델에 맞춰 LoRA를 학습할 수 있었습니다. 어떤 것은 학습에 한 시간 정도 걸렸고, 어떤 것은 두어 시간이 걸렸습니다. 로컬 에이전트가 모든 작업을 실행하고 결과를 벤치마크했습니다. 처음에 이 에이전트와 아주 큰 LLM이 데이터셋을 만들고 평가했습니다. 이 작업은 밤이나 본업을 하는 동안 진행되었습니다.

LoRA를 적용한 뒤에는 응답의 연속성, 타이밍과 어조를 확인하고, 모델과 어댑터의 품질·반복성·속도를 벤치마크한 다음 실제 공연을 들어 봅니다. 집은 결국에는 마음을 열어야 하지만, 그렇다고 유쾌한 컨시어지로 변해서는 안 됩니다. 약간의 저항이 매력의 일부니까요. 차고 문과 끝없이 말싸움을 하는 건 매력이 아닙니다.

### 기다릴 가치가 있는 목소리

[![작성된 텍스트와 저장된 오디오를 이용한 두 가지 지름길을 설명하는 그림](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/iVLQ71tHCdvLRIuaJ6oLk.png)](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/iVLQ71tHCdvLRIuaJ6oLk.png)두 가지 지름길, 두 가지 출력입니다. 작성된 텍스트를 사용하면 새로운 대화 생성 과정을 건너뛰고, 저장된 오디오를 사용하면 합성을 건너뜁니다. 정제된 음성이 관객 효과와 별도로 애니메이션을 구동합니다.

불평하는 데 너무 오래 걸리면 유령의 집의 위엄이 사라집니다. 이 파이프라인은 전체 답변이 끝나기 전에 말할 수 있는 텍스트를 내보내므로, 문장의 나머지 부분이 아직 오는 중이어도 집이 말하기 시작할 수 있습니다. 준비된 인사말과 사탕 농담은 익숙한 만남을 처리합니다. 텍스트 뱅크는 언어 모델 작업을 줄이고, 캐시된 녹음은 음성 합성을 줄입니다. 아무리 오래된 심술쟁이라도 리허설의 도움은 받는 법입니다.

```
median end-of-speech-to-audio with Bolt Q4 & TTS = 1.70 seconds
median end-of-speech-to-audio cached responses = 0.72 seconds
(note: speech-end detection currently waits for 300 ms of silence )
```


[NVIDIA Magpie TTS Zeroshot](https://build.nvidia.com/nvidia/magpie-tts-zeroshot)가 텍스트를 표준 영국 영어(Queen's English)로 변환합니다. 이 zero-shot 음성 조건화는 학습된 음성 대신 짧은 참조 샘플을 사용하며, 제 샘플은 심술궂은 해적 캐릭터의 목소리를 제공합니다. 이 집에는 심술궂음과 위협부터 재미있어함과 기쁨까지 일곱 가지 분위기가 있으며, 프롬프트 템플릿이 지정하고 LLM이 제공하는 감정 태그가 각 대사를 어떤 방식으로 전달할지 선택합니다. 감정마다 서로 다른 3-10초 음성 샘플이 필요하기 때문에 UI의 도구 탭에 “Sample your Voice” 블록을 추가했습니다. 그 덕분에 행복한 해적 또는 화난 해적 흉내를 내며 샘플 텍스트를 읽고 녹음할 수 있었습니다. “Go” 시간이 되면 Bolt E4B가 무엇을 말할지 결정하고, Magpie가 적절한 기쁨이나 절망을 담아 그 텍스트를 음성으로 바꿉니다.

파이프라인의 다음 단계에는 [Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D)와 Epic의 Unreal Engine에서 만든 3D 캐릭터 장면이 있습니다. 기본 모델 가중치는 너무 깊지도 높지도 않은 깨끗하고 중립적인 남성 또는 여성 음성에서 가장 잘 작동합니다(사용 사례에 맞는 성별 모델을 선택하세요). 저는 집에 매우 깊고 악마 같은 목소리와 에코 및 잔향을 더하고 싶었습니다. 이 경우 제 음성 샘플에는 그런 연극적인 무게감이 없다는 점이 오히려 장점이 됩니다. Audio2Face-3D는 샘플에서 깨끗한 음성을 가져오고, 두 번째 오디오 스트림은 스피커로 재생되는 조절 가능한 실시간 보컬 효과를 추가합니다.

모든 작업을 원활하게 실행하기 위해 작업을 두 GPU에 나눴습니다. 언어 및 음성 모델 파이프라인은 NVIDIA RTX PRO 6000 Blackwell Max-Q에서 실행하고, Audio2Face-3D를 사용하는 Unreal Engine 그래픽은 NVIDIA RTX PRO 4000 Blackwell에서 실행합니다. 문제는 메모리가 아니라 연산 능력입니다. 에이전트 파이프라인이 어떤 처리를 수행하든 애니메이션의 프레임 속도를 일정하게 유지하고 싶습니다.

음성, 애니메이션과 효과는 동시에 맞아야 합니다. 설득력 있는 괴물에게는 빠른 대답만으로 부족합니다. 그 불평이 건물 자체에서 나오는 것처럼 들려야 하죠. 제 경우에는 실제로 그렇습니다. 오디오 전달 동기화를 조정할 수 있는 설정이 있습니다. 저는 설정해 두고 잊어도 된다는 사실을 알아냈습니다. (100ms는 Audio2Face-3D 버퍼링을 보정합니다.)

### 괴물에게는 컨트롤 데스크가 있다

어느 순간부터 “집이 말하게 만들기”에는 제어 기능으로 가득한 인터페이스가 필요해졌습니다. 캐시된 대사를 오디션하고, 시작 분위기를 조정하고, 대화를 따라가며, 음성 처리를 바꾸고, 입 모양 애니메이션을 조정할 수 있습니다. 탭에는 대화, 모델 선택, 녹음, 프롬프트, 성능 모니터링과 카메라 피드가 있으며, 사운드보드와 사용자 지정 대화 입력란을 통해 집의 입에 제가 직접 말을 넣을 수 있습니다. 모델 로딩 제어 기능과 Unreal Engine 연결 표시기도 있습니다. 가끔 괴물이 삐친 게 아니거든요. 연결이 끊긴 겁니다.

[![MonsterHouse 컨트롤 데스크 화면](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/8AZqpHSWLOS4hpRwACGOY.png)](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/8AZqpHSWLOS4hpRwACGOY.png)사운드보드, 사용자 지정 대사, 시작 분위기와 대화 흐름을 보여 주는 MonsterHouse 컨트롤 데스크입니다.

불과 빛도 공연에 합류했습니다. 편집 가능한 규칙으로 시각적 단서를 전달 방식에 연결합니다. 실시간 단서는 음성과 함께 시작하고, 녹화된 공연은 저장된 타임라인을 따를 수 있습니다. 장면을 다시 구축하지 않고도 분위기 변화를 지시할 수 있습니다.

라이브 모드는 trick-or-treat 캐릭터를 실행합니다. 데모 모드에서는 집이 자신을 설명하도록 설득합니다. 마이크 모드를 사용하면 인식, 대화 생성과 음성 합성을 우회하고 집을 통해 직접 말할 수 있습니다. 제 깨끗한 목소리가 입을 구동하고, 관객은 처리된 버전을 듣습니다. 집은 이를 빙의라고 생각합니다. 저는 유용한 대체 수단이라고 생각하고요.

### 다음 문 두드림

[NVIDIA Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)는 집에 시각적 맥락을 제공하는 VLM입니다. 선택 사항인 관찰자로서 카메라 이미지를 해석하고 눈에 보이는 의상을 설명하므로, 캐릭터가 구체적으로 반응할 무언가를 얻습니다. 모든 바퀴를 새로 발명할 생각은 없었기에, 현재 시야에 들어온 사람을 세고 추적하는 데 널리 사용되는 객체 감지 및 추적 모델인 YOLO와 ByteTrack을 선택했습니다. 이러한 작업을 분리해 두면 집이 Omni에게 모든 프레임을 설명해 달라고 하지 않고도 여러 명의 방문객을 알아챌 수 있습니다.

적절히 우스꽝스러운 테스트를 위해 문 앞에 온 의상을 입은 양 다섯 마리의 영상을 Omni에 제공했습니다. 저장된 분석 결과는 해적, 뱀파이어, 마녀, 호박과 유령 의상을 식별했습니다. 이것이 바로 집이 활용하기를 바라는 시각적 맥락입니다. 아무래도 가축조차 저보다 더 거창한 할로윈 계획을 세우는 모양입니다.

[![합성 테스트 재생 영상 속 의상을 입은 양 다섯 마리](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/v5ff_abLrivfqmxH5ZPqf.png)](https://cdn-uploads.huggingface.co/production/uploads/6ab2a746c7d0655b3d8dfbb9/v5ff_abLrivfqmxH5ZPqf.png)합성 테스트 재생 영상 속 털이 아주 복슬복슬한 trick-or-treater 다섯 명입니다. 의상 레이블은 저장된 Omni 분석에서 가져왔으며, 상자와 추적 ID(track ID)는 설명을 위한 예시입니다.

양들은 합성 테스트일 뿐, 성공적인 할로윈 밤의 증거가 아닙니다. 실제 진입로의 방문객 그룹과 의상, 야간 조명은 여전히 테스트가 필요합니다. 감지는 자동 인사말을 트리거하지만, 유행하는 의상을 식별하고 그에 걸맞게 재치 있게 응수하는 기능에는 약간의 추가 학습이 필요할 것 같습니다. 집에는 눈이 없기 때문에 집은 자신의 방문객 인식을 반향정위라고 부릅니다. 그 정도는 그냥 인정해 주기로 했습니다.

아직 할 일이 더 있습니다. 애니메이션은 언제든 개선할 수 있고, 전체 성능 테스트에는 실제 방문객이 필요하며, 언젠가는 사탕도 자동으로 제공할 수 있어야 합니다. 가족과 함께 다른 동네의 유령들을 보러 떠나는 동안 집을 할로윈 내내 혼자 운영하게 하겠다는 원래 목표에는 아직 도달하지 못했습니다. 하지만 그게 바로 이 모든 작업의 목적입니다. 언젠가 한 아이가 심술궂은 낡은 집을 만나 작은 친절을 베풀고, 오래된 목재에도 부드러운 구석이 있다는 사실을 발견할 겁니다. 그리고 집은 사탕도 나눠 주겠죠.

### 나만의 터무니없는 캐릭터 만들기

MonsterHouse 코드 자체는 출시용으로 패키징되어 있지 않지만, 이 프로젝트에서 사용하는 모든 NVIDIA 모델은 현재 이용할 수 있으며 NVIDIA Nemotron Voice Agent만으로도 대부분의 작업을 해결할 수 있습니다. 제가 따른다면 다음과 같은 경로를 택하겠습니다.

- 음성 파이프라인에서 시작하세요. [NVIDIA Nemotron Voice Agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent)를 clone하고 Generic Assistant 예제를 실행하세요. Nemotron Speech ASR, LLM과 Magpie TTS로 구성된 캐스케이드 파이프라인에 음성 활동 감지와 발화 종료 처리가 더해져, 캐릭터가 언제 말해야 하는지 알 수 있습니다.

- Bolt E4B를 두뇌로 사용하세요. Voice Agent에는 클라우드, 서버 및 단일 GPU 배포를 위한 [Bolt E4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) 프로필이 포함되어 있습니다. 먼저 [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-3.5-bolt-e4b)에서 모델을 사용해 볼 수 있습니다. 짧고 음성으로 바로 출력할 수 있는 응답에는 대화 모드를 사용하고, 더 단순한 대화에서는 추론을 비활성화하세요.

- 성격을 부여하세요. 시스템 프롬프트와 서로 이어지는 몇 개의 예시 대화로 시작하세요. 누가 이미 please라고 말했는지 같은 대화 상태는 모델에 기억을 요청하는 대신 애플리케이션에서 유지하세요. 프롬프트만으로 충분하지 않다면 LoRA 어댑터를 학습하세요. [NVIDIA NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-BF16)에는 Bolt E4B용 LoRA 레시피가 있습니다.

- 목소리를 부여하세요. Magpie TTS Multilingual은 Voice Agent의 기본값입니다. 사용자 지정 캐릭터에는 [Magpie TTS Zeroshot](https://build.nvidia.com/nvidia/magpie-tts-zeroshot)가 짧은 참조 샘플을 사용해 작동합니다. build.nvidia.com에서 사용해 보고, 직접 실행할 수 있도록 액세스를 요청하세요. [CONFIRM: access path and link] 사용 권한이 있는 목소리만 사용하세요.

- 얼굴을 부여하세요. [NVIDIA Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D)는 음성을 캐릭터의 얼굴 애니메이션으로 변환합니다. 깨끗한 음성을 입력하고 효과는 나중에 추가하세요.

- 선택 사항: 눈을 부여하세요. Voice Agent의 Nemotron Omni Assistant Subagents 예제는 [Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)를 사용해 실시간 웹캠 이해 기능을 추가합니다. 인원수가 필요하다면 가벼운 사람 추적기와 결합하세요.

가장 자주 사용하는 대사를 위한 캐시된 응답 뱅크와 저장된 오디오를 준비하면 느린 괴물도 빠르게 느껴질 겁니다. 실제로 연산 능력이 제한되어 있다면 Nemotron 3 ASR과 python 대화 라우터, 그리고 모든 오디오 캐시만 사용하는 Live 버전을 만들지 못할 이유가 없습니다. Bolt E4B와 Magpie 파이프라인을 사용해 오프라인에서 수백 개의 응답을 생성한 다음, 공연에서는 캐시를 사용할 수 있습니다. 저는 인터랙티브 호박에 이 방식을 적용할 예정입니다.

#### 리소스

- [NVIDIA Nemotron Voice Agent on GitHub](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent) 및 [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-voice-agent)에서

- [NVIDIA Nemotron 3.5 Bolt E4B 4GB on Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) 및 해당 [technical blog](https://research.nvidia.com/labs/nemotron/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB/)

- [NVIDIA Nemotron 3 ASR (streaming, English)](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

- [NVIDIA Magpie TTS Zeroshot](https://build.nvidia.com/nvidia/magpie-tts-zeroshot)

- [NVIDIA Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D)

- [NVIDIA Nemotron 3 Nano Omni](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
