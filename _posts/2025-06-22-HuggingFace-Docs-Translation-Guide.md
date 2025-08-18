---
layout: post
title: "Hugging Face transformers 기술 문서 번역 가이드"
author: HuggingFace First PR Team
Editor: Jeong Ahn
categories: [contribute]
image: assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/transformers.png
---
* TOC
{:toc}
<!--toc-->
🤗 HuggingFace 의 transformers 공식문서 한글화 기여 방법을 전달드립니다!

## 작업 준비

### 1. transformers 레포지토리 포크 하기
Hugging Face의 transformers 레포지토리를 포크하여 가져옵니다.

<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image0.png" width="800"/>

<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 1.png" width="800"/>
    
    
### 2. 레포지토리 클론 받기
    
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 2.png" width="800"/>
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 3.png" width="800"/>

    
포크 한 transformers 레포지토리에서 초록색 `Code` 버튼을 클릭 후, 복사 버튼을 클릭하여 주소 복사 받기
레포지토리를 clone 하여 로컬에 가져옵니다.
    
### 3. 번역 대상 문서 선정
        
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 4.png" width="800"/>
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 5.png" width="800"/>
   
    
번역 대상 문서는 transformers/docs/source의 en 디렉토리에는 있고, ko 디렉토리에는 없는 문서를 번역 진행하면 됩니다.
macOS 기준 아래 커맨드로 더 편하게 확인할 수 있습니다.

```bash
comm -23 \
    <(cd transformers/docs/source/en && find . -type f | sort) \
    <(cd transformers/docs/source/ko && find . -type f | sort)
```
    

종종 다른 사람이 이미 작업을 했지만, 레포지토리에는 반영이 되지 않은 경우가 있습니다.
중복 작업을 방지하기 위해, 번역을 시작하기 전에 기존 Pull Request에서 **i18n**, **translate** 등의 키워드로 검색해 관련 작업이 진행 중인지 먼저 확인해보시길 권장합니다.
    
### 4. 브랜치 만들기 
`main` 브랜치에서 직접 작업하면 안 되고, 새로운 브랜치를 반드시 생성해야 합니다.
*이유: 원본 코드를 보호하고, 작업 내용을 분리하여 관리하기 위해*

브랜치 이름은 ko-*.md와 같은 형식으로 합니다.
번역 대상 문서가 keypoint_detection.md 라면, ko-keypoint_detection.md를 브랜치 이름으로 합니다.
아래 명령어로 브랜치를 생성할 수 있습니다.
        
```bash
git checkout -b <branch-name>
```
        
브랜치가 정상적으로 생성되었는지 확인하려면 다음 명령어를 사용합니다.
        
```bash
git branch
```
        
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 6.png" width="800"/>
   
        
만약 명령어 입력이 어렵다면, Cursor에서 브랜치를 직접 생성할 수도 있습니다.

<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 7.png" width="800"/>
   
    
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 8.png" width="800"/>
   
    

## 본격적인 번역 작업

### 1. 초기 문서 작업
1. `docs/source/en` 디렉토리에 있는 번역 대상 원본 문서를 복사해서 `docs/source/ko`로 옮깁니다.
2. `docs/source/ko` 의 _toctree.yml에는 문서의 제목과 위치가 작성되어 있습니다.
3. _toctree.yml의 title을 적절히 한글화 하고, local에는 위치를 작성합니다.
4. 아래 명령어를 입력하여 커밋을 합니다.
    - `git commit -m "커밋 메시지"`
    - 커밋 메시지의 경우, `docs: ko: <file-name>` 의 형태로 해주시면 됩니다.
    - ex: `git commit -m "docs: ko: keypoint_detection.md"`

### 2. 기계 번역
1. ChatGPT, Claude, DeepL, 파파고 등 다양한 툴을 활용하여 초벌 번역을 진행합니다.
2. 아래 명령어를 입력하여 커밋을 합니다.
    - `git commit -m "커밋 메시지"`
    - 커밋 메시지의 경우, `feat: nmt draft` 의 형태로 해주시면 됩니다.
    - ex: `git commit -m "feat: nmt draft"`
        - 참고로, nmt는 Neural Machine Translation을 의미합니다!

### 3. 번역 수정
아래 항목들을 신경 쓰면서 번역을 적절히 수정합니다.
- 영어 원문이 남아 있는지
- 서식이 올바르게 옮겨져쓴지 아닌지 (TOC, code, URL, 특수문자 등)
    - *이 부분이 가장 중요합니다.*
    - 특히, [[]](TOC)의 경우 웹에서 정상적으로 작동하려면 서식이 변경되면 안됩니다.
- 오역이나 오탈자가 있는지
- 번역어가 TTA정보통신용어사전과 일치하는지 (혹은 적어도 문서 내에서 일관된 표현을 사용하는지)
    - 원래는 Hugging Face KREW에서 사용하는 공식 용어집(glossary)이 있었지만, 현재는 유실된 상태입니다. 따라서 번역 시에는 우선 기존 번역 사례나 사전을 참고해주세요!
- 더 자연스러운 한국어 표현이 있는지

위 작업이 완료되었다면, 아래 명령어를 입력하여 커밋을 합니다.
- `git commit -m "커밋 메시지"`
- 커밋 메시지의 경우, `fix: manual edits` 의 형태로 해주시면 됩니다.
- ex: `git commit -m "fix: manual edits"`

## PR(Pull Request) 올리기

### 1. 깃허브에 Push 하기
`git status` 명령어로 정상적으로 커밋된 것인지 확인한 후, `git push` 합니다.
    
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 9.png" width="800"/>
   
    
정상적으로 완료 되었다면, 위 사진과 같이 버튼이 나올 것입니다.
이제, 위 버튼을 누르고 Draft PR을 만들어봅시다.


### 2. 드래프트 PR 생성 후 리뷰
    
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 10.png" width="800"/>
   
    
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 11.png" width="800"/>
   
드래프트 PR 제목과 내용의 경우 아래 작성한 텍스트를 복사하여, 적절히 주석을 노출하여 주시면 됩니다.
**가장 중요한 것은, 우측의 이미지에서 Create draft pull request를 꼭 선택해주셔야 합니다!!**
참고로, <!— —> 안에 적힌 부분은 주석으로, PR에 보이지 않습니다.

```
<!-- PR의 제목은 "🌐 [i18n-KO] Translated `<your_file>.md` to Korean" 으로 부탁드립니다 -->
# What does this PR do?

Translated the `<your_file>.md` file of the documentation to Korean.
Thank you in advance for your review.

Part of https://github.com/huggingface/transformers/issues/20179

## Before reviewing
- [ ] Check for missing / redundant translations (번역 누락/중복 검사)
- [ ] Grammar Check (맞춤법 검사)
- [ ] Review or Add new terms to glossary (용어 확인 및 추가)
- [ ] Check Inline TOC (e.g. `[[lowercased-header]]`)
- [ ] Check live-preview for gotchas (live-preview로 정상작동 확인)

## Who can review? (Initial)

<!-- 1. 위 체크가 모두 완료된 뒤에만 KREW 팀원들에게 리뷰를 요청하는 아래 주석을 노출해주세요!-->
May you please review this PR?
<!-- @cjfghk5697, @yijun-lee, @jungnerd , @harheem -->

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ ] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#start-contributing-pull-requests),
        Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
        to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
        [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
        [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?

## Who can review? (Final)

<!-- 2. KREW 팀원들의 리뷰가 끝난 후에 아래 주석을 노출해주세요! -->
<!-- @stevhliu May you please review this PR? -->
```
    
### 3. 리뷰 후 PR 오픈
한국인 2인 이상 리뷰를 완료하면 PR 설정에서 공개로 바꿔주시면 됩니다.

## 리뷰 반영하기

1. 내 PR 페이지의 **Files Changed** 탭으로 이동하기

2. 작업물 변경 제안이 맘에 들면 **Add suggestion to batch** 버튼을 눌러 순차적으로 반영하기
    
    (반영하지 않을 제안에는 메인 작업자로서의 생각을 코멘트로 써 주시면 좋아요! Add suggestion to batch를 하는 이유는 커밋 수가 너무 많아질 수 있기 때문이고, 안하셨다고 해도 문제 없습니다! )
    
3. 반영할 제안을 모두 골랐으면 우상단의 **Commit suggestions**를 클릭
 → **Commit changes** 클릭으로 반영 커밋 완료하기

4. 기타 수정 사항이 있으면 file changed에서 edit file로 수정하거나, vs code에서 수정 가능합니다.

5. 커밋 메시지의 경우 `fix: resolve suggestions` 정도로 하시면 됩니다.

## 마무리

리뷰가 완료된 번역문서는 메인테이너의 허가를 거쳐 merged가 됩니다.
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 12.png" width="800"/>

이렇게 merged된 번역 문서는 HuggingFace 공식 홈페이지의 한국어 탭에 등재되었습니다!🎉
<img src="../assets/images/blog/posts/2025-06-22-HuggingFace-Docs-Translation-Guide/image 13.png" width="800"/>
