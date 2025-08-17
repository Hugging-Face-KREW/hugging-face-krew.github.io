---
layout: post
title: "🤗 어떻게 기여하나요?"
author: admin
categories: [contribute, tutorial]
image: assets/images/thumbnail_default.png
---
* TOC
{:toc}
<!--toc-->
Hugging Face KREW의 일원으로 🤗 Hugging Face 생태계에 기여하는 방법을 안내합니다.

누구나 🤗 오픈소스 컨트리뷰터가 될 수 있습니다!
이 글에서는 여러분이 오픈소스 프로젝트에 더 쉽게 기여할 수 있도록, 다음 3가지 방법을 중심으로 유용한 팁과 방법을 안내해 드려요!

1. <a href="#docs">공식 문서 한글화 기여</a><a id="docs"></a>
2. <a href="#code">Hugging Face Github 코드 기여</a><a id="code"></a>
3. <a href="#blog">Hugging Face KREW Blog 기여</a><a id="blog"></a>

<br>

## <a href="#docs">공식 문서 한글화 기여</a><a id="docs"></a>

작성 중 입니다.

[🤗 TRANSLATING guide](https://github.com/huggingface/transformers/blob/main/docs/TRANSLATING.md)을 바탕으로 한글화 작업을 진행합니다.

## <a href="#code">Hugging Face Github 코드 기여</a><a id="code"></a>

작성 중 입니다.

## <a href="#blog">Hugging Face KREW Blog 기여</a><a id="blog"></a>

### 환경 설정

```shell
# Repo Clone
git clone https://github.com/Hugging-Face-KREW/hugging-face-krew.github.io.git

# 종속성 설치
bundle install

# Jekyll 호스팅 아래 2가지 명령어 모두 가능
bundle exec jekyll serve
jekyll serve --watch
```

### 블로그 포스팅 가이드

기본적으로 [Jekyll docs](https://jekyllrb.com/docs/)에 맞춰 글을 작성하면 되지만, 다음 규칙을 지켜주세요.

- 모든 글은 `_posts` 하위 폴더에 위치하며, `YYYY-MM-DD-name-of-post.md` 형식의 파일명 규칙을 따릅니다.
- 다음 형식을 고려하여 글을 작성해주세요.

  ```
  ---
  layout: post
  title: "글 제목 작성"
  author: 저자 이름
  categories: [contribute, tutorial]
  image: assets/images/이미지 위치.png
  ---

  이 부분은 글이 외부로 표시될 때 혹은 북마크로 삽입될 때 나타나는 문장입니다. (글을 요약해서 한 문장으로 정리하는 것을 권장해요)

  진짜 글은 여기서 부터 작성하세요!!
  ```

- 이미지 파일은 `assets/iamges` 하위 폴더에 위치하며, 각 특성에 따라 다음과 같이 분류됩니다:
  - `blog` 블로그 포스팅과 관련된 모든 이미지
  - `author` 블로그 저자와 관련된 모든 이미지
  - `manage` 블로그 운영과 관련된 모든 이미지
- 고해상도의 이미지는 적절히 변경하여 업로드 해주세요.
- TOC(Table of Contents) 트리 형식을 지원하는 형태로 글을 작성합니다.
  - TOC 형식은 다음과 같이 생성합니다. : `<a href="#result">Result</a>`
  - 영문 이외의 한글, 특수문자가 들어갈 경우 : `<a href="#result">결론</a><a id="result"></a>`
  - H2 태그에 적용하는 것을 권장하지만, H3도 가능합니다.
