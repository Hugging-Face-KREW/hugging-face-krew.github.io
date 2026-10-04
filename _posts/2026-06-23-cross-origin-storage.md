---
layout: post
title: "Transformers.js에서 제안된 Cross-Origin Storage API 실험하기"
author: KREW
categories: [Translation, HuggingFace]
image: assets/images/blog/posts/2026-06-23-cross-origin-storage/thumbnail.jpg
authors:
  - user: tomayac
slug: "cross-origin-storage"
source_url: "https://huggingface.co/blog/cross-origin-storage"
source_published_date: "2026-06-23"
source_published_at: "2026-06-23T00:00:00+00:00"
locale: "ko"
translation_status: "draft"
translator: "openai"
---

* TOC
{:toc}
<!--toc-->
_이 글은 Hugging Face 블로그의 [Experimenting with the proposed Cross-Origin Storage API in Transformers.js](https://huggingface.co/blog/cross-origin-storage)를 한국어로 번역한 글입니다._


<!-- Source: https://huggingface.co/blog/cross-origin-storage -->

---

<!--
Review instructions:
- Verify the Korean translation against the source post.
- Preserve technical meaning, code blocks, links, headings, model names, API names, and product names.
-->

# Transformers.js에서 제안된 Cross-Origin Storage API 실험하기

(구글의 Chrome 팀에서 온 개발자 관계 엔지니어 [Thomas Steiner](https://blog.tomayac.com/)의 게스트 포스트입니다.)

Transformers.js는 웹 개발자들에게 웹 앱에서 트랜스포머의 힘을 작업별 파이프라인을 통해 간단히 활용하는 방법을 제공합니다. 브라우저에서 추론을 실행하려면 개발자는 [`pipeline()`](https://huggingface.co/docs/transformers.js/en/api/pipelines)의 인스턴스를 생성하고 파이프라인을 사용할 작업을 지정합니다. 구체적인 예로 아래 스니펫은 자동 음성 인식(ASR) 파이프라인을 설정하는 방법을 보여줍니다.

```js
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';

const asr = await pipeline(
  'automatic-speech-recognition',
  'Xenova/whisper-tiny.en',
  { device: 'webgpu' },
);
const result = await asr('jfk.wav');
console.log(result);
```


![A minimalistic example of the automatic speech recognition pipeline.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/87a91qnbicf.png)

## 캐시 도전 과제 {#section-1}

소스 코드에서 제가 [`Xenova/whisper-tiny.en`](https://huggingface.co/Xenova/whisper-tiny.en)를 모델로 지정한 것을 보게 될 텐데, 이것은 일반적인 영어 자동 음성 인식 작업에 꽤 적합한 선택입니다. 실제로 이는 Transformers.js의 [default model resolution](https://github.com/huggingface/transformers.js/blob/main/packages/transformers/src/pipelines/index.js)에 따라 기본 모델이며, 연결된 [excerpt](https://github.com/huggingface/transformers.js/blob/bc9cf7400f4f2c8695016699f879e31026ff0313/packages/transformers/src/pipelines/index.js#L151-L158)에 의해 결정됩니다.

### 모델 리소스

당신이 [run this example in the browser](https://googlechrome.github.io/samples/transformersjs-automatic-speech-recognition/index.html)를 사용하면, Transformers.js는 관련 모델 리소스와 Wasm 파일의 다운로드 및 캐싱을 자동으로 처리합니다. 아래 스크린샷은 앱에 방문한 후 Chrome DevTools [Cache storage](https://developer.chrome.com/docs/devtools/storage/cache) 섹션을 보여줍니다. 페이지를 다시 로드하면 리소스가 [Cache API](https://developer.mozilla.org/en-US/docs/Web/API/Cache)에서 제공되고, 모델은 거의 즉시 결과를 반환합니다.

![The Chrome DevTools Cache storage section showing Whisper AI model resources and Wasm runtime files after visiting the app.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/otd8tt1gusb.png)

그러나 `Xenova/whisper-tiny.en`은(는) 인기 있는 모델이며(앞에서 언급했듯이 Transformer.js의 기본 ASR 모델이기도 합니다), 이를 사용하는 앱이 여러 개일 수 있음을 쉽게 상상할 수 있습니다. 이 상황을 시뮬레이션하기 위해, 이전의 동일한 예제 앱을 [different origin](https://rawcdn.rawgit.net/GoogleChrome/samples/c4192bd7a3c66fc288a7b22b77acb935df00b8a1/transformersjs-automatic-speech-recognition/index.html)에서 제공하는 것으로 가정합니다. 이 다른 원본(origin) 애플리케이션을 방문하면, 거의 즉시 사용할 수 있도록 하는 대신 브라우저는 모든 모델 리소스를 다시 다운로드하고 캐시해야 하므로 바이트 단위로 동일하더라도 중복 다운로드 및 저장이 발생합니다. 이 toy 예제에서도 이는 누적되어 177 MB의 중복 다운로드 및 저장으로 이어진다는 점을 Chrome DevTools의 Storage 섹션에서 확인할 수 있습니다 [Application panel](https://developer.chrome.com/docs/devtools/application#open_the_application_panel). 이를 상상해 보실 수 있습니다.

![The Chrome DevTools Storage overview showing 177 MB of used storage.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/9byoniem0pw.png)

### Wasm 런타임 리소스

더 악화됩니다. toy 예제에 두 번째 파이프라인으로 감성 분석을 추가해 보겠습니다. 감성 분석 [by default](https://github.com/huggingface/transformers.js/blob/bc9cf7400f4f2c8695016699f879e31026ff0313/packages/transformers/src/pipelines/index.js#L65)은 [`Xenova/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english) 모델을 사용합니다. 모델을 명시하지 않으면 Transformers.js의 기본 모델 해상도는 자동으로 이를 선택합니다.

```js
const classifier = await pipeline('sentiment-analysis');
const sentiment = await classifier(result.text);
pre.append('\n\n' + JSON.stringify(sentiment, null, 2));
```


![image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/le7l1km7o4g.png)

두 개의 완전히 다른 AI 모델이지만, 이들은 Transformers.js가 기반으로 삼는 동일한 4,733 kB `ort-wasm-simd-threaded.asyncify.wasm` WebAssembly(Wasm) 런타임 파일 [from the underlying ONNX Runtime library](https://onnxruntime.ai/docs/api/js/interfaces/Env.WasmFilePaths.html#wasm)에 의존합니다. [extended demo on a different origin](https://rawcdn.rawgit.net/GoogleChrome/samples/d47114a15637383015c274e7bdcd81e1a17b0ccf/transformersjs-automatic-speech-recognition/index2.html)를 열어보면 Wasm 런타임도 다시 다운로드되고 캐시된다는 것을 [**Network** tab](https://developer.chrome.com/docs/devtools/network#load)에서 확인하실 수 있습니다.

![Chrome DevTools Network panel showing the download of the Wasm runtime resource.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/pz12g20fqeg.png)

따라서 동일한 AI 모델을 공유하지 않는 앱이라도 브라우저는 이미 보유 중인 공유 Wasm 리소스에 대해 중복 요청을 하며, 그 위에 다시 캐시하기도 하여 하드 디스크 공간을 차지합니다.

### 캐시 분리

#### AI 모델 리소스 제공

기본적으로, **AI 모델 리소스**는 [Hugging Face Hub](https://huggingface.co/docs/hub/en/models-the-hub)에서 오며 궁극적으로 허깅페이스 CDN에서 제공합니다. 브라우저는 [`https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json`](https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json) 같은 리소스를 요청하고, 이 리소스는 이 경우 최종 CDN URL인 [`https://huggingface.co/api/resolve-cache/models/Xenova/distilbert-base-uncased-finetuned-sst-2-english/0b6928efcb76139cae2c6881d49cda67fe119f42/config.json?%2FXenova%2Fdistilbert-base-uncased-finetuned-sst-2-english%2Fresolve%2Fmain%2Fconfig.json=&etag=%223c36342ef1f74de2797d667c68c6b7b988d0b87c%22`](https://huggingface.co/api/resolve-cache/models/Xenova/distilbert-base-uncased-finetuned-sst-2-english/0b6928efcb76139cae2c6881d49cda67fe119f42/config.json?%2FXenova%2Fdistilbert-base-uncased-finetuned-sst-2-english%2Fresolve%2Fmain%2Fconfig.json=&etag=%223c36342ef1f74de2797d667c68c6b7b988d0b87c%22)로 리다이렉션됩니다.

#### Wasm 런타임 리소스 제공

기본적으로 **Wasm 런타임 리소스**는 [jsDelivr CDN](https://www.jsdelivr.com/)에서 제공됩니다. 예를 들어, `ort-wasm-simd-threaded.asyncify.wasm`은 이 글을 쓰는 시점에 [`https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm`](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm)에서 나옵니다.

다양한 앱이 서로 다른 오리진에서 실행되더라도 결국 동일한 CDN URL에서 리소스를 제공한다면 캐싱 문제는 없을 것이라고 생각할 수 있습니다. 하지만 오랜 기간 브라우저에서의 캐싱 방식은 그렇지 않습니다. 기사 [Gaining security and privacy by partitioning the cache](https://developer.chrome.com/blog/http-cache-partitioning)가 모든 세부 정보를 다룹니다. 본질적으로, 캐시가 오리진별로 분리되어 있어 타이밍 공격을 방지합니다: 웹사이트가 HTTP 요청에 응답하는 데 걸리는 시간은 브라우저가 과거에 같은 리소스에 접근했다는 것을 암시할 수 있어 보안 및 개인정보 유출 취약점을 만들 수 있습니다.

#### Chrome의 구현

구체적인 구현은 브라우저에 따라 다를 수 있지만 Chrome에서는 캐시된 리소스가 **리소스 URL** 외에 네트워크 격리 키(Network Isolation Key)로도 키가 부여됩니다. 네트워크 격리 키는 **최상위 사이트**와 **현재 프레임 사이트**로 구성됩니다. 앞의 toy 예제가 `https://googlechrome.github.io`와 `https://rawcdn.rawgit.net`에서 호스팅되었다고 가정하고, 둘 다 `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm`의 Wasm 런타임을 사용한다면, 캐시 키는 아래 표와 같이 보일 것입니다.

<table>
  <thead>
    <tr>
      <th colspan="2">Network Isolation Key</th>
      <th rowspan="2"><strong>Resource URL</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Top-level site</strong></td>
      <td><strong>Current-frame site</strong></td>
    </tr>
    <tr>
      <td><p><pre>
https://googlechrome.github.io
</pre></p></td>
      <td><p><pre>
https://googlechrome.github.io
</pre></p></td>
      <td><p><pre>
https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm
</pre></p></td>
    </tr>
    <tr>
      <td><p><pre>
https://rawcdn.rawgit.net
</pre></p></td>
      <td><p><pre>
https://rawcdn.rawgit.net
</pre></p></td>
      <td><p><pre>
https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm
</pre></p></td>
    </tr>
  </tbody>
</table>

리소스 URL이 정확히 같더라도 네트워크 격리 키가 일치하지 않으면 캐시 히트가 발생하지 않아 중복 다운로드 및 중복 저장이 발생합니다. 이것이 Cross-Origin Storage 제안이 해결하고자 하는 문제입니다.

## Cross-Origin Storage API 도입 {#section-2}

> **💡 주의:** Cross-Origin Storage API는 초기 단계의 제안으로 최종 확정된 것은 아닙니다. 제안된 API가 아직 어떤 브라우저에서도 기본적으로 구현되어 있지 않더라도, 실험해 보기를 기다릴 필요는 없습니다. [Cross-Origin Storage extension](https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih)를 설치하여 모든 페이지에 `navigator.crossOriginStorage` 폴리필을 주입하고 전체 흐름을 테스트해 보십시오.

제안된 **[Cross-Origin Storage](https://github.com/WICG/cross-origin-storage) (COS) API**는 웹 앱이 원점 경계를 넘어 큰 파일을 저장하고 검색할 수 있도록 하는 전용 `navigator.crossOriginStorage` 인터페이스를 도입합니다. 이는 URL이 아니라 암호학적 해시로 식별됩니다.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/klwb5fryaa.png" alt="The Cross-Origin Storage API logo: a stylized walking person, as typically encountered on crosswalk signs." width="200" height="200">

암호학적 해시에 관한 마지막 포인트가 핵심입니다. COS는 파일을 URL이나 오리진이 아니라 **해시**로 식별하기 때문에, `ort-wasm-simd-threaded.asyncify.wasm` Wasm 런타임이 `https://googlechrome.github.io` 방문 중 다운로드한 것과 `https://rawcdn.rawgit.net`가 막 요청하려는 것이 동일한 것으로 인식됩니다. 어느 오리진이 그것을 가져갔든 상관없이 말이죠. 기본 흐름을 보여주는 아래 코드 조각을 확인해 보세요.

```js
const hash = {
  algorithm: 'SHA-256',
  value: '8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4',
};

try {
  const handle = await navigator.crossOriginStorage.requestFileHandle(hash);
  // Cache hit! Get the file as a Blob and use it directly.
  const fileBlob = await handle.getFile();
} catch (err) {
  // Cache miss. Download from network, then store for next time.
  const fileBlob = await fetch('https://cdn.jsdelivr.net/.../ort-wasm-simd-threaded.asyncify.wasm')
    .then(r => r.blob());
  const handle = await navigator.crossOriginStorage.requestFileHandle(
    hash,
    { create: true, origins: '*' },
  );
  const writableStream = await handle.createWritable();
  await writableStream.write(fileBlob);
  await writableStream.close();  
}
```


리소스가 COS에 있으면, [`FileSystemFileHandle`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemFileHandle)를 받아 이를 통해 [`getFile()`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemFileHandle/getFile)로 Blob 데이터를 직접 읽을 수 있습니다(결과로 생성되는 [`File`](https://developer.mozilla.org/en-US/docs/Web/API/File)는 [`Blob`](https://developer.mozilla.org/en-US/docs/Web/API/Blob)에서 파생됩니다). COS에 리소스가 없으면 네트워크로 폴백하고, 다음 앱이 필요로 할 때를 위해 COS에 리소스를 기록합니다. 이는 귀하의 앱일 수도 있고, 전혀 무관한 다른 앱일 수도 있으며, 완전히 다른 오리진에서의 실행일 수도 있습니다.

이 API는 의도적으로 [File System Standard](https://fs.spec.whatwg.org/)의 [`FileSystemDirectoryHandle.getFileHandle()`](https://developer.mozilla.org/en-US/docs/Web/API/FileSystemDirectoryHandle/getFileHandle)를 본받아 설계되었습니다. 이는 [Origin Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)(OPFS) API에서 익숙하게 보신 부분과 같습니다. `hash` 매개변수는 OPFS의 `name` 매개변수와 같은 역할을 하여 자원을 고유하게 식별합니다. `options.create` 플래그도 같은 방식으로 작동합니다: 읽기 전용 접근의 경우 부재 또는 `false`이며, 쓰려는 경우 `true`가 활성화됩니다.

### 누가 무엇을 읽을 수 있는지 제어

모든 리소스를 전역적으로 공유해서는 안됩니다. COS는 파일 저장 시 가시성을 제어하기 위해 `origins` 옵션을 통해 개발자에게 정밀한 제어를 제공합니다.

* `origins: '*'`를 설정하면 파일이 **전역적으로 사용 가능**해집니다. 어떤 오리진이든 해시로 그것을 찾을 수 있습니다. 이것은 AI 모델 리소스나 Transformers.js 예제의 Wasm 런타임에 적합한 선택입니다: 웹의 모든 앱이 하나의 캐시된 복사본으로부터 이익을 얻는 것이 핵심입니다.
* 특정 오리진 목록, 예를 들어 `origins: ['https://write.example.com', 'https://calculate.example.com']`과 같은 목록은 이 사이트들에 대한 접근을 **제한**합니다. 이는 서로의 소유 자산 간에 공유되지만 다른 누구도 검색될 필요가 없는 리소스와 같은 사례에 잘 맞습니다. 예: 상업용 오피스 도구에서 사용되는 독점 교정 AI 모델.
* `origins`를 완전히 생략하면 파일은 오직 **[same-site](https://web.dev/articles/same-site-same-origin#same-site-cross-site) 오리진**에서만 사용할 수 있습니다. 이는 조직의 모든 하위 도메인 간에 공유되는 리소스에 대한 합리적인 기본값이지만, 조직 경계를 넘지 않도록 의도된 경우가 많습니다.

하나 중요한 규칙: 가시성은 상향은 가능하지만 하향은 불가능합니다. 파일이 이미 전역적으로 공유 가능하면, 나중에 제한된 `origins` 목록으로 저장하려는 시도는 묵시적으로 무시됩니다. 이는 악의적인 행위자가 공개 리소스를 재저장하고 가용성을 축소하는 것을 방지합니다. 반대로도 가능합니다: 처음에 제한된 `origins` 목록으로 저장된 파일은 나중에 더 관대하게 설정될 수 있습니다. 어떤 사이트든, 원래 저장자뿐 아니라, 같은 해시(`requestFileHandle()`)에 대해 같은 해시를 가진 리소스에 대해 더 넓은 `origins` 값을 갖고 호출할 수 있으며, 브라우저가 해시가 일치하는지 확인하면 그 리소스는 그 시점부터 더 넓은 대중에게 제공됩니다. 업그레이드가 이루어져도 반환된 핸들을 통해 전체 파일을 여전히 써야 한다는 점에 주의하십시오. 이 요건은 COS에 특정 파일이 이미 저장되어 있는지 여부를 악용하려는 사이드 채널을 방지하기 위해 존재합니다.

### Integrity by design

COS의 미묘하지만 중요한 속성 중 하나는 브라우저가 파일을 저장할 때 해시를 검증한다는 점입니다. 저장하려는 데이터가 선언된 해시와 일치하지 않으면 기록은 오류로 실패합니다. 이로써 무결성 검사는 자동으로 이루어집니다: COS에서 파일을 읽는 앱은 기대한 정확한 바이트를 받고 있다고 확신할 수 있습니다. 네트워크 다운로드 후 직접 해시를 계산했다면 얻을 수 있었던 보장과 동일합니다.

이것은 Transformers.js 시나리오에서 이중으로 유용합니다. 오늘날 모델 가중치를 다운로드한 후에는 CDN이 올바른 바이트를 서비스했는지 확인하는 실용적인 방법이 거의 없습니다. COS를 사용하면 스토어의 모든 파일이 기록 시점에 암묵적으로 검증되므로, 출처가 허깅페이스 CDN이든 임의의 사이트의 자체 호스팅 미러이든 관계없이 검증됩니다.

### 프라이버시를 해치지 않는 유용성

물론 교차 출처 공유 캐시는 분할된 HTTP 캐시와 같은 문제를 역으로 제기합니다: 어떤 사이트든 해시로 파일의 존재 여부를 확인할 수 있다면, 예를 들어 게임 엔진 Wasm 모듈이 캐시되어 있는지 확인함으로써 사용자의 브라우징 이력에 대해 뭔가를 알아낼 수 있지 않을까요?

COS는 이를 두 가지 보완적 메커니즘으로 해결합니다:

- 먼저, `origins` 필드: 전역적으로 탐지 가능하지 않는 독점 리소스는 단순히 `origins: '*'`로 저장해서는 안 됩니다. 필요할 때마다 개발자 교육을 통해 고려되도록 권장됩니다.
- 둘째, 접근 가능성 게이팅(availability gating): 전역으로 선언된 파일이라도 충분한 수의 서로 다른 오리진에서 충분히 많이 만나지 않았다면 브라우저가 파일의 존재 여부 확인을 억제할 수 있습니다. 한두 사이트에만 나타나는 파일도 교차 사이트 식별자로 작용할 수 있기 때문에, 브라우저는 파일이 존재하지 않는 것처럼 오류를 반환할 수 있습니다. Chrome 팀은 잘 알려지지 않은 리소스에서 발생할 수 있는 프라이버시 누수를 의식하고 있으며, 캐시될 수 있는 정확한 리소스의 제한을 통해 일반적으로 이를 완화할 계획입니다. 구체적인 완화 방법은 아직 확정 중입니다.

중요한 점은 에러가 결정적인 답이 아니라는 점입니다. 이는 "저장되지 않음"일 수도 있고, "저장되었지만 브라우저가 알려주지 않음"일 수도 있습니다. 애플리케이션은 항상 같은 방식으로 처리해야 하며 네트워크로의 폴백을 수행해야 합니다.

### Transformers.js 예제에 대한 시사점

토이 예제에서 돌아가 보자면: `ort-wasm-simd-threaded.asyncify.wasm` 런타임은 4,733 kB 크기로, 어떤 AI 모델을 사용하든 Transformers.js로 구동되는 모든 앱에서 공유됩니다. COS를 사용하면 이를 처음 로드하는 앱이 한 번 다운로드하고 그 해시의 SHA-256로 `origins: '*'`에 저장합니다. 이후의 모든 앱은 `https://googlechrome.github.io`나 `https://rawcdn.rawgit.net` 또는 다른 오리진에서도 즉시 COS에서 이를 찾습니다. 177 MB의 중복 Whisper 모델 가중치? 같은 이야기입니다: `Xenova/whisper-tiny.en`가 한 번 다운로드되고 두 번째 실행에서 해시로 인식되며 COS에서 밀리초 단위로 제공합니다. 그리고 물론 같은 일은 `Xenova/distilbert-base-uncased-finetuned-sst-2-english`에서도 발생합니다.

Transformers.js 자체도 COS API를 라이브러리 차원에서 이미 시도 중입니다. [Pull request #1549](https://github.com/huggingface/transformers.js/pull/1549)은 선택적 플래그(opt-in) 뒤에 실험적 COS 캐시 백엔드를 도입했습니다. 파이프라인을 설정하기 전에 한 줄의 명령으로 활성화합니다:

```js
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

// 👇 Opt in to the experimental Cross-Origin Storage cache backend.
env.experimental_useCrossOriginStorage = true;

const asr = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en', { device: 'webgpu' });
const result = await asr('jfk.wav');
console.log(result);
```


해당 플래그를 설정하면, 각 [Xet-tracked](https://huggingface.co/docs/hub/en/xet/index) 모델 파일(대형 ONNX 가중치 파일)의 SHA-256 해시를 원시 포인터 [example raw pointer file](https://huggingface.co/Xenova/whisper-tiny.en/raw/main/onnx/decoder_model.onnx)를 가져와서 `oid sha256:` 필드를 추출해 얻은 해시를 `navigator.crossOriginStorage`의 키로 사용합니다. 만약 다른 사이트가 먼저 COS에 저장해 두었다면 네트워크 왕복 없이 즉시 제공됩니다. 그렇지 않으면 일반 다운로드로 폴백하고 다음 호출자를 위해 COS에 결과를 저장합니다. 토이 예제에서 실제 이점은 `Xenova/whisper-tiny.en`와 `Xenova/distilbert-base-uncased-finetuned-sst-2-english`(그리고 물론 `ort-wasm-simd-threaded.asyncify.wasm`)가 서로 다른 오리진에서 요청하더라도 한 번만 교차하면 된다는 점입니다.

해당 플래그의 `experimental_` 접두사는 의도적이며, 기반 브라우저 API가 아직 표준화되지 않았고 메이저 버전 증가 없이도 변경될 수 있음을 시사합니다.

### 지금 바로 시도해 보기

COS API는 아직 어떤 브라우저에서도 기본적으로 구현되어 있지 않지만, 실험해 볼 수 있는 기회는 있습니다. 모든 페이지에 `navigator.crossOriginStorage` 폴리필을 주입하기 위해 [Cross-Origin Storage extension](https://chromewebstore.google.com/detail/cross-origin-storage/denpnpcgjgikjpoglpjefakmdcbmlgih)를 설치하고 전체 흐름을 테스트해 보십시오. 시작하려면 [source code of the extension](https://github.com/web-ai-community/cross-origin-storage-extension)를 확인하고 [usage instructions](https://github.com/web-ai-community/cross-origin-storage-extension?tab=readme-ov-file#usage)를 따라가 보세요.

![Chrome Web Store page for the Cross-Origin Storage extension.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/0q3rowmy67ta.png)

확장 프로그램이 설치되면 지금 바로 전체 엔드투엔드 경험을 시도해 볼 수 있습니다: 첫 번째 [toy example with COS enabled](https://googlechrome.github.io/samples/transformersjs-automatic-speech-recognition/index3.html)를 열고 `Xenova/whisper-tiny.en`가 로드되도록 한 뒤 [toy example with COS enabled from the second origin](https://rawcdn.rawgit.net/GoogleChrome/samples/1e4f2b8c10adc394352c6ec8327bb503bac7aba1/transformersjs-automatic-speech-recognition/index3.html)를 열어보세요. 이전에 보셨던 177 MB의 재다운로드 대신, 모델은 COS에서 밀리초 단위로 제공합니다. 확장 팝업 창을 열면 COS가 작동하는 것을 볼 수 있습니다. 만약 **리소스로 보기(View by Resource)**를 클릭하면 SHA-256 해시 `950978b1dbcbf250335358c1236053ba19a7f7849b33dc777f4421b72b7626fa`가 있는 리소스가 `https://googlechrome.github.io` 및 `https://rawcdn.rawgit.net`에 걸쳐 공유되는 것을 볼 수 있습니다. 눈에 띄지 않을 수 있지만 허깅페이스에서 SHA-256 해시를 비교해 보면 [`https://huggingface.co/Xenova/whisper-tiny.en/blob/main/onnx/decoder_model_merged.onnx`](https://huggingface.co/Xenova/whisper-tiny.en/blob/main/onnx/decoder_model_merged.onnx)를 보고 계신다는 것을 확인할 수 있습니다. 지금은 이 확장이 주로 당신과 같은 파워 유저를 위한 것입니다. 브라우저에 구현되면 브라우저의 **설정(Settings)** 페이지에 더 친숙한 통합이 제공될 것입니다. 아래 스크린샷은 활성화된 **리소스로 보기** 탭이 있는 확장 팝업 창으로, 공유 리소스와 그것의 해시, COS 캐시에 이를 보유한 두 원점(origin)을 확인할 수 있습니다.

![A resource seen in the Cross-Origin Storage extension, showing it's shared between two origins.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cross-origin-storage/usg5dq7dhm.png)

## 행동 촉구 {#section-3}

자신의 Transformers.js 앱을 구축 중이라면, 행동 촉구는 간단합니다: 첫 번째 `pipeline()` 호출 전에 `env.experimental_useCrossOriginStorage = true`를 추가하고, 확장 프로그램을 설치한 뒤 네트워크 탭에서 중복 다운로드가 사라지는 것을 지켜보세요. 참여하는 모든 사이트는 다른 사이트의 사용자들에게 더 빠르고 저렴한 경험을 제공합니다. 참여는 완전히 위험이 없으며: COS API가 지원되지 않는 경우(사용자가 COS 확장 설치를 하지 않아서) 코드는 기본 경로( [Web Cache](https://developer.mozilla.org/en-US/docs/Web/API/Cache) API)로 폴백합니다.

Transformers.js는 COS를 실험하는 데 있어 독립적이지 않습니다. [WebLLM](https://webllm.mlc.ai/)(옵트인, 자세한 내용은 [documentation](https://webllm.mlc.ai/docs/user/advanced_usage.html#using-cross-origin-storage-cache) 참조)과 [wllama](https://github.com/ngxson/wllama)(자동, 자세한 내용은 [PR](https://github.com/ngxson/wllama/pull/248) 참조) 역시 이 제안된 API에 대해 기대를 가지고 있습니다.

크롬 팀은 브라우저에서 이를 네이티브로 구현하기 위해 노력하고 있습니다 [considering implementing the COS API](https://chromestatus.com/feature/5163371507875840). 초기 단계의 제안으로, API와 제안의 형태에 대한 피드백을 환영합니다. 이슈를 제기하려면 [Cross-Origin Storage repository](https://github.com/WICG/cross-origin-storage)를 이용하고, [express support](https://github.com/WICG/cross-origin-storage/labels/expression%20of%20support)를 확인하거나 PR을 여세요.

