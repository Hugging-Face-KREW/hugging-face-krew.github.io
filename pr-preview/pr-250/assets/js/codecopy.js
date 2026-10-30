// 코드블럭에 헤더 추가 및 복사 기능
(function () {
  // 복사 아이콘 SVG
  var copyIcon = '<svg aria-hidden="true" role="graphics-symbol" viewBox="0 0 14 16" class="copy" style="width: 16px; height: 16px; display: block; fill: inherit; flex-shrink: 0; padding-right: 4px;"><path d="M2.404 15.322h5.701c1.26 0 1.887-.662 1.887-1.927V12.38h1.154c1.254 0 1.91-.662 1.91-1.928V5.555c0-.774-.158-1.266-.626-1.74L9.512.837C9.066.387 8.545.21 7.865.21H5.463c-1.254 0-1.91.662-1.91 1.928v1.084H2.404c-1.254 0-1.91.668-1.91 1.933v8.239c0 1.265.656 1.927 1.91 1.927zm7.588-6.62c0-.792-.1-1.161-.592-1.665L6.225 3.814c-.452-.462-.844-.58-1.5-.591V2.215c0-.533.28-.832.843-.832h2.38v2.883c0 .726.386 1.113 1.107 1.113h2.83v4.998c0 .539-.276.832-.844.832H9.992V8.701zm-.79-4.29c-.206 0-.288-.088-.288-.287V1.594l2.771 2.818H9.201zM2.503 14.15c-.563 0-.844-.293-.844-.832V5.232c0-.539.281-.837.85-.837h1.91v3.187c0 .85.416 1.26 1.26 1.26h3.14v4.476c0 .54-.28.832-.843.832H2.504zM5.79 7.816c-.24 0-.346-.105-.346-.345V4.547l3.223 3.27H5.791z"></path></svg>복사';
  
  // 언어명 추출 함수
  function getLanguage(block) {
    // Rouge가 생성하는 부모 요소에서 언어 클래스 찾기
    var parent = block.parentElement;
    if (parent && parent.className) {
      var match = parent.className.match(/language-(\w+)/);
      if (match) {
        return match[1];
      }
    }
    // 클래스명에서 직접 찾기
    if (block.className) {
      var match = block.className.match(/language-(\w+)/);
      if (match) {
        return match[1];
      }
    }
    return 'plain text';
  }
  
  // 코드블럭에 헤더 및 복사 버튼 추가
  document.addEventListener('DOMContentLoaded', function() {
    // Rouge가 생성하는 최상위 .highlight만 선택 (중첩된 것 제외)
    document.querySelectorAll('.highlighter-rouge .highlight, .highlight:not(.highlighter-rouge .highlight .highlight)').forEach(function(block) {
      // 이미 헤더가 있거나 부모에 헤더가 있으면 건너뛰기
      if (block.querySelector('.code-header') || block.closest('.highlighter-rouge').querySelector('.code-header')) return;
      
      // 언어명 추출
      var language = getLanguage(block);
      
      // 헤더 생성
      var header = document.createElement('div');
      header.className = 'code-header';
      
      // 언어명 표시
      var langSpan = document.createElement('span');
      langSpan.className = 'code-language';
      langSpan.textContent = language;
      
      // 복사 버튼 생성
      var btn = document.createElement('button');
      btn.className = 'copy-code-button';
      btn.type = 'button';
      btn.innerHTML = copyIcon;
      
      // 헤더에 요소들 추가
      header.appendChild(langSpan);
      header.appendChild(btn);
      
      // 코드블럭 맨 앞에 헤더 삽입
      block.insertBefore(header, block.firstChild);
    });
  });

  // 복사 기능
  document.addEventListener('click', function(e) {
    if (!e.target.classList.contains('copy-code-button')) return;
    var btn = e.target;
    var codeBlock = btn.closest('.highlight');
    var codeElem = codeBlock ? codeBlock.querySelector('code') : null;
    if (!codeElem) return;
    var codeText = codeElem.innerText;
    if (navigator.clipboard) {
      navigator.clipboard.writeText(codeText).then(function() {
        var original = btn.innerHTML;
        btn.innerHTML = '코드 복사 완료';
        setTimeout(function() { btn.innerHTML = original; }, 1200);
      });
    } else {
      var textarea = document.createElement('textarea');
      textarea.value = codeText;
      document.body.appendChild(textarea);
      textarea.select();
      try {
        document.execCommand('copy');
        var original = btn.innerHTML;
        btn.innerHTML = '코드 복사 완료';
        setTimeout(function() { btn.innerHTML = original; }, 1200);
      } catch (err) {}
      document.body.removeChild(textarea);
    }
  });
})(); 