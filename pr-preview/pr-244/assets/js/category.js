/**
 * Category Filter for Blog Posts
 * Hugging Face KREW Blog
 */
document.addEventListener('DOMContentLoaded', function() {
  const filterBtns = document.querySelectorAll('.filter-btn');
  const postItems = document.querySelectorAll('.post-item');
  
  // 1개 이상의 필터가 있어야 필터 기능 적용
  if (filterBtns.length === 0 || postItems.length === 0) {
    return;
  }
  
  // 카테고리 필터링 함수
  function filterPostsByCategory(category, updateHistory = true) {
    filterBtns.forEach(btn => {
      btn.classList.remove('active');
      btn.blur();
    });
    
    const targetBtn = document.querySelector(`[data-category="${category}"]`);
    if (targetBtn) {
      targetBtn.classList.add('active');
    }
    
    postItems.forEach(item => {
      const categories = item.dataset.categories;
      
      if (category === 'all' || categories.includes(category)) {
        item.style.display = 'block';
      } else {
        item.style.display = 'none';
      }
    });
    
    if (updateHistory) {
      if (category !== 'all') {
        window.history.pushState({category: category}, null, '#category=' + category);
      } else {
        window.history.pushState({category: 'all'}, null, window.location.pathname);
      }
    }
  }
  
  filterBtns.forEach(btn => {
    btn.addEventListener('click', function() {
      const category = this.dataset.category;
      filterPostsByCategory(category, true);
    });
  });
  
  // URL 해시에서 카테고리 초기화
  function initializeFromHash() {
    const urlHash = window.location.hash;
    if (urlHash.startsWith('#category=')) {
      const category = urlHash.replace('#category=', '');
      // update history without filtering to avoid infinite loop
      filterPostsByCategory(category, false);
    } else {
      // if there is no hash, initialize with 'all' category
      filterPostsByCategory('all', false);
    }
  }
  
  initializeFromHash();
  
  window.addEventListener('popstate', function(event) {
    initializeFromHash();
  });
}); 