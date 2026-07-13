(function ($) {"use strict";

$(function () {
  var $window = $(window);

  var header = $(".start-style");

  var $sidebar = $(".sidebar");
  var $article = $(".mainheading");

  var $endScroll = $('.post-top-meta');

  if($endScroll.length == 0){
    $endScroll = $('.footer-card-container');

    if($endScroll.length == 0){
      $endScroll = $('.contribute_link');

      if($endScroll.length == 0){
        $endScroll = $('footer.footer');
      }
    }
  }

  var $toc = $('.toc-content #markdown-toc');

  var scrollTop = $('html,body').scrollTop();

  if($article.length){
    var article_top = $article.offset().top;

    if(scrollTop >= article_top){
      $sidebar.addClass("sticky")
    }else{
      $sidebar.removeClass("sticky")
    }

    var windowHeight = $window.height();
    var marginTop = windowHeight < $toc.height() ? ($toc.height() - windowHeight) + 90 : 0
    var article_top = $article.offset().top + marginTop;
    var article_end = $endScroll.offset().top - windowHeight - marginTop;

    if(scrollTop >= article_end){
      $sidebar.css("top",article_end - $window.height())
      $sidebar.removeClass("sticky")
    }else if(scrollTop >= article_top){
      $sidebar.removeAttr("style")
      $sidebar.addClass("sticky")
    }
  }

  $window.scroll(function () {
    var scrollTop = $('html,body').scrollTop();

    if (scrollTop > 0) {
      header.removeClass('start-style').addClass("scroll-on");
    } else {
      header.removeClass("scroll-on").addClass('start-style');
    }


    if($article.length){
      var windowHeight = $window.height();
      var marginTop = windowHeight < $toc.height() ? ($toc.height() - windowHeight) + 90 : 0
      var article_top = $article.offset().top;
      var article_end = $endScroll.offset().top - windowHeight - marginTop;

      if(scrollTop >= article_end){
        $sidebar.css("top",article_end - $window.height())
        $sidebar.removeClass("sticky")
        return
      }else{
        $sidebar.removeAttr("style")
        $sidebar.addClass("sticky")
      }

      if(scrollTop >= article_top){
          $sidebar.addClass("sticky")
        }else{
          $sidebar.removeClass("sticky")
        }
      }

    });

    $('.article-post table').wrap('<div class="table-container"></div>');
});

//Animation

$(document).ready(function () {
  $('body.hero-anime').removeClass('hero-anime');
});

//Menu On Hover

$('body').on('mouseenter mouseleave', '.nav-item', function (e) {
  if ($(window).width() > 750) {
    var _d = $(e.target).closest('.nav-item');_d.addClass('show');
    setTimeout(function () {
      _d[_d.is(':hover') ? 'addClass' : 'removeClass']('show');
    }, 1);
  }
});

// Blank Target External Links
$(document.links).filter(function() {
return this.hostname != window.location.hostname;
}).attr('target', '_blank');

// Sidebar Toggle
$('.sidebar-toggle-btn').on('click', function() {
  $('.sidebar-wrapper').toggleClass('show');
});

})(jQuery);