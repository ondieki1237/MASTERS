
/**
 * theme.js
 *
 * @package     theme_uonbi
 * @copyright   2022 onwards Catalyst IT Europe <http://catalyst-eu.net>
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
(function($){var img=$("header#header").find('.avatars').find('img[src$="/u/f2"]');var src=img.attr('src');img.attr('src',src+'_white');var msg=$("header#header").find('#nav-message-popover-container .nav-link').find("img[src$='t/message']");var msgsrc=msg.attr('src');msg.attr('src',msgsrc+"_white");var note=$("header#header").find('#nav-notification-popover-container .nav-link').find("img[src$='i/notifications']");var notesrc=note.attr('src');note.attr('src',notesrc+"_white");if($("#header .navbar-nav button").attr('aria-expanded')==="true"){$("#header .navbar-nav").find('button').addClass('is-active')}
$("#header .navbar-nav button").click(function(){var $this=$(this);setTimeout(function(){if($this.attr('aria-expanded')==="true"){$("#header .navbar-nav").find('button').addClass('is-active')}else{$("#header .navbar-nav").find('button').removeClass('is-active')}},200)});window.metismenu_observer=new MutationObserver(function(mutations,observer){for(var i=0;i<mutations.length;i++){mutation=mutations[i];if((!mutation.oldValue||mutation.oldValue.match(/\bclosed\b/))&&mutation.target.classList&&!mutation.target.classList.contains('closed')){$('#nav-drawer .metismenu a span').hide();setTimeout(function(){$('#nav-drawer .metismenu a span').show()},350)}}});if($('#nav-drawer').length){window.metismenu_observer.observe(document.querySelector('#nav-drawer'),{attributes:!0,attrubuteOldValue:!0,attributeFilter:['class']})}
if($('#page-my-index').length){$('.block_myprofile .userpicture').each(function(){var src=$(this).attr('src');$(this).attr('src',src.replace('f2','f1'))})}})(jQuery)