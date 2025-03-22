// 导航栏平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    });
});

// 表单提交处理
const contactForm = document.getElementById('contact-form');
if(contactForm) {
    contactForm.addEventListener('submit', function(e) {
        e.preventDefault();
        // 此处需替换为实际表单处理逻辑
        alert('Form submitted successfully! Thank you for your message.');
        this.reset();
    });
}


// 分页初始化
const projectList = document.querySelector('.project-list');
const cards = Array.from(document.querySelectorAll('.project-card'));
const pageHeight = window.innerHeight;

// 创建分页容器
while(projectList.firstChild) {
  projectList.removeChild(projectList.firstChild);
}

// 将卡片分配到各页
// 精确分配每页两个项目
for(let i=0; i<cards.length; i+=2) {
  if(!cards[i]) break;
  const page = document.createElement('div');
  page.className = 'project-page' + (i===0 ? ' active' : '');
  page.appendChild(cards[i]);
  if(cards[i+1]) page.appendChild(cards[i+1]);
  projectList.appendChild(page);
}

// 滚动事件监听
let lastScroll = 0;
// 添加防抖的滚动处理
let isScrolling;
window.addEventListener('wheel', (e) => {
  window.clearTimeout(isScrolling);
  isScrolling = setTimeout(() => {
  const currentPage = document.querySelector('.project-page.active');
  const nextPage = currentPage.nextElementSibling;
  const prevPage = currentPage.previousElementSibling;

  if(e.deltaY > 0 && nextPage) {
    currentPage.classList.remove('active');
    nextPage.classList.add('active');
    window.scrollTo({
  top: 0,
  behavior: 'smooth'
});
  } else if(e.deltaY < 0 && prevPage) {
    currentPage.classList.remove('active');
    prevPage.classList.add('active');
    window.scrollTo({
  top: 0,
  behavior: 'smooth'
});
  }
  }, 300);
}, { passive: true });

// 画廊交互逻辑
document.querySelectorAll('.grid-item').forEach(item => {
    item.addEventListener('click', () => {
        // 移除所有激活状态
        document.querySelectorAll('.grid-item').forEach(i => i.classList.remove('active'));
        // 设置当前激活状态
        item.classList.add('active');
        
        // 获取对应的轮播容器
        const carouselId = item.closest('.grid-container').id.replace('grid', 'carousel');
        const carousel = document.getElementById(carouselId);
        const index = Array.from(item.parentElement.children).indexOf(item);
        
        // 滚动到对应图片
        carousel.style.transform = `translateX(-${index * 100}%)`;
    });
});

// 响应式布局调整
window.addEventListener('resize', () => {
    document.querySelectorAll('.project-page').forEach(page => {
        page.style.minHeight = window.innerHeight + 'px';
    });
});