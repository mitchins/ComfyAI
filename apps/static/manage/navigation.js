// Shared navigation component for ComfyAI UI pages
// Automatically injects a consistent navigation bar at the top of pages

(function() {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initNavigation);
    } else {
        initNavigation();
    }
    
    function initNavigation() {
        const navHTML = `
            <nav class="comfyai-nav">
                <div class="nav-container">
                    <div class="nav-brand">
                        <a href="/" class="brand-link">🤖 ComfyAI</a>
                    </div>
                    <div class="nav-links">
                        <a href="/" class="nav-link" title="Home">🏠 Home</a>
                        <a href="/manage/ui/" class="nav-link" title="Model Management">🌐 Models</a>
                        <a href="/manage/ui/vision-test.html" class="nav-link" title="Vision & Face Testing">🎯 Test</a>
                        <a href="/docs" class="nav-link" title="API Documentation">📚 Docs</a>
                    </div>
                </div>
            </nav>
        `;
        
        const navStyles = `
            <style>
                body {
                    background: #f8f9fa !important;
                    margin: 0;
                    padding: 0;
                }
                .comfyai-nav {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    position: sticky;
                    top: 0;
                    z-index: 1000;
                    margin-bottom: 0;
                }
                .nav-container {
                    max-width: 1200px;
                    margin: 0 auto;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 0 1rem;
                    height: 60px;
                }
                .nav-brand .brand-link {
                    color: white;
                    text-decoration: none;
                    font-size: 1.5rem;
                    font-weight: bold;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                .nav-brand .brand-link:hover {
                    color: rgba(255,255,255,0.9);
                }
                .nav-links {
                    display: flex;
                    gap: 0.5rem;
                    align-items: center;
                }
                .nav-link {
                    color: rgba(255,255,255,0.9);
                    text-decoration: none;
                    padding: 0.5rem 1rem;
                    border-radius: 6px;
                    transition: all 0.3s ease;
                    font-size: 0.9rem;
                    white-space: nowrap;
                }
                .nav-link:hover {
                    background: rgba(255,255,255,0.1);
                    color: white;
                    transform: translateY(-1px);
                }
                .nav-link.active {
                    background: rgba(255,255,255,0.2);
                    color: white;
                }
                
                /* Mobile responsive */
                @media (max-width: 768px) {
                    .nav-container {
                        padding: 0 0.5rem;
                        height: auto;
                        min-height: 60px;
                        flex-wrap: wrap;
                    }
                    .nav-links {
                        gap: 0.25rem;
                        flex-wrap: wrap;
                    }
                    .nav-link {
                        padding: 0.4rem 0.8rem;
                        font-size: 0.8rem;
                    }
                }
                
                /* Adjust body padding if navigation is sticky */
                body.has-nav {
                    padding-top: 0;
                }
                
                /* Integration with existing Materialize CSS pages */
                .comfyai-nav + .container {
                    margin-top: 2rem;
                }
                
                /* Professional content styling - no cards for consistency */
                .main-content {
                    max-width: 1200px;
                    margin: 2rem auto;
                    padding: 0 1rem;
                }
                .main-container {
                    max-width: 1200px;
                    margin: 2rem auto;
                    padding: 0 2rem;
                }
                
                /* Integration with all page types */
                .comfyai-nav + .container,
                .comfyai-nav + .main-container {
                    margin-top: 0;
                }
                .comfyai-nav + .container .header,
                .comfyai-nav + .main-container .page-header {
                    border-radius: 0;
                    margin-top: 0;
                }
            </style>
        `;
        
        // Insert navigation at the beginning of body (only if not already present)
        if (!document.querySelector('.comfyai-nav')) {
            document.head.insertAdjacentHTML('beforeend', navStyles);
            document.body.insertAdjacentHTML('afterbegin', navHTML);
            document.body.classList.add('has-nav');
        }
        
        // Highlight current page
        highlightCurrentPage();
    }
    
    function highlightCurrentPage() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === currentPath || 
                (href === '/manage/ui/' && currentPath.includes('/manage/ui/')) ||
                (href === '/docs' && currentPath.includes('/docs'))) {
                link.classList.add('active');
            }
        });
    }
})();