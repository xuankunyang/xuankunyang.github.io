(function() {
    // Configuration
    const CONFIG = {
        particleCount: 5500,
        colors: {
            cyan:   { h: 185, s: 100, l: 60 },
            purple: { h: 270, s: 95,  l: 65 },
            green:  { h: 150, s: 90,  l: 60 },
            orange: { h: 30,  s: 100, l: 60 },
            rose:   { h: 340, s: 95,  l: 60 },
        }
    };

    class ParticleAvatar {
        constructor(canvasId) {
            console.log('ParticleAvatar constructor called with ID:', canvasId);
            this.canvas = document.getElementById(canvasId);
            if (!this.canvas) {
                console.error('Canvas element not found:', canvasId);
                return;
            }
            
            this.ctx = this.canvas.getContext('2d', { alpha: false });
            this.particles = [];
            this.animationFrame = null;
            this.mouse = { x: -1000, y: -1000 };
            this.time = 0;
            
            // State
            this.mood = 'neutral'; // happy, sad, thinking, surprised, neutral
            this.themeColor = 'cyan';
            this.isDarkMode = false;
            
            // Bind methods
            this.resize = this.resize.bind(this);
            this.handleMouseMove = this.handleMouseMove.bind(this);
            this.handleMouseOut = this.handleMouseOut.bind(this);

            this.init();
        }

        init() {
            this.resize();
            window.addEventListener('resize', this.resize);
            window.addEventListener('mousemove', this.handleMouseMove);
            window.addEventListener('mouseout', this.handleMouseOut);
            
            // Check initial dark mode
            this.checkDarkMode();
            
            // Start loop
            this.animate();
        }

        handleMouseMove(e) {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        }

        handleMouseOut() {
            this.mouse.x = -1000;
            this.mouse.y = -1000;
        }
        
        checkDarkMode() {
            // Check for 'dark' class on body or html (HugoBlox standard)
            // Also check for 'dark-mode' just in case
            const body = document.body;
            const html = document.documentElement;
            const isDark = body.classList.contains('dark') || 
                           html.classList.contains('dark') ||
                           body.classList.contains('dark-mode');
            this.isDarkMode = isDark;
        }

        resize() {
            if (!this.canvas) return;
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
            this.initParticles();
        }

        initParticles() {
            const width = this.canvas.width;
            const height = this.canvas.height;
            const cx = width / 2;
            const cy = height / 2;
            const TWO_PI = Math.PI * 2;
            
            this.particles = [];
            
            for (let i = 0; i < CONFIG.particleCount; i++) {
                const angle = Math.random() * TWO_PI;
                const rRandom = Math.random();
                const radius = rRandom * rRandom * (Math.min(width, height) * 0.55);
                const size = Math.random() * 1.6 + 0.2;
                
                this.particles.push({
                    x: cx + Math.cos(angle) * radius,
                    y: cy + Math.sin(angle) * radius,
                    baseX: cx,
                    baseY: cy,
                    angle: angle,
                    radius: radius,
                    speed: 0.0005 + Math.random() * 0.002,
                    size: size,
                    alpha: Math.random() * 0.5 + 0.1,
                    targetAlpha: Math.random() * 0.8 + 0.2,
                    hueOffset: Math.random() * 60 - 30,
                    orbitOffset: Math.random() * TWO_PI,
                    brightness: size > 1.2 ? 1.3 : 0.7
                });
            }
        }

        update() {
            const width = this.canvas.width;
            const height = this.canvas.height;
            const cx = width / 2;
            const cy = height / 2;
            const breathSpeed = 0.008;
            const breathAmplitude = 0.08;
            const breathingScale = 1 + Math.sin(this.time * breathSpeed) * breathAmplitude;
            
            this.particles.forEach(p => {
                let targetRadius = p.radius;
                let moveSpeed = p.speed;
                
                switch (this.mood) {
                    case 'happy':
                        targetRadius = p.radius * 1.5;
                        moveSpeed = p.speed * 3.0;
                        p.angle += moveSpeed * 0.5;
                        break;
                    case 'sad':
                        targetRadius = p.radius * 0.3;
                        moveSpeed = p.speed * 0.5;
                        p.angle += moveSpeed * 3.0;
                        break;
                    case 'thinking':
                        const ring = (Math.floor(p.orbitOffset * 10) % 4) + 1;
                        targetRadius = 90 * ring * (1 + Math.sin(this.time * 0.02 + p.orbitOffset) * 0.15);
                        p.angle += moveSpeed * (ring % 2 === 0 ? 1 : -1) * 1.5;
                        break;
                    case 'surprised':
                        targetRadius = p.radius + Math.sin(this.time * 0.05 + p.orbitOffset) * 60;
                        p.angle += Math.sin(this.time * 0.03) * 0.005;
                        break;
                    case 'neutral':
                    default:
                        p.angle += moveSpeed;
                        break;
                }
                
                const r = targetRadius * breathingScale;
                
                let mx = cx + Math.cos(p.angle) * r;
                let my = cy + Math.sin(p.angle) * r;
                
                const dx = this.mouse.x - mx;
                const dy = this.mouse.y - my;
                const dist = Math.sqrt(dx*dx + dy*dy);
                const interactRadius = 300;
                
                let pushX = 0;
                let pushY = 0;
                
                if (dist < interactRadius) {
                    const force = (interactRadius - dist) / interactRadius;
                    const angleToMouse = Math.atan2(dy, dx);
                    const pushStrength = 50;
                    pushX = -Math.cos(angleToMouse) * force * pushStrength;
                    pushY = -Math.sin(angleToMouse) * force * pushStrength;
                }
                
                const desiredX = cx + Math.cos(p.angle) * r + pushX;
                const desiredY = cy + Math.sin(p.angle) * r + pushY;
                
                p.x += (desiredX - p.x) * 0.035;
                p.y += (desiredY - p.y) * 0.035;
                
                p.alpha += (p.targetAlpha - p.alpha) * 0.03;
                if (Math.random() > 0.99) {
                    p.targetAlpha = Math.random() * 0.8 + 0.1;
                }
            });
            
            this.time += 1;
        }

        draw() {
            if (!this.ctx) return;
            const width = this.canvas.width;
            const height = this.canvas.height;
            const theme = CONFIG.colors[this.themeColor];
            
            this.checkDarkMode();
            
            this.ctx.fillStyle = this.isDarkMode ? 'rgba(0, 0, 5, 0.2)' : 'rgba(255, 255, 255, 0.2)';
            this.ctx.fillRect(0, 0, width, height);
            
            this.ctx.globalCompositeOperation = this.isDarkMode ? 'lighter' : 'source-over';
            
            this.particles.forEach(p => {
                this.ctx.beginPath();
                const brightnessBoost = p.brightness * 15;
                const l = this.isDarkMode 
                    ? theme.l + brightnessBoost + (p.alpha * 5)
                    : theme.l - 25;
                const a = this.isDarkMode
                    ? p.alpha * 0.6
                    : p.alpha * 0.7;
                    
                this.ctx.fillStyle = `hsla(${Math.floor(theme.h + p.hueOffset)}, ${theme.s}%, ${Math.floor(l)}%, ${a.toFixed(2)})`;
                this.ctx.arc(p.x, p.y, p.size * (this.isDarkMode ? 1.3 : 1.1), 0, Math.PI * 2);
                this.ctx.fill();
            });
            
            this.ctx.globalCompositeOperation = 'source-over';
        }

        animate() {
            this.update();
            this.draw();
            this.animationFrame = requestAnimationFrame(() => this.animate());
        }

        setMood(mood) {
            this.mood = mood;
        }
        
        setThemeColor(color) {
            if (CONFIG.colors[color]) {
                this.themeColor = color;
            }
        }
        
        destroy() {
             if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
             window.removeEventListener('resize', this.resize);
             window.removeEventListener('mousemove', this.handleMouseMove);
             window.removeEventListener('mouseout', this.handleMouseOut);
        }
    }

    window.ParticleAvatar = ParticleAvatar;
})();
