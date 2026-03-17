(function () {
    const TWO_PI = Math.PI * 2;
    const COLOR_MAP = {
        cyan: { h: 185, s: 100, l: 60 },
        purple: { h: 270, s: 95, l: 65 },
        green: { h: 150, s: 90, l: 60 },
        orange: { h: 30, s: 100, l: 60 },
        rose: { h: 340, s: 95, l: 60 }
    };
    const PATTERN_PRESETS = [
        "neutral",
        "happy",
        "sad",
        "thinking",
        "surprised",
        "halo",
        "ripple",
        "helix",
        "bloom",
        "wave",
        "nebula",
        "binary",
        "pulse"
    ];

    class ParticleAvatar {
        constructor(canvasId, options) {
            this.canvas = document.getElementById(canvasId);
            if (!this.canvas) return;

            this.ctx = this.canvas.getContext('2d', { alpha: false });
            if (!this.ctx) return;

            const config = options || {};

            this.particles = [];
            this.animationFrame = null;
            this.time = 0;
            this.mouse = { x: -1000, y: -1000 };
            this.mood = PATTERN_PRESETS.includes(config.mood) ? config.mood : 'neutral';
            this.themeColor = COLOR_MAP[config.themeColor] ? config.themeColor : 'cyan';
            this.isDarkMode = Boolean(config.isDarkMode);
            this.reducedMotion = Boolean(config.reducedMotion);
            this.particleCount = 0;
            this.interactionRadius = this.reducedMotion ? 140 : 260;
            this.pushStrength = this.reducedMotion ? 18 : 42;

            this.resize = this.resize.bind(this);
            this.handleMouseMove = this.handleMouseMove.bind(this);
            this.handleMouseOut = this.handleMouseOut.bind(this);
            this.handleThemeChange = this.handleThemeChange.bind(this);

            this.themeObserver = null;
            this.init();
        }

        init() {
            this.syncThemeMode();
            this.resize();
            window.addEventListener('resize', this.resize);
            window.addEventListener('mousemove', this.handleMouseMove, { passive: true });
            window.addEventListener('mouseout', this.handleMouseOut);
            this.observeThemeMode();
            this.animate();
        }

        observeThemeMode() {
            if (!window.MutationObserver) return;

            this.themeObserver = new MutationObserver(this.handleThemeChange);
            this.themeObserver.observe(document.documentElement, {
                attributes: true,
                attributeFilter: ['class']
            });
            this.themeObserver.observe(document.body, {
                attributes: true,
                attributeFilter: ['class']
            });
        }

        handleThemeChange() {
            this.syncThemeMode();
        }

        syncThemeMode() {
            const body = document.body;
            const html = document.documentElement;
            this.isDarkMode =
                html.classList.contains('dark') ||
                body.classList.contains('dark') ||
                body.classList.contains('dark-mode');
        }

        getAdaptiveParticleCount(width, height) {
            const shortestSide = Math.min(width, height);
            const isTouchDevice = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;

            if (this.reducedMotion) return 280;
            if (isTouchDevice || shortestSide < 640) return 1100;
            if (shortestSide < 960) return 2200;
            return 3600;
        }

        resize() {
            if (!this.canvas) return;

            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
            this.particleCount = this.getAdaptiveParticleCount(this.canvas.width, this.canvas.height);
            this.initParticles();
        }

        initParticles() {
            const width = this.canvas.width;
            const height = this.canvas.height;
            const cx = width / 2;
            const cy = height / 2;
            const maxRadius = Math.min(width, height) * 0.54;

            this.particles = [];

            for (let i = 0; i < this.particleCount; i += 1) {
                const angle = Math.random() * TWO_PI;
                const randomRadius = Math.random();
                const radius = randomRadius * randomRadius * maxRadius;
                const size = this.reducedMotion
                    ? Math.random() * 1.2 + 0.35
                    : Math.random() * 1.5 + 0.18;

                this.particles.push({
                    x: cx + Math.cos(angle) * radius,
                    y: cy + Math.sin(angle) * radius,
                    angle: angle,
                    radius: radius,
                    speed: this.reducedMotion
                        ? 0.00025 + Math.random() * 0.0007
                        : 0.00045 + Math.random() * 0.0017,
                    size: size,
                    alpha: Math.random() * 0.45 + 0.08,
                    targetAlpha: Math.random() * 0.75 + 0.15,
                    hueOffset: Math.random() * 56 - 28,
                    orbitOffset: Math.random() * TWO_PI,
                    brightness: size > 1.15 ? 1.2 : 0.75
                });
            }
        }

        handleMouseMove(event) {
            this.mouse.x = event.clientX;
            this.mouse.y = event.clientY;
        }

        handleMouseOut() {
            this.mouse.x = -1000;
            this.mouse.y = -1000;
        }

        update() {
            const width = this.canvas.width;
            const height = this.canvas.height;
            const cx = width / 2;
            const cy = height / 2;
            const breathSpeed = this.reducedMotion ? 0.003 : 0.008;
            const breathAmplitude = this.reducedMotion ? 0.03 : 0.08;
            const breathingScale = 1 + Math.sin(this.time * breathSpeed) * breathAmplitude;

            this.particles.forEach((particle) => {
                let targetRadius = particle.radius;
                let moveSpeed = particle.speed;

                switch (this.mood) {
                    case 'happy':
                        targetRadius = particle.radius * 1.35;
                        moveSpeed = particle.speed * 2.2;
                        particle.angle += moveSpeed * 0.45;
                        break;
                    case 'sad':
                        targetRadius = particle.radius * 0.45;
                        moveSpeed = particle.speed * 0.65;
                        particle.angle += moveSpeed * 1.8;
                        break;
                    case 'thinking': {
                        const ring = (Math.floor(particle.orbitOffset * 10) % 4) + 1;
                        targetRadius = 78 * ring * (1 + Math.sin(this.time * 0.018 + particle.orbitOffset) * 0.12);
                        particle.angle += moveSpeed * (ring % 2 === 0 ? 1 : -1) * 1.25;
                        break;
                    }
                    case 'surprised':
                        targetRadius = particle.radius + Math.sin(this.time * 0.045 + particle.orbitOffset) * 42;
                        particle.angle += Math.sin(this.time * 0.028) * 0.004;
                        break;
                    case 'halo':
                        targetRadius = Math.min(width, height) * 0.24 + Math.sin(this.time * 0.015 + particle.orbitOffset) * 18;
                        particle.angle += moveSpeed * 1.8;
                        break;
                    case 'ripple':
                        targetRadius = particle.radius * 0.78 + Math.sin(this.time * 0.035 + particle.orbitOffset * 2.4) * 54;
                        particle.angle += moveSpeed * 0.9;
                        break;
                    case 'helix': {
                        const coil = Math.sin(this.time * 0.02 + particle.orbitOffset * 3.2);
                        targetRadius = Math.min(width, height) * 0.14 + particle.radius * 0.45 + coil * 48;
                        particle.angle += moveSpeed * 1.9;
                        break;
                    }
                    case 'bloom': {
                        const petals = 5 + (Math.floor(particle.orbitOffset * 10) % 4);
                        targetRadius = Math.min(width, height) * 0.12 +
                            Math.abs(Math.sin(particle.angle * petals + this.time * 0.012)) * Math.min(width, height) * 0.28;
                        particle.angle += moveSpeed * 1.15;
                        break;
                    }
                    case 'wave':
                        targetRadius = particle.radius * 0.58 +
                            Math.sin(this.time * 0.022 + particle.orbitOffset * 4.5) * 38 +
                            Math.cos(particle.angle * 3.5 + this.time * 0.01) * 26;
                        particle.angle += moveSpeed * 0.85;
                        break;
                    case 'nebula': {
                        const armCount = 3;
                        const swirl = particle.radius * 0.42;
                        const armPhase = (particle.orbitOffset % 1) * armCount * TWO_PI;
                        const spiralWave = Math.sin(this.time * 0.012 + armPhase + particle.radius * 0.025);
                        targetRadius = Math.min(width, height) * 0.08 + swirl + spiralWave * 54;
                        particle.angle += moveSpeed * (1.5 + particle.radius / Math.max(Math.min(width, height), 1)) + 0.0018;
                        break;
                    }
                    case 'binary': {
                        const pivot = Math.sin(this.time * 0.01 + particle.orbitOffset);
                        const lobe = particle.orbitOffset > Math.PI ? 1 : -1;
                        const coreOffset = Math.min(width, height) * 0.16;
                        const coreX = cx + lobe * coreOffset * Math.cos(this.time * 0.008);
                        const coreY = cy + coreOffset * 0.36 * Math.sin(this.time * 0.014 + lobe);
                        const localRadius = Math.min(width, height) * 0.045 + particle.radius * 0.28 + Math.abs(pivot) * 24;
                        const localAngle = particle.angle + (lobe * 0.018);
                        const desiredX = coreX + Math.cos(localAngle) * localRadius;
                        const desiredY = coreY + Math.sin(localAngle) * localRadius;
                        particle.angle += moveSpeed * (1.55 + lobe * 0.08);
                        particle.x += (desiredX - particle.x) * (this.reducedMotion ? 0.018 : 0.032);
                        particle.y += (desiredY - particle.y) * (this.reducedMotion ? 0.018 : 0.032);
                        particle.alpha += (particle.targetAlpha - particle.alpha) * 0.03;
                        if (Math.random() > (this.reducedMotion ? 0.997 : 0.99)) {
                            particle.targetAlpha = Math.random() * 0.7 + 0.12;
                        }
                        return;
                    }
                    case 'pulse': {
                        const beat = Math.abs(Math.sin(this.time * 0.035));
                        const shell = Math.floor((particle.orbitOffset / TWO_PI) * 5) + 1;
                        targetRadius = 18 * shell + beat * (20 + shell * 14) + Math.sin(this.time * 0.018 + particle.orbitOffset * 6) * 8;
                        particle.angle += moveSpeed * (0.75 + shell * 0.08);
                        break;
                    }
                    case 'neutral':
                    default:
                        particle.angle += moveSpeed;
                        break;
                }

                const effectiveRadius = targetRadius * breathingScale;
                const orbitX = cx + Math.cos(particle.angle) * effectiveRadius;
                const orbitY = cy + Math.sin(particle.angle) * effectiveRadius;

                const dx = this.mouse.x - orbitX;
                const dy = this.mouse.y - orbitY;
                const dist = Math.sqrt(dx * dx + dy * dy);

                let pushX = 0;
                let pushY = 0;

                if (dist < this.interactionRadius) {
                    const force = (this.interactionRadius - dist) / this.interactionRadius;
                    const angleToMouse = Math.atan2(dy, dx);
                    pushX = -Math.cos(angleToMouse) * force * this.pushStrength;
                    pushY = -Math.sin(angleToMouse) * force * this.pushStrength;
                }

                const desiredX = orbitX + pushX;
                const desiredY = orbitY + pushY;
                const easing = this.reducedMotion ? 0.02 : 0.035;

                particle.x += (desiredX - particle.x) * easing;
                particle.y += (desiredY - particle.y) * easing;

                particle.alpha += (particle.targetAlpha - particle.alpha) * 0.03;
                if (Math.random() > (this.reducedMotion ? 0.997 : 0.99)) {
                    particle.targetAlpha = Math.random() * 0.7 + 0.12;
                }
            });

            this.time += 1;
        }

        draw() {
            const width = this.canvas.width;
            const height = this.canvas.height;
            const theme = COLOR_MAP[this.themeColor] || COLOR_MAP.cyan;

            this.ctx.fillStyle = this.isDarkMode ? 'rgba(2, 5, 15, 0.22)' : 'rgba(248, 250, 252, 0.22)';
            this.ctx.fillRect(0, 0, width, height);

            this.ctx.globalCompositeOperation = this.isDarkMode ? 'lighter' : 'source-over';

            this.particles.forEach((particle) => {
                const brightnessBoost = particle.brightness * 14;
                const lightness = this.isDarkMode
                    ? theme.l + brightnessBoost + particle.alpha * 5
                    : theme.l - 18;
                const alpha = this.isDarkMode ? particle.alpha * 0.58 : particle.alpha * 0.66;

                this.ctx.beginPath();
                this.ctx.fillStyle = 'hsla(' +
                    Math.floor(theme.h + particle.hueOffset) + ', ' +
                    theme.s + '%, ' +
                    Math.floor(lightness) + '%, ' +
                    alpha.toFixed(2) + ')';
                this.ctx.arc(
                    particle.x,
                    particle.y,
                    particle.size * (this.isDarkMode ? 1.28 : 1.08),
                    0,
                    TWO_PI
                );
                this.ctx.fill();
            });

            this.ctx.globalCompositeOperation = 'source-over';
        }

        animate() {
            this.update();
            this.draw();
            this.animationFrame = window.requestAnimationFrame(this.animate.bind(this));
        }

        setMood(mood) {
            this.mood = PATTERN_PRESETS.includes(mood) ? mood : 'neutral';
        }

        setPattern(pattern) {
            this.setMood(pattern);
        }

        setThemeColor(color) {
            if (COLOR_MAP[color]) {
                this.themeColor = color;
            }
        }

        destroy() {
            if (this.animationFrame) {
                window.cancelAnimationFrame(this.animationFrame);
                this.animationFrame = null;
            }
            window.removeEventListener('resize', this.resize);
            window.removeEventListener('mousemove', this.handleMouseMove);
            window.removeEventListener('mouseout', this.handleMouseOut);
            if (this.themeObserver) {
                this.themeObserver.disconnect();
                this.themeObserver = null;
            }
        }
    }

    window.ParticleAvatar = ParticleAvatar;
})();
