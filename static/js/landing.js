/**
 * PREVEE - Landing Page JavaScript
 * Animaciones: Particles.js, Red Neuronal Canvas, Contadores, AOS
 */

// ========================================
// Inicialización cuando el DOM está listo
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    // Inicializar AOS (Animate On Scroll)
    AOS.init({
        duration: 800,
        once: true,
        offset: 100,
        easing: 'ease-out-cubic'
    });
    
    // Inicializar Particles.js
    initParticles();
    
    // Inicializar Red Neuronal en Canvas
    initNeuralNetwork();
    
    // Inicializar Contadores Animados
    initCounters();
    
    // Navbar scroll effect
    initNavbar();
});

// ========================================
// PARTICLES.JS - Red de Partículas Conectadas
// ========================================
function initParticles() {
    if (typeof particlesJS === 'undefined') {
        console.warn('Particles.js no está cargado');
        return;
    }
    
    particlesJS('particles-js', {
        particles: {
            number: {
                value: 80,
                density: {
                    enable: true,
                    value_area: 800
                }
            },
            color: {
                value: ['#22c55e', '#16a34a', '#86efac']
            },
            shape: {
                type: 'circle',
                stroke: {
                    width: 0,
                    color: '#000000'
                }
            },
            opacity: {
                value: 0.5,
                random: true,
                anim: {
                    enable: true,
                    speed: 1,
                    opacity_min: 0.1,
                    sync: false
                }
            },
            size: {
                value: 3,
                random: true,
                anim: {
                    enable: true,
                    speed: 2,
                    size_min: 0.1,
                    sync: false
                }
            },
            line_linked: {
                enable: true,
                distance: 150,
                color: '#22c55e',
                opacity: 0.4,
                width: 1
            },
            move: {
                enable: true,
                speed: 2,
                direction: 'none',
                random: true,
                straight: false,
                out_mode: 'out',
                bounce: false,
                attract: {
                    enable: true,
                    rotateX: 600,
                    rotateY: 1200
                }
            }
        },
        interactivity: {
            detect_on: 'canvas',
            events: {
                onhover: {
                    enable: true,
                    mode: 'grab'
                },
                onclick: {
                    enable: true,
                    mode: 'push'
                },
                resize: true
            },
            modes: {
                grab: {
                    distance: 200,
                    line_linked: {
                        opacity: 0.8
                    }
                },
                push: {
                    particles_nb: 4
                }
            }
        },
        retina_detect: true
    });
}

// ========================================
// RED NEURONAL CANVAS - Animación Personalizada
// ========================================
function initNeuralNetwork() {
    const canvas = document.getElementById('neural-network');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;
    
    // Nodos de la red neuronal
    const nodes = [];
    const nodeCount = 30;
    
    // Clase Nodo
    class Node {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.radius = Math.random() * 2 + 1;
            this.color = this.getRandomColor();
        }
        
        getRandomColor() {
            const colors = ['#22c55e', '#16a34a', '#86efac'];
            return colors[Math.floor(Math.random() * colors.length)];
        }
        
        update() {
            this.x += this.vx;
            this.y += this.vy;
            
            // Rebote en los bordes
            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;
        }
        
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.shadowBlur = 10;
            ctx.shadowColor = this.color;
            ctx.fill();
        }
    }
    
    // Crear nodos
    for (let i = 0; i < nodeCount; i++) {
        nodes.push(new Node());
    }
    
    // Función de animación
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        // Actualizar y dibujar nodos
        nodes.forEach(node => {
            node.update();
            node.draw();
        });
        
        // Dibujar conexiones entre nodos cercanos
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 150) {
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.strokeStyle = `rgba(34, 197, 94, ${1 - distance / 150})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        
        requestAnimationFrame(animate);
    }
    
    animate();
    
    // Redimensionar canvas
    window.addEventListener('resize', function() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });
}

// ========================================
// CONTADORES ANIMADOS
// ========================================
function initCounters() {
    const counters = document.querySelectorAll('.stat-number');
    const speed = 200; // Velocidad de la animación
    
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const counter = entry.target;
                const target = parseFloat(counter.getAttribute('data-target'));
                
                animateCounter(counter, target);
                counterObserver.unobserve(counter);
            }
        });
    }, { threshold: 0.5 });
    
    counters.forEach(counter => {
        counterObserver.observe(counter);
    });
    
    function animateCounter(element, target) {
        let current = 0;
        const increment = target / speed;
        const isDecimal = target % 1 !== 0;
        
        const timer = setInterval(() => {
            current += increment;
            
            if (current >= target) {
                clearInterval(timer);
                current = target;
            }
            
            if (isDecimal) {
                element.textContent = current.toFixed(2);
            } else {
                element.textContent = Math.floor(current);
            }
        }, 10);
    }
}

// ========================================
// NAVBAR SCROLL EFFECT
// ========================================
function initNavbar() {
    const navbar = document.getElementById('navbar');
    let lastScroll = 0;
    
    window.addEventListener('scroll', function() {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        lastScroll = currentScroll;
    });
}

// ========================================
// SMOOTH SCROLL PARA ENLACES INTERNOS
// ========================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        
        // Solo aplicar smooth scroll si no es un ancla vacío
        if (href !== '#' && href !== '') {
            e.preventDefault();
            
            const target = document.querySelector(href);
            if (target) {
                const offsetTop = target.offsetTop - 80; // 80px para el navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        }
    });
});

// ========================================
// EFECTOS ADICIONALES
// ========================================

// Efecto de parallax simple en el hero
window.addEventListener('scroll', function() {
    const scrolled = window.pageYOffset;
    const heroContent = document.querySelector('.hero-content');
    
    if (heroContent && scrolled < window.innerHeight) {
        heroContent.style.transform = `translateY(${scrolled * 0.5}px)`;
        heroContent.style.opacity = 1 - (scrolled / window.innerHeight);
    }
});

// Efecto hover en las tarjetas con mouse tracking
document.querySelectorAll('.glass-card').forEach(card => {
    card.addEventListener('mousemove', function(e) {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        const rotateX = (y - centerY) / 20;
        const rotateY = (centerX - x) / 20;
        
        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-5px)`;
    });
    
    card.addEventListener('mouseleave', function() {
        card.style.transform = '';
    });
});

// Animación de escritura para el título (opcional)
function typeWriter(element, text, speed = 100) {
    let i = 0;
    element.textContent = '';
    
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    type();
}

// Detectar cuando el usuario llega al footer
const footer = document.getElementById('footer');
if (footer) {
    const footerObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                footer.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });
    
    footerObserver.observe(footer);
}

// Log de carga exitosa
console.log('%c✨ Prevee Landing Page Cargada ✨', 'color: #22c55e; font-size: 20px; font-weight: bold;');
console.log('%cSistema de Predicción de Rendimiento Académico con IA', 'color: #16a34a; font-size: 14px;');

