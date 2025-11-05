import React, { useEffect, useRef } from 'react';


const debounce = (func, delay) => {
  let timeoutId;
  return function() {
    const context = this;
    const args = arguments;
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(context, args), delay);
  }
};

const GeometricAnimation = ({ paused }) => {
  const canvasRef = useRef(null);
  const particlesRef = useRef([]);
  const animationIdRef = useRef();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // This function recreates all particles. It's expensive to run repeatedly.
    const createParticles = () => {
      const area = window.innerWidth * window.innerHeight;
      const count = area < 500000 ? 80 : 150;
      const baseSize = window.innerWidth < 500 ? 0.8 : window.innerWidth < 900 ? 1.2 : 1.5;

      particlesRef.current = Array.from({ length: count }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.7,
        vy: (Math.random() - 0.5) * 0.7,
        size: baseSize + Math.random() * 1.5,
        opacity: Math.random() * 0.5 + 0.2,
        hue: Math.random() * 30 + 240,
        shape: Math.random() < 0.1 ? 'triangle' : (Math.random() < 0.2 ? 'square' : 'circle')
      }));
    };
    
    // --- FIX: Debounce the expensive resize operation for performance ---
    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      createParticles();
    };
    
    // Create a debounced version that will only run 250ms after the user stops resizing.
    const debouncedResize = debounce(handleResize, 250);

    const drawShape = (ctx, x, y, size, type = 'circle') => {
      ctx.beginPath();
      if (type === 'triangle') {
        ctx.moveTo(x, y - size);
        ctx.lineTo(x - size, y + size);
        ctx.lineTo(x + size, y + size);
        ctx.closePath();
      } else if (type === 'square') {
        ctx.rect(x - size, y - size, size * 2, size * 2);
      } else {
        ctx.arc(x, y, size, 0, Math.PI * 2);
      }
      ctx.fill();
    };

    const animate = (time) => {
      // The `paused` prop effectively stops the animation loop.
      if (paused) {
        animationIdRef.current = requestAnimationFrame(animate);
        return;
      }

      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const particles = particlesRef.current;

      particles.forEach((p1, i) => {
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 70) {
            ctx.strokeStyle = `hsla(${p1.hue}, 90%, 60%, 0.2)`;
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      });

      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        ctx.save();
        ctx.globalAlpha = p.opacity * (0.8 + 0.2 * Math.sin(time * 0.004 + i));
        ctx.fillStyle = `hsl(${p.hue}, 100%, 60%)`; // richer/brighter color
        drawShape(ctx, p.x, p.y, p.size, p.shape);
        ctx.restore();
      });

      animationIdRef.current = requestAnimationFrame(animate);
    };

    handleResize(); // Call once initially to set up the canvas
    window.addEventListener('resize', debouncedResize); 
    animationIdRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationIdRef.current);
      window.removeEventListener('resize', debouncedResize); // Clean up the debounced listener
    };
  }, [paused]); // The dependency is correct

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none opacity-80 z-10"
      style={{ background: 'transparent' }}
      aria-hidden="true"
    />
  );
};

export default GeometricAnimation;