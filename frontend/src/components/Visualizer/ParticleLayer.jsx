import React, { useRef, useEffect } from 'react';
import { worldToCanvas } from '../../utils/geometry';

/**
 * ParticleLayer
 * -------------
 * A transparent canvas overlay that renders active particle swarm positions
 * during the optimization process. Each candidate node is drawn as a tiny
 * semi-transparent purple dot.
 *
 * Props
 * -----
 *  particles   {array}   Nested arrays of particle coordinates [[[x, y], ...], ...]
 *  areaWidth   {number}  World area width in meters.
 *  areaHeight  {number}  World area height in meters.
 */
export default function ParticleLayer({ particles = [], areaWidth = 100, areaHeight = 100 }) {
  const canvasRef = useRef(null);

  // Redraw particles when they change or when width/height bounds change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Clear previous frame
    ctx.clearRect(0, 0, W, H);

    if (!particles || particles.length === 0) return;

    // Draw each node coordinates for all swarm particles
    particles.forEach((particle) => {
      if (!Array.isArray(particle)) return;
      
      particle.forEach((node) => {
        if (!node || node.length < 2) return;
        const [wx, wy] = node;
        const { px, py } = worldToCanvas(wx, wy, areaWidth, areaHeight, W, H, 16);

        ctx.beginPath();
        ctx.arc(px, py, 1.8, 0, Math.PI * 2);
        // Vibrant neon purple with slight alpha to show particle density
        ctx.fillStyle = 'rgba(168, 85, 247, 0.45)';
        ctx.fill();
      });
    });
  }, [particles, areaWidth, areaHeight]);

  // Handle resizing to always match the container bounding box
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;

      // Force redraw after canvas buffer resize
      const ctx = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      if (particles && particles.length > 0) {
        particles.forEach((particle) => {
          if (!Array.isArray(particle)) return;
          particle.forEach((node) => {
            if (!node || node.length < 2) return;
            const [wx, wy] = node;
            const { px, py } = worldToCanvas(wx, wy, areaWidth, areaHeight, W, H, 16);
            ctx.beginPath();
            ctx.arc(px, py, 1.8, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(168, 85, 247, 0.45)';
            ctx.fill();
          });
        });
      }
    };

    resizeCanvas();
    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(canvas.parentElement);

    return () => observer.disconnect();
  }, [particles, areaWidth, areaHeight]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 5, // Overlay over GridCanvas draw buffer, under labels
      }}
    />
  );
}
