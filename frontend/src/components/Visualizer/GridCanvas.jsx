import React, { useRef, useEffect, useMemo, useCallback } from 'react';
import { valueToViridisRGB } from '../../utils/colormap';
import { worldToCanvas, buildCommLinks } from '../../utils/geometry';

/**
 * GridCanvas
 * -----------
 * Renders the sensor deployment area onto an HTML Canvas element.
 *
 * Layers (painted in order, back → front):
 *  1. Coverage heatmap  — each cell coloured by detection probability using the
 *                         viridis perceptually-uniform palette.
 *  2. Sensing rings     — translucent circles at each sensor showing Rs radius.
 *  3. Communication links — thin lines between sensors within Rc of each other.
 *  4. Sensor nodes      — filled circles with a highlight dot.
 *
 * Props
 * -----
 *  result        {object}   Full OptimizationResult from the backend.
 *                           Expected fields:
 *                             best_positions   [N × [x,y]]  world coords
 *                             coverage_map     [rows × cols] values in [0,1]
 *  areaWidth     {number}   World width in metres  (default 100)
 *  areaHeight    {number}   World height in metres (default 100)
 *  sensingRadius {number}   Rs in metres           (default 10)
 *  commRadius    {number}   Rc in metres           (default 20)
 *
 * The canvas is responsive: it fills its CSS container width and preserves a
 * square aspect ratio (1:1) matching the world area.
 */
export default function GridCanvas({
  result,
  areaWidth = 100,
  areaHeight = 100,
  sensingRadius = 10,
  commRadius = 20,
  children,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  // ─── Derived data ──────────────────────────────────────────────────────────
  const positions = useMemo(() => result?.best_positions ?? [], [result]);
  const coverageMap = useMemo(() => result?.coverage_map ?? null, [result]);
  const failedIndices = useMemo(() => result?.failed_indices ?? [], [result]);

  // Pre-build communication edges so we don't recompute on every render.
  const commLinks = useMemo(
    () => buildCommLinks(positions, commRadius),
    [positions, commRadius]
  );

  // ─── Drawing logic ─────────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    // Clear
    ctx.clearRect(0, 0, W, H);

    const PADDING = 16;
    const innerW = Math.max(1, W - 2 * PADDING);
    const innerH = Math.max(1, H - 2 * PADDING);

    // ── Layer 1: Coverage heatmap ───────────────────────────────────────────
    if (coverageMap && coverageMap.length > 0) {
      const rows = coverageMap.length;
      const cols = coverageMap[0].length;
      const cellW = innerW / cols;
      const cellH = innerH / rows;

      ctx.fillStyle = '#0b0d17';
      ctx.fillRect(0, 0, W, H);

      const imageData = ctx.createImageData(W, H);
      const data = imageData.data;

      // Fill background dark first
      for (let i = 0; i < data.length; i += 4) {
        data[i]     = 11;
        data[i + 1] = 13;
        data[i + 2] = 23;
        data[i + 3] = 255;
      }

      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const val = coverageMap[row][col];
          const [r, g, b] = valueToViridisRGB(val);

          const x0 = Math.round(PADDING + col * cellW);
          const y0 = Math.round(PADDING + row * cellH);
          const x1 = Math.round(PADDING + (col + 1) * cellW);
          const y1 = Math.round(PADDING + (row + 1) * cellH);

          for (let py = y0; py < y1; py++) {
            if (py < 0 || py >= H) continue;
            for (let px = x0; px < x1; px++) {
              if (px < 0 || px >= W) continue;
              const idx = (py * W + px) * 4;
              data[idx]     = r;
              data[idx + 1] = g;
              data[idx + 2] = b;
              data[idx + 3] = 220; // slight transparency for depth
            }
          }
        }
      }
      ctx.putImageData(imageData, 0, 0);
    } else {
      // No coverage map — paint a dark grid background
      ctx.fillStyle = '#0b0d17';
      ctx.fillRect(0, 0, W, H);

      // Grid lines
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.lineWidth = 0.5;
      const gridLines = 20;
      for (let i = 0; i <= gridLines; i++) {
        const x = PADDING + (i / gridLines) * innerW;
        const y = PADDING + (i / gridLines) * innerH;
        ctx.beginPath(); ctx.moveTo(x, PADDING); ctx.lineTo(x, H - PADDING); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(PADDING, y); ctx.lineTo(W - PADDING, y); ctx.stroke();
      }
    }

    if (positions.length === 0) return;

    // ── Layer 2: Sensing radius rings ───────────────────────────────────────
    const rsPixels = (sensingRadius / areaWidth) * innerW;
    positions.forEach(([wx, wy], idx) => {
      if (failedIndices.includes(idx)) return;
      const { px, py } = worldToCanvas(wx, wy, areaWidth, areaHeight, W, H, PADDING);
      ctx.beginPath();
      ctx.arc(px, py, rsPixels, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = 'rgba(99, 102, 241, 0.05)';
      ctx.fill();
    });

    // ── Layer 3: Communication links ────────────────────────────────────────
    ctx.strokeStyle = 'rgba(14, 165, 233, 0.35)';
    ctx.lineWidth = 0.8;
    commLinks.forEach(([i, j]) => {
      if (failedIndices.includes(i) || failedIndices.includes(j)) return;
      const { px: x1, py: y1 } = worldToCanvas(
        positions[i][0], positions[i][1], areaWidth, areaHeight, W, H, PADDING
      );
      const { px: x2, py: y2 } = worldToCanvas(
        positions[j][0], positions[j][1], areaWidth, areaHeight, W, H, PADDING
      );
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });

    // ── Layer 4: Sensor nodes ───────────────────────────────────────────────
    const nodeRadius = Math.max(4, Math.min(8, W / 80));
    positions.forEach(([wx, wy], idx) => {
      const { px, py } = worldToCanvas(wx, wy, areaWidth, areaHeight, W, H, PADDING);

      if (failedIndices.includes(idx)) {
        // Red halo behind failed node
        ctx.beginPath();
        ctx.arc(px, py, nodeRadius * 1.6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.35)';
        ctx.fill();

        // Draw red ✕ marker for failed node
        ctx.beginPath();
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2.5;
        const size = nodeRadius;
        ctx.moveTo(px - size, py - size);
        ctx.lineTo(px + size, py + size);
        ctx.moveTo(px - size, py + size);
        ctx.lineTo(px + size, py - size);
        ctx.stroke();
      } else {
        // Glow halo
        const glow = ctx.createRadialGradient(px, py, 0, px, py, nodeRadius * 2.5);
        glow.addColorStop(0, 'rgba(99, 102, 241, 0.35)');
        glow.addColorStop(1, 'rgba(99, 102, 241, 0)');
        ctx.beginPath();
        ctx.arc(px, py, nodeRadius * 2.5, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();

        // Node body
        ctx.beginPath();
        ctx.arc(px, py, nodeRadius, 0, Math.PI * 2);
        ctx.fillStyle = '#6366f1';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Highlight dot
        ctx.beginPath();
        ctx.arc(px - nodeRadius * 0.3, py - nodeRadius * 0.3, nodeRadius * 0.28, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.fill();
      }
    });
  }, [positions, coverageMap, commLinks, areaWidth, areaHeight, sensingRadius, failedIndices]);

  // ─── Resize observer ───────────────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeCanvas = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = container.getBoundingClientRect();
      const size = rect.width; // square canvas
      canvas.width = size;
      canvas.height = size;
      draw();
    };

    resizeCanvas();
    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(container);
    return () => observer.disconnect();
  }, [draw]);

  // ─── Redraw on data change ─────────────────────────────────────────────────
  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div ref={containerRef} className="grid-canvas-container">
      <canvas
        ref={canvasRef}
        id="grid-canvas"
        style={{ display: 'block', width: '100%', height: '100%', borderRadius: '8px' }}
      />
      {positions.length > 0 && (
        <div className="canvas-legend">
          <span>Low</span>
          <div className="legend-gradient" />
          <span>High</span>
          <span className="legend-label">Coverage Probability</span>
        </div>
      )}
      {children}
    </div>
  );
}
