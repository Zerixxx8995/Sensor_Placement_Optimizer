/**
 * geometry.js
 * Pure geometric utility functions used by canvas rendering and sensor network
 * calculations.  No React, no side-effects.
 */

/**
 * Euclidean distance between two 2-D points.
 *
 * @param {number} x1
 * @param {number} y1
 * @param {number} x2
 * @param {number} y2
 * @returns {number}
 */
export function euclideanDistance(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Convert a world-space coordinate (metres) to canvas pixel coordinates.
 *
 * @param {number} worldX      - X position in world units
 * @param {number} worldY      - Y position in world units
 * @param {number} worldWidth  - Total world width (metres)
 * @param {number} worldHeight - Total world height (metres)
 * @param {number} canvasW     - Canvas pixel width
 * @param {number} canvasH     - Canvas pixel height
 * @returns {{ px: number, py: number }}
 */
export function worldToCanvas(worldX, worldY, worldWidth, worldHeight, canvasW, canvasH, padding = 16) {
  const innerW = Math.max(1, canvasW - 2 * padding);
  const innerH = Math.max(1, canvasH - 2 * padding);
  return {
    px: padding + (worldX / worldWidth) * innerW,
    py: padding + (worldY / worldHeight) * innerH,
  };
}

/**
 * Clamp a value between min and max (inclusive).
 *
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/**
 * Build a list of communication edges (pairs of sensor indices) where the
 * two sensors are within communication radius of each other.
 *
 * @param {Array<[number,number]>} positions - Array of [x, y] world positions
 * @param {number} commRadius               - Communication radius in world units
 * @returns {Array<[number, number]>}       - Array of [i, j] index pairs
 */
export function buildCommLinks(positions, commRadius) {
  const edges = [];
  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      const [x1, y1] = positions[i];
      const [x2, y2] = positions[j];
      if (euclideanDistance(x1, y1, x2, y2) <= commRadius) {
        edges.push([i, j]);
      }
    }
  }
  return edges;
}
