import React from 'react';

/**
 * LiveMetrics
 * -----------
 * Displays the four key outcome metrics from an optimization result in a
 * compact, styled row. Designed to live just below the GridCanvas.
 *
 * Props
 * -----
 *  result         {object|null}  OptimizationResult (final results).
 *  status         {string|null}  Active job status ('running', 'complete', etc.).
 *  liveIteration  {number}       Current iteration number during active run.
 *  liveFitness    {number|null}  Current global best fitness value.
 */
export default function LiveMetrics({ result, status, liveIteration = 0, liveFitness = null }) {
  const isOptimizing = status === 'pending' || status === 'running';

  if (!result && !isOptimizing) {
    return (
      <div className="live-metrics live-metrics--empty">
        <span>Run optimization to see live metrics.</span>
      </div>
    );
  }

  // Extract metrics or fallback depending on running state
  const bestPositions = isOptimizing ? (result?.best_positions ?? []) : (result?.best_positions ?? []);
  const iterationsRun = isOptimizing ? liveIteration : (result?.iterations_run ?? 0);

  const coverageVal = isOptimizing ? '--%' : `${((result?.coverage_ratio ?? 0) * 100).toFixed(1)}%`;
  const connectivityVal = isOptimizing ? '--%' : `${((result?.connectivity_ratio ?? 0) * 100).toFixed(1)}%`;
  const energyVal = isOptimizing ? '-- J' : `${(result?.avg_energy ?? 0).toFixed(3)} J`;

  const sensorsVal = bestPositions.length;
  const sensorsSub = isOptimizing
    ? `Iter ${iterationsRun} · Fit: ${liveFitness !== null ? liveFitness.toFixed(4) : '--'}`
    : `${iterationsRun} iters · ${(result?.compute_time_seconds ?? 0).toFixed(2)}s`;

  const metrics = [
    {
      id: 'coverage',
      label: 'Coverage',
      value: coverageVal,
      sub: 'Target ≥ 91%',
      color: 'var(--color-success)',
      glow: 'var(--color-success-glow)',
      icon: '◉',
    },
    {
      id: 'connectivity',
      label: 'Connectivity',
      value: connectivityVal,
      sub: 'Nodes connected',
      color: 'var(--color-info)',
      glow: 'rgba(14,165,233,0.2)',
      icon: '⬡',
    },
    {
      id: 'energy',
      label: 'Avg Energy',
      value: energyVal,
      sub: 'Per node remaining',
      color: 'var(--color-warning)',
      glow: 'rgba(251,191,36,0.2)',
      icon: '⚡',
    },
    {
      id: 'nodes',
      label: 'Sensors',
      value: sensorsVal,
      sub: sensorsSub,
      color: '#a78bfa',
      glow: 'rgba(167,139,250,0.2)',
      icon: '⬤',
    },
  ];

  return (
    <div className="live-metrics">
      {metrics.map((m) => (
        <div
          key={m.id}
          id={`metric-${m.id}`}
          className="live-metric-card"
          style={{ '--metric-color': m.color, '--metric-glow': m.glow }}
        >
          <span className="metric-icon">{m.icon}</span>
          <div className="metric-content">
            <span className="metric-label">{m.label}</span>
            <span className="metric-value" style={{ color: m.color }}>{m.value}</span>
            <span className="metric-sub">{m.sub}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
