import React from 'react';

const STRATEGY_LABELS = {
  random: 'Random Placement',
  grid: 'Grid Placement',
  pso: 'Standard PSO Swarm',
  pso_vdcoa: 'PSO-VDCOA Hybrid',
  rl: 'RL Placement (PPO)',
};

export default function ComparisonTable({ results, activeStrategy }) {
  if (!results || results.length === 0) return null;

  // Find the best values in the comparison results to highlight them dynamically
  const bestCoverage = Math.max(...results.map(r => r.coverage_ratio ?? 0));
  const bestConnectivity = Math.max(...results.map(r => r.connectivity_ratio ?? 0));
  const bestEnergy = Math.max(...results.map(r => r.avg_energy ?? 0));
  const bestTime = Math.min(...results.map(r => r.compute_time_seconds ?? Infinity));

  return (
    <div className="comparison-table-card glass-card">
      <div className="compare-header">
        <h2 className="compare-title" style={{ fontFamily: 'var(--font-title)', fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
          Strategy Comparison Benchmarks
        </h2>
        <p className="compare-subtitle" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
          Synchronous benchmarking of standard baselines, swarm models, and RL placement policy on this configuration.
        </p>
      </div>

      <div className="table-responsive" style={{ overflowX: 'auto' }}>
        <table className="compare-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)', fontWeight: 600 }}>
              <th style={{ padding: '0.75rem 0.5rem' }}>Strategy</th>
              <th style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>Road Coverage</th>
              <th style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>Connectivity</th>
              <th style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>Deployment Cost</th>
              <th style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>Compute Time</th>
            </tr>
          </thead>
          <tbody>
            {results.map((row) => {
              const isActive = row.strategy === activeStrategy;
              
              const isBestCov = row.coverage_ratio === bestCoverage && bestCoverage > 0;
              const isBestConn = row.connectivity_ratio === bestConnectivity && bestConnectivity > 0;
              const isBestEnergy = row.avg_energy === bestEnergy && bestEnergy > 0;
              const isBestTime = row.compute_time_seconds === bestTime && bestTime < Infinity;

              return (
                <tr 
                  key={row.strategy} 
                  className={isActive ? 'row-active' : ''}
                  data-testid={`row-${row.strategy}`}
                  style={{
                    borderBottom: '1px solid var(--border-color)',
                    background: isActive ? 'rgba(99, 102, 241, 0.08)' : 'transparent',
                    transition: 'background var(--transition-fast)'
                  }}
                >
                  <td className="strategy-name" style={{ padding: '1rem 0.5rem', fontWeight: isActive ? 600 : 500, color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    {STRATEGY_LABELS[row.strategy] || row.strategy}
                    {isActive && (
                      <span className="active-badge" style={{
                        fontSize: '0.7rem',
                        background: 'rgba(99, 102, 241, 0.2)',
                        color: 'var(--color-primary)',
                        padding: '1px 6px',
                        borderRadius: '4px',
                        border: '1px solid rgba(99, 102, 241, 0.4)',
                        fontWeight: 600
                      }}>
                        Active
                      </span>
                    )}
                  </td>
                  <td className="text-right" style={{
                    padding: '1rem 0.5rem',
                    textAlign: 'right',
                    fontFamily: 'monospace',
                    fontWeight: isBestCov ? '700' : 'normal',
                    color: isBestCov ? 'var(--color-success)' : 'var(--text-secondary)'
                  }}>
                    {((row.coverage_ratio ?? 0) * 100).toFixed(1)}%
                    {isBestCov && <span className="best-star" style={{ marginLeft: '4px', color: 'var(--color-success)' }}>★</span>}
                  </td>
                  <td className="text-right" style={{
                    padding: '1rem 0.5rem',
                    textAlign: 'right',
                    fontFamily: 'monospace',
                    fontWeight: isBestConn ? '700' : 'normal',
                    color: isBestConn ? 'var(--color-success)' : 'var(--text-secondary)'
                  }}>
                    {((row.connectivity_ratio ?? 0) * 100).toFixed(1)}%
                    {isBestConn && <span className="best-star" style={{ marginLeft: '4px', color: 'var(--color-success)' }}>★</span>}
                  </td>
                  <td className="text-right" style={{
                    padding: '1rem 0.5rem',
                    textAlign: 'right',
                    fontFamily: 'monospace',
                    fontWeight: isBestEnergy ? '700' : 'normal',
                    color: isBestEnergy ? 'var(--color-success)' : 'var(--text-secondary)'
                  }}>
                    {(row.avg_energy ?? 0).toFixed(3)} J
                    {isBestEnergy && <span className="best-star" style={{ marginLeft: '4px', color: 'var(--color-success)' }}>★</span>}
                  </td>
                  <td className="text-right" style={{
                    padding: '1rem 0.5rem',
                    textAlign: 'right',
                    fontFamily: 'monospace',
                    fontWeight: isBestTime ? '700' : 'normal',
                    color: isBestTime ? 'var(--color-success)' : 'var(--text-secondary)'
                  }}>
                    {(row.compute_time_seconds ?? 0).toFixed(3)}s
                    {isBestTime && <span className="best-star" style={{ marginLeft: '4px', color: 'var(--color-success)' }}>★</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
