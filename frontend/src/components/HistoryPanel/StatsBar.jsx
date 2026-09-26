/**
 * components/HistoryPanel/StatsBar.jsx
 * --------------------------------------
 * Displays aggregate run statistics at the top of the History tab.
 * Pure display — receives props, no data fetching.
 */

import React from 'react';

function StatCard({ label, value, icon, color }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)',
      border: '1px solid var(--border-color)',
      borderRadius: '12px',
      padding: '1rem 1.25rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem',
      flex: '1',
      minWidth: '150px',
      transition: 'background 0.2s',
    }}>
      <div style={{
        width: '40px',
        height: '40px',
        borderRadius: '10px',
        background: `${color}22`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '1.25rem',
        color,
        flexShrink: 0,
      }}>
        {icon}
      </div>
      <div>
        <div style={{
          fontSize: '1.35rem',
          fontWeight: 700,
          fontFamily: 'var(--font-title)',
          color: 'var(--text-primary)',
          lineHeight: 1.1,
        }}>
          {value}
        </div>
        <div style={{
          fontSize: '0.78rem',
          color: 'var(--text-muted)',
          marginTop: '2px',
          letterSpacing: '0.02em',
        }}>
          {label}
        </div>
      </div>
    </div>
  );
}

export default function StatsBar({ stats }) {
  if (!stats) {
    return (
      <div style={{
        display: 'flex',
        gap: '0.75rem',
        flexWrap: 'wrap',
        marginBottom: '1.25rem',
      }}>
        {[1,2,3,4].map(i => (
          <div key={i} style={{
            flex: '1',
            minWidth: '150px',
            height: '72px',
            borderRadius: '12px',
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid var(--border-color)',
            animation: 'pulse 1.5s infinite',
          }} />
        ))}
      </div>
    );
  }

  const bestCoverage = stats.best_coverage_ever != null
    ? `${(stats.best_coverage_ever * 100).toFixed(1)}%`
    : '—';

  const avgTime = stats.avg_compute_time != null
    ? `${stats.avg_compute_time.toFixed(1)}s`
    : '—';

  const strategy = stats.most_used_strategy
    ? stats.most_used_strategy.toUpperCase().replace('_', '-')
    : '—';

  return (
    <div style={{
      display: 'flex',
      gap: '0.75rem',
      flexWrap: 'wrap',
      marginBottom: '1.25rem',
    }}>
      <StatCard
        label="Total Runs"
        value={stats.total_runs ?? 0}
        icon="📊"
        color="#6366f1"
      />
      <StatCard
        label="Best Coverage"
        value={bestCoverage}
        icon="🎯"
        color="#10b981"
      />
      <StatCard
        label="Avg Compute Time"
        value={avgTime}
        icon="⏱️"
        color="#f59e0b"
      />
      <StatCard
        label="Most Used Strategy"
        value={strategy}
        icon="🏆"
        color="#8b5cf6"
      />
    </div>
  );
}
