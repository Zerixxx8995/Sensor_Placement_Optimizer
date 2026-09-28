import React from 'react';

export default function PPOStatusCard({ statusData, isTraining }) {
  const isTrained = statusData?.trained ?? false;
  const episodes = statusData?.episodes_trained ?? 0;
  const bestCov = statusData?.best_coverage ?? 0.0;

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          PPO Placement Policy
        </span>
        <span
          style={{
            fontSize: '0.75rem',
            padding: '2px 8px',
            borderRadius: '12px',
            fontWeight: 600,
            background: isTraining
              ? 'rgba(245, 158, 11, 0.15)'
              : isTrained
              ? 'rgba(34, 197, 94, 0.15)'
              : 'rgba(239, 68, 68, 0.15)',
            color: isTraining
              ? '#f59e0b'
              : isTrained
              ? '#22c55e'
              : '#ef4444',
            border: `1px solid ${
              isTraining
                ? 'rgba(245, 158, 11, 0.3)'
                : isTrained
                ? 'rgba(34, 197, 94, 0.3)'
                : 'rgba(239, 68, 68, 0.3)'
            }`,
          }}
        >
          {isTraining ? 'Training...' : isTrained ? 'Trained' : 'Not Trained'}
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem' }}>
        <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.5rem', borderRadius: '4px' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Episodes Trained</div>
          <div style={{ fontSize: '1rem', fontWeight: 600 }}>{episodes}</div>
        </div>
        <div style={{ background: 'rgba(255,255,255,0.02)', padding: '0.5rem', borderRadius: '4px' }}>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Best Training Coverage</div>
          <div style={{ fontSize: '1rem', fontWeight: 600, color: '#38bdf8' }}>
            {(bestCov * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
