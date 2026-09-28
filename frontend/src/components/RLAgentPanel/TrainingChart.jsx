import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';

export default function TrainingChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div
        style={{
          height: '140px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.8rem',
          border: '1px dashed var(--border-color)',
          borderRadius: '6px',
        }}
      >
        No training progress data yet. Click "Train Agent" to start.
      </div>
    );
  }

  const chartData = data.map((d) => ({
    episode: d.episode,
    reward: parseFloat(d.reward.toFixed(2)),
    coverage: parseFloat((d.coverage * 100).toFixed(1)),
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          Reward per Episode
        </span>
        <div style={{ width: '100%', height: 110, marginTop: '4px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="episode" stroke="var(--text-muted)" fontSize={10} />
              <YAxis stroke="var(--text-muted)" fontSize={10} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '4px', fontSize: '11px' }}
              />
              <Line type="monotone" dataKey="reward" stroke="#818cf8" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>
          Coverage Achieved (%)
        </span>
        <div style={{ width: '100%', height: 110, marginTop: '4px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="episode" stroke="var(--text-muted)" fontSize={10} />
              <YAxis stroke="var(--text-muted)" fontSize={10} domain={[0, 100]} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '4px', fontSize: '11px' }}
              />
              <Line type="monotone" dataKey="coverage" stroke="#38bdf8" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
