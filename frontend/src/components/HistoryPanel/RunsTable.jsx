/**
 * components/HistoryPanel/RunsTable.jsx
 * ---------------------------------------
 * Paginated table of past optimization runs.
 *
 * Props:
 *   runs       – array of run summary objects
 *   page       – current page
 *   totalPages – total number of pages
 *   total      – total run count
 *   isLoading  – shows skeleton rows when true
 *   onDelete   – (jobId: string) => void
 *   onLoadRun  – (jobId: string) => void  — loads full result into visualizer
 *   onPageChange – (page: number) => void
 */

import React, { useState } from 'react';

const STRATEGY_COLORS = {
  pso:       { bg: 'rgba(99,102,241,0.15)', text: '#818cf8' },
  pso_vdcoa: { bg: 'rgba(139,92,246,0.15)', text: '#a78bfa' },
  random:    { bg: 'rgba(245,158,11,0.15)', text: '#fbbf24' },
  grid:      { bg: 'rgba(16,185,129,0.15)', text: '#34d399' },
};

function StrategyBadge({ strategy }) {
  const s = strategy?.toLowerCase() || 'pso';
  const colors = STRATEGY_COLORS[s] || { bg: 'rgba(148,163,184,0.15)', text: '#94a3b8' };
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 10px',
      borderRadius: '99px',
      fontSize: '0.75rem',
      fontWeight: 600,
      background: colors.bg,
      color: colors.text,
      letterSpacing: '0.03em',
      textTransform: 'uppercase',
    }}>
      {s.replace('_', '-')}
    </span>
  );
}

function SkeletonRow() {
  return (
    <tr style={{ animation: 'pulse 1.5s infinite' }}>
      {[1,2,3,4,5,6,7].map(i => (
        <td key={i} style={{ padding: '0.875rem 1rem' }}>
          <div style={{
            height: '14px',
            borderRadius: '4px',
            background: 'rgba(255,255,255,0.07)',
            width: i === 1 ? '80%' : i === 7 ? '60%' : '90%',
          }} />
        </td>
      ))}
    </tr>
  );
}

export default function RunsTable({
  runs = [],
  page = 1,
  totalPages = 1,
  total = 0,
  isLoading = false,
  onDelete,
  onLoadRun,
  onPageChange,
}) {
  const [deletingId, setDeletingId] = useState(null);
  const [confirmId, setConfirmId] = useState(null);

  const handleDelete = async (jobId) => {
    if (confirmId !== jobId) {
      setConfirmId(jobId);
      return;
    }
    setConfirmId(null);
    setDeletingId(jobId);
    try {
      await onDelete(jobId);
    } finally {
      setDeletingId(null);
    }
  };

  const formatDate = (iso) => {
    if (!iso) return '—';
    try {
      return new Date(iso).toLocaleString(undefined, {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch {
      return iso;
    }
  };

  const thStyle = {
    padding: '0.75rem 1rem',
    textAlign: 'left',
    fontSize: '0.75rem',
    fontWeight: 600,
    color: 'var(--text-muted)',
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    borderBottom: '1px solid var(--border-color)',
    whiteSpace: 'nowrap',
  };

  const tdStyle = {
    padding: '0.875rem 1rem',
    fontSize: '0.875rem',
    color: 'var(--text-primary)',
    borderBottom: '1px solid rgba(255,255,255,0.04)',
    verticalAlign: 'middle',
  };

  return (
    <div>
      <div style={{ overflowX: 'auto', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.03)' }}>
              <th style={thStyle}>Date</th>
              <th style={thStyle}>Strategy</th>
              <th style={thStyle}>Nodes</th>
              <th style={thStyle}>Coverage</th>
              <th style={thStyle}>Compute Time</th>
              <th style={thStyle}>GPU</th>
              <th style={{ ...thStyle, textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)
            ) : runs.length === 0 ? (
              <tr>
                <td colSpan={7} style={{
                  padding: '3rem',
                  textAlign: 'center',
                  color: 'var(--text-muted)',
                  fontSize: '0.9rem',
                }}>
                  No optimization runs recorded yet.
                  <br />
                  <span style={{ fontSize: '0.8rem', marginTop: '0.5rem', display: 'block' }}>
                    Runs are saved automatically after each optimization completes.
                  </span>
                </td>
              </tr>
            ) : (
              runs.map((run) => (
                <tr
                  key={run.job_id}
                  style={{
                    transition: 'background 0.15s',
                    cursor: 'default',
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ ...tdStyle, color: 'var(--text-secondary)', fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                    {formatDate(run.created_at)}
                  </td>
                  <td style={tdStyle}>
                    <StrategyBadge strategy={run.strategy} />
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>
                    {run.num_nodes ?? '—'}
                  </td>
                  <td style={tdStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div style={{
                        flex: 1,
                        height: '6px',
                        borderRadius: '3px',
                        background: 'rgba(255,255,255,0.08)',
                        maxWidth: '80px',
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${((run.coverage_ratio || 0) * 100).toFixed(0)}%`,
                          background: run.coverage_ratio > 0.9 ? '#10b981' : run.coverage_ratio > 0.7 ? '#f59e0b' : '#ef4444',
                          borderRadius: '3px',
                          transition: 'width 0.4s',
                        }} />
                      </div>
                      <span style={{ fontSize: '0.85rem', fontWeight: 600, color: run.coverage_ratio > 0.9 ? '#10b981' : run.coverage_ratio > 0.7 ? '#f59e0b' : '#ef4444' }}>
                        {run.coverage_ratio != null ? `${(run.coverage_ratio * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                  </td>
                  <td style={{ ...tdStyle, color: 'var(--text-secondary)' }}>
                    {run.compute_time_seconds != null ? `${run.compute_time_seconds.toFixed(2)}s` : '—'}
                  </td>
                  <td style={tdStyle}>
                    <span style={{
                      display: 'inline-block',
                      width: '10px',
                      height: '10px',
                      borderRadius: '50%',
                      background: run.gpu_used ? '#10b981' : '#6b7280',
                      boxShadow: run.gpu_used ? '0 0 6px #10b981' : 'none',
                    }} title={run.gpu_used ? 'GPU' : 'CPU'} />
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>
                    <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
                      {onLoadRun && (
                        <button
                          id={`btn-load-run-${run.job_id}`}
                          onClick={() => onLoadRun(run.job_id)}
                          style={{
                            padding: '4px 12px',
                            borderRadius: '6px',
                            border: '1px solid rgba(99,102,241,0.4)',
                            background: 'rgba(99,102,241,0.1)',
                            color: '#818cf8',
                            fontSize: '0.78rem',
                            cursor: 'pointer',
                            fontWeight: 500,
                            transition: 'all 0.15s',
                          }}
                          onMouseEnter={e => { e.target.style.background = 'rgba(99,102,241,0.2)'; }}
                          onMouseLeave={e => { e.target.style.background = 'rgba(99,102,241,0.1)'; }}
                        >
                          Load
                        </button>
                      )}
                      <button
                        id={`btn-delete-run-${run.job_id}`}
                        onClick={() => handleDelete(run.job_id)}
                        disabled={deletingId === run.job_id}
                        style={{
                          padding: '4px 10px',
                          borderRadius: '6px',
                          border: `1px solid ${confirmId === run.job_id ? 'rgba(239,68,68,0.6)' : 'rgba(239,68,68,0.25)'}`,
                          background: confirmId === run.job_id ? 'rgba(239,68,68,0.2)' : 'transparent',
                          color: '#f87171',
                          fontSize: '0.78rem',
                          cursor: deletingId === run.job_id ? 'not-allowed' : 'pointer',
                          fontWeight: 500,
                          transition: 'all 0.15s',
                          opacity: deletingId === run.job_id ? 0.5 : 1,
                        }}
                        title={confirmId === run.job_id ? 'Click again to confirm' : 'Delete run'}
                      >
                        {deletingId === run.job_id ? '…' : confirmId === run.job_id ? 'Confirm?' : '✕'}
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: '1rem',
          fontSize: '0.85rem',
          color: 'var(--text-muted)',
        }}>
          <span>{total} total run{total !== 1 ? 's' : ''}</span>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button
              id="btn-history-prev-page"
              onClick={() => onPageChange(page - 1)}
              disabled={page <= 1}
              style={{
                padding: '4px 12px',
                borderRadius: '6px',
                border: '1px solid var(--border-color)',
                background: 'transparent',
                color: page <= 1 ? 'var(--text-muted)' : 'var(--text-primary)',
                cursor: page <= 1 ? 'not-allowed' : 'pointer',
                fontSize: '0.8rem',
              }}
            >
              ← Prev
            </button>
            <span>Page {page} / {totalPages}</span>
            <button
              id="btn-history-next-page"
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages}
              style={{
                padding: '4px 12px',
                borderRadius: '6px',
                border: '1px solid var(--border-color)',
                background: 'transparent',
                color: page >= totalPages ? 'var(--text-muted)' : 'var(--text-primary)',
                cursor: page >= totalPages ? 'not-allowed' : 'pointer',
                fontSize: '0.8rem',
              }}
            >
              Next →
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
