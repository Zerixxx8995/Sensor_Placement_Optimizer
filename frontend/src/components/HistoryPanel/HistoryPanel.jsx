/**
 * components/HistoryPanel/HistoryPanel.jsx
 * -----------------------------------------
 * Parent component for the History tab.
 *
 * Owns table state via useHistory hook.
 * Renders StatsBar + RunsTable.
 *
 * Props:
 *   onLoadResult – (result: object, config: object) => void
 *                  Called when user clicks "Load" on a past run.
 *                  Parent (App) uses this to push the result into the visualizer.
 */

import React, { useCallback } from 'react';
import { useHistory } from '../../hooks/useHistory';
import StatsBar from './StatsBar';
import RunsTable from './RunsTable';
import { getRun } from '../../api/historyService';

export default function HistoryPanel({ onLoadResult }) {
  const {
    runs,
    stats,
    page,
    totalPages,
    total,
    isLoading,
    error,
    deleteRun,
    goToPage,
    refresh,
  } = useHistory();

  const handleLoadRun = useCallback(async (jobId) => {
    try {
      const doc = await getRun(jobId);
      if (onLoadResult) {
        // Shape the result to match what the visualizer expects
        onLoadResult(
          { ...doc.result, job_id: doc.job_id },
          doc.config,
        );
      }
    } catch (err) {
      console.error('Failed to load run:', err);
    }
  }, [onLoadResult]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <div>
          <h2 style={{
            fontFamily: 'var(--font-title)',
            fontSize: '1.4rem',
            fontWeight: 700,
            color: 'var(--text-primary)',
            margin: 0,
          }}>
            Experiment History
          </h2>
          <p style={{
            fontSize: '0.875rem',
            color: 'var(--text-muted)',
            margin: '0.25rem 0 0',
          }}>
            All completed optimization runs — click Load to replay in the visualizer
          </p>
        </div>
        <button
          id="btn-history-refresh"
          onClick={refresh}
          disabled={isLoading}
          style={{
            padding: '6px 14px',
            borderRadius: '8px',
            border: '1px solid var(--border-color)',
            background: 'rgba(255,255,255,0.05)',
            color: 'var(--text-secondary)',
            fontSize: '0.82rem',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            transition: 'all 0.15s',
          }}
          onMouseEnter={e => { if (!isLoading) e.currentTarget.style.background = 'rgba(255,255,255,0.09)'; }}
          onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
        >
          <span style={{ display: 'inline-block', animation: isLoading ? 'spin 0.8s linear infinite' : 'none' }}>↻</span>
          Refresh
        </button>
      </div>

      {/* Error banner */}
      {error && (
        <div style={{
          padding: '0.75rem 1rem',
          borderRadius: '8px',
          border: '1px solid rgba(239,68,68,0.3)',
          background: 'rgba(239,68,68,0.1)',
          color: '#f87171',
          fontSize: '0.875rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <span>⚠ {error}</span>
          <button
            onClick={refresh}
            style={{
              background: 'none',
              border: 'none',
              color: '#f87171',
              cursor: 'pointer',
              fontSize: '0.8rem',
              textDecoration: 'underline',
            }}
          >
            Retry
          </button>
        </div>
      )}

      {/* Aggregate stats */}
      <StatsBar stats={stats} />

      {/* Runs table */}
      <RunsTable
        runs={runs}
        page={page}
        totalPages={totalPages}
        total={total}
        isLoading={isLoading}
        onDelete={deleteRun}
        onLoadRun={handleLoadRun}
        onPageChange={goToPage}
      />
    </div>
  );
}
