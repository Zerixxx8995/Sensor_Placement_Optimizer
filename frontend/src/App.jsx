import React, { useState } from 'react';
import ConfigPanel from './components/ConfigPanel/ConfigPanel';
import StatusBadge from './components/shared/StatusBadge';
import ErrorBanner from './components/shared/ErrorBanner';
import Visualizer from './components/Visualizer/Visualizer';
import HistoryPanel from './components/HistoryPanel/HistoryPanel';
import SurrogateStatusCard from './components/SurrogateStatus/SurrogateStatusCard';
import RedeployPanel from './components/RedeployPanel/RedeployPanel';
import { useOptimizationJob } from './hooks/useOptimizationJob';

const TABS = [
  { id: 'optimizer', label: '⚡ Optimizer' },
  { id: 'history',   label: '📋 History' },
];

export default function App() {
  const { jobId, status, result, error, isLoading, submitJob, resetJob } = useOptimizationJob();
  const [activeTab, setActiveTab] = useState('optimizer');
  const [lastConfig, setLastConfig] = useState(null);

  // Holds a result loaded from history (so it can be replayed in the visualizer)
  const [historyResult, setHistoryResult] = useState(null);
  const [historyConfig, setHistoryConfig] = useState(null);

  // Holds dead node indices if fault injection was run
  const [deadNodeIndices, setDeadNodeIndices] = useState([]);

  // Holds redeployed positions if user triggered DQN redeployment
  const [redeployedPositions, setRedeployedPositions] = useState(null);

  const handleSubmit = (config) => {
    setLastConfig(config);
    setHistoryResult(null); // clear any history-loaded result
    setRedeployedPositions(null);
    setDeadNodeIndices([]);
    submitJob(config);
  };

  // When user clicks "Load" in History tab, clear active job state & switch to Optimizer tab
  const handleLoadFromHistory = (resultDoc, config) => {
    resetJob();
    setHistoryResult(resultDoc);
    setHistoryConfig(config);
    setRedeployedPositions(null);
    setDeadNodeIndices([]);
    setActiveTab('optimizer');
  };

  const handleRedeployComplete = (redeployedData) => {
    if (redeployedData && redeployedData.new_positions) {
      setRedeployedPositions(redeployedData.new_positions);
      setDeadNodeIndices([]);
    }
  };

  // Which result to show: live job result or history-loaded, with optional redeployed positions
  let displayResult = result || historyResult;
  if (displayResult && redeployedPositions) {
    displayResult = {
      ...displayResult,
      best_positions: redeployedPositions,
    };
  }

  const displayConfig = lastConfig || historyConfig;

  let errorMsg = null;
  let errorDetail = null;
  if (error) {
    errorMsg = error.message || 'Optimization job failed';
    errorDetail = error.detail || null;
  }

  return (
    <div className="app-container">
      <header>
        <h1 className="app-title">PSO Sensor Placement Optimizer</h1>
        <p className="app-subtitle">
          Multi-objective coverage &amp; connectivity optimization using standard and chaos-based algorithms
        </p>

        {/* ── Tab navigation ─────────────────────────────────────────── */}
        <nav style={{
          display: 'flex',
          gap: '0.25rem',
          marginTop: '1.25rem',
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '0',
        }}>
          {TABS.map(tab => (
            <button
              key={tab.id}
              id={`tab-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '0.6rem 1.25rem',
                border: 'none',
                borderBottom: activeTab === tab.id
                  ? '2px solid var(--color-primary)'
                  : '2px solid transparent',
                background: 'transparent',
                color: activeTab === tab.id
                  ? 'var(--color-primary)'
                  : 'var(--text-muted)',
                fontFamily: 'var(--font-body)',
                fontSize: '0.9rem',
                fontWeight: activeTab === tab.id ? 600 : 400,
                cursor: 'pointer',
                borderRadius: '4px 4px 0 0',
                transition: 'all 0.18s',
                letterSpacing: '0.01em',
                marginBottom: '-1px',
              }}
            >
              {tab.label}
              {tab.id === 'optimizer' && jobId && status === 'running' && (
                <span style={{
                  display: 'inline-block',
                  width: '7px',
                  height: '7px',
                  borderRadius: '50%',
                  background: '#f59e0b',
                  marginLeft: '6px',
                  animation: 'pulse 1.2s infinite',
                  verticalAlign: 'middle',
                }} />
              )}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {/* ── Optimizer Tab ───────────────────────────────────────────── */}
        {activeTab === 'optimizer' && (
          <div className="workspace-grid">
            {/* Left panel: configuration & ML status panels */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <ConfigPanel
                onSubmit={handleSubmit}
                isLoading={isLoading}
                error={null}
              />
              <SurrogateStatusCard
                isOptimizing={isLoading || status === 'running'}
                surrogateUsed={displayResult?.surrogate_used}
                iterationsTotal={displayConfig?.pso_params?.iterations || 500}
              />
              <RedeployPanel
                positions={displayResult?.best_positions}
                deadNodeIndices={deadNodeIndices}
                area={displayConfig?.area || { width: 100, height: 100 }}
                sensingRadius={displayConfig?.sensing_radius || 15}
                onRedeployComplete={handleRedeployComplete}
              />
            </div>



            {/* Right panel: status + visualizer */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {errorMsg && (
                <ErrorBanner message={errorMsg} detail={errorDetail} />
              )}

              {/* History result banner */}
              {historyResult && !jobId && (
                <div style={{
                  padding: '0.65rem 1rem',
                  borderRadius: '8px',
                  border: '1px solid rgba(99,102,241,0.3)',
                  background: 'rgba(99,102,241,0.08)',
                  color: '#a5b4fc',
                  fontSize: '0.85rem',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}>
                  <span>📋 Viewing a past run from history</span>
                  <button
                    onClick={() => { setHistoryResult(null); setHistoryConfig(null); }}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#a5b4fc',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      textDecoration: 'underline',
                    }}
                  >
                    Clear
                  </button>
                </div>
              )}

              {jobId ? (
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  {/* Job header */}
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    borderBottom: '1px solid var(--border-color)',
                    paddingBottom: '0.75rem',
                  }}>
                    <div>
                      <h2 style={{ fontFamily: 'var(--font-title)', fontSize: '1.4rem', fontWeight: 600 }}>
                        Optimization Status
                      </h2>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        Job ID: {jobId}
                      </span>
                    </div>
                    <StatusBadge status={status} />
                  </div>

                  {/* Success, Running, or Pending: Visualizer */}
                  {(status === 'pending' || status === 'running' || status === 'complete') && (
                    <Visualizer
                      result={displayResult}
                      config={displayConfig}
                      jobId={jobId}
                      status={status}
                      onFaultChange={setDeadNodeIndices}
                    />
                  )}
                </div>
              ) : historyResult ? (
                /* History replay view */
                <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    borderBottom: '1px solid var(--border-color)',
                    paddingBottom: '0.75rem',
                  }}>
                    <div>
                      <h2 style={{ fontFamily: 'var(--font-title)', fontSize: '1.4rem', fontWeight: 600 }}>
                        Historical Run
                      </h2>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                        Job ID: {historyResult.job_id}
                      </span>
                    </div>
                    <StatusBadge status="complete" />
                  </div>
                  <Visualizer
                    result={historyResult}
                    config={historyConfig}
                    jobId={historyResult.job_id}
                    status="complete"
                    onFaultChange={setDeadNodeIndices}
                  />
                </div>
              ) : (
                /* Empty state */
                <div className="glass-panel" style={{
                  display: 'flex', flexDirection: 'column',
                  justifyContent: 'center', alignItems: 'center',
                  minHeight: '400px', gap: '1.5rem', textAlign: 'center',
                }}>
                  <div style={{
                    width: '64px', height: '64px', borderRadius: '50%',
                    background: 'rgba(99, 102, 241, 0.1)',
                    display: 'flex', justifyContent: 'center', alignItems: 'center',
                    fontSize: '2rem', color: 'var(--color-primary)',
                  }}>
                    &#x2699;
                  </div>
                  <div>
                    <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                      No Active Optimization
                    </h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', maxWidth: '320px' }}>
                      Configure your network deployment parameters in the sidebar panel and click{' '}
                      <strong>Run Optimization</strong> to launch a task, or view a past run in the{' '}
                      <button
                        onClick={() => setActiveTab('history')}
                        style={{
                          background: 'none', border: 'none', color: 'var(--color-primary)',
                          cursor: 'pointer', textDecoration: 'underline', fontSize: 'inherit',
                          fontWeight: 600, padding: 0,
                        }}
                      >
                        History tab
                      </button>.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── History Tab ─────────────────────────────────────────────── */}
        {activeTab === 'history' && (
          <div className="glass-panel" style={{ maxWidth: '1200px', margin: '0 auto' }}>
            <HistoryPanel onLoadResult={handleLoadFromHistory} />
          </div>
        )}
      </main>
    </div>
  );
}
