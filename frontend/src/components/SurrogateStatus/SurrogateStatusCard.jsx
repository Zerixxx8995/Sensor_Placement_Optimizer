import React, { useState, useEffect } from 'react';
import { getSurrogateStatus, trainSurrogate } from '../../api/surrogateService';

export default function SurrogateStatusCard({ isOptimizing, surrogateUsed, iterationsTotal }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [msg, setMsg] = useState(null);

  const fetchStatus = async () => {
    try {
      const data = await getSurrogateStatus();
      setStatus(data);
    } catch (err) {
      console.error("Failed to fetch surrogate status", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 10000);
    return () => clearInterval(timer);
  }, []);

  const handleRetrain = async () => {
    try {
      setIsTraining(true);
      setMsg(null);
      const res = await trainSurrogate(50);
      setMsg(res.message || "Training started!");
      setTimeout(fetchStatus, 2000);
    } catch (err) {
      setMsg(err.response?.data?.detail || "Training request failed");
    } finally {
      setIsTraining(false);
    }
  };

  if (loading) {
    return (
      <div className="glass-panel" style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        Loading Surrogate Model status…
      </div>
    );
  }

  const { trained, runs_available, runs_needed, last_trained } = status || {};
  const switchIter = Math.floor((iterationsTotal || 500) * 0.7);

  return (
    <div className="glass-panel" style={{ padding: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '1.1rem' }}>🧠</span>
          <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '1rem', fontWeight: 600, margin: 0 }}>
            PyTorch Surrogate Model
          </h3>
        </div>
        <span style={{
          padding: '0.2rem 0.6rem',
          borderRadius: '12px',
          fontSize: '0.75rem',
          fontWeight: 600,
          background: trained ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
          color: trained ? '#10b981' : '#f59e0b',
          border: trained ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(245, 158, 11, 0.3)',
        }}>
          {trained ? 'Trained' : 'Not Trained'}
        </span>
      </div>

      {/* Banner during optimization */}
      {isOptimizing && (
        <div style={{
          padding: '0.6rem 0.8rem',
          borderRadius: '6px',
          fontSize: '0.8rem',
          background: trained
            ? 'rgba(99, 102, 241, 0.12)'
            : 'rgba(239, 68, 68, 0.1)',
          border: trained
            ? '1px solid rgba(99, 102, 241, 0.3)'
            : '1px solid rgba(239, 68, 68, 0.2)',
          color: trained ? '#a5b4fc' : '#fca5a5',
        }}>
          {trained
            ? `⚡ Fast surrogate active (iterations 1–${switchIter}), switching to analytical at ${switchIter + 1}`
            : `⚠️ Surrogate model not yet trained — ${runs_needed} more runs needed`}
        </div>
      )}

      {/* Stats summary */}
      <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Runs in DB:</span>
          <strong style={{ color: 'var(--text-primary)' }}>{runs_available} / 5</strong>
        </div>
        {!trained && runs_needed > 0 && (
          <div style={{ width: '100%', height: '4px', background: 'var(--border-color)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{
              width: `${Math.min(100, (runs_available / 5) * 100)}%`,
              height: '100%',
              background: 'var(--color-primary)',
            }} />
          </div>
        )}
        {last_trained && (
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            <span>Last trained:</span>
            <span>{new Date(last_trained).toLocaleString()}</span>
          </div>
        )}
      </div>

      {msg && (
        <div style={{ fontSize: '0.8rem', color: '#10b981', background: 'rgba(16,185,129,0.1)', padding: '0.4rem', borderRadius: '4px' }}>
          {msg}
        </div>
      )}

      {/* Retrain button */}
      <button
        onClick={handleRetrain}
        disabled={isTraining || runs_available < 5}
        style={{
          padding: '0.5rem 0.8rem',
          border: '1px solid var(--border-color)',
          borderRadius: '6px',
          background: runs_available >= 5 ? 'rgba(99, 102, 241, 0.15)' : 'rgba(255,255,255,0.03)',
          color: runs_available >= 5 ? 'var(--color-primary)' : 'var(--text-muted)',
          fontSize: '0.82rem',
          fontWeight: 600,
          cursor: runs_available >= 5 && !isTraining ? 'pointer' : 'not-allowed',
          transition: 'all 0.15s ease',
        }}
      >
        {isTraining ? 'Training…' : 'Retrain Now'}
      </button>
    </div>
  );
}
