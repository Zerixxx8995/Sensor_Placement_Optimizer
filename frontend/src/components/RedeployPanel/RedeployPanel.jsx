import React, { useState, useEffect } from 'react';
import { getRedeployStatus, trainRedeployAgent, redeploySensors } from '../../api/redeployService';

export default function RedeployPanel({
  positions,
  deadNodeIndices = [],
  area = { width: 100, height: 100 },
  sensingRadius = 15,
  onRedeployComplete,
}) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isTraining, setIsTraining] = useState(false);
  const [isRedeploying, setIsRedeploying] = useState(false);
  const [redeployResult, setRedeployResult] = useState(null);
  const [msg, setMsg] = useState(null);

  const fetchStatus = async () => {
    try {
      const data = await getRedeployStatus();
      setStatus(data);
    } catch (err) {
      console.error("Failed to fetch DQN status", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const timer = setInterval(fetchStatus, 10000);
    return () => clearInterval(timer);
  }, []);

  const handleTrain = async () => {
    try {
      setIsTraining(true);
      setMsg(null);
      const res = await trainRedeployAgent(500);
      setMsg(res.message || "Training started!");
      setTimeout(fetchStatus, 2000);
    } catch (err) {
      setMsg(err.response?.data?.detail || "Training request failed");
    } finally {
      setIsTraining(false);
    }
  };

  const handleRedeploy = async () => {
    if (!positions || positions.length === 0) return;
    try {
      setIsRedeploying(true);
      setMsg(null);
      const res = await redeploySensors({
        positions,
        dead_node_indices: deadNodeIndices,
        area,
        sensing_radius: sensingRadius,
      });
      setRedeployResult(res);
      if (onRedeployComplete) {
        onRedeployComplete(res);
      }
    } catch (err) {
      setMsg(err.response?.data?.detail || "Redeployment failed");
    } finally {
      setIsRedeploying(false);
    }
  };

  if (loading) {
    return (
      <div className="glass-panel" style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        Loading DQN Agent status…
      </div>
    );
  }

  const { agent_trained, episodes_trained } = status || {};
  const hasFailures = deadNodeIndices && deadNodeIndices.length > 0;

  return (
    <div className="glass-panel" style={{ padding: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '1.1rem' }}>🤖</span>
          <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '1rem', fontWeight: 600, margin: 0 }}>
            DQN Adaptive Redeployment
          </h3>
        </div>
        <span style={{
          padding: '0.2rem 0.6rem',
          borderRadius: '12px',
          fontSize: '0.75rem',
          fontWeight: 600,
          background: agent_trained ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          color: agent_trained ? '#10b981' : '#ef4444',
          border: agent_trained ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)',
        }}>
          {agent_trained ? `Trained (${episodes_trained} eps)` : 'Not Trained'}
        </span>
      </div>

      {!agent_trained && (
        <div style={{ fontSize: '0.82rem', color: '#f59e0b', background: 'rgba(245,158,11,0.1)', padding: '0.5rem', borderRadius: '6px' }}>
          ⚠️ DQN agent not trained yet — click <strong>Train Agent</strong> first.
        </div>
      )}

      {msg && (
        <div style={{ fontSize: '0.8rem', color: '#10b981', background: 'rgba(16,185,129,0.1)', padding: '0.4rem', borderRadius: '4px' }}>
          {msg}
        </div>
      )}

      {/* Action controls */}
      <div style={{ display: 'flex', gap: '0.6rem' }}>
        <button
          onClick={handleTrain}
          disabled={isTraining}
          style={{
            flex: 1,
            padding: '0.55rem 0.8rem',
            border: '1px solid var(--border-color)',
            borderRadius: '6px',
            background: 'rgba(99, 102, 241, 0.15)',
            color: 'var(--color-primary)',
            fontSize: '0.82rem',
            fontWeight: 600,
            cursor: isTraining ? 'not-allowed' : 'pointer',
          }}
        >
          {isTraining ? 'Training Agent…' : 'Train Agent (500 eps)'}
        </button>

        <button
          onClick={handleRedeploy}
          disabled={isRedeploying || !hasFailures}
          style={{
            flex: 1,
            padding: '0.55rem 0.8rem',
            border: 'none',
            borderRadius: '6px',
            background: hasFailures ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)' : 'rgba(255,255,255,0.05)',
            color: hasFailures ? '#ffffff' : 'var(--text-muted)',
            fontSize: '0.82rem',
            fontWeight: 600,
            cursor: hasFailures && !isRedeploying ? 'pointer' : 'not-allowed',
          }}
        >
          {isRedeploying ? 'Redeploying…' : '⚡ Redeploy with DQN'}
        </button>
      </div>

      {/* Redeployment results */}
      {redeployResult && (
        <div style={{
          marginTop: '0.4rem',
          padding: '0.75rem',
          borderRadius: '8px',
          background: 'rgba(16, 185, 129, 0.08)',
          border: '1px solid rgba(16, 185, 129, 0.25)',
          fontSize: '0.82rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.35rem',
        }}>
          <div style={{ fontWeight: 600, color: '#10b981' }}>
            ✓ Redeployment Complete
          </div>
          <div>
            Coverage before: <strong>{(redeployResult.coverage_before * 100).toFixed(1)}%</strong> → after DQN: <strong style={{ color: '#10b981' }}>{(redeployResult.coverage_after * 100).toFixed(1)}%</strong>
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Moved nodes: {redeployResult.moved_nodes.length > 0 ? redeployResult.moved_nodes.join(', ') : 'None'}
          </div>
        </div>
      )}
    </div>
  );
}
