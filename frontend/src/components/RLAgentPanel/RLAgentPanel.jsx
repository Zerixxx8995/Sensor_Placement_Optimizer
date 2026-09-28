import React, { useState, useEffect } from 'react';
import PPOStatusCard from './PPOStatusCard';
import TrainingChart from './TrainingChart';
import { usePPOTraining } from '../../hooks/usePPOTraining';
import { getPPOStatus } from '../../api/ppoService';
import Button from '../shared/Button';

export default function RLAgentPanel() {
  const [statusData, setStatusData] = useState(null);
  const [trainEpisodes, setTrainEpisodes] = useState(50);
  const {
    isTraining,
    history,
    currentEpisode,
    latestReward,
    latestCoverage,
    bestCoverage,
    startTraining,
    stopTraining,
  } = usePPOTraining();

  const fetchStatus = async () => {
    try {
      const data = await getPPOStatus();
      setStatusData(data);
    } catch (e) {
      console.error('Error fetching PPO status:', e);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  useEffect(() => {
    if (!isTraining) {
      fetchStatus();
    }
  }, [isTraining]);

  const handleStart = () => {
    startTraining(trainEpisodes);
  };

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
        <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '1.2rem', fontWeight: 600 }}>
          🤖 RL Placement Agent (PPO)
        </h3>
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Continuous action-space policy trained on simulated city grid layouts
        </p>
      </div>

      <PPOStatusCard statusData={statusData} isTraining={isTraining} />

      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Episodes to train</label>
          <input
            type="number"
            min="10"
            max="1000"
            step="10"
            value={trainEpisodes}
            onChange={(e) => setTrainEpisodes(parseInt(e.target.value, 10) || 50)}
            disabled={isTraining}
            className="form-input"
            style={{ padding: '4px 8px', fontSize: '0.8rem' }}
          />
        </div>

        {isTraining ? (
          <Button variant="danger" onClick={stopTraining} style={{ alignSelf: 'flex-end' }}>
            Stop Training
          </Button>
        ) : (
          <Button variant="primary" onClick={handleStart} style={{ alignSelf: 'flex-end' }}>
            Train Agent
          </Button>
        )}
      </div>

      {isTraining && (
        <div style={{ fontSize: '0.8rem', color: 'var(--color-primary)', display: 'flex', justifyContent: 'space-between' }}>
          <span>Episode {currentEpisode} / {trainEpisodes}</span>
          <span>Reward: {latestReward.toFixed(2)} | Coverage: {(latestCoverage * 100).toFixed(1)}%</span>
        </div>
      )}

      <TrainingChart data={history} />
    </div>
  );
}
