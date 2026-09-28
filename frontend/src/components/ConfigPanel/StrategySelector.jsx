import React, { useEffect, useState } from 'react';
import { useConfigStore } from '../../store/configStore';
import { getPPOStatus } from '../../api/ppoService';

export default function StrategySelector() {
  const { config, updateField, updateNestedField } = useConfigStore();
  const { strategy, use_gpu, pso_params, method = 'pso' } = config;
  const [ppoTrained, setPpoTrained] = useState(false);

  useEffect(() => {
    getPPOStatus()
      .then((res) => setPpoTrained(res.trained))
      .catch(() => setPpoTrained(false));
  }, []);

  const handleMethodChange = (newMethod) => {
    updateField('method', newMethod);
    if (newMethod === 'rl') {
      updateField('strategy', 'rl');
    } else if (strategy === 'rl') {
      updateField('strategy', 'pso');
    }
  };

  const handleStrategyChange = (e) => {
    const val = e.target.value;
    updateField('strategy', val);
    if (val === 'rl') {
      updateField('method', 'rl');
    } else {
      updateField('method', 'pso');
    }
  };

  const handleGpuToggle = (e) => {
    updateField('use_gpu', e.target.checked);
  };

  const handlePsoParamChange = (param) => (e) => {
    const isFloat = ['inertia', 'c1', 'c2'].includes(param);
    let value = isFloat ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
    if (isNaN(value)) value = 0;
    updateNestedField('pso_params', param, value);
  };

  const currentMethod = strategy === 'rl' || method === 'rl' ? 'rl' : 'pso';
  const showPsoParams = (strategy === 'pso' || strategy === 'pso_vdcoa') && currentMethod === 'pso';

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '1.1rem', fontWeight: 600, borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
        Execution Settings
      </h3>

      {/* Method Selector Toggle */}
      <div className="form-group">
        <label className="form-label">Placement Method</label>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            type="button"
            onClick={() => handleMethodChange('pso')}
            style={{
              flex: 1,
              padding: '6px 10px',
              fontSize: '0.8rem',
              fontWeight: 600,
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              cursor: 'pointer',
              background: currentMethod === 'pso' ? 'var(--color-primary)' : 'rgba(255, 255, 255, 0.05)',
              color: '#fff',
              transition: 'all 0.2s',
            }}
          >
            PSO Optimization
          </button>
          <button
            type="button"
            onClick={() => handleMethodChange('rl')}
            style={{
              flex: 1,
              padding: '6px 10px',
              fontSize: '0.8rem',
              fontWeight: 600,
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              cursor: 'pointer',
              background: currentMethod === 'rl' ? 'var(--color-primary)' : 'rgba(255, 255, 255, 0.05)',
              color: '#fff',
              transition: 'all 0.2s',
            }}
          >
            RL Placement (PPO)
          </button>
        </div>
      </div>

      {currentMethod === 'rl' ? (
        <div
          style={{
            padding: '0.75rem',
            background: 'rgba(99, 102, 241, 0.08)',
            border: '1px solid rgba(99, 102, 241, 0.25)',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.85rem',
          }}
        >
          <p style={{ fontWeight: 600, marginBottom: '0.25rem', color: '#a5b4fc' }}>
            🤖 PPO Reinforcement Learning Policy
          </p>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
            PPO agent will place cameras using learned policy.
          </p>
          {!ppoTrained && (
            <div
              style={{
                marginTop: '0.5rem',
                padding: '0.5rem',
                background: 'rgba(245, 158, 11, 0.1)',
                border: '1px solid rgba(245, 158, 11, 0.3)',
                borderRadius: '4px',
                color: '#f59e0b',
                fontSize: '0.75rem',
              }}
            >
              ⚠️ PPO agent not trained — click "Train Agent" in the RL panel.
            </div>
          )}
        </div>
      ) : (
        <div className="form-group">
          <label className="form-label">Optimization Algorithm</label>
          <select
            value={strategy}
            onChange={handleStrategyChange}
            className="form-input"
            style={{ cursor: 'pointer' }}
          >
            <option value="pso">Particle Swarm Optimization (PSO)</option>
            <option value="pso_vdcoa">PSO-VDCOA Hybrid (Chaos Refined)</option>
            <option value="random">Random Placement Baseline</option>
            <option value="grid">Grid Placement Baseline</option>
          </select>
        </div>
      )}

      {currentMethod === 'pso' && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: showPsoParams ? '1px solid var(--border-color)' : 'none' }}>
          <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>Use GPU Acceleration</span>
          <label style={{
            position: 'relative',
            display: 'inline-block',
            width: '44px',
            height: '24px',
            cursor: 'pointer'
          }}>
            <input
              type="checkbox"
              checked={use_gpu}
              onChange={handleGpuToggle}
              style={{ opacity: 0, width: 0, height: 0 }}
            />
            <span style={{
              position: 'absolute',
              top: 0, left: 0, right: 0, bottom: 0,
              backgroundColor: use_gpu ? 'var(--color-primary)' : 'rgba(255, 255, 255, 0.1)',
              transition: '0.3s',
              borderRadius: '24px',
              boxShadow: use_gpu ? '0 0 8px var(--color-primary-glow)' : 'none'
            }}>
              <span style={{
                position: 'absolute',
                height: '18px',
                width: '18px',
                left: use_gpu ? '22px' : '3px',
                bottom: '3px',
                borderRadius: '50%',
                transition: '0.3s',
                background: '#fff'
              }} />
            </span>
          </label>
        </div>
      )}

      {showPsoParams && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.25rem' }}>
          <h4 style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
            PSO Hyperparameters
          </h4>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Swarm Size</label>
              <input
                type="number"
                min="10"
                max="1000"
                value={pso_params.swarm_size}
                onChange={handlePsoParamChange('swarm_size')}
                className="form-input"
              />
            </div>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Iterations</label>
              <input
                type="number"
                min="1"
                max="10000"
                value={pso_params.iterations}
                onChange={handlePsoParamChange('iterations')}
                className="form-input"
              />
            </div>
          </div>

          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Inertia Weight (&omega;)</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={pso_params.inertia}
              onChange={handlePsoParamChange('inertia')}
              className="form-input"
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Cognitive (c1)</label>
              <input
                type="number"
                min="0"
                max="10"
                step="0.1"
                value={pso_params.c1}
                onChange={handlePsoParamChange('c1')}
                className="form-input"
              />
            </div>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Social (c2)</label>
              <input
                type="number"
                min="0"
                max="10"
                step="0.1"
                value={pso_params.c2}
                onChange={handlePsoParamChange('c2')}
                className="form-input"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
