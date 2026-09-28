import React, { useState } from 'react';
import { useConfigStore } from '../../store/configStore';
import Slider from '../shared/Slider';

export default function WeightSliders() {
  const { config, updateNestedField } = useConfigStore();
  const { w1, w2, w3 } = config.weights;
  const [autoBalance, setAutoBalance] = useState(true);

  const sum = parseFloat((w1 + w2 + w3).toFixed(6));
  const isValid = Math.abs(sum - 1.0) < 1e-6;

  const handleSliderChange = (sliderKey) => (e) => {
    const newVal = parseFloat(e.target.value);
    
    if (!autoBalance) {
      updateNestedField('weights', sliderKey, newVal);
      return;
    }

    // Proportional autobalance math
    let otherKeys;
    if (sliderKey === 'w1') otherKeys = ['w2', 'w3'];
    else if (sliderKey === 'w2') otherKeys = ['w1', 'w3'];
    else otherKeys = ['w1', 'w2'];

    const [keyA, keyB] = otherKeys;
    const valA = config.weights[keyA];
    const valB = config.weights[keyB];

    const sumOthers = valA + valB;
    const remaining = 1.0 - newVal;

    let newValA, newValB;
    if (sumOthers > 0) {
      newValA = parseFloat((remaining * (valA / sumOthers)).toFixed(4));
      newValB = parseFloat((remaining - newValA).toFixed(4));
    } else {
      newValA = parseFloat((remaining / 2).toFixed(4));
      newValB = parseFloat((remaining - newValA).toFixed(4));
    }

    // Keep bounds safe
    newValA = Math.max(0, Math.min(1, newValA));
    newValB = Math.max(0, Math.min(1, newValB));

    updateNestedField('weights', sliderKey, newVal);
    updateNestedField('weights', keyA, newValA);
    updateNestedField('weights', keyB, newValB);
  };

  const handleEqualize = () => {
    updateNestedField('weights', 'w1', 0.34);
    updateNestedField('weights', 'w2', 0.33);
    updateNestedField('weights', 'w3', 0.33);
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
        <h3 style={{ fontFamily: 'var(--font-title)', fontSize: '1.1rem', fontWeight: 600 }}>
          Objective Weights
        </h3>
        <button
          type="button"
          onClick={handleEqualize}
          style={{
            background: 'transparent',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-sm)',
            padding: '2px 8px',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            cursor: 'pointer',
            transition: 'all var(--transition-fast)'
          }}
          className="btn-secondary"
        >
          Equalize
        </button>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <input
          type="checkbox"
          id="autobalance"
          checked={autoBalance}
          onChange={(e) => setAutoBalance(e.target.checked)}
          style={{ cursor: 'pointer' }}
        />
        <label htmlFor="autobalance" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
          Auto-balance sliders (keep sum = 1.0)
        </label>
      </div>

      <Slider
        label="Road Coverage Weight (w1)"
        min={0}
        max={1}
        step={0.01}
        value={w1}
        onChange={handleSliderChange('w1')}
      />

      <Slider
        label="Deployment Cost Weight (w2)"
        min={0}
        max={1}
        step={0.01}
        value={w2}
        onChange={handleSliderChange('w2')}
      />

      <Slider
        label="Network Connectivity Weight (w3)"
        min={0}
        max={1}
        step={0.01}
        value={w3}
        onChange={handleSliderChange('w3')}
      />

      <div style={{
        marginTop: '0.5rem',
        padding: '0.5rem 0.75rem',
        borderRadius: 'var(--radius-sm)',
        background: isValid ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
        border: `1px solid ${isValid ? 'rgba(16, 185, 129, 0.2)' : 'rgba(244, 63, 94, 0.2)'}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '0.85rem'
      }}>
        <span>Total Sum: <strong style={{ color: isValid ? 'var(--color-success)' : 'var(--color-danger)' }}>{sum.toFixed(3)}</strong></span>
        {!isValid && (
          <span style={{ color: 'var(--color-danger)', fontSize: '0.8rem' }}>
            Weights must sum to 1.0
          </span>
        )}
      </div>
    </div>
  );
}
