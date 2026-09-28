import React from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useConfigStore } from '../src/store/configStore';
import WeightSliders from '../src/components/ConfigPanel/WeightSliders';

describe('useConfigStore - weights state', () => {
  beforeEach(() => {
    useConfigStore.getState().resetConfig();
  });

  it('should initialize with default weights summing to 1.0', () => {
    const { weights } = useConfigStore.getState().config;
    expect(weights.w1).toBe(0.7);
    expect(weights.w2).toBe(0.15);
    expect(weights.w3).toBe(0.15);
    expect(weights.w1 + weights.w2 + weights.w3).toBe(1.0);
  });

  it('should update weights correctly when updateNestedField is called', () => {
    useConfigStore.getState().updateNestedField('weights', 'w1', 0.8);
    useConfigStore.getState().updateNestedField('weights', 'w2', 0.1);
    useConfigStore.getState().updateNestedField('weights', 'w3', 0.1);

    const { weights } = useConfigStore.getState().config;
    expect(weights.w1).toBe(0.8);
    expect(weights.w2).toBe(0.1);
    expect(weights.w3).toBe(0.1);
    expect(weights.w1 + weights.w2 + weights.w3).toBe(1.0);
  });
});

describe('WeightSliders Component', () => {
  beforeEach(() => {
    useConfigStore.getState().resetConfig();
  });

  it('renders all three objective sliders and sum indicator', () => {
    render(<WeightSliders />);
    expect(screen.getByText('Coverage Weight (w1)')).toBeDefined();
    expect(screen.getByText('Energy Weight (w2)')).toBeDefined();
    expect(screen.getByText('Connectivity Weight (w3)')).toBeDefined();
    
    expect(screen.getByText('Total Sum:')).toBeDefined();
    expect(screen.getByText('1.000')).toBeDefined();
  });

  it('equalizes weights to 0.34, 0.33, 0.33 when Equalize button is clicked', async () => {
    render(<WeightSliders />);
    const equalizeBtn = screen.getByText('Equalize');
    fireEvent.click(equalizeBtn);

    const { weights } = useConfigStore.getState().config;
    expect(weights.w1).toBe(0.34);
    expect(weights.w2).toBe(0.33);
    expect(weights.w3).toBe(0.33);
    expect(weights.w1 + weights.w2 + weights.w3).toBe(1.0);
  });
});
