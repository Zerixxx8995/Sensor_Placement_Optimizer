import React, { useState, useEffect, useCallback } from 'react';
import GridCanvas from './GridCanvas';
import LiveMetrics from './LiveMetrics';
import ParticleLayer from './ParticleLayer';
import ConvergenceChart from '../ConvergenceChart/ConvergenceChart';
import FaultInjector from '../FaultInjector/FaultInjector';
import ComparisonTable from '../ComparisonTable/ComparisonTable';
import { compareService } from '../../api/compareService';
import { useSSEStream } from '../../hooks/useSSEStream';

/**
 * Visualizer
 * ----------
 * Parent component that owns the full visualisation panel.
 * Composes GridCanvas (the canvas renderer) and LiveMetrics (the stats bar).
 * Supports real-time visualization of running jobs using Server-Sent Events (SSE).
 *
 * Props
 * -----
 *  result        {object|null}  OptimizationResult from the backend (when complete).
 *  config        {object|null}  The OptimizationConfig that was submitted.
 *  jobId         {string|null}  UUID of the active job.
 *  status        {string|null}  Job execution status ('running', 'complete', etc.).
 */
export default function Visualizer({ result, config, jobId, status, onFaultChange }) {
  const isOptimizing = status === 'pending' || status === 'running';

  const [liveHistory, setLiveHistory] = useState([]);
  const [faultResult, setFaultResult] = useState(null);

  const [compareResult, setCompareResult] = useState(null);
  const [isComparing, setIsComparing] = useState(false);
  const [compareError, setCompareError] = useState(null);

  // Clear live history and fault simulation results when jobId changes
  useEffect(() => {
    setLiveHistory([]);
    setFaultResult(null);
    setCompareResult(null);
    setIsComparing(false);
    setCompareError(null);
    if (onFaultChange) {
      onFaultChange([]);
    }
  }, [jobId]);

  // Synchronously fetch baseline comparison metrics on completion
  useEffect(() => {
    if (status !== 'complete' || !config || !jobId) {
      setCompareResult(null);
      setIsComparing(false);
      setCompareError(null);
      return;
    }

    let isMounted = true;

    const fetchComparison = async () => {
      setIsComparing(true);
      setCompareError(null);
      try {
        const res = await compareService.runComparison(config);
        if (isMounted) {
          setCompareResult(res);
        }
      } catch (err) {
        console.error('Error running comparison:', err);
        if (isMounted) {
          setCompareError(err.message || 'Comparison failed');
        }
      } finally {
        if (isMounted) {
          setIsComparing(false);
        }
      }
    };

    fetchComparison();

    return () => {
      isMounted = false;
    };
  }, [status, config, jobId]);

  const handleIteration = useCallback((data) => {
    setLiveHistory((prev) => {
      if (prev.some((item) => item.iteration === data.iteration)) {
        return prev;
      }
      return [...prev, { iteration: data.iteration, fitness: data.best_fitness }];
    });
  }, []);

  // Connect to SSE stream if the job is active
  const {
    particles,
    iteration,
    bestFitness,
    bestPositions,
  } = useSSEStream(isOptimizing ? jobId : null, {
    onIteration: handleIteration,
  });

  // Map fitness history: liveHistory if optimizing, result.fitness_history if complete
  const chartData = isOptimizing
    ? liveHistory
    : (result?.fitness_history || []).map((val, idx) => ({
        iteration: idx,
        fitness: val,
      }));

  const areaWidth = config?.area?.width ?? 100;
  const areaHeight = config?.area?.height ?? 100;
  const sensingRadius = config?.sensing_radius ?? 10;
  const commRadius = config?.comm_radius ?? 20;

  // Render live positions when optimizing (Global Best), fallback to final result when complete
  const displayResult = isOptimizing
    ? { best_positions: (bestPositions && bestPositions.length > 0 ? bestPositions : (result?.best_positions || [])), coverage_map: null }
    : faultResult
      ? {
          ...result,
          coverage_map: faultResult.coverage_map,
          failed_indices: faultResult.failed_indices,
        }
      : result;

  return (
    <div className="visualizer-container">
      {/* ─── Header ─────────────────────────────────────────────────── */}
      <div className="visualizer-header">
        <div>
          <h2 className="visualizer-title">
            {isOptimizing ? 'Optimizing Deployments...' : 'Deployment Visualizer'}
          </h2>
          <p className="visualizer-subtitle">
            {isOptimizing
              ? 'Real-time swarm convergence · Candidate nodes · Live updates'
              : 'Coverage heatmap · Sensor nodes · Communication links'}
          </p>
        </div>
        {status === 'complete' && result && (
          <div className="vis-badge-group">
            {result.gpu_used && (
              <span className="vis-badge vis-badge--gpu">⚡ GPU</span>
            )}
            <span className="vis-badge vis-badge--complete">● Complete</span>
          </div>
        )}
        {isOptimizing && (
          <div className="vis-badge-group">
            <span
              className="vis-badge"
              style={{
                background: 'rgba(99, 102, 241, 0.15)',
                color: 'var(--color-primary)',
                border: '1px solid rgba(99, 102, 241, 0.3)',
                animation: 'pulse 1.5s infinite',
              }}
            >
              ● Optimizing
            </span>
            <style>{`
              @keyframes pulse {
                0%   { opacity: 0.6; }
                50%  { opacity: 1; }
                100% { opacity: 0.6; }
              }
            `}</style>
          </div>
        )}
      </div>

      {/* ─── Canvas Layer Stack ─────────────────────────────────────── */}
      <GridCanvas
        result={displayResult}
        areaWidth={areaWidth}
        areaHeight={areaHeight}
        sensingRadius={sensingRadius}
        commRadius={commRadius}
      >
        {isOptimizing && (
          <ParticleLayer
            particles={particles}
            areaWidth={areaWidth}
            areaHeight={areaHeight}
          />
        )}
      </GridCanvas>

      {/* ─── Metrics bar ────────────────────────────────────────────── */}
      <LiveMetrics
        result={result}
        status={status}
        liveIteration={iteration}
        liveFitness={bestFitness}
      />

      {/* ─── Convergence Chart ──────────────────────────────────────── */}
      <ConvergenceChart data={chartData} />

      {/* ─── Fault Injector Simulation ───────────────────────────────── */}
      {status === 'complete' && result && (
        <FaultInjector
          jobId={jobId}
          originalResult={result}
          onInject={(res) => {
            setFaultResult(res);
            if (onFaultChange) {
              onFaultChange(res?.failed_indices || []);
            }
          }}
          onReset={() => {
            setFaultResult(null);
            if (onFaultChange) {
              onFaultChange([]);
            }
          }}
        />
      )}

      {/* ─── Strategy Comparison Benchmarks ─────────────────────────── */}
      {status === 'complete' && (
        <div style={{ marginTop: '1.5rem' }}>
          {isComparing && (
            <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '120px', gap: '1rem' }}>
              <div className="spinner" style={{
                width: '30px',
                height: '30px',
                border: '2px solid rgba(99, 102, 241, 0.1)',
                borderTop: '2px solid var(--color-primary)',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
              }} />
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Running baseline strategy comparisons…</span>
            </div>
          )}
          {compareError && (
            <div className="glass-card" style={{ padding: '1rem', border: '1px solid rgba(244, 63, 94, 0.2)', background: 'rgba(244, 63, 94, 0.05)', color: 'var(--color-danger)', fontSize: '0.85rem' }}>
              Failed to load strategy comparison benchmarks: {compareError}
            </div>
          )}
          {compareResult && (
            <ComparisonTable
              results={compareResult.results}
              activeStrategy={config?.strategy}
            />
          )}
        </div>
      )}
    </div>
  );
}
