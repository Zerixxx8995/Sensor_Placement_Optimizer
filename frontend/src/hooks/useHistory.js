/**
 * hooks/useHistory.js
 * --------------------
 * Stateful hook for the History tab.
 *
 * Responsibilities:
 *   - Fetch paginated run list from /history
 *   - Fetch aggregate stats from /history/stats
 *   - Expose deleteRun() and loadIntoVisualizer()
 *   - Manage loading/error states
 *
 * No direct rendering — pure state and side-effects.
 */

import { useState, useEffect, useCallback } from 'react';
import { listRuns, deleteRun as apiDeleteRun, getStats } from '../api/historyService';

export function useHistory({ onLoadResult } = {}) {
  const [runs, setRuns] = useState([]);
  const [stats, setStats] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const PAGE_SIZE = 20;

  const fetchRuns = useCallback(async (p = page) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listRuns(p, PAGE_SIZE);
      setRuns(data.runs || []);
      setTotal(data.total || 0);
      setTotalPages(data.total_pages || 1);
      setPage(p);
    } catch (err) {
      setError(err.message || 'Failed to load history');
    } finally {
      setIsLoading(false);
    }
  }, [page]);

  const fetchStats = useCallback(async () => {
    try {
      const data = await getStats();
      setStats(data);
    } catch {
      // stats are non-critical — silently fail
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchRuns(1);
    fetchStats();
  }, []);

  const deleteRun = useCallback(async (jobId) => {
    try {
      await apiDeleteRun(jobId);
      // Refresh current page; if it becomes empty go back one
      const newTotal = total - 1;
      const newTotalPages = Math.max(1, Math.ceil(newTotal / PAGE_SIZE));
      const targetPage = Math.min(page, newTotalPages);
      await fetchRuns(targetPage);
      await fetchStats();
    } catch (err) {
      setError(err.message || 'Failed to delete run');
    }
  }, [total, page, fetchRuns, fetchStats]);

  const goToPage = useCallback((p) => {
    if (p >= 1 && p <= totalPages) {
      fetchRuns(p);
    }
  }, [fetchRuns, totalPages]);

  const refresh = useCallback(() => {
    fetchRuns(page);
    fetchStats();
  }, [fetchRuns, fetchStats, page]);

  return {
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
  };
}
