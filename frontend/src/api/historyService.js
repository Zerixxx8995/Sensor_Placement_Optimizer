/**
 * api/historyService.js
 * ----------------------
 * API calls for the experiment history feature.
 *
 * All functions return plain JS objects/arrays — no framework-specific types.
 * Throws on non-2xx responses so callers can catch uniformly.
 */

import { httpClient } from './httpClient';

/**
 * List past optimization runs (paginated).
 * @param {number} page      - 1-indexed page number
 * @param {number} pageSize  - items per page (max 100)
 * @returns {Promise<{ runs: object[], total: number, page: number, page_size: number, total_pages: number }>}
 */
export async function listRuns(page = 1, pageSize = 20) {
  return httpClient.get(`/history?page=${page}&page_size=${pageSize}`);
}

/**
 * Fetch the full result document for a single past run.
 * @param {string} jobId
 * @returns {Promise<object>}
 */
export async function getRun(jobId) {
  return httpClient.get(`/history/${jobId}`);
}

/**
 * Delete a past run from history.
 * @param {string} jobId
 * @returns {Promise<{ deleted: boolean, job_id: string }>}
 */
export async function deleteRun(jobId) {
  return httpClient.delete(`/history/${jobId}`);
}

/**
 * Fetch aggregate statistics across all stored runs.
 * @returns {Promise<{ total_runs: number, best_coverage_ever: number|null, avg_compute_time: number|null, runs_per_strategy: object, most_used_strategy: string|null }>}
 */
export async function getStats() {
  return httpClient.get('/history/stats');
}
