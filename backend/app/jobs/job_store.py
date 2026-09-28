"""
jobs/job_store.py
-----------------
In-memory job registry. Maps job_id → {status, result, error}.

Thread-safe via threading.Lock so FastAPI BackgroundTasks (which run in
worker threads) can safely write while request threads read.

No I/O, no HTTP, no business logic — pure state storage.
"""

import threading
import queue
from typing import Any

# Module-level store — lives for the lifetime of the process.
_store: dict[str, dict[str, Any]] = {}
_subscribers: dict[str, list[queue.Queue]] = {}
_history_events: dict[str, list[dict]] = {}
_lock = threading.Lock()


def create_job(job_id: str) -> None:
    """Register a new job in PENDING state."""
    with _lock:
        _store[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
        }
        _subscribers[job_id] = []
        _history_events[job_id] = []


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return the job record or None if not found. Returns a shallow copy."""
    with _lock:
        record = _store.get(job_id)
        return dict(record) if record is not None else None


def set_running(job_id: str) -> None:
    """Transition job to RUNNING state."""
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "running"


def set_complete(job_id: str, result: dict[str, Any]) -> None:
    """Transition job to COMPLETE, store its result payload, and notify subscribers."""
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "complete"
            _store[job_id]["result"] = result
        if job_id in _subscribers:
            for q in _subscribers[job_id]:
                q.put({"event": "complete", "result": result})


def set_failed(job_id: str, error: str) -> None:
    """Transition job to FAILED, record the error message, and notify subscribers."""
    with _lock:
        if job_id in _store:
            _store[job_id]["status"] = "failed"
            _store[job_id]["error"] = error
        if job_id in _subscribers:
            for q in _subscribers[job_id]:
                q.put({"event": "failed", "error": error})


def subscribe_job(job_id: str) -> queue.Queue:
    """Subscribe to receive real-time iteration events for a job."""
    with _lock:
        q = queue.Queue()
        # Replay any iteration events that occurred before subscription
        if job_id in _history_events:
            for event_data in _history_events[job_id]:
                q.put(event_data)
        if job_id not in _subscribers:
            _subscribers[job_id] = []
        _subscribers[job_id].append(q)
        return q


def unsubscribe_job(job_id: str, q: queue.Queue) -> None:
    """Unsubscribe a queue from job events."""
    with _lock:
        if job_id in _subscribers:
            try:
                _subscribers[job_id].remove(q)
            except ValueError:
                pass


def publish_iteration(job_id: str, data: dict) -> None:
    """Send an iteration update to all active subscribers for this job."""
    with _lock:
        if job_id in _history_events:
            _history_events[job_id].append(data)
        if job_id in _subscribers:
            for q in _subscribers[job_id]:
                q.put(data)


def clear_all() -> None:
    """
    Wipe the entire store. Intended for use in tests only.
    Not exposed via any API endpoint.
    """
    with _lock:
        _store.clear()
        _subscribers.clear()
        _history_events.clear()
