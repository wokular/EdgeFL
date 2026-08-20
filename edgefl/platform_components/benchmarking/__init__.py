"""Benchmarking package: the Benchmarker metrics pipeline (see PLAN.md) plus an
env-driven singleton accessor used by the server instrumentation call-sites."""

import os
import threading

from platform_components.lib.logger.error_handling import get_logger

logger = get_logger(__name__)

_benchmarker = None
_lock = threading.Lock()


def _is_true(value, default=False):
    """Env flags are written as True/False/1/0/yes/no across the env files."""
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def _resolve_endpoint():
    """The REST address this process streams metrics to, or "" if none resolves.

    BENCHMARK_REST_CONN is the explicit collector (the default deployment points every
    process at one operator). Fallback to this process's own EXTERNAL_IP is never
    implicit — it requires BENCHMARK_FALLBACK=True — because a node's own address is
    often a master node, which has no Operator process and would silently drop every
    metric rather than fail loudly.
    """
    conn = os.getenv("BENCHMARK_REST_CONN", "").strip()
    if conn:
        return conn
    if _is_true(os.getenv("BENCHMARK_FALLBACK")):
        own = os.getenv("EXTERNAL_IP", "").strip()
        if own:
            logger.info(f"BENCHMARK_REST_CONN unset; falling back to EXTERNAL_IP ({own})")
            return own
        logger.warning("BENCHMARK_FALLBACK=True but EXTERNAL_IP is unset.")
    return ""


def get_benchmarker():
    """Process-wide Benchmarker, configured from env (BENCHMARK_ENABLED /
    BENCHMARK_REST_CONN / BENCHMARK_FALLBACK). Built lazily on first metric call so
    construction happens after the harness's env is injected and the target operator is
    up. Never raises — a broken benchmarker must not take training down with it."""
    global _benchmarker
    if _benchmarker is None:
        with _lock:
            if _benchmarker is None:
                from platform_components.benchmarking.benchmarker import Benchmarker
                conn = _resolve_endpoint()
                enabled = _is_true(os.getenv("BENCHMARK_ENABLED")) and bool(conn)
                if _is_true(os.getenv("BENCHMARK_ENABLED")) and not conn:
                    logger.warning(
                        "Benchmarking enabled but no target resolved; disabling. Set "
                        "BENCHMARK_REST_CONN to an operator's REST address, or "
                        "BENCHMARK_FALLBACK=True to use this process's EXTERNAL_IP."
                    )
                try:
                    _benchmarker = Benchmarker(f"http://{conn}", enabled=enabled)
                except Exception as e:
                    logger.warning(f"Benchmarker init failed; metrics disabled: {e}")
                    _benchmarker = Benchmarker("", enabled=False)
    return _benchmarker
