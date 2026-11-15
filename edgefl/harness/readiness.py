"""Readiness gates: poll real signals instead of fixed sleeps.

The current scripts sleep 30 and hope, which flakes at scale. Each gate polls a real
signal with a timeout and returns True/False, so the lifecycle can fail fast and report
which layer wasn't ready.

Signals (confirmed against the code/Makefile):
  uvicorn server: answers an HTTP GET on its port once bound and app-started.
  EdgeLake operator/master: REST `get status` returns 200 once the node is up.

A deeper blockchain-sync gate is deferred until we have real `get status` output to match
against (the first live run).
"""

import time

import requests

from platform_components.lib.logger.error_handling import get_logger

logger = get_logger(__name__)


ANYLOG_UA = {"User-Agent": "AnyLog/1.23"}


def _poll(predicate, timeout_s: float, interval_s: float, label: str, abort_if=None) -> bool:
    """Call predicate() until it returns True or the timeout hits. Swallows request
    exceptions (connection refused mid-startup is expected, not fatal). abort_if() is
    checked after each failed probe; a True return gives up at once (the thing being
    waited for can no longer happen, so waiting out the timeout is pointless)."""
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            if predicate():
                logger.info(f"readiness OK: {label} (after {attempt} probe(s))")
                return True
        except requests.RequestException:
            pass
        if abort_if is not None and abort_if():
            logger.warning(f"readiness ABORT: {label} (it can no longer succeed)")
            return False
        time.sleep(interval_s)
    logger.warning(f"readiness TIMEOUT: {label} after {timeout_s}s")
    return False


def wait_http_up(url: str, timeout_s: float = 30.0, interval_s: float = 1.0) -> bool:
    """Wait until an HTTP server answers at url at all (any status = bound and serving).
    Used for uvicorn node_server / aggregator readiness."""
    def up():
        requests.get(url, timeout=2)
        return True
    return _poll(up, timeout_s, interval_s, f"http up {url}")


def wait_edgelake_status(
    rest_endpoint: str, timeout_s: float = 120.0, interval_s: float = 2.0
) -> bool:
    """Wait until an EdgeLake node answers `get status` with 200. rest_endpoint is
    host:port, e.g. 127.0.0.1:32149."""
    url = f"http://{rest_endpoint}"
    def ready():
        r = requests.get(url, headers={**ANYLOG_UA, "command": "get status"}, timeout=3)
        return r.status_code == 200
    return _poll(ready, timeout_s, interval_s, f"edgelake status {rest_endpoint}")


def wait_edgelake_process(
    rest_endpoint: str, process_name: str, timeout_s: float = 120.0, interval_s: float = 5.0,
    abort_if=None,
) -> bool:
    """Wait until `get processes` reports `process_name` as Running. This is the deep
    readiness signal: a half-booted node answers `get status` while its TCP/Operator
    process never started (the image's deployment script is one-shot and can lose a
    boot-time race)."""
    url = f"http://{rest_endpoint}"
    def running():
        r = requests.get(url, headers={**ANYLOG_UA, "command": "get processes"}, timeout=5)
        if r.status_code != 200:
            return False
        for line in r.text.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if parts and parts[0].lower() == process_name.lower():
                return len(parts) > 1 and parts[1].lower().startswith("running")
        return False
    return _poll(running, timeout_s, interval_s, f"process '{process_name}' at {rest_endpoint}",
                 abort_if=abort_if)


def wait_all_http_up(urls: list[str], timeout_s: float = 60.0, interval_s: float = 1.0) -> list[str]:
    """Wait for a batch of HTTP servers. Returns the URLs that did not come up within the
    timeout (empty means all ready)."""
    failed = []
    for url in urls:
        if not wait_http_up(url, timeout_s=timeout_s, interval_s=interval_s):
            failed.append(url)
    return failed
