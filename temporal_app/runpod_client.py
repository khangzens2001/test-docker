"""Thin synchronous RunPod Serverless client (run in a thread from async activities)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from temporal_app.config import RunPodSettings

logger = logging.getLogger(__name__)

TERMINAL_OK = {"COMPLETED"}
TERMINAL_FAIL = {"FAILED", "CANCELLED", "TIMED_OUT"}


class RunPodError(Exception):
    def __init__(self, message: str, *, retryable: bool = True, status: Optional[str] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class RunPodStatus:
    status: str
    output: Any = None
    error: Optional[str] = None
    stage_events: list[dict] = None  # type: ignore[assignment]
    raw: dict = None  # type: ignore[assignment]


def _parse_stage_events(payload: dict) -> list[dict]:
    """Best-effort extraction of {"stage","pct"} progress events from a /status payload.

    RunPod progress_update() text may surface in `output`, `stream` or `progress` depending on
    runtime version, so look at all of them defensively and ignore anything that is not our JSON.
    """
    candidates: list[Any] = []
    for key in ("output", "stream", "progress"):
        val = payload.get(key)
        if isinstance(val, list):
            candidates.extend(val)
        elif val is not None:
            candidates.append(val)
    events: list[dict] = []
    for c in candidates:
        if isinstance(c, dict) and "output" in c:  # stream item wrapper
            c = c["output"]
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except ValueError:
                continue
        if isinstance(c, dict) and "stage" in c:
            events.append(c)
    return events


class RunPodClient:
    def __init__(self, settings: RunPodSettings, session: Optional[requests.Session] = None):
        if not settings.api_key or not settings.endpoint_id:
            raise RunPodError("RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID not configured", retryable=False)
        self.s = settings
        self.http = session or requests.Session()
        self.http.headers.update({"Authorization": f"Bearer {settings.api_key}"})

    def _url(self, path: str) -> str:
        return f"{self.s.base_url}/{self.s.endpoint_id}/{path}"

    def submit(self, payload: dict) -> str:
        body = {
            "input": payload,
            "policy": {
                "executionTimeout": self.s.execution_timeout_s * 1000,
                "ttl": self.s.queue_ttl_s * 1000,
            },
        }
        r = self.http.post(self._url("run"), json=body, timeout=self.s.http_timeout_s)
        if r.status_code in (401, 403):
            raise RunPodError(f"RunPod auth error {r.status_code}", retryable=False)
        if r.status_code >= 400:
            raise RunPodError(f"RunPod submit failed {r.status_code}: {r.text[:200]}",
                              retryable=r.status_code >= 500 or r.status_code == 429)
        return r.json()["id"]

    def status(self, job_id: str) -> RunPodStatus:
        r = self.http.get(self._url(f"status/{job_id}"), timeout=self.s.http_timeout_s)
        if r.status_code >= 400:
            raise RunPodError(f"RunPod status failed {r.status_code}", retryable=True)
        p = r.json()
        return RunPodStatus(
            status=p.get("status", "UNKNOWN"),
            output=p.get("output"),
            error=p.get("error"),
            stage_events=_parse_stage_events(p),
            raw=p,
        )

    def cancel(self, job_id: str) -> None:
        try:
            self.http.post(self._url(f"cancel/{job_id}"), timeout=self.s.http_timeout_s)
        except requests.RequestException as exc:  # best effort
            logger.warning("RunPod cancel failed for %s: %s", job_id, exc)

    def sleep(self) -> None:
        time.sleep(self.s.poll_interval_s)
