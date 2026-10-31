"""In-process fake of the RunPod serverless HTTP API (/run, /status, /cancel) for local E2E runs."""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STAGES = [("ingestion", 2, 1), ("vio", 10, 2), ("keyframe_selection", 20, 3), ("depth_completion", 35, 5),
          ("tsdf_integration", 60, 6), ("vggt_prior", 72, 7), ("plane_segmentation", 82, 8),
          ("export_artifacts", 90, 9)]


class FakeRunPod:
    def __init__(self, floorplan_json_path: str, stage_seconds: float = 0.5, host: str = "127.0.0.1",
                 public_base: str | None = None):
        self.floorplan = floorplan_json_path
        self.stage_seconds = stage_seconds
        self.public_base = public_base
        self.jobs: dict[str, dict] = {}
        self.cancelled: list[str] = []
        self.submissions: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path.endswith("/run"):
                    jid = str(uuid.uuid4())
                    outer.jobs[jid] = {"t0": time.time(), "input": body["input"]}
                    outer.submissions.append(body["input"])
                    print(f"SUBMIT job={jid} input_job_id={body['input'].get('job_id')}", flush=True)
                    return self._send(200, {"id": jid, "status": "IN_QUEUE"})
                if "/cancel/" in self.path:
                    outer.cancelled.append(self.path.rsplit("/", 1)[1])
                    return self._send(200, {"status": "CANCELLED"})
                self._send(404, {})

            def do_GET(self):
                if self.path.endswith("/floorplan.json"):  # lets a containerised worker download the checkpoint
                    data = open(outer.floorplan, "rb").read()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                jid = self.path.rsplit("/", 1)[1]
                job = outer.jobs.get(jid)
                if job is None:
                    return self._send(404, {})
                if jid in outer.cancelled:
                    return self._send(200, {"id": jid, "status": "CANCELLED"})
                el = time.time() - job["t0"]
                done = [s for i, s in enumerate(STAGES) if el >= i * outer.stage_seconds]
                events = [json.dumps({"stage": n, "pct": p, "stage_index": k}) for n, p, k in done]
                total = len(STAGES) * outer.stage_seconds
                zip_url = job["input"].get("zip_url", "")
                if "hang" in zip_url:
                    return self._send(200, {"id": jid, "status": "IN_PROGRESS", "stream": [{"output": e} for e in events]})
                if el < total:
                    return self._send(200, {"id": jid, "status": "IN_PROGRESS", "stream": [{"output": e} for e in events]})
                if "bad" in zip_url:
                    out = {"status": "error", "error": "LiDAR telemetry must contain at least 2 rows.",
                           "error_type": "invalid_input", "retryable": False}
                else:
                    out = {
                        "status": "success", "job_id": job["input"].get("job_id"), "mode": "multisensor_lidar",
                        "pipeline_mode": job["input"].get("mode"), "session_prefix": f"b/{job['input'].get('job_id')}",
                        "floorplan_files": {"floorplan_json": {"presigned_url": (
                            f"{outer.public_base}/floorplan.json" if outer.public_base else f"file://{outer.floorplan}")}},
                        "room_metrics": None, "timings": {"vio_sec": 1.0},
                        "warnings": [{"stage": "portals", "message": "fake warning"}],
                        "clean_ply": {"presigned_url": None, "r2_key": None},
                    }
                self._send(200, {"id": jid, "status": "COMPLETED", "output": out})

        self.server = ThreadingHTTPServer((host, int(__import__("os").getenv("FAKE_RUNPOD_PORT", "0"))), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
