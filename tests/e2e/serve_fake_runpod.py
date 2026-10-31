"""Run the fake RunPod API for compose-based tests:  python -m tests.e2e.serve_fake_runpod <floorplan.json> [port]"""
import sys
import time

from tests.e2e.fake_runpod import FakeRunPod

fp, port = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "8765"
import os
os.environ["FAKE_RUNPOD_PORT"] = port
f = FakeRunPod(fp, stage_seconds=float(os.getenv("STAGE_SECONDS", "2")), host="0.0.0.0",
               public_base=f"http://host.docker.internal:{port}")
print("fake runpod on", f.url, flush=True)
while True:
    time.sleep(3600)
