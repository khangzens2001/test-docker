"""handler.py contract tests with runpod/torch/backend stubbed (no GPU)."""
import importlib
import sys
import types

import pytest

pytest.importorskip("fastapi")


def _load_handler(monkeypatch, tmp_path, download):
    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(progress_update=lambda *a, **k: None, start=lambda *a, **k: None)
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    backend = types.ModuleType("backend_api_extended_manhattan")
    backend.DEFAULT_CLEAN_VOXEL, backend.DEFAULT_CLEAN_STAT_NEIGHBORS, backend.DEFAULT_CLEAN_STAT_STD = 0.01, 16, 2.8
    backend.safe_object_part = lambda v: "".join(c if c.isalnum() or c in "-_" else "_" for c in v)
    backend.download_and_extract_zip = download
    backend.upload_to_r2 = lambda p, k: {"key": k, "presigned_url": "u:" + k}
    for name, mod in (("runpod", runpod), ("torch", torch), ("backend_api_extended_manhattan", backend)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    sys.modules.pop("handler", None)
    return importlib.import_module("handler")


def test_error_has_type_traceback_and_uses_given_job_id(monkeypatch, tmp_path):
    def boom(url, job_dir, img_dir):
        raise ValueError("LiDAR telemetry must contain at least 2 rows.")

    h = _load_handler(monkeypatch, tmp_path, boom)
    out = h.handler({"input": {"zip_url": "x", "job_id": "t-ABC", "keep_session": True}})
    assert out["status"] == "error" and out["job_id"] == "t-ABC"
    assert out["error_type"] == "invalid_input" and out["retryable"] is False
    assert "traceback (tail)" in out["error"] and "ValueError" in out["error"]
    assert (tmp_path / "t-ABC").exists()  # keep_session=True -> not cleaned up


def test_reconstruct_only_success_path(monkeypatch, tmp_path):
    def fake_download(url, job_dir, img_dir):
        import os
        os.makedirs(img_dir, exist_ok=True)
        open(os.path.join(img_dir, "manifest.json"), "w").write("{}")
        open(os.path.join(img_dir, "floorplan.json"), "w").write("{}")
        open(os.path.join(img_dir, "reconstructed.ply"), "w").write("x")
        return []

    h = _load_handler(monkeypatch, tmp_path, fake_download)
    seen = {}
    runner = types.ModuleType("app.pipeline.runner")

    def run_pipeline(session_dir, **kw):
        seen.update(kw)
        kw["stage_callback"](3, "keyframe_selection", 20)
        return types.SimpleNamespace(success=True, floorplan={}, warnings=[{"stage": "x", "message": "m"}],
                                     metrics=types.SimpleNamespace(timings={"vio_sec": 1}))

    runner.run_pipeline = run_pipeline
    contract = types.ModuleType("app.services.cad.contract")
    contract.layout_to_metrics_mm = lambda fp: {"area_m2": 1}
    monkeypatch.setitem(sys.modules, "app.pipeline.runner", runner)
    monkeypatch.setitem(sys.modules, "app.services.cad.contract", contract)
    out = h.handler({"input": {"zip_url": "x", "job_id": "t-1", "mode": "reconstruct_only", "enable_parallel": False}})
    assert out["status"] == "success" and out["pipeline_mode"] == "reconstruct_only"
    assert seen["skip_materials"] is True and seen["enable_parallel"] is False
    assert "floorplan_json" in out["floorplan_files"] and out["warnings"][0]["stage"] == "x"
