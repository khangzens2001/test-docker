import os

import pytest

pytest.importorskip("fastapi")

from app.core.exceptions import SensorDataIngestionError, classify_pipeline_error  # noqa: E402
from app.services.artifact_upload import upload_session_artifacts  # noqa: E402


def test_classify():
    assert classify_pipeline_error(SensorDataIngestionError("x")) == ("SensorDataIngestionError", False)
    assert classify_pipeline_error(ValueError("LiDAR telemetry must contain at least 2 rows.")) == ("invalid_input", False)
    assert classify_pipeline_error(RuntimeError("CUDA out of memory"))[1] is True
    assert classify_pipeline_error(ConnectionError("x")) == ("transient_io", True)
    assert classify_pipeline_error(RuntimeError("boom")) == ("internal_error", True)


def test_upload_session_artifacts(tmp_path):
    for n in ("reconstructed.ply", "floorplan.json", "FloorPlan_A3.png", "Walls.png"):
        (tmp_path / n).write_bytes(b"x")
    calls = []

    def fake(path, key):
        calls.append(key)
        return {"key": key, "presigned_url": "u:" + key}

    clean, files = upload_session_artifacts(str(tmp_path), "b/j", fake)
    assert clean["key"].endswith("project_point_cloud_clean.ply")
    assert {"floorplan_json", "floorplan_png", "wall_elevation_png"} <= set(files)
    assert any(k.endswith("/walls.png") for k in calls)
