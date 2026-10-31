from temporal_app.runpod_client import _parse_stage_events


def test_parse_stage_events_variants():
    assert _parse_stage_events({"output": '{"stage":"vio","pct":10}'}) == [{"stage": "vio", "pct": 10}]
    assert _parse_stage_events({"stream": [{"output": '{"stage":"tsdf_integration","pct":60}'}]})[0]["pct"] == 60
    assert _parse_stage_events({"output": {"status": "success"}}) == []
    assert _parse_stage_events({"output": "not json"}) == []


def test_classify_terminal():
    from temporal_app.activities.gpu import _classify_terminal as c
    assert c("FAILED", "CUDA out of memory") == "oom"
    assert c("FAILED", "expected scalar type Float but found BFloat16") == "PipelineFailed"
    assert c("TIMED_OUT", None) == "RunPodTimedOut"
    assert c("FAILED", "worker unavailable") == "RunPodWorkerLost"
