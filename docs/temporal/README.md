# Temporal orchestration for the 3D LiDAR pipeline

## Topology
```
NestJS BE ──start/signal/query──► Temporal (self-host) ◄── worker (CPU host: queues lidar-cpu + lidar-gpu-bridge)
                                                              │ runpod_poll = submit/poll/cancel RunPod job
                                                              ▼
                                            RunPod Serverless (handler.py, mode=reconstruct_only)
                                            stages 2-9, progress_update, uploads artifacts to R2
```
Workflow `RoomScanWorkflow` (id `scan-{task_id}`): `resolve_input → runpod_poll → render_materials(CPU) →
build_result → notify_backend → cleanup(finally)`.
Photo ZIPs (VGGT branch) are detected *inside* the GPU job (`mode=vggt_photo`), so the workflow skips
`render_materials` for them (single workflow instead of a second one).

Stage-level trace: GPU stage events -> `report_stage` signals (visible in history) + `CurrentStage` search
attribute + heartbeat details + `get_status` query. Metric `lidar_stage_duration_seconds{stage}`.

## Run locally
```bash
cp .env.temporal.example .env.temporal   # fill RUNPOD_*, R2_*
docker compose -f docker-compose.temporal.yml --env-file .env.temporal up -d --build
python -m temporal_app.starter start --task-id T1 --zip-url https://.../scan.zip --wait
python -m temporal_app.starter status --task-id T1
python -m temporal_app.starter cancel --task-id T1
```

## Tests
`pytest tests/temporal` (needs `temporalio pytest pytest-asyncio fastapi`; downloads the Temporal test server on first run).

## Runbook
| Symptom | Check | Action |
|---|---|---|
| Workflows stuck in `runpod_poll` | UI pending activity: heartbeat details show `runpod_job_id`, `stage` | RunPod console for that job; `starter cancel` cancels the RunPod job too |
| `RunPodTimedOut` repeatedly | `RUNPOD_EXECUTION_TIMEOUT_S` (1200s), input size | raise timeout or reject oversize ZIP in `resolve_input` |
| High schedule-to-start on `lidar-gpu-bridge` | `GPU_MAX_CONCURRENT_ACTIVITIES` vs RunPod max workers | raise both together |
| Workflow failed `invalid_input` etc. | non-retryable by design | fix data, start a new task id (failed ids may be reused: `ALLOW_DUPLICATE_FAILED_ONLY`) |
| Worker deploy | change workflow code | use `workflow.patched()`; run replay tests first; drain old workers last |

## Rollout
1. Deploy handler.py (backward compatible: default `mode=legacy_full`). 2. Deploy worker. 3. BE flag `USE_TEMPORAL` for % of users.
4. Keep legacy path until two weeks of parity on `room_metrics`.
