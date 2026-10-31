# Test plan & results

| Level | What | How | Status |
|---|---|---|---|
| L0 unit | RunPod stage-event parser, error classification, shared upload | `pytest tests/temporal` | PASS (8 tests incl. 5 time-skipping workflow tests) |
| L1 local E2E | real Temporal dev server + real worker + **real CAD/material rendering** on a synthetic 4.2x3.6x2.45 m room + fake RunPod HTTP API | `pytest tests/e2e -s` | PASS (3 tests, ~2 min) |
| L2 compose | Temporal + Postgres + UI + worker via `docker-compose.temporal.yml` | needs Docker daemon running (it was not, in the authoring session) | NOT RUN |
| L3 real RunPod | `mode=reconstruct_only` on GPU with a real ZIP | deploy image, set `RUNPOD_*`, `starter start --wait` | NOT RUN |
| L4 parity | `room_metrics` / artifacts vs `legacy_full` for 3 scans | compare `metrics.json` | NOT RUN |
| L5 chaos/load | kill worker mid-job, RunPod FAILED/TIMED_OUT, N concurrent jobs | see below | NOT RUN |

## L1 assertions (all verified)
- stage timeline relayed once per stage and in order; `materials`, `build_result`, `notify_backend`, `done` present
- 21 artifacts incl. floorplan, wall elevation, CF/Neda/Plywood/Tiling, quality report; `room_metrics.area_m2 == 15.12`
- `invalid_input` from GPU -> workflow fails, exactly 1 RunPod submission (no retry)
- cancel signal -> RunPod `/cancel` called, workflow returns `cancelled`
- search attribute query `TaskId = '...'` returns the workflow; local workdir removed by `cleanup`
- re-starting a succeeded `scan-{task_id}` raises `WorkflowAlreadyStarted` (no duplicate GPU spend)

## Next runs (commands)
```bash
# L2 (start Docker Desktop first)
cp .env.temporal.example .env.temporal && docker compose -f docker-compose.temporal.yml --env-file .env.temporal up -d --build
docker compose -f docker-compose.temporal.yml logs temporal-init   # 5 search attributes created
# L3
python -m temporal_app.starter start --task-id REAL1 --zip-url <presigned zip> --wait
# L5 chaos
docker compose -f docker-compose.temporal.yml kill worker   # mid runpod_poll; restart: same runpod_job_id must be resumed (check logs/submissions)
```
Pass criteria L3: stage signals visible in UI history; L5: after worker restart only one RunPod job exists for the task.
