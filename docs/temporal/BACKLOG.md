# Remaining work (honest status)

Verified in this workspace: 8 unit/workflow tests + 3 local E2E tests (real Temporal dev server, real CAD/material render on a
synthetic room, fake RunPod) - see TEST_PLAN.md; docker-compose syntax; CPU import smoke (no torch/onnx/open3d needed).

| # | Item | Why open | Steps | Done when |
|---|---|---|---|---|
| 1 | RunPod `progress_update` visibility via `/status` | parser is defensive; real payload shape unverified | run one real job, capture `/status` JSON, fix `_parse_stage_events`, add fixture test | stage signals appear in Temporal history for a real job |
| 2 | `render_materials` parity on real scans | PASSED on synthetic room (21 artifacts, area 15.12). Not yet on real scan data | run against 3 captured floorplan.json; compare with legacy | outputs identical to `legacy_full` |
| 3 | Docker image tags / builds | registry unreachable here, images not built | `docker compose build && up`; pin digests | stack healthy, `temporal-init` registered 5 search attributes |
| 4 | Real RunPod run of `mode=reconstruct_only` | handler path untested on GPU | deploy image; run starter on sample ZIP; compare `room_metrics` | parity with legacy within tolerance |
| 5 | NestJS integration | BE repo absent | apply `NESTJS_INTEGRATION.md`, flag rollout | BE creates jobs via Temporal in staging |
| 6 | Prod hardening | needs infra decisions | Helm Temporal, Postgres HA+PITR, mTLS, UI auth, retention, archival; KEDA metric names verified against /metrics | game-day: kill worker/DB failover passes |
| 7 | Remaining fail-open stages | calibration inner, ingestion warnings not surfaced | add `_warn` in other swallowed `except` blocks of runner (lines ~246-296, 876-920) | warnings list covers all swallowed errors |
| 8 | Replay tests in CI | no CI history yet | export histories, `Replayer` test in GitHub workflow | CI fails on non-determinism |
| 9 | Per-stage GPU retry | single GPU job; resume needs checkpoints | optional: split job + R2 checkpoints | only if GPU cost of full retry matters |
| 10 | R2 temp data lifecycle / PII | policy decision | lifecycle rule on `ply_clean/*` or prefix TTL | documented retention |
| 11 | Backend `upload_to_r2` duplicate | `app/services/r2_storage.py` mirrors it | make backend import the shared helper (needs torch-image regression test) | single implementation |
| 12 | Photo branch stage events | `run_inference_pipeline` emits none | add progress_update calls | photo jobs show stages |

## Findings from the Docker Desktop run (compose, fake RunPod)
| # | Finding | Status |
|---|---|---|
| A | Worker started before namespace `default` existed and never recovered | FIXED: init waits for namespace; worker `depends_on: service_completed_successfully`; verified clean boot has 0 errors |
| B | Kill -9 worker mid-job -> restart -> exactly 1 RunPod submission, result OK | PASS |
| C | 5 concurrent scans produced duplicate RunPod submissions | FIXED: bridge split into `runpod_submit` (id stored in history) + `runpod_poll(job_id)` (idempotent); at most 1 resubmission, sequential mode, only for job-level failures; CAD render moved to a separate process pool. Verified: 8 concurrent scans -> 8 COMPLETED, exactly 8 RunPod submissions |
| D | Stage log has no GPU stages when resumed after the job already finished (COMPLETED has no stream) | Expected; low priority |
