# Handoff: what belongs to whom

Scope of the Temporal work (done, tested): orchestration, retries, cancel, stage tracing, CPU materials worker, Docker stack.
Everything below is **outside** that scope and needs an owner.

## A. GPU / AI-pipeline owner (image running on RunPod)
| # | Item | Evidence | Needed |
|---|---|---|---|
| A1 | **Job FAILS on the deployed endpoint**: `expected scalar type Float but found BFloat16` | Same error for the plain `curl .../runsync` (no Temporal) with ZIP `1zAMg2Yp-...`, worker `nngzsgxsbuxrq4`; 3 of 3 attempts. RunPod ids: `57c2b3f7-...-e2`, `c6321aff-...-e2`, `sync-1631e214-...-e2` | Find root cause. Only `backend_api_extended_manhattan.py:42` uses bfloat16 (casts whole VGGT to bf16 on Ampere+), so the ZIP most likely runs the photo branch. Try `VGGT_DTYPE=float16`; if it still fails read the new traceback in the job `error` |
| A2 | Deploy the updated image | `handler.py`, `app/pipeline/runner.py`, `backend_api_extended_manhattan.py`, `app/services/artifact_upload.py` changed (not built/pushed) | Push `main` (CI builds), point the RunPod endpoint to the new image, set env `VGGT_DTYPE=float16` for the test |
| A3 | Confirm which branch the test ZIP takes | Branch A needs `manifest.json`/`lidar.csv`/...; otherwise photo branch | Check ZIP contents; if it is meant to be a LiDAR session the detection or the ZIP is wrong |
| A4 | Verify new handler modes on GPU | `mode=reconstruct_only` (skips materials; Temporal's CPU worker does them), `job_id` passthrough, `keep_session`, `enable_parallel`, `progress_update` stage events | One real job, then check Temporal UI: stages appear, `floorplan_json` uploaded |
| A5 | Optional: confirm RunPod `/status` shape for `progress_update` | parser handles `output`/`stream`/`progress` defensively | Share one in-progress `/status` payload |

Pass criterion A: one real scan finishes `COMPLETED` through
`python -m temporal_app.starter start --task-id X --zip-url <zip> --wait`, with a non-empty `room_metrics` and files.

## B. Server / infra owner
| # | Item | Needed |
|---|---|---|
| B1 | Run the stack | `docs/temporal/DEPLOY.md` (Docker + Compose, outbound internet, ~4 GB RAM) |
| B2 | Secrets | Create `.env.temporal` on the server: `TEMPORAL_DB_PASSWORD`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`. **The RunPod key used during testing was pasted in chat: revoke and re-issue it** |
| B3 | UI access | UI/Temporal bind to 127.0.0.1 and have no auth: use SSH tunnel or an authenticated reverse proxy |
| B4 | Backup (if history matters) | Postgres volume `temporal-db`; otherwise history is lost with the volume |

## C. Not requested (explicitly out of scope)
NestJS integration, K8s/mTLS/HA, Prometheus/Grafana, parity comparison with the legacy flow.
(Specs kept for later: `NESTJS_INTEGRATION.md`.)

## D. Known limits of the Temporal layer (by design)
- At most 1 resubmission per scan, only for OOM / timeout / cancel / worker-lost; deterministic pipeline errors are not retried (GPU cost guard).
- Cancel stops the RunPod job; failed ids can be reused, succeeded ids cannot (`scan-{task_id}`).
- Photo (VGGT) branch has no stage events (handler emits none); workflow still works.
