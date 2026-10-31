# Deploy on a server (Docker only)
Needs: Docker + Compose, outbound internet (RunPod API, Google Drive/R2). No GPU on this host.
```bash
cp .env.temporal.example .env.temporal      # fill TEMPORAL_DB_PASSWORD, RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
docker compose -f docker-compose.temporal.yml --env-file .env.temporal up -d --build
docker compose -f docker-compose.temporal.yml --env-file .env.temporal ps
```
Containers: temporal, temporal-db (volume `temporal-db`), temporal-ui, temporal-init (one-shot), worker. ~3-4 GB RAM total.
UI/Temporal listen on 127.0.0.1 only: `ssh -L 8080:localhost:8080 user@server` then open http://localhost:8080.
Start a scan: `docker compose ... exec worker python -m temporal_app.starter start --task-id T1 --zip-url <url> --wait`
(set `TEMPORAL_ADDRESS=temporal:7233` is already in the worker env).
Update: `git pull && docker compose ... up -d --build worker`. Logs: `docker compose ... logs -f worker`.
