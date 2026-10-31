# NestJS (`UT-3D-Lidar-BE`) integration spec
*The BE repo was not available in this workspace; this is a spec + reference code, not applied code.*

```ts
// temporal.service.ts  (npm i @temporalio/client)
@Injectable()
export class TemporalService implements OnModuleInit {
  client: Client;
  async onModuleInit() {
    const conn = await Connection.connect({ address: process.env.TEMPORAL_ADDRESS, tls: process.env.TEMPORAL_TLS === '1' });
    this.client = new Client({ connection: conn, namespace: process.env.TEMPORAL_NAMESPACE ?? 'default' });
  }
  startScan(taskId: string, zipUrl: string, batchId: string, userId: string) {
    return this.client.workflow.start('RoomScanWorkflow', {
      taskQueue: 'lidar-cpu', workflowId: `scan-${taskId}`,
      workflowIdConflictPolicy: WorkflowIdConflictPolicy.USE_EXISTING,
      workflowIdReusePolicy: WorkflowIdReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
      args: [{ task_id: taskId, zip_url: zipUrl, batch_id: batchId, user_id: userId,
               gpu_task_queue: 'lidar-gpu-bridge' }],
      searchAttributes: { TaskId: [taskId], BatchId: [batchId], UserId: [userId] },
    });
  }
  status(taskId: string) { return this.client.workflow.getHandle(`scan-${taskId}`).query('get_status'); }
  cancel(taskId: string) { return this.client.workflow.getHandle(`scan-${taskId}`).signal('cancel_scan'); }
  override(taskId: string, p: Record<string, unknown>) { return this.client.workflow.getHandle(`scan-${taskId}`).signal('override_params', p); }
}
```
Mapping from today's code:
| Today | With Temporal |
|---|---|
| `POST /api/v2/ply/project-floorplan/jobs` creates `Reconstruct3DJob` + calls RunPod | create row, `startScan` (idempotent) |
| `pollPlyJob()` + `jobCache` | `status()` query for progress; final result via webhook `notify_backend` or `handle.result()` |
| SSE `/jobs/:id/events` | RxJS interval over `status()` (stage, pct, warnings) or push on webhook |
| error mapping | failed workflow -> `error` from `get_status`; `error_type` is in ApplicationError type |

Webhook: `POST BACKEND_CALLBACK_URL` body `{workflow_id, ...result}`, header `X-Signature = HMAC-SHA256(body, BACKEND_CALLBACK_SECRET)`; verify with timing-safe compare; must be idempotent (activity retries).
Feature flag `USE_TEMPORAL` per user/percentage. Data contract of `result` equals `new_update.md` section 5 plus `warnings[]`.
