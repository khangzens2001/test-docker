from __future__ import annotations

from typing import Optional

from temporalio.client import Client
from temporalio.runtime import Runtime

from temporal_app.config import TemporalSettings


async def connect(cfg: TemporalSettings, runtime: Optional[Runtime] = None) -> Client:
    kwargs = dict(namespace=cfg.namespace, runtime=runtime)
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.tls or cfg.api_key:
        kwargs["tls"] = True
    return await Client.connect(cfg.address, **kwargs)
