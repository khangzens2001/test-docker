"""Torch-free R2 helpers for CPU-only workers (mirrors upload_to_r2 in backend_api_extended_manhattan)."""
from __future__ import annotations

import os
import shutil
import threading
from urllib.parse import urlparse

R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")
R2_BUCKET = os.getenv("R2_BUCKET", "3d-ply")
R2_PRESIGN_EXPIRES = int(os.getenv("R2_PRESIGN_EXPIRES", str(7 * 24 * 3600)))

_client = None
_lock = threading.Lock()


def r2_configured() -> bool:
    return bool(R2_ENDPOINT_URL and os.getenv("R2_ACCESS_KEY_ID") and os.getenv("R2_SECRET_ACCESS_KEY"))


def _get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                import boto3
                from botocore.config import Config

                _client = boto3.client(
                    "s3",
                    endpoint_url=R2_ENDPOINT_URL,
                    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                    region_name="auto",
                    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}, max_pool_connections=16),
                )
    return _client


def upload_to_r2(local_path: str, key: str) -> dict:
    size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    if not r2_configured():
        return {
            "bucket": "local", "key": key, "size": size, "etag": "local",
            "presigned_url": f"file://{os.path.abspath(local_path)}", "expires_in": 0,
            "local_path": os.path.abspath(local_path),
        }
    client = _get_client()
    with open(local_path, "rb") as f:
        put = client.put_object(Bucket=R2_BUCKET, Key=key, Body=f, ContentType="application/octet-stream")
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=R2_PRESIGN_EXPIRES
    )
    return {"bucket": R2_BUCKET, "key": key, "size": int(size), "etag": str(put.get("ETag", "")).strip('"'),
            "presigned_url": url, "expires_in": R2_PRESIGN_EXPIRES}


def download_to_file(url: str, dest: str, timeout: float = 120.0) -> str:
    """Download a presigned/http(s)/file:// URL to dest."""
    parsed = urlparse(url)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if parsed.scheme == "file":
        shutil.copy2(parsed.path, dest)
        return dest
    import requests

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return dest
