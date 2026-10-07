from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from typing import Literal, Optional
import uuid

try:
    from cachetools import TTLCache
except ImportError:
    class TTLCache(dict):  # type: ignore[no-redef]
        def __init__(self, maxsize: int = 10_000, ttl: int = 3600):
            super().__init__()

from app.core.config import settings

logger = logging.getLogger("app.storage")

# ---- Rate limit ------------------------------------------------------------
# In-memory bounded cache; Redis path used when settings.REDIS_URL is set.
# We store `(count, window_start_epoch)` tuples so the hourly window resets
# after 3600 s regardless of how often the key is written (a bare
# `TTLCache[key] = count` refreshes the TTL on every write, indefinitely
# extending the window for anyone under sustained pressure).
#
# CRITICAL: keys are the **tuple** `(client_ip, user_id)` — never a delimited
# string. `f"rate:{client_ip}:{user_id}"` would collide for values such as
# `("a:b", "c")` and `("a", "b:c")`; a real tuple gives Python's dict/hash
# machinery unambiguous identity and requires no escaping.
rate_limit_cache: TTLCache = TTLCache(maxsize=10_000, ttl=3600)
_rate_limit_lock = asyncio.Lock()
_UPLOADS_PER_HOUR = 5
_RATE_WINDOW_SECONDS = 3600

_redis_client = None

# Lua script for atomic INCR + conditional EXPIRE. Redis executes each EVAL
# body as a single atomic operation, so the TTL is guaranteed to be set on
# the same freshly-created key that INCR returned 1 for — even if the client
# process crashes between the two logical commands. Returns the new count.
_RATE_LIMIT_LUA = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return n
"""

# Same idea for disk reservation: read current value, verify effective free
# >= min, INCRBY on success. All under one Redis atomic block so two workers
# cannot both admit. Returns 1 on success, 0 on refusal.
# KEYS[1] = disk:reserved_bytes, KEYS[2] = disk:active_tokens
# ARGV[1] = n, ARGV[2] = free, ARGV[3] = min_bytes, ARGV[4] = token_id
_DISK_ADMIT_LUA = """
local reserved = tonumber(redis.call('GET', KEYS[1]) or '0')
local n = tonumber(ARGV[1])
local free = tonumber(ARGV[2])
local min_bytes = tonumber(ARGV[3])
local token_id = ARGV[4]
local effective_free = free - reserved - n
if effective_free < min_bytes then
    return 0
end
redis.call('INCRBY', KEYS[1], n)
redis.call('SADD', KEYS[2], token_id)
return 1
"""

# Atomically release the token. Verify the token is in the set using SREM.
# If present (returns 1), read current reserved bytes, subtract n, clamp to 0,
# and write back. If not present (returns 0), do nothing.
# KEYS[1] = disk:reserved_bytes, KEYS[2] = disk:active_tokens
# ARGV[1] = n, ARGV[2] = token_id
_DISK_RELEASE_LUA = """
local token_removed = redis.call('SREM', KEYS[2], ARGV[2])
if token_removed == 1 then
    local reserved = tonumber(redis.call('GET', KEYS[1]) or '0')
    local n = tonumber(ARGV[1])
    local new_reserved = reserved - n
    if new_reserved < 0 then
        new_reserved = 0
    end
    redis.call('SET', KEYS[1], tostring(new_reserved))
    return 1
else
    return 0
end
"""


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not settings.REDIS_URL:
        return None
    import redis.asyncio as _aioredis  # local import so redis is optional in dev

    _redis_client = _aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


async def close_redis() -> None:
    """Called from `_lifespan` shutdown so we don't leak Redis sockets."""
    global _redis_client
    if _redis_client is None:
        return
    try:
        await _redis_client.aclose()
    except Exception:
        logger.exception("Failed to close Redis client cleanly")
    finally:
        _redis_client = None


def _rate_limit_redis_key(client_ip: str, user_id: str) -> str:
    """Collision-proof Redis key.

    We hash a JSON payload of the tuple components so that no combination
    of `:`-containing IPs or user_ids can collide with another tuple. The
    `rate:` prefix is retained for Redis observability.
    """
    payload = json.dumps(
        {"ip": client_ip, "uid": user_id}, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"rate:{digest}"


async def check_rate_limit(client_ip: str, user_id: str) -> bool:
    """Atomic: increment first, then check. Local uses asyncio.Lock; prod uses Redis.

    Redis path runs a Lua script so INCR + conditional EXPIRE happen in one
    atomic step (see `_RATE_LIMIT_LUA`). This closes the R3 gap where a
    crash between `INCR` and `EXPIRE` could leave a TTL-less key that
    permanently locks out the `(ip, user_id)` tuple.
    """
    r = _get_redis()
    if r is not None:
        try:
            key = _rate_limit_redis_key(client_ip, user_id)
            count = int(
                await r.eval(_RATE_LIMIT_LUA, 1, key, _RATE_WINDOW_SECONDS)
            )
            return count <= _UPLOADS_PER_HOUR
        except Exception:
            # Redis outage → degrade to the local cache path so uploads still
            # rate-limit within a single worker rather than fail-open.
            logger.exception("Redis rate-limit path failed; falling back to memory")

    # Local key is a real tuple — collision-proof by construction.
    key: tuple[str, str] = (client_ip, user_id)
    async with _rate_limit_lock:
        now = time.time()
        entry = rate_limit_cache.get(key)
        if entry is None or now - entry[1] >= _RATE_WINDOW_SECONDS:
            rate_limit_cache[key] = (1, now)
            return True
        new_count = entry[0] + 1
        # Preserve the original `window_start` so TTL/window semantics do not
        # extend on every write.
        rate_limit_cache[key] = (new_count, entry[1])
        return new_count <= _UPLOADS_PER_HOUR


# ---- Disk reservation (atomic admission) -----------------------------------
# In-memory counter guarded by an asyncio.Lock so multiple concurrent uploads
# in the same worker cannot race. Redis path (`disk:reserved_bytes`) is used
# when settings.REDIS_URL is set so multi-worker deployments share the total.
_reserved_bytes: int = 0
_active_reservation_token_ids: set[str] = set()
_reserve_lock = asyncio.Lock()
_REDIS_RESERVATION_KEY = "disk:reserved_bytes"
_REDIS_ACTIVE_TOKENS_KEY = "disk:active_tokens"


@dataclass(frozen=True)
class ReservationToken:
    """Handle returned by `try_reserve_disk_bytes` on successful admission.

    The `backend` field pins the release path — Redis reservations always
    release via Redis; memory reservations always release via memory. This
    closes the R3 backend-consistency gap where a reservation made in one
    backend could be released against a different backend after a transient
    Redis outage.
    """

    bytes: int
    backend: Literal["redis", "memory"]
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))


async def try_reserve_disk_bytes(
    path: str, n: int
) -> Optional[ReservationToken]:
    """Atomic: compare `disk_usage(free) - reserved` to `MIN_DISK_SPACE_GB`
    and increment the reservation ONLY when the effective free space would
    still satisfy the floor. Returns a `ReservationToken` on success and
    `None` on refusal (the caller must return `503 Insufficient storage`).

    Local backend: reads `shutil.disk_usage` and the in-memory counter
    under `_reserve_lock` so two concurrent uploads in the same worker
    cannot both admit.

    Redis backend: runs the `_DISK_ADMIT_LUA` script so the read + compare
    + increment happens in a single Redis atomic block across ALL workers.

    We compute the local `shutil.disk_usage(path)` in either backend since
    Redis stores only the shared **reservation** counter, not the raw free
    space. That is fine — every worker sees the same host filesystem in
    Phase 1.
    """
    global _reserved_bytes
    n = max(int(n), 0)
    min_bytes = int(settings.MIN_DISK_SPACE_GB * (1024**3))
    _, _, free = shutil.disk_usage(path)

    r = _get_redis()
    if r is not None:
        try:
            token_id = str(uuid.uuid4())
            ok = int(
                await r.eval(
                    _DISK_ADMIT_LUA,
                    2,
                    _REDIS_RESERVATION_KEY,
                    _REDIS_ACTIVE_TOKENS_KEY,
                    n,
                    free,
                    min_bytes,
                    token_id,
                )
            )
            if ok == 1:
                return ReservationToken(bytes=n, backend="redis", token_id=token_id)
            return None
        except Exception:
            # Redis outage during admission.
            # Production (REDIS_URL configured, ENV_MODE != local): FAIL CLOSED
            # — return None so the endpoint answers 503. Falling back to
            # per-worker memory would re-open multi-worker over-admission.
            # Local/test: fall back to memory so offline unit tests still work.
            logger.exception("Redis try_reserve path failed")
            if settings.REDIS_URL and settings.ENV_MODE != "local":
                return None

    token_id = str(uuid.uuid4())
    async with _reserve_lock:
        effective_free = free - _reserved_bytes - n
        if effective_free < min_bytes:
            return None
        _reserved_bytes = _reserved_bytes + n
        _active_reservation_token_ids.add(token_id)
        return ReservationToken(bytes=n, backend="memory", token_id=token_id)


async def release_disk_bytes(token: Optional[ReservationToken]) -> None:
    """Release a reservation through THE SAME backend it was made in.

    Passing `None` is a safe no-op so `finally` blocks can call this
    unconditionally regardless of whether admission succeeded.

    Production behaviour on a Redis release failure: we DO NOT silently
    fall back to the memory counter for a Redis reservation — that would
    leave the Redis key permanently inflated and later cause spurious 503s
    when Redis recovers. Instead we log at `ERROR` so operators can
    reconcile the counter, and re-raise in production so the caller sees
    the failure. In local mode we log-and-swallow so tests are not
    disrupted by a flaky Redis mock.
    """
    global _reserved_bytes
    if token is None:
        return
    n = max(int(token.bytes), 0)

    if token.backend == "redis":
        r = _get_redis()
        if r is None:
            # Redis client disappeared between reserve and release — never
            # mutate the memory counter for a Redis token.
            logger.error(
                "Redis release skipped (client unavailable) for %d bytes; "
                "operator must reconcile %s.",
                n,
                _REDIS_RESERVATION_KEY,
            )
            if settings.ENV_MODE != "local":
                # Fail closed in production (same policy as DECRBY failure).
                raise RuntimeError(
                    f"Redis client unavailable during release of {n} bytes "
                    f"on {_REDIS_RESERVATION_KEY}; reservation stranded."
                )
            return
        try:
            await r.eval(
                _DISK_RELEASE_LUA,
                2,
                _REDIS_RESERVATION_KEY,
                _REDIS_ACTIVE_TOKENS_KEY,
                n,
                token.token_id,
            )
        except Exception:
            logger.error(
                "Redis release script failed for %d bytes on %s; the reservation "
                "remains outstanding and must be reconciled.",
                n,
                _REDIS_RESERVATION_KEY,
                exc_info=True,
            )
            if settings.ENV_MODE != "local":
                # Fail closed in production so the operator sees the leak.
                raise
        return

    # Memory backend: verify the token is active before releasing.
    async with _reserve_lock:
        if token.token_id in _active_reservation_token_ids:
            _active_reservation_token_ids.remove(token.token_id)
            _reserved_bytes = max(0, _reserved_bytes - n)


async def _current_reservation() -> int:
    """Advisory read used only by `check_disk_space` (status/tests)."""
    r = _get_redis()
    if r is not None:
        try:
            return int(await r.get(_REDIS_RESERVATION_KEY) or 0)
        except Exception:
            logger.exception("Redis GET reservation failed; falling back to memory")
    return _reserved_bytes


async def check_disk_space(path: str) -> bool:
    """Advisory-only free-space check (NOT used for admission).

    Retained so status endpoints and unit tests can inspect effective free
    space without racing an admission call. Production admission MUST go
    through `try_reserve_disk_bytes` — using `check_disk_space` then
    `reserve_disk_bytes` sequentially reintroduces the TOCTOU class this
    plan just closed.
    """
    _, _, free = shutil.disk_usage(path)
    reserved = await _current_reservation()
    effective_free = max(0, free - reserved)
    return (effective_free / (1024**3)) >= settings.MIN_DISK_SPACE_GB


# ---- Zip validation --------------------------------------------------------
def is_safe_zip(zip_path: str, target_dir: str) -> tuple[bool, Optional[str]]:
    """Pre-flight ZIP validation. Extraction happens later with measured ratio checks."""
    try:
        target_abspath = os.path.abspath(target_dir)
        total_size = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            infolist = zf.infolist()
            if len(infolist) > settings.MAX_ZIP_ENTRIES:
                return (
                    False,
                    f"ZIP contains {len(infolist)} entries which exceeds "
                    f"MAX_ZIP_ENTRIES={settings.MAX_ZIP_ENTRIES}.",
                )

            for info in infolist:
                # Reject symlinks
                unix_mode = info.external_attr >> 16
                if (unix_mode & 0o170000) == 0o120000:
                    return (
                        False,
                        f"ZIP contains forbidden symbolic links: {info.filename}",
                    )

                # Zip-slip check via commonpath
                # Reject absolute paths and any that resolve outside target_abspath.
                joined = os.path.join(target_abspath, info.filename)
                resolved_path = os.path.abspath(joined)
                try:
                    common = os.path.commonpath([target_abspath, resolved_path])
                except ValueError:
                    # Different drives on Windows, absolute vs relative, etc.
                    return (
                        False,
                        f"Directory traversal attempt detected: {info.filename}",
                    )
                if common != target_abspath:
                    return (
                        False,
                        f"Directory traversal attempt detected: {info.filename}",
                    )

                total_size += info.file_size

            if total_size > settings.EXTRACT_MAX_SIZE:
                return (
                    False,
                    f"Total expanded size {total_size} exceeds "
                    f"EXTRACT_MAX_SIZE={settings.EXTRACT_MAX_SIZE} bytes.",
                )
        return True, None
    except Exception as exc:
        return False, f"Unable to validate ZIP contents: {exc}"
