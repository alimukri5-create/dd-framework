"""Small TTL JSON cache for Velocity Scanner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CACHE_PATH = Path(__file__).resolve().parent / "cache" / "velocity_scanner_cache.json"


@dataclass(frozen=True)
class CacheResult:
    value: Any
    age_seconds: float


def _read_cache(path: Path = CACHE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(data: dict[str, Any], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, separators=(",", ":"), sort_keys=True)
        tmp_path.replace(path)
    except PermissionError:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, separators=(",", ":"), sort_keys=True)


def get(key: str, ttl_seconds: int, path: Path = CACHE_PATH) -> CacheResult | None:
    """Return a cached value only when it is still inside its TTL."""
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    cache = _read_cache(path)
    item = cache.get(key)
    if not isinstance(item, dict):
        return None
    saved_at = item.get("saved_at")
    if not isinstance(saved_at, (int, float)):
        return None
    age = time.time() - float(saved_at)
    if age < 0 or age > ttl_seconds:
        return None
    return CacheResult(value=item.get("value"), age_seconds=age)


def set(key: str, value: Any, path: Path = CACHE_PATH) -> None:
    cache = _read_cache(path)
    cache[key] = {"saved_at": time.time(), "value": value}
    _write_cache(cache, path)


def format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"
