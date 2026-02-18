from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import random

import httpx


MODELS_URL = "https://openrouter.ai/api/v1/models"


def _cache_path() -> Path:
    return Path("eyeofai/.cache/models.json")


def fetch_models(api_key: str, refresh: bool = False, ttl_seconds: int = 1800) -> list[dict[str, Any]]:
    cache = _cache_path()
    if not refresh and cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age < ttl_seconds:
            data = json.loads(cache.read_text())
            return data.get("data", [])

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            MODELS_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://eyeofai.local",
                "X-Title": "EyeOfAI",
            },
        )
        resp.raise_for_status()
        payload = resp.json()

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2))
    return payload.get("data", [])


def free_vision_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in models:
        model_id = model.get("id", "")
        input_modalities = model.get("architecture", {}).get("input_modalities") or []
        is_vision = "image" in input_modalities
        is_free = model_id.endswith(":free") or model_id == "openrouter/free"
        if is_vision and is_free:
            out.append(model)
    return out


def free_models(
    models: list[dict[str, Any]],
    include_openrouter_free: bool = False,
    include_zero_priced: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in models:
        model_id = model.get("id", "")
        is_free_suffix = model_id.endswith(":free")
        is_openrouter_alias = include_openrouter_free and model_id == "openrouter/free"
        is_zero_priced = False
        if include_zero_priced:
            pricing = model.get("pricing", {}) or {}
            try:
                is_zero_priced = float(pricing.get("prompt", "999")) == 0.0 and float(pricing.get("completion", "999")) == 0.0
            except Exception:
                is_zero_priced = False

        if is_free_suffix or is_openrouter_alias or is_zero_priced:
            out.append(model)
    return out


def is_zero_priced(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing", {}) or {}
    try:
        return float(pricing.get("prompt", "999")) == 0.0 and float(pricing.get("completion", "999")) == 0.0
    except Exception:
        return False


def is_free_model(
    model: dict[str, Any],
    *,
    include_openrouter_free: bool = False,
    include_zero_priced: bool = False,
) -> bool:
    model_id = model.get("id", "")
    if model_id.endswith(":free"):
        return True
    if include_openrouter_free and model_id == "openrouter/free":
        return True
    if include_zero_priced and is_zero_priced(model):
        return True
    return False


def billing_filtered_models(
    models: list[dict[str, Any]],
    *,
    billing_mode: str,
    include_openrouter_free: bool = False,
    include_zero_priced: bool = False,
) -> list[dict[str, Any]]:
    if billing_mode == "all":
        return list(models)
    out: list[dict[str, Any]] = []
    for model in models:
        free = is_free_model(
            model,
            include_openrouter_free=include_openrouter_free,
            include_zero_priced=include_zero_priced,
        )
        if billing_mode == "free" and free:
            out.append(model)
        if billing_mode == "paid" and not free:
            out.append(model)
    return out


def image_capable(model: dict[str, Any]) -> bool:
    input_modalities = model.get("architecture", {}).get("input_modalities") or []
    return "image" in input_modalities


def select_models(
    models: list[dict[str, Any]],
    max_models: int,
    strategy: str = "fastest",
) -> list[str]:
    if strategy == "all":
        ranked = models
    elif strategy == "random":
        ranked = list(models)
        random.shuffle(ranked)
    elif strategy == "largest-context":
        ranked = sorted(models, key=lambda m: m.get("context_length", 0), reverse=True)
    else:
        ranked = sorted(models, key=lambda m: m.get("context_length", 0))
    return [m.get("id", "") for m in ranked[:max_models] if m.get("id")]
