from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

import httpx

from eyeofai.schemas import BBox, ModelResult


CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _extract_json_block(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part and isinstance(part["text"], str):
                    chunks.append(part["text"])
            elif isinstance(part, str):
                chunks.append(part)
        return "\n".join(chunks)
    return str(content)


def _to_bbox(parsed: dict[str, Any], width: int, height: int) -> BBox | None:
    if "bbox" in parsed and isinstance(parsed["bbox"], dict):
        parsed = parsed["bbox"]

    if all(k in parsed for k in ["x", "y", "width", "height"]):
        x_min = float(parsed["x"])
        y_min = float(parsed["y"])
        x_max = x_min + float(parsed["width"])
        y_max = y_min + float(parsed["height"])
    elif all(k in parsed for k in ["x_min", "y_min", "x_max", "y_max"]):
        x_min = float(parsed["x_min"])
        y_min = float(parsed["y_min"])
        x_max = float(parsed["x_max"])
        y_max = float(parsed["y_max"])
    else:
        return None

    values = [x_min, y_min, x_max, y_max]
    if max(values) <= 1.0:
        x_min *= width
        y_min *= height
        x_max *= width
        y_max *= height

    x_min = max(0.0, min(float(width), x_min))
    x_max = max(0.0, min(float(width), x_max))
    y_min = max(0.0, min(float(height), y_min))
    y_max = max(0.0, min(float(height), y_max))
    if x_max <= x_min or y_max <= y_min:
        return None
    return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


async def localize_with_model(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    model_id: str,
    image_bytes: bytes,
    width: int,
    height: int,
    query: str,
    timeout_seconds: float,
) -> ModelResult:
    started = time.perf_counter()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    instruction_prompt = (
        "You localize objects in images and return strict JSON only. "
        "No markdown and no extra text. "
        "Return this object: "
        '{"bbox":{"x_min":number,"y_min":number,"x_max":number,"y_max":number},'
        '"confidence":number,"reason":"short text"}. '
        "If not found, return bbox as null and confidence 0."
    )

    def make_body(use_system: bool) -> dict[str, Any]:
        user_text = f"{instruction_prompt}\n\nQuery: {query}" if not use_system else f"Query: {query}"
        messages: list[dict[str, Any]] = []
        if use_system:
            messages.append({"role": "system", "content": instruction_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        )
        return {
            "model": model_id,
            "temperature": 0,
            "max_tokens": 250,
            "messages": messages,
        }

    async def send_and_parse(body: dict[str, Any]) -> ModelResult:
        resp = await client.post(
            CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://eyeofai.local",
                "X-Title": "EyeOfAI",
            },
            json=body,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        content_raw = payload["choices"][0]["message"].get("content", "")
        content = _stringify_content(content_raw)
        parsed = _extract_json_block(content)
        latency_ms = int((time.perf_counter() - started) * 1000)
        if not parsed:
            return ModelResult(
                model=model_id,
                bbox=None,
                confidence=0,
                reason="model output was not valid JSON",
                latency_ms=latency_ms,
                raw=payload,
                error="parse_error",
            )

        bbox = _to_bbox(parsed, width, height)
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        reason = str(parsed.get("reason", ""))
        return ModelResult(
            model=model_id,
            bbox=bbox,
            confidence=max(0.0, min(1.0, confidence)),
            reason=reason,
            latency_ms=latency_ms,
            raw=payload,
            error=None if bbox else "not_found",
        )

    try:
        return await send_and_parse(make_body(use_system=True))
    except httpx.HTTPStatusError as exc:
        text = exc.response.text
        if (
            exc.response.status_code == 400
            and "Developer instruction is not enabled" in text
        ):
            try:
                return await send_and_parse(make_body(use_system=False))
            except Exception:
                pass

        latency_ms = int((time.perf_counter() - started) * 1000)
        return ModelResult(
            model=model_id,
            bbox=None,
            confidence=0,
            reason="request failed",
            latency_ms=latency_ms,
            raw={"status": exc.response.status_code, "body": text[:2000]},
            error=f"http_{exc.response.status_code}",
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        error_code = "request_failed"
        if isinstance(exc, httpx.TimeoutException):
            error_code = "timeout"
        elif isinstance(exc, httpx.TransportError):
            error_code = "transport_error"
        return ModelResult(
            model=model_id,
            bbox=None,
            confidence=0,
            reason="request failed",
            latency_ms=latency_ms,
            raw={"exception": str(exc)[:2000]},
            error=error_code,
        )
