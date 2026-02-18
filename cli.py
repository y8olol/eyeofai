from __future__ import annotations

import asyncio
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from eyeofai.consensus import consensus_bbox, iou
from eyeofai.input_sources import capture_screen, load_from_path
from eyeofai.models_registry import (
    billing_filtered_models,
    fetch_models,
    image_capable,
    is_free_model,
    select_models,
)
from eyeofai.openrouter_client import localize_with_model
from eyeofai.schemas import BBox, InputFrame, ModelResult
from eyeofai.ui import print_eye_header, print_result_summary, print_run_info


app = typer.Typer(help="EyeOfAI - coordinate-level vision localization across OpenRouter free and paid models")
console = Console()


def _resolve_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        env_file = Path(".env.local")
        if env_file.exists():
            for raw_line in env_file.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise typer.BadParameter("Missing OpenRouter API key. Use --api-key or set OPENROUTER_API_KEY")
    return key


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _timestamp_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return Path("eyeofai/runs") / f"{ts}.json"


def _pick_model_ids(
    *,
    api_key: str,
    models_raw: Optional[str],
    max_models: int,
    strategy: str,
    refresh_models: bool,
    model_scope: str,
    billing_mode: str,
    require_image: bool,
    include_openrouter_free: bool,
    include_zero_priced: bool,
) -> tuple[list[str], dict[str, int], int]:
    catalog = fetch_models(api_key, refresh=refresh_models)
    all_pool = billing_filtered_models(
        catalog,
        billing_mode=billing_mode,
        include_openrouter_free=include_openrouter_free,
        include_zero_priced=include_zero_priced,
    )
    vision_pool = [m for m in all_pool if image_capable(m)]
    chosen_pool = all_pool if model_scope == "all" else vision_pool

    explicit_selection = bool(models_raw)
    if models_raw:
        requested = [m.strip() for m in models_raw.split(",") if m.strip()]
        available = {m.get("id", "") for m in all_pool}
        selected = [m for m in requested if m in available]
        if not selected:
            raise typer.BadParameter("None of the requested models are available in the selected billing mode")
        selected_models = [m for m in all_pool if m.get("id") in selected]
    else:
        selected_ids = select_models(chosen_pool, max_models=max_models, strategy=strategy)
        selected_models = [m for m in chosen_pool if m.get("id") in set(selected_ids)]

    skipped_non_vision = 0
    if require_image:
        filtered = [m for m in selected_models if image_capable(m)]
        skipped_non_vision = len(selected_models) - len(filtered)

        if (not explicit_selection) and len(filtered) < max_models:
            selected_ids = {m.get("id") for m in filtered}
            for candidate in vision_pool:
                cid = candidate.get("id")
                if not cid or cid in selected_ids:
                    continue
                filtered.append(candidate)
                selected_ids.add(cid)
                if len(filtered) >= max_models:
                    break
        selected_models = filtered

    if not selected_models:
        raise typer.BadParameter("No image-capable models selected. Try --refresh-models or change --billing-mode")

    selected_ids = [m.get("id", "") for m in selected_models if m.get("id")]
    free_total = len(
        [
            m
            for m in catalog
            if is_free_model(
                m,
                include_openrouter_free=include_openrouter_free,
                include_zero_priced=include_zero_priced,
            )
        ]
    )
    paid_total = max(0, len(catalog) - free_total)
    selected_free = len(
        [
            m
            for m in selected_models
            if is_free_model(
                m,
                include_openrouter_free=include_openrouter_free,
                include_zero_priced=include_zero_priced,
            )
        ]
    )
    selected_paid = max(0, len(selected_models) - selected_free)
    stats = {
        "catalog_total": len(catalog),
        "catalog_free": free_total,
        "catalog_paid": paid_total,
        "pool_total": len(all_pool),
        "pool_vision": len(vision_pool),
        "selected_total": len(selected_models),
        "selected_free": selected_free,
        "selected_paid": selected_paid,
    }
    return selected_ids[:max_models], stats, skipped_non_vision


async def _run_models(
    api_key: str,
    frame: InputFrame,
    query: str,
    model_ids: list[str],
    timeout_seconds: float,
    concurrency: int,
    max_retries: int,
    retry_base_delay: float,
    inter_request_delay: float,
    on_update: Optional[Callable[[ModelResult], None]] = None,
) -> list[ModelResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(model_id: str) -> ModelResult:
        attempt = 0
        last: ModelResult | None = None
        while attempt <= max_retries:
            async with semaphore:
                async with httpx.AsyncClient() as client:
                    result = await localize_with_model(
                        client,
                        api_key=api_key,
                        model_id=model_id,
                        image_bytes=frame.image_bytes,
                        width=frame.width,
                        height=frame.height,
                        query=query,
                        timeout_seconds=timeout_seconds,
                    )
                if inter_request_delay > 0:
                    await asyncio.sleep(inter_request_delay)

            last = result
            retriable = result.error in {"http_429", "timeout", "transport_error", "request_failed", "http_503", "http_502", "http_500"}
            if not retriable or attempt >= max_retries:
                return result

            backoff = retry_base_delay * (2**attempt)
            jitter = random.uniform(0.0, retry_base_delay)
            await asyncio.sleep(backoff + jitter)
            attempt += 1

        return last if last is not None else ModelResult(
            model=model_id,
            bbox=None,
            confidence=0,
            reason="request failed",
            latency_ms=0,
            error="request_failed",
        )

    tasks = [asyncio.create_task(run_one(model_id)) for model_id in model_ids]
    results: list[ModelResult] = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        if on_update:
            on_update(result)
    return results


def _make_progress_table(rows: list[dict[str, str]]) -> Table:
    table = Table(
        title="[bold white]Per Model Results[/bold white]",
        border_style="#ff4d4d",
        show_lines=False,
        expand=True,
    )
    table.add_column("Model", style="white", overflow="ellipsis", no_wrap=True)
    table.add_column("BBox", style="white", overflow="ellipsis")
    table.add_column("Confidence", style="#ff8a8a", justify="right", overflow="ellipsis", no_wrap=True)
    table.add_column("Latency", style="white", justify="right", overflow="ellipsis", no_wrap=True)
    table.add_column("Error", style="#ff8a8a", overflow="ellipsis", no_wrap=True)
    table.add_column("State", style="white", overflow="ellipsis", no_wrap=True)
    for row in rows:
        table.add_row(
            row["model"],
            row["bbox"],
            row["confidence"],
            row["latency"],
            row["error"],
            row["state"],
        )
    return table


def _make_rank_table(rows: list[dict[str, str]]) -> Table:
    table = Table(
        title="[bold white]Vision Model Leaderboard (Live)[/bold white]",
        border_style="#ff4d4d",
        show_lines=False,
        expand=True,
    )
    table.add_column("Rank", style="#ff8a8a", justify="right", no_wrap=True)
    table.add_column("Model", style="white", overflow="ellipsis", no_wrap=True)
    table.add_column("Score", style="white", justify="right", no_wrap=True)
    table.add_column("IoU", style="white", justify="right", no_wrap=True)
    table.add_column("Hit@0.5", style="white", justify="right", no_wrap=True)
    table.add_column("Reliability", style="white", justify="right", no_wrap=True)
    table.add_column("p50(ms)", style="white", justify="right", no_wrap=True)
    table.add_column("Done", style="#ff8a8a", justify="right", no_wrap=True)
    for row in rows:
        table.add_row(
            row["rank"],
            row["model"],
            row["score"],
            row["iou"],
            row["hit"],
            row["reliability"],
            row["latency"],
            row["done"],
        )
    return table


def _to_global(bbox: Optional[BBox], frame: InputFrame) -> Optional[dict[str, float]]:
    if not bbox:
        return None
    return {
        "x_min": bbox.x_min + frame.offset_x,
        "y_min": bbox.y_min + frame.offset_y,
        "x_max": bbox.x_max + frame.offset_x,
        "y_max": bbox.y_max + frame.offset_y,
    }


@app.command()
def models(
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key"),
    refresh: bool = typer.Option(False, help="Force-refresh model catalog from API"),
    billing_mode: str = typer.Option("free", help="Billing mode: free|paid|all"),
    include_openrouter_free: bool = typer.Option(False, help="Include openrouter/free alias model"),
    include_zero_priced: bool = typer.Option(False, help="Also include zero-priced non-:free models"),
) -> None:
    """List models by billing mode and image capability."""
    key = _resolve_api_key(api_key)
    if billing_mode not in {"free", "paid", "all"}:
        raise typer.BadParameter("--billing-mode must be free, paid, or all")
    catalog = fetch_models(key, refresh=refresh)
    scoped = billing_filtered_models(
        catalog,
        billing_mode=billing_mode,
        include_openrouter_free=include_openrouter_free,
        include_zero_priced=include_zero_priced,
    )

    table = Table(title="[bold white]EyeOfAI Model Catalog[/bold white]", border_style="#ff4d4d")
    table.add_column("Model", style="white")
    table.add_column("Billing", style="#ff8a8a")
    table.add_column("Image", style="#ff8a8a")
    table.add_column("Modality", style="white")
    table.add_column("Context", style="white")
    for item in scoped:
        free = is_free_model(
            item,
            include_openrouter_free=include_openrouter_free,
            include_zero_priced=include_zero_priced,
        )
        vision = "yes" if image_capable(item) else "no"
        table.add_row(
            item.get("id", ""),
            "free" if free else "paid",
            vision,
            item.get("architecture", {}).get("modality", ""),
            str(item.get("context_length", "")),
        )

    free_count = len(
        [
            m
            for m in catalog
            if is_free_model(
                m,
                include_openrouter_free=include_openrouter_free,
                include_zero_priced=include_zero_priced,
            )
        ]
    )
    paid_count = max(0, len(catalog) - free_count)
    scoped_vision = [m for m in scoped if image_capable(m)]
    stats = (
        f"Total models: {len(catalog)}\n"
        f"Catalog free models: {free_count}\n"
        f"Catalog paid models: {paid_count}\n"
        f"Selected ({billing_mode}) models: {len(scoped)}\n"
        f"Selected image-capable models: {len(scoped_vision)}"
    )
    console.print(table)
    console.print(Panel(stats, border_style="#ff4d4d", title="[bold white]Stats[/bold white]"))


@app.command()
def find(
    query: str = typer.Option(..., help="What to locate in the image/document"),
    input_path: Optional[str] = typer.Option(None, "--input", help="Image, PDF, or folder path"),
    recursive: bool = typer.Option(False, help="Recursively scan folder inputs"),
    capture: Optional[str] = typer.Option(None, help="Capture mode: screen or region"),
    region: Optional[str] = typer.Option(None, help="Region for capture mode: x,y,w,h"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key"),
    models_raw: Optional[str] = typer.Option(None, "--models", help="Comma-separated model IDs to use"),
    max_models: int = typer.Option(10, min=1, max=200, help="How many models to run"),
    strategy: str = typer.Option("fastest", help="Selection strategy: fastest|largest-context|all|random"),
    free_scope: str = typer.Option("all", help="Model scope: all|vision"),
    billing_mode: str = typer.Option("free", help="Billing mode: free|paid|all"),
    include_openrouter_free: bool = typer.Option(False, help="Include openrouter/free alias model"),
    include_zero_priced: bool = typer.Option(False, help="Also include zero-priced non-:free models"),
    timeout_seconds: float = typer.Option(35.0, help="Per-model timeout"),
    concurrency: int = typer.Option(2, min=1, max=20, help="Max concurrent model requests"),
    max_retries: int = typer.Option(2, min=0, max=6, help="Retries for rate-limit/transient errors"),
    retry_base_delay: float = typer.Option(1.5, min=0.1, max=30.0, help="Base retry backoff seconds"),
    inter_request_delay: float = typer.Option(0.15, min=0.0, max=5.0, help="Delay after each model request"),
    min_agreement: float = typer.Option(0.4, min=0.0, max=1.0, help="Minimum agreement against attempted models"),
    refresh_models: bool = typer.Option(False, help="Refresh model list from API now"),
    no_eye: bool = typer.Option(False, help="Disable EyeOfAI boxed art header"),
    out_json: Optional[str] = typer.Option(None, help="Write full raw JSON result to this file"),
    save_raw: bool = typer.Option(True, help="Save full raw JSON to timestamped run file"),
) -> None:
    """Locate query targets in images, PDFs, folders, or screen captures."""
    key = _resolve_api_key(api_key)
    if not input_path and not capture:
        raise typer.BadParameter("Provide --input <path> or --capture <screen|region>")
    if input_path and capture:
        raise typer.BadParameter("Use either --input or --capture, not both")
    if strategy not in {"fastest", "largest-context", "all", "random"}:
        raise typer.BadParameter("--strategy must be one of: fastest, largest-context, all, random")
    if free_scope not in {"all", "vision"}:
        raise typer.BadParameter("--free-scope must be all or vision")
    if billing_mode not in {"free", "paid", "all"}:
        raise typer.BadParameter("--billing-mode must be free, paid, or all")

    selected_models, selection_stats, skipped_non_vision = _pick_model_ids(
        api_key=key,
        models_raw=models_raw,
        max_models=max_models,
        strategy=strategy,
        refresh_models=refresh_models,
        model_scope=free_scope,
        billing_mode=billing_mode,
        require_image=True,
        include_openrouter_free=include_openrouter_free,
        include_zero_priced=include_zero_priced,
    )

    if capture:
        frames = [capture_screen(capture, region=region)]
    else:
        frames = load_from_path(Path(input_path).expanduser().resolve(), recursive=recursive)
    if not frames:
        raise typer.BadParameter("No supported files found. Supported: images + PDFs")

    if not no_eye:
        print_eye_header(console)

    all_results: list[dict[str, Any]] = []
    for frame in frames:
        print_run_info(
            console,
            query=query,
            source=frame.source,
            selected_models=selected_models,
            free_scope=free_scope,
            billing_mode=billing_mode,
            strategy=strategy,
            attempted=len(selected_models),
            skipped_non_vision=skipped_non_vision,
            concurrency=concurrency,
            max_retries=max_retries,
        )

        rows: list[dict[str, str]] = [
            {
                "model": model_id,
                "bbox": "-",
                "confidence": "-",
                "latency": "-",
                "error": "-",
                "state": "pending",
            }
            for model_id in selected_models
        ]
        row_index = {row["model"]: idx for idx, row in enumerate(rows)}

        def on_update(model_result: ModelResult) -> None:
            idx = row_index.get(model_result.model)
            if idx is None:
                return
            rows[idx]["bbox"] = json.dumps(model_result.bbox.as_dict()) if model_result.bbox else "-"
            rows[idx]["confidence"] = f"{model_result.confidence:.3f}"
            rows[idx]["latency"] = f"{model_result.latency_ms}ms"
            rows[idx]["error"] = model_result.error or "-"
            rows[idx]["state"] = "done"

        with Live(_make_progress_table(rows), console=console, refresh_per_second=8, screen=True) as live:
            def update_and_render(model_result: ModelResult) -> None:
                on_update(model_result)
                live.update(_make_progress_table(rows))

            per_model = asyncio.run(
                _run_models(
                    key,
                    frame,
                    query,
                    selected_models,
                    timeout_seconds,
                    concurrency,
                    max_retries,
                    retry_base_delay,
                    inter_request_delay,
                    update_and_render,
                )
            )
            live.update(_make_progress_table(rows))

        consensus, agreement_attempted, agreement_successful = consensus_bbox(per_model, min_agreement=min_agreement)
        failed_count = len([m for m in per_model if m.error not in {None, "not_found", "parse_error"}])

        winner_global = _to_global(consensus.winner, frame)
        print_result_summary(
            console,
            winner=winner_global,
            confidence=consensus.confidence,
            agreement_attempted=agreement_attempted,
            agreement_successful=agreement_successful,
            chosen_models=consensus.chosen_models,
            uncertain_reason=consensus.uncertain_reason,
            failed_count=failed_count,
        )

        all_results.append(
            {
                "source": frame.source,
                "page": frame.page,
                "query": query,
                "offset": {"x": frame.offset_x, "y": frame.offset_y},
                "size": {"width": frame.width, "height": frame.height},
                "models": [{**m.as_dict(), "bbox_global": _to_global(m.bbox, frame)} for m in per_model],
                "consensus": {
                    **consensus.as_dict(),
                    "winner_global": winner_global,
                    "agreement_attempted": agreement_attempted,
                    "agreement_successful": agreement_successful,
                },
            }
        )

    payload = {
        "tool": "EyeOfAI",
        "meta": {
            "model_scope": free_scope,
            "billing_mode": billing_mode,
            "catalog_total": selection_stats["catalog_total"],
            "catalog_free": selection_stats["catalog_free"],
            "catalog_paid": selection_stats["catalog_paid"],
            "pool_total": selection_stats["pool_total"],
            "pool_vision": selection_stats["pool_vision"],
            "selected_total": selection_stats["selected_total"],
            "selected_free": selection_stats["selected_free"],
            "selected_paid": selection_stats["selected_paid"],
            "skipped_non_vision": skipped_non_vision,
            "selected_models": selected_models,
        },
        "results": all_results,
    }

    if out_json:
        explicit_path = Path(out_json)
        _write_json(explicit_path, payload)
        console.print(f"[bold white]Saved raw JSON:[/bold white] {explicit_path}")
    if save_raw:
        run_path = _timestamp_path()
        _write_json(run_path, payload)
        console.print(f"[bold white]Saved run JSON:[/bold white] {run_path}")


@app.command()
def rank(
    dataset: str = typer.Option(..., help="Path to JSON dataset file"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key"),
    max_models: int = typer.Option(25, min=1, max=200, help="Max models to rank"),
    strategy: str = typer.Option("all", help="Selection strategy: fastest|largest-context|all|random"),
    model_scope: str = typer.Option("vision", help="Model scope: all|vision"),
    billing_mode: str = typer.Option("all", help="Billing mode for discovery: free|paid|all"),
    include_openrouter_free: bool = typer.Option(False, help="Include openrouter/free alias model"),
    include_zero_priced: bool = typer.Option(False, help="Include zero-priced non-:free models"),
    timeout_seconds: float = typer.Option(35.0, help="Per-model timeout"),
    concurrency: int = typer.Option(1, min=1, max=20, help="Max concurrent requests"),
    max_retries: int = typer.Option(2, min=0, max=6, help="Retries for transient errors"),
    retry_base_delay: float = typer.Option(1.5, min=0.1, max=30.0, help="Base retry backoff"),
    inter_request_delay: float = typer.Option(0.2, min=0.0, max=5.0, help="Delay after each model request"),
    w_iou: float = typer.Option(0.65, min=0.0, max=1.0, help="Weight for IoU metric"),
    w_reliability: float = typer.Option(0.25, min=0.0, max=1.0, help="Weight for reliability metric"),
    w_speed: float = typer.Option(0.10, min=0.0, max=1.0, help="Weight for speed metric"),
    out_json: str = typer.Option("eyeofai-rank.json", help="Output leaderboard JSON"),
) -> None:
    """Discover and rank best vision models using a labeled dataset."""
    key = _resolve_api_key(api_key)
    if billing_mode not in {"free", "paid", "all"}:
        raise typer.BadParameter("--billing-mode must be free, paid, or all")
    if model_scope not in {"all", "vision"}:
        raise typer.BadParameter("--model-scope must be all or vision")

    examples = json.loads(Path(dataset).read_text())
    if not examples:
        raise typer.BadParameter("Dataset is empty")

    model_ids, selection_stats, skipped_non_vision = _pick_model_ids(
        api_key=key,
        models_raw=None,
        max_models=max_models,
        strategy=strategy,
        refresh_models=False,
        model_scope=model_scope,
        billing_mode=billing_mode,
        require_image=True,
        include_openrouter_free=include_openrouter_free,
        include_zero_priced=include_zero_priced,
    )

    model_metrics: dict[str, dict[str, Any]] = {
        mid: {
            "model": mid,
            "ious": [],
            "hits": 0,
            "success": 0,
            "fail": 0,
            "latencies": [],
            "done": 0,
        }
        for mid in model_ids
    }

    def compute_score(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
        done = max(1, metrics["done"])
        mean_iou = (sum(metrics["ious"]) / len(metrics["ious"])) if metrics["ious"] else 0.0
        hit = metrics["hits"] / done
        reliability = metrics["success"] / done
        p50 = 0.0
        if metrics["latencies"]:
            lat = sorted(metrics["latencies"])
            p50 = float(lat[len(lat) // 2])
        speed_score = 1.0 / (1.0 + (p50 / 4000.0)) if p50 > 0 else 0.0
        composite = (w_iou * mean_iou) + (w_reliability * reliability) + (w_speed * speed_score)
        return composite, mean_iou, hit, reliability, p50

    def leaderboard_rows() -> list[dict[str, str]]:
        scored: list[tuple[str, float, float, float, float, float, int]] = []
        for mid, m in model_metrics.items():
            score, mean_iou, hit, reliability, p50 = compute_score(m)
            scored.append((mid, score, mean_iou, hit, reliability, p50, m["done"]))
        scored.sort(key=lambda x: x[1], reverse=True)
        rows: list[dict[str, str]] = []
        for idx, (mid, score, mean_iou, hit, reliability, p50, done) in enumerate(scored, start=1):
            rows.append(
                {
                    "rank": str(idx),
                    "model": mid,
                    "score": f"{score:.3f}",
                    "iou": f"{mean_iou:.3f}",
                    "hit": f"{hit:.3f}",
                    "reliability": f"{reliability:.3f}",
                    "latency": f"{int(p50)}",
                    "done": str(done),
                }
            )
        return rows

    print_eye_header(console)
    console.print(
        Panel(
            (
                f"Dataset examples: {len(examples)}\n"
                f"Billing mode: {billing_mode}\n"
                f"Model scope: {model_scope}\n"
                f"Selected models: {len(model_ids)}\n"
                f"Skipped non-vision: {skipped_non_vision}"
            ),
            border_style="#ff4d4d",
            title="[bold white]Rank Setup[/bold white]",
        )
    )

    with Live(_make_rank_table(leaderboard_rows()), console=console, refresh_per_second=8, screen=True) as live:
        for item in examples:
            frame = load_from_path(Path(item["input"]).expanduser().resolve(), recursive=False)[0]
            gt = item["gt"]
            gt_box = BBox(gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"])
            for mid in model_ids:
                results = asyncio.run(
                    _run_models(
                        key,
                        frame,
                        item["query"],
                        [mid],
                        timeout_seconds,
                        concurrency,
                        max_retries,
                        retry_base_delay,
                        inter_request_delay,
                    )
                )
                result = results[0]
                mm = model_metrics[mid]
                mm["done"] += 1
                mm["latencies"].append(result.latency_ms)
                if result.error in {None, "not_found", "parse_error"}:
                    mm["success"] += 1
                else:
                    mm["fail"] += 1
                if result.bbox is not None:
                    score_iou = iou(result.bbox, gt_box)
                    mm["ious"].append(score_iou)
                    if score_iou >= 0.5:
                        mm["hits"] += 1
                live.update(_make_rank_table(leaderboard_rows()))

    leaderboard: list[dict[str, Any]] = []
    for row in leaderboard_rows():
        mid = row["model"]
        mm = model_metrics[mid]
        leaderboard.append(
            {
                "rank": int(row["rank"]),
                "model": mid,
                "score": float(row["score"]),
                "mean_iou": float(row["iou"]),
                "hit_at_0_5": float(row["hit"]),
                "reliability": float(row["reliability"]),
                "p50_latency_ms": int(row["latency"]),
                "done": mm["done"],
                "success": mm["success"],
                "fail": mm["fail"],
            }
        )

    output = {
        "tool": "EyeOfAI",
        "mode": "rank",
        "meta": {
            "billing_mode": billing_mode,
            "model_scope": model_scope,
            "weights": {"iou": w_iou, "reliability": w_reliability, "speed": w_speed},
            **selection_stats,
            "skipped_non_vision": skipped_non_vision,
        },
        "leaderboard": leaderboard,
    }
    _write_json(Path(out_json), output)
    console.print(Panel(f"Saved leaderboard: {out_json}", border_style="#ff4d4d", title="[bold white]Rank Complete[/bold white]"))


@app.command()
def bench(
    dataset: str = typer.Option(..., help="Path to JSON dataset file"),
    api_key: Optional[str] = typer.Option(None, help="OpenRouter API key"),
    max_models: int = typer.Option(10, min=1, max=200),
    strategy: str = typer.Option("fastest"),
    free_scope: str = typer.Option("all"),
    billing_mode: str = typer.Option("free"),
    include_openrouter_free: bool = typer.Option(False),
    include_zero_priced: bool = typer.Option(False),
    concurrency: int = typer.Option(2, min=1, max=20),
    max_retries: int = typer.Option(2, min=0, max=6),
    retry_base_delay: float = typer.Option(1.5, min=0.1, max=30.0),
    inter_request_delay: float = typer.Option(0.15, min=0.0, max=5.0),
    out_json: str = typer.Option("eyeofai-benchmark.json"),
) -> None:
    """Run a lightweight benchmark on labeled examples."""
    key = _resolve_api_key(api_key)
    if billing_mode not in {"free", "paid", "all"}:
        raise typer.BadParameter("--billing-mode must be free, paid, or all")
    examples = json.loads(Path(dataset).read_text())
    model_ids, _, _ = _pick_model_ids(
        api_key=key,
        models_raw=None,
        max_models=max_models,
        strategy=strategy,
        refresh_models=False,
        model_scope=free_scope,
        billing_mode=billing_mode,
        require_image=True,
        include_openrouter_free=include_openrouter_free,
        include_zero_priced=include_zero_priced,
    )

    records: list[dict[str, Any]] = []
    for item in examples:
        frame = load_from_path(Path(item["input"]).expanduser().resolve(), recursive=False)[0]
        result = asyncio.run(
            _run_models(
                key,
                frame,
                item["query"],
                model_ids,
                35.0,
                concurrency,
                max_retries,
                retry_base_delay,
                inter_request_delay,
            )
        )
        consensus, agreement_attempted, agreement_successful = consensus_bbox(result)
        gt = item["gt"]
        gt_box = BBox(gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"])
        score = iou(consensus.winner, gt_box) if consensus.winner else 0.0
        records.append(
            {
                "input": item["input"],
                "query": item["query"],
                "iou": score,
                "agreement_attempted": agreement_attempted,
                "agreement_successful": agreement_successful,
                "consensus": consensus.as_dict(),
            }
        )

    avg_iou = sum(r["iou"] for r in records) / max(1, len(records))
    output = {"models": model_ids, "avg_iou": avg_iou, "records": records}
    _write_json(Path(out_json), output)
    console.print(Panel(f"avg_iou={avg_iou:.3f}\nSaved to {out_json}", border_style="#ff4d4d", title="[bold white]Benchmark[/bold white]"))


if __name__ == "__main__":
    app()
