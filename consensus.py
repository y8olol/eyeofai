from __future__ import annotations

from dataclasses import dataclass

from eyeofai.schemas import BBox, ConsensusResult, ModelResult


def iou(a: BBox, b: BBox) -> float:
    x_left = max(a.x_min, b.x_min)
    y_top = max(a.y_min, b.y_min)
    x_right = min(a.x_max, b.x_max)
    y_bottom = min(a.y_max, b.y_max)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Cluster:
    entries: list[ModelResult]


def _weighted_mean_box(items: list[ModelResult]) -> BBox:
    total = 0.0
    sx_min = sy_min = sx_max = sy_max = 0.0
    for item in items:
        if not item.bbox:
            continue
        w = max(0.05, item.confidence)
        total += w
        sx_min += item.bbox.x_min * w
        sy_min += item.bbox.y_min * w
        sx_max += item.bbox.x_max * w
        sy_max += item.bbox.y_max * w
    if total == 0:
        first = items[0].bbox
        assert first is not None
        return first
    return BBox(sx_min / total, sy_min / total, sx_max / total, sy_max / total)


def consensus_bbox(
    results: list[ModelResult],
    *,
    iou_threshold: float = 0.5,
    min_agreement: float = 0.4,
) -> tuple[ConsensusResult, float, float]:
    valid = [r for r in results if r.bbox is not None]
    attempted = len(results)
    successful = len([r for r in results if r.error not in {"http_400", "http_401", "http_403", "http_404", "http_429", "http_500", "http_502", "http_503", "timeout", "transport_error", "request_failed"}])
    if not valid:
        return (
            ConsensusResult(
            winner=None,
            confidence=0.0,
            agreement=0.0,
            chosen_models=[],
            uncertain_reason="no model returned a valid box",
            ),
            0.0,
            0.0,
        )

    clusters: list[_Cluster] = []
    for result in valid:
        placed = False
        for cluster in clusters:
            centroid = _weighted_mean_box(cluster.entries)
            if iou(result.bbox, centroid) >= iou_threshold:
                cluster.entries.append(result)
                placed = True
                break
        if not placed:
            clusters.append(_Cluster(entries=[result]))

    best = max(clusters, key=lambda c: sum(max(0.05, e.confidence) for e in c.entries))
    winner = _weighted_mean_box(best.entries)
    agreement_successful = len(best.entries) / max(1, successful)
    agreement_attempted = len(best.entries) / max(1, attempted)
    confidence = min(
        1.0,
        0.5 * agreement_attempted + 0.5 * (sum(e.confidence for e in best.entries) / len(best.entries)),
    )
    models = [e.model for e in best.entries]

    if agreement_attempted < min_agreement:
        return (
            ConsensusResult(
            winner=winner,
            confidence=confidence,
            agreement=agreement_attempted,
            chosen_models=models,
            uncertain_reason="weak agreement across models",
            ),
            agreement_attempted,
            agreement_successful,
        )

    return (
        ConsensusResult(
            winner=winner,
            confidence=confidence,
            agreement=agreement_attempted,
            chosen_models=models,
            uncertain_reason=None,
        ),
        agreement_attempted,
        agreement_successful,
    )
