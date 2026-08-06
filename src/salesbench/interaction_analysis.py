"""Private, non-causal analysis of snapshot interaction profiles.

This module is deliberately independent from Gold/Evidence and VQA compilation.
Its outputs are diagnostic research artifacts and must never become model inputs,
reference answers, Judge evidence, or leaderboard components.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .io_utils import read_records, write_json, write_jsonl
from .utils import clean_text, parse_number


INTERACTION_KEYS = ("likes", "comments", "shares", "collects")
CONTROL_KEYS = ("product_bucket", "followers_total", "publish_date", "video_duration_s")
DEFAULT_VISIBLE_FEATURES = (
    "video_duration_s",
    "speech_rate",
    "text_density",
    "text_magnitude",
    "hashtag_count",
)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile for an empty sequence")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _tertile(value: float, q33: float, q67: float) -> str:
    if value <= q33:
        return "low"
    if value <= q67:
        return "medium"
    return "high"


def _publish_week(value: object) -> str:
    text = clean_text(value)
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError:
        return "unknown"
    year, week, _ = parsed.isocalendar()
    return f"{year}-W{week:02d}"


def _band(value: float | None, cut1: float, cut2: float) -> str:
    if value is None:
        return "unknown"
    return _tertile(value, cut1, cut2)


def build_private_analysis_metadata(
    records: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build private snapshot profiles and coarse controlled strata.

    Tertiles are descriptive sampling aids only. They do not define quality,
    performance, or a model prediction target.
    """

    follower_logs = [
        math.log1p(value)
        for record in records
        if (value := parse_number(record.get("followers_total"))) is not None and value >= 0
    ]
    duration_values = [
        value
        for record in records
        if (value := parse_number(record.get("video_duration_s"))) is not None and value >= 0
    ]
    follower_cuts = (
        _quantile(follower_logs, 1 / 3),
        _quantile(follower_logs, 2 / 3),
    ) if follower_logs else (0.0, 0.0)
    duration_cuts = (
        _quantile(duration_values, 1 / 3),
        _quantile(duration_values, 2 / 3),
    ) if duration_values else (0.0, 0.0)

    metric_values: dict[str, list[float]] = {key: [] for key in INTERACTION_KEYS}
    for record in records:
        for key in INTERACTION_KEYS:
            value = parse_number(record.get(key))
            if value is not None and value >= 0:
                metric_values[key].append(math.log1p(value))
    thresholds = {
        key: {
            "q33": _quantile(values, 1 / 3),
            "q67": _quantile(values, 2 / 3),
        }
        for key, values in metric_values.items()
        if values
    }

    private_records: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: clean_text(item.get("video_id"))):
        video_id = clean_text(record.get("video_id"))
        follower_value = parse_number(record.get("followers_total"))
        duration_value = parse_number(record.get("video_duration_s"))
        controls = {
            "product_bucket": clean_text(record.get("product_bucket")) or "unknown",
            "follower_band": _band(
                math.log1p(follower_value) if follower_value is not None and follower_value >= 0 else None,
                *follower_cuts,
            ),
            "publish_week": _publish_week(record.get("publish_date")),
            "duration_band": _band(duration_value, *duration_cuts),
        }
        log_counts: dict[str, float] = {}
        raw_counts: dict[str, float] = {}
        strata: dict[str, str] = {}
        for key in INTERACTION_KEYS:
            value = parse_number(record.get(key))
            if value is None or value < 0 or key not in thresholds:
                continue
            transformed = math.log1p(value)
            raw_counts[key] = value
            log_counts[key] = round(transformed, 6)
            strata[key] = _tertile(transformed, thresholds[key]["q33"], thresholds[key]["q67"])
        private_records.append(
            {
                "video_id": video_id,
                "analysis_type": "snapshot_interaction_profile",
                "raw_snapshot_counts": raw_counts,
                "log1p_counts": log_counts,
                "interaction_strata": strata,
                "matching_controls": controls,
                "cohort_key": "|".join(str(controls[key]) for key in sorted(controls)),
                "limitations": [
                    "unknown_standardized_observation_window",
                    "no_video_exposure_denominator",
                    "no_watch_time_or_sales_conversion",
                    "descriptive_non_causal_only",
                ],
            }
        )

    meta = {
        "analysis_type": "snapshot_interaction_profile",
        "visibility": "private_analysis_only",
        "included_in_vqa": False,
        "included_in_leaderboard": False,
        "interaction_keys": list(INTERACTION_KEYS),
        "control_keys": list(CONTROL_KEYS),
        "transform": "log1p",
        "stratification": "global_tertiles_with_matching_controls",
        "thresholds": thresholds,
        "limitations": [
            "Interaction counts are snapshot associations, not content quality labels.",
            "No causal or predictive interpretation is permitted.",
        ],
    }
    return private_records, meta


def _bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(max(1, samples)))
    low = estimates[int((len(estimates) - 1) * 0.025)]
    high = estimates[int((len(estimates) - 1) * 0.975)]
    return round(low, 6), round(high, 6)


def interaction_diagnostic_slices(
    judge_details: list[dict[str, object]],
    private_metadata: list[dict[str, object]],
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Slice already-computed VQA scores by private interaction strata."""

    metadata_lookup = {clean_text(row.get("video_id")): row for row in private_metadata}
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for detail in judge_details:
        video_id = clean_text(detail.get("video_id"))
        score = parse_number(detail.get("score"))
        if score is None or video_id not in metadata_lookup:
            continue
        task = clean_text(detail.get("task_type")).upper()
        for metric, stratum in dict(metadata_lookup[video_id].get("interaction_strata") or {}).items():
            buckets[(task, clean_text(metric), clean_text(stratum))].append(score)

    rows: list[dict[str, object]] = []
    for index, ((task, metric, stratum), values) in enumerate(sorted(buckets.items())):
        ci_low, ci_high = _bootstrap_mean_ci(values, bootstrap_samples, seed + index)
        rows.append(
            {
                "task_type": task,
                "interaction_metric": metric,
                "stratum": stratum,
                "sample_count": len(values),
                "mean_score": round(mean(values), 6),
                "bootstrap_ci95": [ci_low, ci_high],
            }
        )
    return {
        "role": "diagnostic_only",
        "included_in_leaderboard": False,
        "interpretation": "robustness_association_not_causation",
        "slices": rows,
    }


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = average_rank
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return numerator / denominator if denominator else None


def noncausal_associations(
    source_records: list[dict[str, object]],
    visible_features: Iterable[str] = DEFAULT_VISIBLE_FEATURES,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for feature in visible_features:
        for metric in INTERACTION_KEYS:
            pairs: list[tuple[float, float]] = []
            for record in source_records:
                feature_value = parse_number(record.get(feature))
                metric_value = parse_number(record.get(metric))
                if feature_value is None or metric_value is None or metric_value < 0:
                    continue
                pairs.append((feature_value, math.log1p(metric_value)))
            correlation = _pearson(_rank([x for x, _ in pairs]), _rank([y for _, y in pairs]))
            rows.append(
                {
                    "feature": feature,
                    "interaction_metric": metric,
                    "sample_count": len(pairs),
                    "spearman_rho": None if correlation is None else round(correlation, 6),
                }
            )
    return {
        "analysis_type": "noncausal_association",
        "allowed_language": ["correlation", "association", "co-occurrence"],
        "prohibited_claims": ["causation", "interaction_prediction_capability", "content_quality"],
        "associations": rows,
    }


def run_interaction_analysis(
    records_path: Path,
    output_dir: Path,
    judge_details_path: Path | None = None,
    bootstrap_samples: int = 1000,
) -> dict[str, object]:
    records = read_records(records_path)
    private_records, metadata = build_private_analysis_metadata(records)
    associations = noncausal_associations(records)
    diagnostics = None
    if judge_details_path is not None:
        diagnostics = interaction_diagnostic_slices(
            read_records(judge_details_path),
            private_records,
            bootstrap_samples=bootstrap_samples,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    private_path = output_dir / "private_analysis_metadata.jsonl"
    report_path = output_dir / "interaction_analysis.json"
    write_jsonl(private_path, private_records)
    report = {
        "metadata": metadata,
        "associations": associations,
        "diagnostics": diagnostics,
        "outputs": {
            "private_analysis_metadata": str(private_path),
            "report": str(report_path),
        },
    }
    write_json(report_path, report)
    return report
