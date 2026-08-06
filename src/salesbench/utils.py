from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import re


MISSING_TOKENS = {"", "-", "—", "暂无", "N/A", "NA", "None", "null", "nan"}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_missing(value: object) -> bool:
    text = clean_text(value)
    if not text:
        return True
    return text in MISSING_TOKENS or text.lower() in {"none", "null", "nan"}


def parse_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)

    text = clean_text(value).replace(",", "").replace("，", "")
    if not text or text in MISSING_TOKENS or text.lower() in {"none", "null", "nan"}:
        return None

    if "~" in text:
        parts = [parse_number(part) for part in text.split("~")]
        values = [part for part in parts if part is not None]
        if values:
            return sum(values) / len(values)
        return None

    if text.endswith("%"):
        parsed = parse_number(text[:-1])
        return None if parsed is None else parsed / 100.0

    multiplier = 1.0
    unit_map = {
        "w": 10_000.0,
        "W": 10_000.0,
        "万": 10_000.0,
        "亿": 100_000_000.0,
        "k": 1_000.0,
        "K": 1_000.0,
    }
    if text and text[-1] in unit_map:
        multiplier = unit_map[text[-1]]
        text = text[:-1].strip()

    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", text):
        return float(text) * multiplier
    return None


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def log1p_or_zero(value: float | None) -> float:
    if value is None or value <= -1:
        return 0.0
    return math.log1p(value)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    return percentile(values, 0.5)


def iqr(values: list[float]) -> float:
    return percentile(values, 0.75) - percentile(values, 0.25)


def excel_date_to_iso(value: object) -> str:
    numeric = parse_number(value)
    if numeric is not None:
        base = date(1899, 12, 30)
        return (base + timedelta(days=int(numeric))).isoformat()

    text = clean_text(value)
    if not text:
        return ""

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return text


def excel_time_fraction_to_hms(value: object) -> str:
    numeric = parse_number(value)
    if numeric is None:
        text = clean_text(value)
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", text):
            return text if len(text.split(":")) == 3 else f"{text}:00"
        return ""

    seconds = int(round((numeric % 1.0) * 24 * 60 * 60))
    if seconds == 24 * 60 * 60:
        seconds = 0
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remain = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remain:02d}"


def normalize_robust(value: float | None, p5: float, p95: float) -> float:
    if value is None:
        return 0.0
    if p95 <= p5:
        return 0.0
    return clip((value - p5) / (p95 - p5), 0.0, 1.0)


def duration_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "未知"
    if seconds <= 30:
        return "<=30s"
    if seconds <= 60:
        return "31-60s"
    if seconds <= 120:
        return "61-120s"
    return ">120s"


def rankdata_average(values: list[float]) -> list[float]:
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor
        while end + 1 < len(indexed) and indexed[end + 1][0] == indexed[cursor][0]:
            end += 1
        average_rank = (cursor + end + 2) / 2.0
        for offset in range(cursor, end + 1):
            _, original_index = indexed[offset]
            ranks[original_index] = average_rank
        cursor = end + 1
    return ranks


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    mean_x = mean(x_values)
    mean_y = mean(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def spearman_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    return pearson_correlation(rankdata_average(x_values), rankdata_average(y_values))


def roc_auc_score(binary_labels: list[int], scores: list[float]) -> float | None:
    if len(binary_labels) != len(scores) or not binary_labels:
        return None
    positives = sum(binary_labels)
    negatives = len(binary_labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranks = rankdata_average(scores)
    positive_rank_sum = sum(rank for label, rank in zip(binary_labels, ranks) if label == 1)
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return auc


def macro_f1_score(true_labels: list[str], pred_labels: list[str]) -> float:
    if len(true_labels) != len(pred_labels) or not true_labels:
        return 0.0
    labels = sorted(set(true_labels) | set(pred_labels))
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(true_labels, pred_labels) if truth == label and pred != label)
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return mean(f1_values)


def brier_score(probabilities: list[float], outcomes: list[int]) -> float | None:
    if len(probabilities) != len(outcomes) or not probabilities:
        return None
    return mean([(clip(probability, 0.0, 1.0) - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes)])
