from __future__ import annotations

from collections import Counter

from .config import BenchmarkConfig
from .schema import FAN_SEGMENTS_LOW_TO_HIGH, NUMERIC_FIELD_MAP, PRODUCT_BUCKET_CODE_MAP, PRODUCT_BUCKET_COLUMNS, VIDEO_ID_COLUMN
from .utils import clean_text, duration_bucket, excel_date_to_iso, excel_time_fraction_to_hms, parse_number, percentile, safe_div


def normalize_gender(raw_value: object) -> tuple[str, int | None]:
    text = clean_text(raw_value).replace("/", "")
    return {
        "女": ("female", 1),
        "男": ("male", 0),
        "1": ("female", 1),
        "0": ("male", 0),
        "-": ("unknown", None),
        "": ("unknown", None),
    }.get(text, ("unknown", None))


def is_verified(raw_value: object) -> bool:
    text = clean_text(raw_value)
    return bool(text and text not in {"0", "-"} and "未认证" not in text)


def infer_product_bucket(record: dict[str, object]) -> str:
    for column, bucket in PRODUCT_BUCKET_COLUMNS.items():
        value = parse_number(record.get(column))
        if value is not None and value > 0:
            return bucket
    for code_column in ("产品类型", "product_type"):
        code = clean_text(record.get(code_column))
        if code in PRODUCT_BUCKET_CODE_MAP:
            return PRODUCT_BUCKET_CODE_MAP[code]
    return clean_text(record.get("商品分类")) or "未知"


def _build_base_record(
    raw_record: dict[str, str],
    asset_index: dict[str, dict[str, list[str]]],
) -> dict[str, object]:
    record: dict[str, object] = dict(raw_record)
    for raw_field, clean_field in NUMERIC_FIELD_MAP.items():
        record[clean_field] = parse_number(raw_record.get(raw_field))

    video_id = clean_text(raw_record.get(VIDEO_ID_COLUMN))
    video_text = clean_text(raw_record.get("视频文本"))
    title = clean_text(raw_record.get("视频标题"))
    gender_clean, gender_code = normalize_gender(raw_record.get("性别"))
    followers_total = parse_number(raw_record.get("粉丝总数"))
    followers_w = parse_number(raw_record.get("粉丝数/w"))
    followers_million = parse_number(raw_record.get("粉丝数/ml"))
    if followers_total is None and followers_w is not None:
        followers_total = followers_w * 10_000.0
    if followers_total is None and followers_million is not None:
        followers_total = followers_million * 1_000_000.0

    assets = asset_index.get(video_id, {"video_paths": [], "sales_paths": []})
    video_paths = list(assets["video_paths"])
    sales_paths = list(assets["sales_paths"])
    duration = record.get("video_duration_s")
    text_chars = parse_number(raw_record.get("视频文本总字符数"))
    if text_chars is None and video_text:
        text_chars = float(len(video_text))

    record.update(
        {
            "video_id": video_id,
            "publish_date": excel_date_to_iso(raw_record.get("发布日期")),
            "publish_time": excel_time_fraction_to_hms(raw_record.get("发布时间")),
            "author_name": clean_text(raw_record.get("作者")),
            "douyin_handle": clean_text(raw_record.get("抖音号")),
            "title": title,
            "video_text": video_text,
            "gender_clean": gender_clean,
            "gender_code": gender_code,
            "product_bucket": infer_product_bucket(raw_record),
            "influencer_type_clean": clean_text(raw_record.get("达人类型")) or "未知",
            "is_verified": is_verified(raw_record.get("认证信息")),
            "followers_total": followers_total,
            "video_paths": video_paths,
            "primary_video_path": video_paths[0] if video_paths else "",
            "has_video_asset": int(bool(video_paths)),
            "sales_image_paths": sales_paths,
            "primary_sales_image_path": sales_paths[0] if sales_paths else "",
            "has_sales_asset": int(bool(sales_paths)),
            "video_text_chars": text_chars,
            "text_density": safe_div(text_chars, duration),
            "duration_bucket": duration_bucket(duration),
        }
    )
    return record


def _assign_fan_segments(records: list[dict[str, object]]) -> dict[str, float]:
    followers = [float(record["followers_total"]) for record in records if record.get("followers_total") is not None]
    if not followers:
        for record in records:
            record["fan_segment"] = "未知"
        return {"q25": 0.0, "q50": 0.0, "q75": 0.0}
    cuts = {"q25": percentile(followers, 0.25), "q50": percentile(followers, 0.5), "q75": percentile(followers, 0.75)}
    for record in records:
        value = record.get("followers_total")
        if value is None:
            record["fan_segment"] = "未知"
        elif value <= cuts["q25"]:
            record["fan_segment"] = FAN_SEGMENTS_LOW_TO_HIGH[0]
        elif value <= cuts["q50"]:
            record["fan_segment"] = FAN_SEGMENTS_LOW_TO_HIGH[1]
        elif value <= cuts["q75"]:
            record["fan_segment"] = FAN_SEGMENTS_LOW_TO_HIGH[2]
        else:
            record["fan_segment"] = FAN_SEGMENTS_LOW_TO_HIGH[3]
    return cuts


def build_benchmark_dataset(
    raw_records: list[dict[str, str]],
    asset_index: dict[str, dict[str, list[str]]],
    asset_manifest: dict[str, object],
    config: BenchmarkConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build internal assets without creating any outcome-prediction labels."""
    records = [_build_base_record(raw_record, asset_index) for raw_record in raw_records]
    fan_thresholds = _assign_fan_segments(records)
    profile = {
        "dataset_name": config.dataset_name,
        "benchmark_type": "multimodal_vqa",
        "public_tasks": ["BP", "CM", "SS", "AE"],
        "record_count": len(records),
        "records_with_video_asset": sum(int(record["has_video_asset"]) for record in records),
        "records_with_sales_asset": sum(int(record["has_sales_asset"]) for record in records),
        "fan_segment_thresholds": fan_thresholds,
        "product_bucket_distribution": dict(sorted(Counter(clean_text(record.get("product_bucket")) for record in records).items())),
        "fan_segment_distribution": dict(sorted(Counter(clean_text(record.get("fan_segment")) for record in records).items())),
        "asset_manifest_summary": asset_manifest.get("summary", {}),
        "interaction_fields": {
            "visibility": "private_analysis_only",
            "fields": ["likes", "comments", "shares", "collects"],
        },
    }
    return records, [], [], profile
