from __future__ import annotations

import re

from .assets import build_asset_index
from .config import BenchmarkConfig
from .excel_reader import load_sheet_records
from .io_utils import write_json, write_jsonl
from .preprocess import build_benchmark_dataset
from .utils import clean_text, mean, percentile, safe_div


CTA_KEYWORDS = (
    "下单",
    "去拍",
    "拍下",
    "购买",
    "入手",
    "冲",
    "囤",
    "抢",
    "带回家",
    "点链接",
    "别错过",
)

URGENCY_KEYWORDS = (
    "现在",
    "今天",
    "限时",
    "最后",
    "就这几天",
    "赶紧",
    "抓紧",
    "马上",
    "仅限",
    "只要",
)

SOCIAL_PROOF_KEYWORDS = (
    "口碑",
    "好评",
    "销量",
    "回购",
    "爆单",
    "都在买",
    "都在用",
    "推荐",
    "万人",
    "热销",
)

BENEFIT_KEYWORDS = (
    "好用",
    "好吃",
    "方便",
    "划算",
    "省力",
    "干净",
    "显白",
    "提亮",
    "留香",
    "舒服",
    "耐用",
)

RISK_REDUCTION_KEYWORDS = (
    "放心",
    "官方",
    "正品",
    "包邮",
    "安全",
    "无添加",
    "不伤",
    "保障",
    "售后",
    "退",
)

TEXT_STOP_PUNCT = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


def _normalize_text(value: object) -> str:
    return TEXT_STOP_PUNCT.sub("", clean_text(value)).lower()


def _char_ngrams(value: object, n: int = 2) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _contains(text: object, phrase: object) -> int:
    normalized_text = _normalize_text(text)
    normalized_phrase = _normalize_text(phrase)
    if not normalized_text or not normalized_phrase:
        return 0
    return 1 if normalized_phrase in normalized_text else 0


def _contains_any(text: object, keywords: tuple[str, ...]) -> int:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return 0
    return 1 if any(_normalize_text(keyword) in normalized_text for keyword in keywords) else 0


def _count_any(text: object, keywords: tuple[str, ...]) -> int:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return 0
    return sum(normalized_text.count(_normalize_text(keyword)) for keyword in keywords)


def _sentence_count(text: object) -> int:
    raw = clean_text(text)
    if not raw:
        return 0
    parts = [part for part in re.split(r"[。！？!?；;…\n]+", raw) if clean_text(part)]
    return max(1, len(parts))


def _punctuation_count(text: object) -> int:
    raw = clean_text(text)
    if not raw:
        return 0
    return len(re.findall(r"[，。！？!?；;：:、,\.\-]", raw))


def _question_count(text: object) -> int:
    raw = clean_text(text)
    return raw.count("?") + raw.count("？")


def _exclamation_count(text: object) -> int:
    raw = clean_text(text)
    return raw.count("!") + raw.count("！")


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _speech_rate_bucket(records: list[dict[str, object]]) -> tuple[float, float]:
    values = sorted(
        float(record["speech_rate"])
        for record in records
        if record.get("speech_rate") is not None
    )
    if not values:
        return 0.0, 0.0
    return percentile(values, 0.25), percentile(values, 0.75)


def _bucket_speech_rate(value: object, low: float, high: float) -> str:
    if value is None:
        return "unknown"
    numeric = float(value)
    if numeric <= low:
        return "slow"
    if numeric >= high:
        return "fast"
    return "medium"


def _category_terms(category: object) -> list[str]:
    raw = clean_text(category)
    if not raw:
        return []
    pieces = re.split(r"[/>｜|·\s]+", raw)
    return [piece for piece in pieces if len(_normalize_text(piece)) >= 2]


def _consistency_summary(values: list[float]) -> float:
    usable = [value for value in values if value is not None]
    return mean(usable) if usable else 0.0


def build_input_artifacts(config: BenchmarkConfig) -> dict[str, object]:
    raw_records = load_sheet_records(config.research_workbook)
    asset_index, asset_manifest = build_asset_index(config.raw_video_dir, config.raw_sales_dir)
    processed_records, _, _, _ = build_benchmark_dataset(
        raw_records=raw_records,
        asset_index=asset_index,
        asset_manifest=asset_manifest,
        config=config,
    )

    low_speech_rate, high_speech_rate = _speech_rate_bucket(processed_records)

    raw_video_records: list[dict[str, object]] = []
    visual_feature_records: list[dict[str, object]] = []
    audio_speech_records: list[dict[str, object]] = []
    text_language_records: list[dict[str, object]] = []
    publish_context_records: list[dict[str, object]] = []
    cross_modal_records: list[dict[str, object]] = []

    for record in processed_records:
        video_id = record["video_id"]
        title = clean_text(record.get("title"))
        video_text = clean_text(record.get("video_text"))
        product_title = clean_text(record.get("商品标题"))
        category = clean_text(record.get("商品分类"))
        brand_name = clean_text(record.get("brand_name"))
        small_blue_word = clean_text(record.get("小蓝词"))
        title_chars = int(record.get("title_chars") or 0)
        video_text_chars = int(record.get("video_text_chars") or 0)
        title_sentence_count = _sentence_count(title)
        video_text_sentence_count = _sentence_count(video_text)

        title_has_cta = _contains_any(title, CTA_KEYWORDS)
        text_has_cta = _contains_any(video_text, CTA_KEYWORDS)
        title_has_urgency = _contains_any(title, URGENCY_KEYWORDS)
        text_has_urgency = _contains_any(video_text, URGENCY_KEYWORDS)
        title_has_social_proof = _contains_any(title, SOCIAL_PROOF_KEYWORDS)
        text_has_social_proof = _contains_any(video_text, SOCIAL_PROOF_KEYWORDS)
        title_has_benefit = _contains_any(title, BENEFIT_KEYWORDS)
        text_has_benefit = _contains_any(video_text, BENEFIT_KEYWORDS)
        title_has_risk_reduction = _contains_any(title, RISK_REDUCTION_KEYWORDS)
        text_has_risk_reduction = _contains_any(video_text, RISK_REDUCTION_KEYWORDS)

        title_text_jaccard = _jaccard(_char_ngrams(title), _char_ngrams(video_text))
        title_product_jaccard = _jaccard(_char_ngrams(title), _char_ngrams(product_title))
        text_product_jaccard = _jaccard(_char_ngrams(video_text), _char_ngrams(product_title))

        category_terms = _category_terms(category)
        category_in_title = 1 if any(_contains(title, term) for term in category_terms) else 0
        category_in_text = 1 if any(_contains(video_text, term) for term in category_terms) else 0

        raw_video_records.append(
            {
                "video_id": video_id,
                "primary_video_path": record.get("primary_video_path"),
                "video_paths": record.get("video_paths"),
                "has_video_asset": record.get("has_video_asset"),
                "primary_sales_image_path": record.get("primary_sales_image_path"),
                "sales_image_paths": record.get("sales_image_paths"),
                "has_sales_asset": record.get("has_sales_asset"),
                "video_duration_s": record.get("video_duration_s"),
                "duration_bucket": record.get("duration_bucket"),
                "title": title,
                "video_text": video_text,
                "product_title": product_title,
                "author_name": record.get("author_name"),
                "douyin_handle": record.get("douyin_handle"),
                "product_bucket": record.get("product_bucket"),
                "recommended_frame_fps": 1,
                "recommended_hook_window_s": 3,
                "recommended_clip_strategy": "uniform_1fps_plus_first3s",
            }
        )

        visual_feature_records.append(
            {
                "video_id": video_id,
                "primary_video_path": record.get("primary_video_path"),
                "has_video_asset": record.get("has_video_asset"),
                "face_rate": record.get("face_rate"),
                "face_count_max": record.get("face_count_max"),
                "face_beauty": record.get("face_beauty"),
                "face_quality": record.get("face_quality"),
                "face_area": record.get("face_area"),
                "if_smile": record.get("if_smile"),
                "smile_rate": record.get("smile_rate"),
                "emotion_pos": record.get("emotion_pos"),
                "emotion_neg": record.get("emotion_neg"),
                "emotion_richness": record.get("emotion_richness"),
                "emotion_naturalness": record.get("emotion_naturalness"),
                "emotion_naturalness_missing": record.get("emotion_naturalness_missing"),
                "pitch_angle_std": record.get("pitch_angle_std"),
                "roll_angle_std": record.get("roll_angle_std"),
                "yaw_angle_std": record.get("yaw_angle_std"),
                "health": record.get("health"),
                "not_health": record.get("not_health"),
                "not_health_average": record.get("not_health_average"),
                "stain": record.get("stain"),
                "dark_circle": record.get("dark_circle"),
                "acne": record.get("acne"),
                "blurness": record.get("blurness"),
                "color_value": record.get("color_value"),
                "color_saturation": record.get("color_saturation"),
                "color_hue": record.get("color_hue"),
                "colorfulness": record.get("colorfulness"),
            }
        )

        audio_speech_records.append(
            {
                "video_id": video_id,
                "primary_video_path": record.get("primary_video_path"),
                "has_video_asset": record.get("has_video_asset"),
                "video_duration_s": record.get("video_duration_s"),
                "duration_bucket": record.get("duration_bucket"),
                "speech_rate": record.get("speech_rate"),
                "speech_rate_bucket": _bucket_speech_rate(record.get("speech_rate"), low_speech_rate, high_speech_rate),
                "video_text_chars": record.get("video_text_chars"),
                "text_density": record.get("text_density"),
                "is_text_missing": record.get("is_text_missing"),
                "transcript_sentence_count": video_text_sentence_count,
                "transcript_question_count": _question_count(video_text),
                "transcript_exclamation_count": _exclamation_count(video_text),
                "transcript_punctuation_count": _punctuation_count(video_text),
                "transcript_avg_sentence_length": _safe_ratio(video_text_chars, video_text_sentence_count),
                "cta_keyword_count_in_text": _count_any(video_text, CTA_KEYWORDS),
                "urgency_keyword_count_in_text": _count_any(video_text, URGENCY_KEYWORDS),
            }
        )

        text_language_records.append(
            {
                "video_id": video_id,
                "title": title,
                "video_text": video_text,
                "product_title": product_title,
                "small_blue_word": small_blue_word,
                "title_chars": record.get("title_chars"),
                "video_text_chars": record.get("video_text_chars"),
                "title_sentence_count": title_sentence_count,
                "video_text_sentence_count": video_text_sentence_count,
                "hashtag_count": record.get("hashtag_count"),
                "title_score": record.get("title_score"),
                "title_magnitude": record.get("title_magnitude"),
                "title_readability": record.get("title_readability"),
                "title_entropy": record.get("title_entropy"),
                "text_score": record.get("text_score"),
                "text_magnitude": record.get("text_magnitude"),
                "video_text_readability": record.get("video_text_readability"),
                "video_text_entropy": record.get("video_text_entropy"),
                "title_first_person": record.get("视频标题_first_person"),
                "title_second_person": record.get("视频标题_2nd"),
                "title_third_person": record.get("视频标题_3rd"),
                "text_first_person": record.get("视频文本_first_person"),
                "text_second_person": record.get("视频文本_2nd"),
                "text_third_person": record.get("视频文本_3rd"),
                "concrete1": record.get("concrete1"),
                "concrete2": record.get("concrete2"),
                "info_value_1": record.get("IV"),
                "info_value_2": record.get("IV2"),
                "info_value_3": record.get("IV3"),
                "title_has_cta": title_has_cta,
                "text_has_cta": text_has_cta,
                "title_has_urgency": title_has_urgency,
                "text_has_urgency": text_has_urgency,
                "title_has_social_proof": title_has_social_proof,
                "text_has_social_proof": text_has_social_proof,
                "title_has_benefit": title_has_benefit,
                "text_has_benefit": text_has_benefit,
                "title_has_risk_reduction": title_has_risk_reduction,
                "text_has_risk_reduction": text_has_risk_reduction,
                "title_question_count": _question_count(title),
                "text_question_count": _question_count(video_text),
                "title_exclamation_count": _exclamation_count(title),
                "text_exclamation_count": _exclamation_count(video_text),
            }
        )

        publish_context_records.append(
            {
                "video_id": video_id,
                "publish_date": record.get("publish_date"),
                "publish_time": record.get("publish_time"),
                "publish_hour": record.get("publish_hour"),
                "if_holiday": record.get("if_holiday"),
                "is_weekend": record.get("is_weekend"),
                "author_name": record.get("author_name"),
                "douyin_handle": record.get("douyin_handle"),
                "influencer_type": record.get("influencer_type_clean"),
                "gender_clean": record.get("gender_clean"),
                "is_verified": record.get("is_verified"),
                "followers_total": record.get("followers_total"),
                "followers_w": record.get("followers_w"),
                "followers_million": record.get("followers_million"),
                "fan_group_size": record.get("fan_group_size"),
                "fan_segment": record.get("fan_segment"),
                "commerce_reputation": record.get("commerce_reputation"),
                "video_sales_power": record.get("video_sales_power"),
                "product_bucket": record.get("product_bucket"),
                "product_category": category,
                "product_title": product_title,
                "brand_name": brand_name,
                "brand_present": record.get("brand_present"),
                "goods_count": record.get("goods_count"),
                "product_views_30d_w": record.get("product_views_30d_w"),
                "shop_name": clean_text(record.get("小店")),
                "small_blue_word": small_blue_word,
            }
        )

        cta_alignment = 1 if title_has_cta == text_has_cta else 0
        urgency_alignment = 1 if title_has_urgency == text_has_urgency else 0
        social_proof_alignment = 1 if title_has_social_proof == text_has_social_proof else 0
        benefit_alignment = 1 if title_has_benefit == text_has_benefit else 0
        risk_reduction_alignment = 1 if title_has_risk_reduction == text_has_risk_reduction else 0

        cross_modal_records.append(
            {
                "video_id": video_id,
                "title_text_bigram_jaccard": title_text_jaccard,
                "title_product_bigram_jaccard": title_product_jaccard,
                "text_product_bigram_jaccard": text_product_jaccard,
                "title_text_sentiment_gap": abs(float(record.get("title_score") or 0.0) - float(record.get("text_score") or 0.0)),
                "title_text_magnitude_gap": abs(float(record.get("title_magnitude") or 0.0) - float(record.get("text_magnitude") or 0.0)),
                "title_text_length_ratio": safe_div(record.get("title_chars"), record.get("video_text_chars")),
                "small_blue_word_in_title": _contains(title, small_blue_word),
                "small_blue_word_in_text": _contains(video_text, small_blue_word),
                "brand_in_title": _contains(title, brand_name),
                "brand_in_text": _contains(video_text, brand_name),
                "product_title_in_title": _contains(title, product_title),
                "product_title_in_text": _contains(video_text, product_title),
                "category_in_title": category_in_title,
                "category_in_text": category_in_text,
                "title_has_cta": title_has_cta,
                "text_has_cta": text_has_cta,
                "cta_alignment": cta_alignment,
                "title_has_urgency": title_has_urgency,
                "text_has_urgency": text_has_urgency,
                "urgency_alignment": urgency_alignment,
                "title_has_social_proof": title_has_social_proof,
                "text_has_social_proof": text_has_social_proof,
                "social_proof_alignment": social_proof_alignment,
                "title_has_benefit": title_has_benefit,
                "text_has_benefit": text_has_benefit,
                "benefit_alignment": benefit_alignment,
                "title_has_risk_reduction": title_has_risk_reduction,
                "text_has_risk_reduction": text_has_risk_reduction,
                "risk_reduction_alignment": risk_reduction_alignment,
                "cross_modal_consistency_score": _consistency_summary(
                    [
                        title_text_jaccard,
                        title_product_jaccard,
                        text_product_jaccard,
                        float(cta_alignment),
                        float(urgency_alignment),
                        float(social_proof_alignment),
                        float(benefit_alignment),
                        float(risk_reduction_alignment),
                    ]
                ),
            }
        )

    write_jsonl(config.input_raw_video, raw_video_records)
    write_jsonl(config.input_visual_features, visual_feature_records)
    write_jsonl(config.input_audio_speech_features, audio_speech_records)
    write_jsonl(config.input_text_language_features, text_language_records)
    write_jsonl(config.input_publish_context_features, publish_context_records)
    write_jsonl(config.input_cross_modal_consistency_features, cross_modal_records)

    summary = {
        "record_count": len(processed_records),
        "source_research_workbook": str(config.research_workbook),
        "source_raw_video_dir": str(config.raw_video_dir),
        "input_root": str(config.input_dir),
        "folders": {
            "raw_video": str(config.input_raw_video.parent),
            "visual_features": str(config.input_visual_features.parent),
            "audio_speech": str(config.input_audio_speech_features.parent),
            "text_language": str(config.input_text_language_features.parent),
            "publish_context": str(config.input_publish_context_features.parent),
            "cross_modal_consistency": str(config.input_cross_modal_consistency_features.parent),
        },
        "files": {
            "raw_video": str(config.input_raw_video),
            "visual_features": str(config.input_visual_features),
            "audio_speech": str(config.input_audio_speech_features),
            "text_language": str(config.input_text_language_features),
            "publish_context": str(config.input_publish_context_features),
            "cross_modal_consistency": str(config.input_cross_modal_consistency_features),
        },
        "availability": {
            "video_asset_coverage": sum(int(record["has_video_asset"]) for record in raw_video_records) / len(raw_video_records) if raw_video_records else 0.0,
            "sales_image_coverage": sum(int(record["has_sales_asset"]) for record in raw_video_records) / len(raw_video_records) if raw_video_records else 0.0,
            "video_text_coverage": sum(1 for record in text_language_records if clean_text(record["video_text"])) / len(text_language_records) if text_language_records else 0.0,
            "emotion_naturalness_coverage": sum(1 for record in visual_feature_records if record.get("emotion_naturalness") not in (None, "")) / len(visual_feature_records) if visual_feature_records else 0.0,
        },
        "speech_rate_bucket_thresholds": {
            "q25": low_speech_rate,
            "q75": high_speech_rate,
        },
        "asset_manifest_summary": asset_manifest.get("summary", {}),
    }
    write_json(config.input_summary, summary)
    return summary
