from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from salesbench.io_utils import read_jsonl, write_jsonl  # noqa: E402
from salesbench.vlm.api_client import APICallResult  # noqa: E402
from salesbench.vqa.realizer import (  # noqa: E402
    QUESTION_REALIZER_PROMPT_VERSION,
    run_qa_realizer,
    validate_realized_question,
)


class FakeClient:
    model = "gpt-4o"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def call_text_only(self, system_prompt: str, user_text: str, response_format: str | None = None):
        self.calls.append({"system": system_prompt, "user": user_text})
        spec_id = json.loads(user_text)["question_spec"]["spec_id"]
        return APICallResult(
            raw_response=json.dumps(
                {
                    "spec_id": spec_id,
                    "question": "What does the wiping demonstration show, and what claimed effect remains unverified?",
                }
            ),
            model=self.model,
            input_tokens=10,
            output_tokens=10,
            latency_s=0.01,
            cost_usd=0.0,
            success=True,
        )


def _write_evidence_dir(root: Path) -> None:
    write_jsonl(
        root / "video_evidence_dataset.jsonl",
        [
            {
                "video_id": "v1",
                "schema_version": "evidence-dataset-schema-v3",
                "grounded_annotations": [
                    {
                        "annotation_id": "a1",
                        "video_id": "v1",
                        "task_type": "CM",
                        "task_subtype": "CLAIM_DEMONSTRATION_STATUS",
                        "capability": "CLAIM_DEMONSTRATION_STATUS",
                        "reasoning_operator": "CLASSIFY_CLAIM_SUPPORT",
                        "target": {"claim": "the cleaner removes the visible mark"},
                        "gold_value": {
                            "answer": "The wiping demonstration shows the mark becoming lighter, but it does not establish long-term removal."
                        },
                        "evidence_refs": ["e1", "e2"],
                        "commerce_cue_ids": ["c1", "c2"],
                        "commercial_relation_ids": ["r1"],
                        "question_intent": "Ask what the demonstration shows and what remains unverified.",
                        "forbidden_inferences": ["Do not treat the long-term claim as demonstrated."],
                        "quality_status": "DIRECT",
                        "review_status": "verified",
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    )
    write_jsonl(
        root / "evidence_units.jsonl",
        [
            {
                "evidence_id": "e1",
                "video_id": "v1",
                "modality": "asr",
                "content_en": "The speaker claims that the cleaner removes the mark.",
                "source_text_native": "这个可以把污渍去掉",
            },
            {
                "evidence_id": "e2",
                "video_id": "v1",
                "modality": "visual",
                "content_en": "The mark becomes lighter after wiping.",
                "source_text_native": "",
            },
        ],
    )
    write_jsonl(
        root / "commerce_cues.jsonl",
        [
            {"cue_id": "c1", "video_id": "v1", "cue_type": "EFFECT_CLAIM", "content_en": "The speaker claims stain removal.", "evidence_ids": ["e1"]},
            {"cue_id": "c2", "video_id": "v1", "cue_type": "PROCESS_DEMONSTRATION", "content_en": "The host wipes the mark.", "evidence_ids": ["e2"]},
        ],
    )
    write_jsonl(
        root / "commercial_relations.jsonl",
        [
            {
                "relation_id": "r1",
                "video_id": "v1",
                "relation_type": "CLAIM_PARTIALLY_SUPPORTED",
                "rationale_en": "The visible change supports only the immediate part of the claim.",
                "source_cue_ids": ["c1"],
                "target_cue_ids": ["c2"],
                "evidence_ids": ["e1", "e2"],
            }
        ],
    )


def test_realizer_writes_specific_english_question_and_resumes(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    output_dir = tmp_path / "qa"
    _write_evidence_dir(evidence_dir)
    client = FakeClient()

    summary = run_qa_realizer(evidence_dir, output_dir, client, dataset_filename="video_evidence_dataset.jsonl")
    second = run_qa_realizer(evidence_dir, output_dir, client, dataset_filename="video_evidence_dataset.jsonl")
    realized = read_jsonl(output_dir / "qa_realizations.jsonl")
    prompt_payload = json.dumps(client.calls, ensure_ascii=False)

    assert summary["counts"]["realized"] == 1
    assert second["counts"]["resumed"] == 1
    assert len(client.calls) == 1
    assert realized[0]["question"].startswith("What does the wiping demonstration")
    assert "这个可以" not in prompt_payload
    assert "likes" not in prompt_payload
    assert QUESTION_REALIZER_PROMPT_VERSION == "question-realizer-prompt-v1"


def test_realizer_rejects_formulaic_or_non_english_questions():
    assert "FORMULAIC_QUESTION" in validate_realized_question(
        "What mechanism is used? Support the answer with evidence.", "An answer."
    )
    assert "NON_ENGLISH_QUESTION" in validate_realized_question("视频展示了什么？", "An answer.")
