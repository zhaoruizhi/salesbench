"""Evidence-First multi-agent annotation state machine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from ..multiagent.context import public_observation_context
from ..multiagent.schema import VideoContextBundle
from ..utils import clean_text, ensure_speaker_attribution
from ..vlm.api_client import APICallResult, VLMClient
from .commerce_ontology import DEMONSTRATION_CUES, relation_requires_temporal_order
from .commerce_schema import ActionRole, CommerceCue, CommercialRelation, CueType
from .normalizer import (
    derive_source_capabilities,
    normalize_commerce_cues,
    normalize_commercial_relations,
    normalize_evidence_unit,
    normalize_proposals,
    semantic_key,
)
from .ontology import default_reasoning_operator, eligible_question_formats
from .parsing import (
    ModelOutputError,
    parse_adjudication_response,
    parse_commerce_cue_response,
    parse_commercial_relation_response,
    parse_evidence_response,
    parse_proposal_response,
    parse_review_response,
)
from .prompts import (
    build_adjudicator_prompt,
    build_challenger_prompt,
    build_commerce_cue_prompt,
    build_commercial_relation_prompt,
    build_evidence_extractor_prompt,
    build_language_evidence_prompt,
    build_language_evidence_repair_prompt,
    build_proposer_prompt,
    build_visual_evidence_prompt,
    build_visual_evidence_repair_prompt,
    build_visual_commerce_cue_prompt,
)
from .quality_gate import LifecycleStatus, route_quality_record
from .quality_prompts import build_proposal_repair_prompt
from .schema import (
    SCHEMA_VERSION,
    EvidenceModality,
    EvidenceUnit,
    GoldItem,
    GoldProposal,
    GoldTaskType,
    GoldTier,
    ReviewVerdict,
    VideoGoldRecord,
    make_gold_id,
    parse_gold_item,
    parse_gold_review,
    stable_digest,
)
from .semantic_verifier import SemanticVerdict, verify_relations
from .validators import (
    ValidationIssue,
    find_duplicate_and_conflicting_items,
    validate_commerce_cue,
    validate_commercial_relation,
    validate_evidence_unit,
    validate_gold_proposal,
    validate_gold_item,
)


@dataclass
class GoldBankResult:
    video_id: str
    evidence_units: list[dict[str, object]]
    gold_proposals: list[dict[str, object]]
    gold_reviews: list[dict[str, object]]
    video_gold_record: dict[str, object] | None
    human_review_queue: list[dict[str, object]]
    agent_traces: list[dict[str, object]]
    status: str
    commerce_cues: list[dict[str, object]] = field(default_factory=list)
    commercial_relations: list[dict[str, object]] = field(default_factory=list)
    repaired_candidates: list[dict[str, object]] = field(default_factory=list)
    rejected_candidates: list[dict[str, object]] = field(default_factory=list)
    pipeline_diagnostics: list[dict[str, object]] = field(default_factory=list)
    quality_decisions: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Partition legacy queue writes into canonical quality artifacts."""

        actionable: list[dict[str, object]] = []
        known_decision_ids = {
            clean_text(record.get("decision_id")) for record in self.quality_decisions
        }
        for raw in self.human_review_queue:
            record = dict(raw)
            decision = route_quality_record(record)
            record["quality_disposition"] = decision.disposition.value
            record["quality_decision_id"] = decision.decision_id
            if decision.decision_id not in known_decision_ids:
                self.quality_decisions.append(decision.to_dict())
                known_decision_ids.add(decision.decision_id)
            if decision.output_channel == "human_review_queue":
                actionable.append(record)
            elif decision.output_channel == "rejected_candidates":
                self.rejected_candidates.append(record)
            else:
                self.pipeline_diagnostics.append(record)
        self.human_review_queue = actionable


def _trace(
    stage: str,
    agent_name: str,
    result: APICallResult,
    parsed_output: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    payload = {
        "stage": stage,
        "agent_name": agent_name,
        "success": result.success and error is None,
        "raw_response": result.raw_response,
        "parsed_output": parsed_output,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": result.latency_s,
        "cost_usd": result.cost_usd,
    }
    if result.error or error:
        payload["error"] = error or result.error
    return payload


def _empty_result(
    video_id: str,
    status: str,
    traces: list[dict[str, object]],
    quality_records: list[dict[str, object]] | None = None,
) -> GoldBankResult:
    return GoldBankResult(
        video_id=video_id,
        evidence_units=[],
        gold_proposals=[],
        gold_reviews=[],
        video_gold_record=None,
        human_review_queue=list(quality_records or []),
        agent_traces=traces,
        status=status,
        commerce_cues=[],
        commercial_relations=[],
    )


_BP_CUE_CAPABILITIES = {
    CueType.PRODUCT_IDENTITY: "PRODUCT_IDENTITY",
    CueType.PRODUCT_ATTRIBUTE: "ATTRIBUTE_AND_VARIANT",
    CueType.PRODUCT_VARIANT: "ATTRIBUTE_AND_VARIANT",
    CueType.QUANTITY: "QUANTITY_AND_BUNDLE",
    CueType.BUNDLE: "QUANTITY_AND_BUNDLE",
    CueType.PRICE: "PRICE_AND_DISCOUNT",
    CueType.DISCOUNT: "PRICE_AND_DISCOUNT",
    CueType.OFFER_CONDITION: "OFFER_CONDITION",
    CueType.PROCESS_DEMONSTRATION: "USAGE_STEP",
    CueType.BEFORE_AFTER: "DEMONSTRATED_STATE_CHANGE",
    CueType.USAGE_SCENARIO: "USAGE_SCENARIO",
}

_BP_VISUAL_REQUIRED_CUES = {
    CueType.PROCESS_DEMONSTRATION,
    CueType.BEFORE_AFTER,
}


def build_bp_proposals_from_graph(
    video_id: str,
    evidence_units: list[EvidenceUnit],
    commerce_cues: list[CommerceCue],
    commercial_relations: list[CommercialRelation],
) -> list[GoldProposal]:
    proposals: list[GoldProposal] = []
    evidence_by_id = {unit.evidence_id: unit for unit in evidence_units}
    for idx, cue in enumerate(commerce_cues):
        if cue.action_role == ActionRole.BACKGROUND_HANDLING:
            continue
        subtype = _BP_CUE_CAPABILITIES.get(cue.cue_type)
        if subtype is None:
            continue
        if not cue.evidence_ids or any(evidence_id not in evidence_by_id for evidence_id in cue.evidence_ids):
            continue
        cue_evidence = [evidence_by_id[evidence_id] for evidence_id in cue.evidence_ids]
        if cue.cue_type in _BP_VISUAL_REQUIRED_CUES and not any(
            unit.modality.value == "visual" for unit in cue_evidence
        ):
            continue
        asr_only = all(unit.modality.value == "asr" for unit in cue_evidence)
        answer = ensure_speaker_attribution(cue.content_en) if asr_only else cue.content_en
        question_intent = (
            f"Ask what the speaker states about {subtype.lower().replace('_', ' ')}."
            if asr_only
            else f"Ask what the video specifically presents about {subtype.lower().replace('_', ' ')}."
        )
        related = [
            relation
            for relation in commercial_relations
            if cue.cue_id in {*relation.source_cue_ids, *relation.target_cue_ids}
        ]
        proposal_id = f"{video_id}_local_bp_{idx:03d}"
        proposals.append(
            GoldProposal(
                proposal_id=proposal_id,
                video_id=video_id,
                source_agent="bp_compiler",
                task_type=GoldTaskType.BP,
                task_subtype=subtype,
                target={"cue_type": cue.cue_type.value, "specific_focus": cue.content_en},
                proposed_gold={"answer": answer},
                evidence_ids=cue.evidence_ids,
                reasoning_edges=tuple(
                    (evidence_id, cue.content_en, "SUPPORTED") for evidence_id in cue.evidence_ids
                ),
                proposal_confidence=cue.confidence,
                capability=subtype,
                reasoning_operator=default_reasoning_operator(GoldTaskType.BP, subtype),
                commerce_cue_ids=(cue.cue_id,),
                commercial_relation_ids=tuple(relation.relation_id for relation in related),
                question_intent=question_intent,
                forbidden_inferences=(
                    "Do not infer sales, interaction, conversion, or unshown product properties.",
                ),
                assertion_scope=cue.assertion_scope,
            )
        )
    return proposals


def _bp_items_from_proposals(proposals: list[GoldProposal]) -> list[GoldItem]:
    items: list[GoldItem] = []
    for proposal in proposals:
        gold_id = make_gold_id(
            proposal.video_id,
            proposal.task_type,
            proposal.task_subtype,
            proposal.target,
            proposal.proposed_gold,
        )
        items.append(
            GoldItem(
                gold_id=gold_id,
                video_id=proposal.video_id,
                task_type=proposal.task_type,
                task_subtype=proposal.task_subtype,
                target=proposal.target,
                gold_value=proposal.proposed_gold,
                evidence_ids=proposal.evidence_ids,
                reasoning_edges=proposal.reasoning_edges,
                eligible_question_formats=eligible_question_formats(proposal.task_type, proposal.task_subtype),
                source_proposal_ids=(proposal.proposal_id,),
                gold_tier=GoldTier.GOLD_A,
                review_status=LifecycleStatus.AUTO_ACCEPTED_CANDIDATE.value,
                confidence=proposal.proposal_confidence,
                capability=proposal.capability,
                reasoning_operator=proposal.reasoning_operator,
                commerce_cue_ids=proposal.commerce_cue_ids,
                commercial_relation_ids=proposal.commercial_relation_ids,
                question_intent=proposal.question_intent,
                forbidden_inferences=proposal.forbidden_inferences,
                assertion_scope=proposal.assertion_scope,
            )
        )
    return items


def _review_queue_item(
    video_id: str,
    reason: str,
    proposal: dict[str, object] | None = None,
    issues: list[ValidationIssue] | None = None,
    *,
    stage: str | None = None,
    item_type: str | None = None,
    source_proposal_ids: list[str] | None = None,
) -> dict[str, object]:
    candidate = proposal or {}
    proposal_id = clean_text(
        candidate.get("proposal_id") or candidate.get("annotation_id") or candidate.get("gold_id")
    ) or "stage"
    source_ids = list(
        dict.fromkeys(
            source_proposal_ids
            or [clean_text(value) for value in candidate.get("source_proposal_ids", []) or [] if clean_text(value)]
            or ([proposal_id] if proposal_id != "stage" and candidate.get("proposal_id") else [])
        )
    )
    reason_code = clean_text(reason).split(":", 1)[0].upper()
    if stage is None:
        if reason_code.startswith("ADJUDICATOR"):
            stage = "adjudication"
        elif reason_code.startswith("CHALLENGE"):
            stage = "challenge"
        elif reason_code in {"VALIDATION_FAILED", "CONFLICTING_VALUE", "SEMANTIC_DUPLICATE"}:
            stage = "validation"
        else:
            stage = "proposal"
    if item_type is None:
        item_type = "conflict" if reason_code in {"CONFLICTING_VALUE", "SEMANTIC_DUPLICATE"} else "candidate"
    target = dict(candidate.get("target") or {})
    candidate_gold = dict(candidate.get("gold_value") or candidate.get("proposed_gold") or {})
    evidence_refs = list(candidate.get("evidence_refs") or candidate.get("evidence_ids") or [])
    issue_payloads = [issue.to_dict() if isinstance(issue, ValidationIssue) else dict(issue) for issue in issues or []]
    digest = stable_digest(
        {
            "video_id": video_id,
            "stage": stage,
            "reason": reason,
            "proposal_id": proposal_id,
            "source_proposal_ids": source_ids,
            "target": target,
        },
        length=10,
    )
    return {
        "review_item_id": f"hr_{video_id}_{stage}_{digest}",
        "video_id": video_id,
        "stage": stage,
        "item_type": item_type,
        "task_type": clean_text(candidate.get("task_type")).upper(),
        "task_subtype": clean_text(candidate.get("task_subtype")).upper(),
        "reason_code": reason_code,
        "reason": reason,
        "target": target,
        "candidate_gold": candidate_gold,
        "evidence_refs": evidence_refs,
        "source_proposal_ids": source_ids,
        "issues": issue_payloads,
        "proposal_id": proposal_id,
        "candidate_snapshot": dict(candidate),
        "proposal": dict(candidate) if candidate.get("proposal_id") else {},
    }


def _abstention_queue_item(
    video_id: str,
    generator: str,
    abstention: dict[str, object],
    ordinal: int,
) -> dict[str, object]:
    reason = clean_text(abstention.get("reason")) or "The generator did not produce a supported candidate."
    digest = stable_digest({"video_id": video_id, "generator": generator, "ordinal": ordinal, "abstention": abstention}, length=10)
    return {
        "review_item_id": f"hr_{video_id}_proposal_{digest}",
        "video_id": video_id,
        "stage": "proposal",
        "item_type": "abstention",
        "task_type": clean_text(abstention.get("task_type")).upper(),
        "task_subtype": clean_text(abstention.get("task_subtype")).upper(),
        "reason_code": clean_text(abstention.get("reason_code")).upper() or "ABSTENTION",
        "reason": reason,
        "target": {},
        "candidate_gold": {},
        "evidence_refs": [],
        "source_proposal_ids": [],
        "issues": [],
        "proposal_id": "",
        "generator": generator,
    }


def _coverage(video_id: str, items: list[GoldItem]) -> dict[str, object]:
    by_task = {task.value: 0 for task in GoldTaskType}
    for item in items:
        by_task[item.task_type.value] += 1
    coverage = {}
    for task, count in by_task.items():
        if count > 0:
            status = "covered"
        else:
            status = "insufficient_evidence"
        coverage[task] = {"eligible_count": count, "status": status}
    return coverage


def _quality(items: list[GoldItem], queue: list[dict[str, object]]) -> dict[str, int]:
    return {
        "direct_count": sum(1 for item in items if item.gold_tier == GoldTier.GOLD_A),
        "inferred_count": sum(1 for item in items if item.gold_tier == GoldTier.GOLD_B),
        "needs_review_count": len(queue),
        "rejected_count": sum(1 for item in queue if "REJECT" in str(item.get("reason")).upper()),
    }


def _locally_assign_quality(item: GoldItem) -> GoldItem:
    """Ignore model-assigned tiers and apply the benchmark's deterministic rule."""
    tier = GoldTier.GOLD_A if item.task_type in {GoldTaskType.BP, GoldTaskType.CM} else GoldTier.GOLD_B
    return replace(item, gold_tier=tier)


class GoldBankPipeline:
    def __init__(
        self,
        vlm_client: VLMClient,
        llm_client: VLMClient,
        min_confidence: float = 0.70,
        strict_semantic_verification: bool = False,
        semantic_verifier_client: VLMClient | None = None,
    ):
        self.vlm_client = vlm_client
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.strict_semantic_verification = strict_semantic_verification
        self.semantic_verifier_client = semantic_verifier_client or vlm_client

    def run_video(
        self,
        bundle: VideoContextBundle,
        frames_b64: list[str] | None = None,
        frame_urls: list[str] | None = None,
    ) -> GoldBankResult:
        if frames_b64 and frame_urls:
            raise ValueError("frames_b64 and frame_urls are mutually exclusive")
        video_id = bundle.video_id
        traces: list[dict[str, object]] = []
        content_context = public_observation_context(bundle)
        frame_metadata = list(content_context.get("sampled_frames") or [])
        image_blocks: list[dict[str, object]] = []
        frame_images: dict[int, str] = {}
        frame_references = list(frame_urls or [])
        if not frame_references:
            frame_references = [
                f"data:image/jpeg;base64,{image_b64}" for image_b64 in frames_b64 or []
            ]
        for position, image_reference in enumerate(frame_references):
            metadata = frame_metadata[position] if position < len(frame_metadata) else {}
            frame_index = metadata.get("frame_index", position)
            timestamp_s = metadata.get("timestamp_s")
            frame_images[int(frame_index)] = image_reference
            image_blocks.extend(
                [
                    {"type": "text", "text": f"[FRAME frame_index={frame_index} timestamp_s={timestamp_s}]"},
                    {"type": "image_url", "image_url": {"url": image_reference}},
                ]
            )

        evidence_units: list[EvidenceUnit] = []
        evidence_stage_failed = False
        evidence_quality_records: list[dict[str, object]] = []

        def collect_evidence(
            stage: str,
            agent_name: str,
            call: APICallResult,
            allowed_modalities: set[EvidenceModality],
        ) -> tuple[list[EvidenceUnit], list[dict[str, object]], list[dict[str, object]]]:
            if not call.success:
                traces.append(_trace(stage, agent_name, call, error=call.error))
                evidence_quality_records.append(
                    _review_queue_item(
                        video_id,
                        f"{stage}_failed: {call.error or 'model call failed'}",
                        stage=stage,
                        item_type="stage_failure",
                    )
                )
                return [], [], []
            try:
                raw_units = parse_evidence_response(call.raw_response)
                validation_issues: list[dict[str, object]] = []
                accepted: list[EvidenceUnit] = []
                for ordinal, raw_unit in enumerate(raw_units):
                    try:
                        unit = normalize_evidence_unit(video_id, raw_unit, ordinal)
                    except ValueError as exc:
                        issue = ValidationIssue(
                            "EVIDENCE_NORMALIZATION_FAILED",
                            "ERROR",
                            clean_text(raw_unit.get("evidence_id")) or f"{stage}_{ordinal}",
                            str(exc),
                        )
                        validation_issues.append(issue.to_dict())
                        evidence_quality_records.append(
                            _review_queue_item(
                                video_id,
                                f"evidence_validation_failed: {exc}",
                                raw_unit,
                                [issue],
                                stage=stage,
                                item_type="evidence_unit",
                            )
                        )
                        continue
                    issues = validate_evidence_unit(unit)
                    if unit.modality not in allowed_modalities:
                        issues.append(
                            ValidationIssue(
                                "WRONG_EVIDENCE_STAGE_MODALITY",
                                "ERROR",
                                unit.evidence_id,
                                f"{stage} cannot emit {unit.modality.value}",
                            )
                        )
                    if unit.confidence < self.min_confidence:
                        issues.append(
                            ValidationIssue(
                                "EVIDENCE_BELOW_MIN_CONFIDENCE",
                                "ERROR",
                                unit.evidence_id,
                                f"Evidence confidence {unit.confidence:.3f} is below {self.min_confidence:.3f}",
                            )
                        )
                    validation_issues.extend(issue.to_dict() for issue in issues)
                    if any(issue.severity == "ERROR" for issue in issues):
                        reason = (
                            "evidence_below_min_confidence"
                            if any(issue.code == "EVIDENCE_BELOW_MIN_CONFIDENCE" for issue in issues)
                            else "evidence_validation_failed"
                        )
                        evidence_quality_records.append(
                            _review_queue_item(
                                video_id,
                                reason,
                                unit.to_dict(),
                                issues,
                                stage=stage,
                                item_type="evidence_unit",
                            )
                        )
                        continue
                    accepted.append(unit)
                evidence_units.extend(accepted)
                traces.append(
                    _trace(
                        stage,
                        agent_name,
                        call,
                        {
                            "evidence_units": raw_units,
                            "accepted_evidence_units": [unit.to_dict() for unit in accepted],
                            "validation_issues": validation_issues,
                        },
                    )
                )
                return accepted, raw_units, validation_issues
            except (ModelOutputError, ValueError) as exc:
                traces.append(_trace(stage, agent_name, call, error=str(exc)))
                evidence_quality_records.append(
                    _review_queue_item(
                        video_id,
                        f"{stage}_parse_error: {exc}",
                        stage=stage,
                        item_type="stage_failure",
                    )
                )
                return [], [], []

        asr_subtitles = content_context.get("asr_subtitles") or {}
        if asr_subtitles:
            language_system, language_user = build_language_evidence_prompt(video_id, content_context)
            language_call = self.llm_client.call_text_only(
                language_system,
                language_user,
                response_format="json_object",
            )
            language_accepted, rejected_language, language_validation_issues = collect_evidence(
                "language_evidence_extraction",
                "asr_evidence_extractor",
                language_call,
                {EvidenceModality.ASR},
            )
            if language_call.success and not any(
                unit.modality == EvidenceModality.ASR for unit in language_accepted
            ):
                repair_system, repair_user = build_language_evidence_repair_prompt(
                    video_id,
                    content_context,
                    rejected_language,
                    language_validation_issues,
                )
                repair_call = self.llm_client.call_text_only(
                    repair_system,
                    repair_user,
                    response_format="json_object",
                )
                repaired, _, _ = collect_evidence(
                    "language_evidence_repair",
                    "asr_evidence_repairer",
                    repair_call,
                    {EvidenceModality.ASR},
                )
                if not any(unit.modality == EvidenceModality.ASR for unit in repaired):
                    evidence_stage_failed = True
            elif not language_call.success:
                evidence_stage_failed = True

        if image_blocks:
            visual_system, visual_user = build_visual_evidence_prompt(video_id, content_context)
            visual_call = self.vlm_client.call(
                visual_system,
                [*image_blocks, *visual_user],
                response_format="json_object",
            )
            visual_accepted, rejected_visual, visual_validation_issues = collect_evidence(
                "visual_evidence_extraction",
                "visual_ocr_evidence_extractor",
                visual_call,
                {EvidenceModality.VISUAL, EvidenceModality.OCR},
            )
            if visual_call.success and not any(
                unit.modality == EvidenceModality.VISUAL for unit in visual_accepted
            ):
                repair_system, repair_user = build_visual_evidence_repair_prompt(
                    video_id,
                    content_context,
                    rejected_visual,
                    visual_validation_issues,
                )
                repair_call = self.vlm_client.call(
                    repair_system,
                    [*image_blocks, *repair_user],
                    response_format="json_object",
                )
                repaired, _, _ = collect_evidence(
                    "visual_evidence_repair",
                    "visual_evidence_repairer",
                    repair_call,
                    {EvidenceModality.VISUAL},
                )
                if not any(unit.modality == EvidenceModality.VISUAL for unit in repaired):
                    evidence_stage_failed = True
            elif not visual_call.success:
                evidence_stage_failed = True

        if not asr_subtitles and not image_blocks:
            system, user_blocks = build_evidence_extractor_prompt(video_id, content_context)
            evidence_call = self.vlm_client.call(system, user_blocks, response_format="json_object")
            fallback_accepted, _, _ = collect_evidence(
                "evidence_extraction",
                "objective_evidence_extractor",
                evidence_call,
                {EvidenceModality.VISUAL, EvidenceModality.OCR, EvidenceModality.ASR},
            )
            if not fallback_accepted:
                evidence_stage_failed = True

        evidence_units = list({unit.evidence_id: unit for unit in evidence_units}.values())

        if not evidence_units:
            return _empty_result(video_id, "failed", traces, evidence_quality_records)

        evidence_dict = {unit.evidence_id: unit for unit in evidence_units}
        human_review_queue: list[dict[str, object]] = list(evidence_quality_records)
        status = "partial" if evidence_stage_failed else "ok"

        cue_system, cue_user = build_commerce_cue_prompt(
            video_id,
            [unit.to_dict() for unit in evidence_units],
        )
        cue_call = self.llm_client.call_text_only(cue_system, cue_user, response_format="json_object")
        if not cue_call.success:
            traces.append(
                _trace(
                    "commerce_cue_extraction",
                    "commerce_cue_extractor",
                    cue_call,
                    error=cue_call.error,
                )
            )
            return GoldBankResult(
                video_id=video_id,
                evidence_units=[unit.to_dict() for unit in evidence_units],
                gold_proposals=[],
                gold_reviews=[],
                video_gold_record=None,
                human_review_queue=[],
                agent_traces=traces,
                status="partial",
            )
        try:
            cue_raw, cue_abstentions = parse_commerce_cue_response(cue_call.raw_response)
            commerce_cues = []
            for raw_cue in cue_raw:
                try:
                    cue = normalize_commerce_cues(video_id, [raw_cue], evidence_dict)[0]
                except (ValueError, IndexError) as exc:
                    status = "partial"
                    human_review_queue.append(
                        _review_queue_item(
                            video_id,
                            f"commerce_cue_parse_error: {exc}",
                            raw_cue,
                            stage="commerce_cue_extraction",
                            item_type="commerce_cue",
                        )
                    )
                    continue
                issues = validate_commerce_cue(cue, evidence_dict)
                if any(issue.severity == "ERROR" for issue in issues):
                    status = "partial"
                    human_review_queue.append(
                        _review_queue_item(
                            video_id,
                            "commerce_cue_validation_failed",
                            cue.to_dict(),
                            issues,
                            stage="commerce_cue_extraction",
                            item_type="commerce_cue",
                        )
                    )
                    continue
                if cue.confidence < self.min_confidence:
                    human_review_queue.append(
                        _review_queue_item(
                            video_id,
                            "commerce_cue_below_min_confidence",
                            cue.to_dict(),
                            stage="commerce_cue_extraction",
                            item_type="commerce_cue",
                        )
                    )
                    continue
                commerce_cues.append(cue)
            for ordinal, abstention in enumerate(cue_abstentions):
                human_review_queue.append(
                    _abstention_queue_item(video_id, "commerce_cue_extractor", abstention, ordinal)
                )
            traces.append(
                _trace(
                    "commerce_cue_extraction",
                    "commerce_cue_extractor",
                    cue_call,
                    {"commerce_cues": cue_raw, "abstentions": cue_abstentions},
                )
            )
        except (ModelOutputError, ValueError) as exc:
            traces.append(
                _trace(
                    "commerce_cue_extraction",
                    "commerce_cue_extractor",
                    cue_call,
                    error=str(exc),
                )
            )
            return GoldBankResult(
                video_id=video_id,
                evidence_units=[unit.to_dict() for unit in evidence_units],
                gold_proposals=[],
                gold_reviews=[],
                video_gold_record=None,
                human_review_queue=human_review_queue,
                agent_traces=traces,
                status="partial",
            )

        visual_evidence_ids = {
            unit.evidence_id for unit in evidence_units if unit.modality == EvidenceModality.VISUAL
        }
        visual_cue_types = {
            CueType.PRODUCT_IDENTITY,
            CueType.PRODUCT_ATTRIBUTE,
            CueType.PRODUCT_VARIANT,
            CueType.USAGE_SCENARIO,
            CueType.COMPARISON_ANCHOR,
            CueType.CREDIBILITY_SIGNAL,
            *DEMONSTRATION_CUES,
        }
        has_visual_commerce_cue = any(
            cue.cue_type in visual_cue_types and visual_evidence_ids.intersection(cue.evidence_ids)
            for cue in commerce_cues
        )
        if visual_evidence_ids and not has_visual_commerce_cue:
            visual_cue_system, visual_cue_user = build_visual_commerce_cue_prompt(
                video_id,
                [
                    unit.to_dict()
                    for unit in evidence_units
                    if unit.modality in {EvidenceModality.VISUAL, EvidenceModality.OCR}
                ],
            )
            visual_cue_call = self.llm_client.call_text_only(
                visual_cue_system,
                visual_cue_user,
                response_format="json_object",
            )
            if not visual_cue_call.success:
                status = "partial"
                traces.append(
                    _trace(
                        "visual_commerce_cue_repair",
                        "visual_commerce_cue_extractor",
                        visual_cue_call,
                        error=visual_cue_call.error,
                    )
                )
            else:
                try:
                    visual_cue_raw, visual_cue_abstentions = parse_commerce_cue_response(
                        visual_cue_call.raw_response
                    )
                    used_cue_ids = {cue.cue_id for cue in commerce_cues}
                    for raw_cue in visual_cue_raw:
                        try:
                            cue = normalize_commerce_cues(video_id, [raw_cue], evidence_dict)[0]
                        except (ValueError, IndexError) as exc:
                            status = "partial"
                            human_review_queue.append(
                                _review_queue_item(
                                    video_id,
                                    f"visual_commerce_cue_parse_error: {exc}",
                                    raw_cue,
                                    stage="visual_commerce_cue_repair",
                                    item_type="commerce_cue",
                                )
                            )
                            continue
                        issues = validate_commerce_cue(cue, evidence_dict)
                        if any(issue.severity == "ERROR" for issue in issues):
                            status = "partial"
                            human_review_queue.append(
                                _review_queue_item(
                                    video_id,
                                    "visual_commerce_cue_validation_failed",
                                    cue.to_dict(),
                                    issues,
                                    stage="visual_commerce_cue_repair",
                                    item_type="commerce_cue",
                                )
                            )
                            continue
                        if cue.confidence < self.min_confidence or cue.cue_id in used_cue_ids:
                            continue
                        used_cue_ids.add(cue.cue_id)
                        commerce_cues.append(cue)
                    for ordinal, abstention in enumerate(visual_cue_abstentions):
                        human_review_queue.append(
                            _abstention_queue_item(
                                video_id,
                                "visual_commerce_cue_extractor",
                                abstention,
                                ordinal,
                            )
                        )
                    traces.append(
                        _trace(
                            "visual_commerce_cue_repair",
                            "visual_commerce_cue_extractor",
                            visual_cue_call,
                            {
                                "commerce_cues": visual_cue_raw,
                                "abstentions": visual_cue_abstentions,
                            },
                        )
                    )
                except (ModelOutputError, ValueError) as exc:
                    status = "partial"
                    traces.append(
                        _trace(
                            "visual_commerce_cue_repair",
                            "visual_commerce_cue_extractor",
                            visual_cue_call,
                            error=str(exc),
                        )
                    )

        if not commerce_cues:
            return GoldBankResult(
                video_id=video_id,
                evidence_units=[unit.to_dict() for unit in evidence_units],
                gold_proposals=[],
                gold_reviews=[],
                video_gold_record=None,
                human_review_queue=human_review_queue,
                agent_traces=traces,
                status="partial",
            )

        cue_dict = {cue.cue_id: cue for cue in commerce_cues}
        source_capabilities = derive_source_capabilities(evidence_units)
        relation_system, relation_user = build_commercial_relation_prompt(
            video_id,
            [unit.to_dict() for unit in evidence_units],
            [cue.to_dict() for cue in commerce_cues],
            source_capabilities,
        )
        relation_call = self.llm_client.call_text_only(
            relation_system,
            relation_user,
            response_format="json_object",
        )
        relation_response = relation_call.raw_response
        if not relation_call.success:
            status = "partial"
            traces.append(
                _trace(
                    "commercial_relation_building",
                    "commercial_relation_builder",
                    relation_call,
                    error=relation_call.error,
                )
            )
            relation_response = '{"commercial_relations":[],"abstentions":[]}'
        try:
            relation_raw, relation_abstentions = parse_commercial_relation_response(
                relation_response
            )
            commercial_relations = []
            for raw_relation in relation_raw:
                try:
                    relation = normalize_commercial_relations(
                        video_id,
                        [raw_relation],
                        cue_dict,
                        evidence_dict,
                    )[0]
                except (ValueError, IndexError) as exc:
                    status = "partial"
                    human_review_queue.append(
                        _review_queue_item(
                            video_id,
                            f"commercial_relation_parse_error: {exc}",
                            raw_relation,
                            stage="commercial_relation_building",
                            item_type="commercial_relation",
                        )
                    )
                    continue
                if relation_requires_temporal_order(relation.relation_type) and not source_capabilities[
                    "asr_temporal_order"
                ]:
                    human_review_queue.append(
                        _abstention_queue_item(
                            video_id,
                            "commercial_relation_builder",
                            {
                                "reason_code": "TEMPORAL_CAPABILITY_UNAVAILABLE",
                                "reason": "Source ASR has no timestamps, so this temporal relation is intentionally unavailable.",
                                "task_type": "CM",
                                "task_subtype": relation.relation_type.value,
                            },
                            len(relation_abstentions),
                        )
                    )
                    continue
                issues = validate_commercial_relation(relation, cue_dict, evidence_dict)
                if any(issue.severity == "ERROR" for issue in issues):
                    status = "partial"
                    human_review_queue.append(
                        _review_queue_item(
                            video_id,
                            "commercial_relation_validation_failed",
                            relation.to_dict(),
                            issues,
                            stage="commercial_relation_building",
                            item_type="commercial_relation",
                        )
                    )
                    continue
                if relation.confidence < self.min_confidence:
                    human_review_queue.append(
                        _review_queue_item(
                            video_id,
                            "commercial_relation_below_min_confidence",
                            relation.to_dict(),
                            stage="commercial_relation_building",
                            item_type="commercial_relation",
                        )
                    )
                    continue
                commercial_relations.append(relation)
            for ordinal, abstention in enumerate(relation_abstentions):
                human_review_queue.append(
                    _abstention_queue_item(video_id, "commercial_relation_builder", abstention, ordinal)
                )
            if relation_call.success:
                traces.append(
                    _trace(
                        "commercial_relation_building",
                        "commercial_relation_builder",
                        relation_call,
                        {
                            "commercial_relations": relation_raw,
                            "abstentions": relation_abstentions,
                        },
                    )
                )
        except (ModelOutputError, ValueError) as exc:
            status = "partial"
            commercial_relations = []
            traces.append(
                _trace(
                    "commercial_relation_building",
                    "commercial_relation_builder",
                    relation_call,
                    error=str(exc),
                )
            )

        if self.strict_semantic_verification and commercial_relations:
            try:
                verification_batch = verify_relations(
                    self.semantic_verifier_client,
                    video_id,
                    commercial_relations,
                    cue_dict,
                    evidence_dict,
                    frame_images,
                )
                traces.append(
                    _trace(
                        "semantic_relation_verification",
                        "frame_aware_relation_verifier",
                        verification_batch.call,
                        {
                            "verifications": [
                                verification.to_dict()
                                for verification in verification_batch.verifications
                            ]
                        },
                    )
                )
                relation_by_id = {
                    relation.relation_id: relation for relation in commercial_relations
                }
                verified_relations: list[CommercialRelation] = []
                for verification in verification_batch.verifications:
                    relation = relation_by_id[verification.relation_id]
                    if verification.verdict == SemanticVerdict.PASS:
                        verified_relations.append(relation)
                        continue
                    reason = (
                        "semantic_verifier_ambiguous"
                        if verification.verdict == SemanticVerdict.AMBIGUOUS
                        else "semantic_verifier_reject"
                    )
                    snapshot = relation.to_dict()
                    snapshot["semantic_verifier"] = verification.to_dict()
                    human_review_queue.append(
                        _review_queue_item(
                            video_id,
                            reason,
                            snapshot,
                            stage="semantic_relation_verification",
                            item_type="commercial_relation",
                        )
                    )
                commercial_relations = verified_relations
                relation_dict = {
                    relation.relation_id: relation for relation in commercial_relations
                }
            except ValueError as exc:
                status = "partial"
                human_review_queue.append(
                    _review_queue_item(
                        video_id,
                        f"semantic_relation_verification_failed: {exc}",
                        stage="semantic_relation_verification",
                        item_type="stage_failure",
                    )
                )
                commercial_relations = []
                relation_dict = {}

        bp_proposals = build_bp_proposals_from_graph(
            video_id,
            evidence_units,
            commerce_cues,
            commercial_relations,
        )
        all_proposals = list(bp_proposals)
        used_proposal_ids = {proposal.proposal_id for proposal in bp_proposals}

        for generator, stage in (
            ("cm_proposer", "task_proposal"),
            ("ss_proposer", "task_proposal"),
            ("ae_proposer", "task_proposal"),
        ):
            system, user = build_proposer_prompt(
                generator,
                video_id,
                [unit.to_dict() for unit in evidence_units],
                [cue.to_dict() for cue in commerce_cues],
                [relation.to_dict() for relation in commercial_relations],
            )
            call = self.llm_client.call_text_only(system, user, response_format="json_object")
            if not call.success:
                status = "partial"
                traces.append(_trace(stage, generator, call, error=call.error))
                continue
            try:
                proposals_raw, abstentions = parse_proposal_response(call.raw_response)
                proposals: list[GoldProposal] = []
                for ordinal, raw_proposal in enumerate(proposals_raw):
                    try:
                        proposal = normalize_proposals(
                            video_id,
                            generator,
                            [raw_proposal],
                            cue_dict,
                            {relation.relation_id: relation for relation in commercial_relations},
                            set(evidence_dict),
                        )[0]
                    except (ValueError, IndexError) as exc:
                        status = "partial"
                        human_review_queue.append(
                            _review_queue_item(video_id, f"{generator}_proposal_parse_error: {exc}", raw_proposal)
                        )
                        continue
                    if proposal.proposal_id in used_proposal_ids:
                        proposal = replace(
                            proposal,
                            proposal_id=(
                                f"{video_id}_{generator}_{proposal.task_type.value.lower()}_{ordinal:03d}_"
                                f"{stable_digest({'target': proposal.target, 'gold': proposal.proposed_gold})}"
                            ),
                        )
                    used_proposal_ids.add(proposal.proposal_id)
                    proposals.append(proposal)
                all_proposals.extend(proposals)
                for abstention_ordinal, abstention in enumerate(abstentions):
                    human_review_queue.append(
                        _abstention_queue_item(video_id, generator, abstention, abstention_ordinal)
                    )
                traces.append(_trace(stage, generator, call, {"proposals": proposals_raw, "abstentions": abstentions}))
            except ModelOutputError as exc:
                status = "partial"
                traces.append(_trace(stage, generator, call, error=str(exc)))

        proposal_dicts = [proposal.to_dict() for proposal in all_proposals]
        eligible_proposal_dicts: list[dict[str, object]] = []
        relation_dict = {relation.relation_id: relation for relation in commercial_relations}
        for proposal in all_proposals:
            if proposal.proposal_confidence < self.min_confidence:
                human_review_queue.append(_review_queue_item(video_id, "below_min_confidence", proposal.to_dict()))
                continue
            proposal_issues = validate_gold_proposal(
                proposal,
                evidence_dict,
                cue_dict,
                relation_dict,
            )
            if any(issue.severity == "ERROR" for issue in proposal_issues):
                human_review_queue.append(
                    _review_queue_item(
                        video_id,
                        "proposal_graph_validation_failed",
                        proposal.to_dict(),
                        proposal_issues,
                    )
                )
                continue
            eligible_proposal_dicts.append(proposal.to_dict())
        system, user = build_challenger_prompt(
            video_id,
            eligible_proposal_dicts,
            [unit.to_dict() for unit in evidence_units],
            [cue.to_dict() for cue in commerce_cues],
            [relation.to_dict() for relation in commercial_relations],
        )
        challenge_call = self.llm_client.call_text_only(system, user, response_format="json_object")
        if not challenge_call.success:
            traces.append(_trace("challenge", "gold_challenger", challenge_call, error=challenge_call.error))
            return self._review_only_result(
                video_id, evidence_units, commerce_cues, commercial_relations,
                proposal_dicts, [], traces, "partial", "challenge_failed"
            )
        try:
            review_raw = parse_review_response(challenge_call.raw_response)
            reviews = [parse_gold_review(review) for review in review_raw]
            traces.append(_trace("challenge", "gold_challenger", challenge_call, {"reviews": review_raw}))
        except (ModelOutputError, ValueError) as exc:
            traces.append(_trace("challenge", "gold_challenger", challenge_call, error=str(exc)))
            return self._review_only_result(
                video_id, evidence_units, commerce_cues, commercial_relations,
                proposal_dicts, [], traces, "partial", "challenge_parse_failed"
            )

        proposals_by_id = {proposal.proposal_id: proposal for proposal in all_proposals}
        eligible_proposal_ids = {
            clean_text(proposal.get("proposal_id"))
            for proposal in eligible_proposal_dicts
            if clean_text(proposal.get("proposal_id"))
        }
        repaired_proposal_ids: set[str] = set()
        repair_attempted_ids: set[str] = set()
        repaired_candidates: list[dict[str, object]] = []
        for review in reviews:
            if review.verdict == ReviewVerdict.PASS:
                continue
            proposal = proposals_by_id.get(review.proposal_id)
            if (
                review.verdict == ReviewVerdict.REVISE
                and proposal is not None
                and proposal.proposal_id in eligible_proposal_ids
                and proposal.source_agent != "bp_compiler"
                and proposal.proposal_id not in repair_attempted_ids
            ):
                repair_attempted_ids.add(proposal.proposal_id)
                repair_system, repair_user = build_proposal_repair_prompt(
                    video_id,
                    proposal.to_dict(),
                    review.to_dict(),
                    [unit.to_dict() for unit in evidence_units],
                    [cue.to_dict() for cue in commerce_cues],
                    [relation.to_dict() for relation in commercial_relations],
                )
                repair_call = self.llm_client.call_text_only(
                    repair_system,
                    repair_user,
                    response_format="json_object",
                )
                repaired_proposal: GoldProposal | None = None
                repair_error = repair_call.error or ""
                if repair_call.success:
                    try:
                        repair_payload = json.loads(repair_call.raw_response)
                        raw_repaired = (
                            repair_payload.get("repaired_proposal")
                            if isinstance(repair_payload, dict)
                            else None
                        )
                        if not isinstance(raw_repaired, dict):
                            raise ValueError("repairer returned no repaired_proposal")
                        for identity_field, expected in {
                            "proposal_id": proposal.proposal_id,
                            "video_id": proposal.video_id,
                            "source_agent": proposal.source_agent,
                            "task_type": proposal.task_type.value,
                        }.items():
                            if clean_text(raw_repaired.get(identity_field)) != clean_text(expected):
                                raise ValueError(f"repairer changed {identity_field}")
                        supplied_evidence_ids = {
                            clean_text(value)
                            for value in raw_repaired.get("evidence_ids", []) or []
                            if clean_text(value)
                        }
                        supplied_cue_ids = {
                            clean_text(value)
                            for value in raw_repaired.get("commerce_cue_ids", []) or []
                            if clean_text(value)
                        }
                        supplied_relation_ids = {
                            clean_text(value)
                            for value in raw_repaired.get("commercial_relation_ids", []) or []
                            if clean_text(value)
                        }
                        if not supplied_evidence_ids <= set(evidence_dict):
                            raise ValueError("repairer invented an EvidenceUnit ID")
                        if not supplied_cue_ids <= set(cue_dict):
                            raise ValueError("repairer invented a CommerceCue ID")
                        if not supplied_relation_ids <= set(relation_dict):
                            raise ValueError("repairer invented a CommercialRelation ID")
                        repaired_proposal = normalize_proposals(
                            video_id,
                            proposal.source_agent,
                            [raw_repaired],
                            cue_dict,
                            relation_dict,
                            set(evidence_dict),
                        )[0]
                        repaired_proposal = replace(
                            repaired_proposal,
                            proposal_id=proposal.proposal_id,
                        )
                        repair_issues = validate_gold_proposal(
                            repaired_proposal,
                            evidence_dict,
                            cue_dict,
                            relation_dict,
                        )
                        if repaired_proposal.proposal_confidence < self.min_confidence:
                            raise ValueError("repaired proposal is below min_confidence")
                        if any(issue.severity == "ERROR" for issue in repair_issues):
                            codes = ", ".join(issue.code for issue in repair_issues)
                            raise ValueError(f"repaired proposal failed validation: {codes}")
                    except (json.JSONDecodeError, ValueError, IndexError) as exc:
                        repair_error = str(exc)
                        repaired_proposal = None
                traces.append(
                    _trace(
                        "proposal_repair",
                        "proposal_repairer",
                        repair_call,
                        (
                            {"repaired_proposal": repaired_proposal.to_dict()}
                            if repaired_proposal is not None
                            else None
                        ),
                        error=repair_error or None,
                    )
                )
                if repaired_proposal is not None:
                    proposals_by_id[proposal.proposal_id] = repaired_proposal
                    repaired_proposal_ids.add(proposal.proposal_id)
                    repair_record = _review_queue_item(
                        video_id,
                        "challenger_revise_repaired",
                        repaired_proposal.to_dict(),
                        stage="proposal_repair",
                        item_type="candidate",
                        source_proposal_ids=[proposal.proposal_id],
                    )
                    repair_record["challenger_review"] = review.to_dict()
                    repair_record["original_candidate"] = proposal.to_dict()
                    repaired_candidates.append(repair_record)
                    continue
                human_review_queue.append(
                    _review_queue_item(
                        video_id,
                        f"challenger_repair_exhausted: {repair_error or 'invalid repair'}",
                        proposal.to_dict(),
                        stage="proposal_repair",
                        item_type="candidate",
                        source_proposal_ids=[proposal.proposal_id],
                    )
                )
                continue
            human_review_queue.append(
                _review_queue_item(
                    video_id,
                    f"challenger_{review.verdict.value.lower()}",
                    proposal.to_dict() if proposal else {"proposal_id": review.proposal_id},
                    stage="challenge",
                    item_type="candidate",
                    source_proposal_ids=[review.proposal_id] if review.proposal_id else [],
                )
            )

        rejected_proposal_ids = {
            review.proposal_id
            for review in reviews
            if review.proposal_id in eligible_proposal_ids
            and review.verdict in {ReviewVerdict.REJECT, ReviewVerdict.HUMAN_REVIEW}
        }
        passed_proposal_ids = {
            review.proposal_id
            for review in reviews
            if review.proposal_id in eligible_proposal_ids
            and review.verdict == ReviewVerdict.PASS
        } | repaired_proposal_ids
        passed_proposals_by_id = {
            proposal_id: proposal
            for proposal_id, proposal in proposals_by_id.items()
            if proposal_id in passed_proposal_ids
        }

        adjudicator_proposals = [
            proposal.to_dict()
            for proposal_id, proposal in passed_proposals_by_id.items()
            if proposal_id in passed_proposal_ids
            and proposal.task_type != GoldTaskType.BP
        ]
        adjudicator_proposal_ids = {
            clean_text(proposal.get("proposal_id")) for proposal in adjudicator_proposals
        }
        adjudicator_reviews = []
        for review in reviews:
            if review.proposal_id not in adjudicator_proposal_ids:
                continue
            payload = review.to_dict()
            if review.proposal_id in repaired_proposal_ids:
                payload.update(
                    {
                        "verdict": ReviewVerdict.PASS.value,
                        "issues": [],
                        "suggested_revision": None,
                        "checks": {
                            **dict(payload.get("checks") or {}),
                            "automatic_repair_validated": True,
                        },
                    }
                )
            adjudicator_reviews.append(payload)
        system, user = build_adjudicator_prompt(
            video_id,
            adjudicator_proposals,
            adjudicator_reviews,
            [unit.to_dict() for unit in evidence_units],
            [cue.to_dict() for cue in commerce_cues],
            [relation.to_dict() for relation in commercial_relations],
        )
        adjudication_call = self.llm_client.call_text_only(system, user, response_format="json_object")
        if not adjudication_call.success:
            traces.append(_trace("adjudication", "gold_adjudicator", adjudication_call, error=adjudication_call.error))
            return self._review_only_result(
                video_id, evidence_units, commerce_cues, commercial_relations,
                proposal_dicts, [review.to_dict() for review in reviews], traces, "partial", "adjudication_failed"
            )
        try:
            gold_raw, adjudicator_queue = parse_adjudication_response(adjudication_call.raw_response)
            decision_only = any(bool(item.get("_decision_only")) for item in gold_raw)
            parsed_key = "accepted_groups" if decision_only else "grounded_annotations"
            traces.append(
                _trace(
                    "adjudication",
                    "gold_adjudicator",
                    adjudication_call,
                    {parsed_key: gold_raw, "human_review_queue": adjudicator_queue},
                )
            )
        except (ModelOutputError, ValueError) as exc:
            traces.append(_trace("adjudication", "gold_adjudicator", adjudication_call, error=str(exc)))
            return self._review_only_result(
                video_id, evidence_units, commerce_cues, commercial_relations,
                proposal_dicts, [review.to_dict() for review in reviews], traces, "partial", "adjudication_parse_failed"
            )

        accepted_items = _bp_items_from_proposals(
            [proposal for proposal in bp_proposals if proposal.proposal_id in passed_proposal_ids]
        )
        adjudicated_source_ids: set[str] = set()
        for raw_item in gold_raw:
            try:
                local_payload = dict(raw_item)
                decision_only = bool(local_payload.pop("_decision_only", False))
                local_payload.pop("gold_tier", None)
                local_payload.pop("quality_status", None)
                source_ids = {
                    clean_text(value)
                    for value in local_payload.get("source_proposal_ids", []) or []
                    if clean_text(value)
                }
                annotation_id = clean_text(local_payload.get("annotation_id") or local_payload.get("gold_id"))
                if not source_ids and annotation_id in passed_proposal_ids:
                    source_ids = {annotation_id}
                adjudicated_source_ids.update(source_ids)
                source_proposals = [passed_proposals_by_id[source_id] for source_id in source_ids if source_id in passed_proposals_by_id]
                if decision_only and (not source_ids or len(source_proposals) != len(source_ids)):
                    human_review_queue.append(_review_queue_item(video_id, "adjudicator_unknown_source", raw_item))
                    continue
                if source_proposals and all(proposal.task_type == GoldTaskType.BP for proposal in source_proposals):
                    continue
                if decision_only and len(source_proposals) > 1:
                    semantic_keys = {semantic_key(proposal) for proposal in source_proposals}
                    if len(semantic_keys) != 1:
                        human_review_queue.append(
                            _review_queue_item(video_id, "adjudicator_group_not_semantically_equivalent", raw_item)
                        )
                        continue
                if len(source_proposals) == 1 or (decision_only and source_proposals):
                    source = sorted(source_proposals, key=lambda proposal: proposal.proposal_id)[0]
                    local_payload.update(
                        {
                            "video_id": video_id,
                            "task_type": source.task_type.value,
                            "task_subtype": source.task_subtype,
                            "target": dict(source.target),
                            "gold_value": dict(source.proposed_gold),
                            "evidence_refs": list(source.evidence_ids),
                            "reasoning_edges": [list(edge) for edge in source.reasoning_edges],
                            "eligible_question_formats": list(eligible_question_formats(source.task_type, source.task_subtype)),
                            "source_proposal_ids": sorted(source_ids),
                            "confidence": source.proposal_confidence,
                            "capability": source.capability,
                            "reasoning_operator": source.reasoning_operator,
                            "commerce_cue_ids": list(source.commerce_cue_ids),
                            "commercial_relation_ids": list(source.commercial_relation_ids),
                            "question_intent": source.question_intent,
                            "forbidden_inferences": list(source.forbidden_inferences),
                            "annotation_id": make_gold_id(
                                video_id,
                                source.task_type,
                                source.task_subtype,
                                source.target,
                                source.proposed_gold,
                            ),
                        }
                    )
                task_type = GoldTaskType(clean_text(local_payload.get("task_type")).upper())
                local_payload["quality_status"] = "DIRECT" if task_type in {GoldTaskType.BP, GoldTaskType.CM} else "INFERRED"
                local_payload["review_status"] = (
                    LifecycleStatus.AUTO_ACCEPTED_CANDIDATE.value
                )
                item = parse_gold_item(local_payload)
            except ValueError as exc:
                human_review_queue.append(_review_queue_item(video_id, f"parse_error: {exc}", raw_item))
                continue
            source_ids = set(item.source_proposal_ids)
            if not source_ids or not source_ids <= passed_proposal_ids or source_ids & rejected_proposal_ids:
                human_review_queue.append(_review_queue_item(video_id, "challenger_rejected_or_human_review", raw_item))
                continue
            accepted_items.append(_locally_assign_quality(item))

        for queue_item in adjudicator_queue:
            if isinstance(queue_item, dict):
                queue_payload = dict(queue_item)
                queue_source_ids = [
                    clean_text(value)
                    for value in queue_payload.get("source_proposal_ids", []) or []
                    if clean_text(value)
                ]
                adjudicated_source_ids.update(queue_source_ids)
                source_candidate = next(
                    (passed_proposals_by_id[source_id].to_dict() for source_id in queue_source_ids if source_id in passed_proposals_by_id),
                    {},
                )
                human_review_queue.append(
                    _review_queue_item(
                        video_id,
                        clean_text(queue_payload.get("reason")) or "adjudicator_human_review",
                        source_candidate,
                        stage="adjudication",
                        item_type="conflict" if len(queue_source_ids) > 1 else "candidate",
                        source_proposal_ids=queue_source_ids,
                    )
                )

        for proposal_id in sorted(passed_proposal_ids - adjudicated_source_ids):
            proposal = passed_proposals_by_id[proposal_id]
            if proposal.task_type != GoldTaskType.BP:
                human_review_queue.append(_review_queue_item(video_id, "adjudicator_omitted", proposal.to_dict()))

        validated_items: list[GoldItem] = []
        for item in accepted_items:
            item = _locally_assign_quality(item)
            if item.confidence < self.min_confidence:
                human_review_queue.append(_review_queue_item(video_id, "below_min_confidence", item.to_dict()))
                continue
            issues = validate_gold_item(item, evidence_dict, cue_dict, relation_dict)
            errors = [issue for issue in issues if issue.severity == "ERROR"]
            if errors or item.gold_tier not in {GoldTier.GOLD_A, GoldTier.GOLD_B}:
                human_review_queue.append(_review_queue_item(video_id, "validation_failed", item.to_dict(), issues))
            else:
                validated_items.append(item)

        unique_items: list[GoldItem] = []
        seen_semantics: set[tuple[str, str, str, str, str]] = set()
        for item in validated_items:
            key = semantic_key(item)
            if key in seen_semantics:
                human_review_queue.append(_review_queue_item(video_id, "semantic_duplicate", item.to_dict()))
                continue
            seen_semantics.add(key)
            unique_items.append(item)
        validated_items = unique_items

        duplicate_issues = find_duplicate_and_conflicting_items(validated_items)
        if duplicate_issues:
            conflict_ids = {issue.item_id for issue in duplicate_issues if issue.code == "CONFLICTING_VALUE"}
            duplicate_ids = {issue.item_id for issue in duplicate_issues if issue.code == "DUPLICATE_SEMANTICS"}
            filtered: list[GoldItem] = []
            for item in validated_items:
                if item.gold_id in conflict_ids:
                    human_review_queue.append(_review_queue_item(video_id, "conflicting_value", item.to_dict(), duplicate_issues))
                elif item.gold_id in duplicate_ids:
                    human_review_queue.append(_review_queue_item(video_id, "semantic_duplicate", item.to_dict(), duplicate_issues))
                else:
                    filtered.append(item)
            validated_items = filtered

        record = VideoGoldRecord(
            video_id=video_id,
            schema_version=SCHEMA_VERSION,
            evidence_unit_ids=tuple(unit.evidence_id for unit in evidence_units),
            gold_items=tuple(sorted(validated_items, key=lambda item: item.gold_id)),
            private_interaction_ref="",
            coverage=_coverage(video_id, validated_items),
            quality_summary=_quality(validated_items, human_review_queue),
            observation_scope={
                "frame_strategy": "hook_plus_uniform",
                "sampled_frame_count": len(frame_references),
                "source_capabilities": source_capabilities,
                "known_limitations": (
                    ["ASR temporal order is unavailable; temporal tasks are disabled."]
                    if not source_capabilities["asr_temporal_order"]
                    and source_capabilities["spoken_claim_extraction"]
                    else []
                ),
            },
            commerce_cue_ids=tuple(cue.cue_id for cue in commerce_cues),
            commercial_relation_ids=tuple(
                relation.relation_id for relation in commercial_relations
            ),
        )
        return GoldBankResult(
            video_id=video_id,
            evidence_units=[unit.to_dict() for unit in evidence_units],
            gold_proposals=[
                proposals_by_id.get(proposal.proposal_id, proposal).to_dict()
                for proposal in all_proposals
            ],
            gold_reviews=[review.to_dict() for review in reviews],
            video_gold_record=record.to_dict(),
            human_review_queue=human_review_queue,
            agent_traces=traces,
            status=status,
            commerce_cues=[cue.to_dict() for cue in commerce_cues],
            commercial_relations=[relation.to_dict() for relation in commercial_relations],
            repaired_candidates=repaired_candidates,
        )

    def _review_only_result(
        self,
        video_id: str,
        evidence_units: list[EvidenceUnit],
        commerce_cues: list,
        commercial_relations: list,
        proposals: list[dict[str, object]],
        reviews: list[dict[str, object]],
        traces: list[dict[str, object]],
        status: str,
        reason: str,
    ) -> GoldBankResult:
        stage = "adjudication" if reason.startswith("adjudication") else "challenge"
        queue = [
            _review_queue_item(video_id, reason, proposal, stage=stage, item_type="stage_failure")
            for proposal in proposals
        ]
        return GoldBankResult(
            video_id=video_id,
            evidence_units=[unit.to_dict() for unit in evidence_units],
            gold_proposals=proposals,
            gold_reviews=reviews,
            video_gold_record=None,
            human_review_queue=queue,
            agent_traces=traces,
            status=status,
            commerce_cues=[cue.to_dict() for cue in commerce_cues],
            commercial_relations=[relation.to_dict() for relation in commercial_relations],
        )
