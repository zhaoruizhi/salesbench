"""Evidence-First multi-agent annotation state machine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..multiagent.context import public_observation_context
from ..multiagent.schema import VideoContextBundle
from ..utils import clean_text
from ..vlm.api_client import APICallResult, VLMClient
from .normalizer import normalize_evidence_units, normalize_proposals, semantic_key
from .ontology import eligible_question_formats
from .parsing import (
    ModelOutputError,
    parse_adjudication_response,
    parse_evidence_response,
    parse_proposal_response,
    parse_review_response,
)
from .prompts import (
    build_adjudicator_prompt,
    build_challenger_prompt,
    build_evidence_extractor_prompt,
    build_proposer_prompt,
)
from .schema import (
    SCHEMA_VERSION,
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
from .validators import (
    ValidationIssue,
    find_duplicate_and_conflicting_items,
    validate_evidence_unit,
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


def _empty_result(video_id: str, status: str, traces: list[dict[str, object]]) -> GoldBankResult:
    return GoldBankResult(
        video_id=video_id,
        evidence_units=[],
        gold_proposals=[],
        gold_reviews=[],
        video_gold_record=None,
        human_review_queue=[],
        agent_traces=traces,
        status=status,
    )


def build_bp_proposals_from_evidence(video_id: str, evidence_units: list[EvidenceUnit]) -> list[GoldProposal]:
    proposals: list[GoldProposal] = []
    for idx, unit in enumerate(evidence_units):
        if not unit.subject or unit.value in (None, ""):
            continue
        if unit.modality.value == "asr":
            subtype = "ASR_FACT"
        elif unit.modality.value == "ocr":
            subtype = "OCR_FACT"
        elif clean_text(unit.predicate).lower() in {"count", "数量"}:
            subtype = "COUNT_SPATIAL"
        elif clean_text(unit.predicate).lower() in {"action", "does", "do"}:
            subtype = "ACTION"
        else:
            subtype = "ENTITY_ATTRIBUTE"
        proposal_id = f"{video_id}_local_bp_{idx:03d}"
        proposals.append(
            GoldProposal(
                proposal_id=proposal_id,
                video_id=video_id,
                source_agent="local_bp_builder",
                task_type=GoldTaskType.BP,
                task_subtype=subtype,
                target={"subject": unit.subject, "predicate": unit.predicate},
                proposed_gold={"value": unit.value, "text_span": unit.text_span},
                evidence_ids=(unit.evidence_id,),
                reasoning_edges=((unit.evidence_id, f"{unit.subject}:{unit.predicate}", "SUPPORTED"),),
                proposal_confidence=unit.confidence,
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
                review_status="verified",
                confidence=proposal.proposal_confidence,
            )
        )
    return items


def _review_queue_item(
    video_id: str,
    reason: str,
    proposal: dict[str, object] | None = None,
    issues: list[ValidationIssue] | None = None,
) -> dict[str, object]:
    proposal_id = clean_text((proposal or {}).get("proposal_id")) or "stage"
    return {
        "review_item_id": f"hr_{video_id}_{proposal_id}",
        "video_id": video_id,
        "proposal_id": proposal_id,
        "reason": reason,
        "issues": [issue.to_dict() for issue in issues or []],
        "proposal": proposal or {},
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
    ):
        self.vlm_client = vlm_client
        self.llm_client = llm_client
        self.min_confidence = min_confidence

    def run_video(
        self,
        bundle: VideoContextBundle,
        frames_b64: list[str] | None = None,
    ) -> GoldBankResult:
        video_id = bundle.video_id
        traces: list[dict[str, object]] = []
        content_context = public_observation_context(bundle)

        system, user_blocks = build_evidence_extractor_prompt(video_id, content_context)
        frame_metadata = list(content_context.get("sampled_frames") or [])
        image_blocks: list[dict[str, object]] = []
        for position, image_b64 in enumerate(frames_b64 or []):
            metadata = frame_metadata[position] if position < len(frame_metadata) else {}
            frame_index = metadata.get("frame_index", position)
            timestamp_s = metadata.get("timestamp_s")
            image_blocks.extend(
                [
                    {"type": "text", "text": f"[FRAME frame_index={frame_index} timestamp_s={timestamp_s}]"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]
            )
        user_blocks = [*image_blocks, *user_blocks]
        evidence_call = self.vlm_client.call(system, user_blocks, response_format="json_object")
        if not evidence_call.success:
            traces.append(_trace("evidence_extraction", "objective_evidence_extractor", evidence_call, error=evidence_call.error))
            return _empty_result(video_id, "failed", traces)
        try:
            evidence_raw = parse_evidence_response(evidence_call.raw_response)
            evidence_units = normalize_evidence_units(video_id, evidence_raw)
            valid_evidence_units: list[EvidenceUnit] = []
            for unit in evidence_units:
                issues = validate_evidence_unit(unit)
                if any(issue.severity == "ERROR" for issue in issues):
                    continue
                valid_evidence_units.append(unit)
            evidence_units = valid_evidence_units
            traces.append(_trace("evidence_extraction", "objective_evidence_extractor", evidence_call, {"evidence_units": evidence_raw}))
        except (ModelOutputError, ValueError) as exc:
            traces.append(_trace("evidence_extraction", "objective_evidence_extractor", evidence_call, error=str(exc)))
            return _empty_result(video_id, "failed", traces)

        if not evidence_units:
            return _empty_result(video_id, "failed", traces)

        evidence_dict = {unit.evidence_id: unit for unit in evidence_units}
        bp_proposals = build_bp_proposals_from_evidence(video_id, evidence_units)
        all_proposals = list(bp_proposals)
        used_proposal_ids = {proposal.proposal_id for proposal in bp_proposals}
        human_review_queue: list[dict[str, object]] = []
        status = "ok"

        for perspective, stage in (
            ("consumer", "consumer_proposal"),
            ("operator", "operator_proposal"),
            ("strategist", "strategist_proposal"),
        ):
            system, user = build_proposer_prompt(perspective, video_id, [unit.to_dict() for unit in evidence_units])
            call = self.llm_client.call_text_only(system, user, response_format="json_object")
            if not call.success:
                status = "partial"
                traces.append(_trace(stage, perspective, call, error=call.error))
                continue
            try:
                proposals_raw, abstentions = parse_proposal_response(call.raw_response)
                proposals: list[GoldProposal] = []
                for ordinal, raw_proposal in enumerate(proposals_raw):
                    try:
                        proposal = normalize_proposals(video_id, perspective, [raw_proposal])[0]
                    except (ValueError, IndexError) as exc:
                        status = "partial"
                        human_review_queue.append(
                            _review_queue_item(video_id, f"{perspective}_proposal_parse_error: {exc}", raw_proposal)
                        )
                        continue
                    if proposal.proposal_id in used_proposal_ids:
                        proposal = replace(
                            proposal,
                            proposal_id=(
                                f"{video_id}_{perspective}_{proposal.task_type.value.lower()}_{ordinal:03d}_"
                                f"{stable_digest({'target': proposal.target, 'gold': proposal.proposed_gold})}"
                            ),
                        )
                    used_proposal_ids.add(proposal.proposal_id)
                    proposals.append(proposal)
                all_proposals.extend(proposals)
                for abstention in abstentions:
                    human_review_queue.append(
                        {
                            "review_item_id": f"hr_{video_id}_{perspective}_abstain_{len(human_review_queue):03d}",
                            "video_id": video_id,
                            "proposal_id": "",
                            "reason": "ABSTENTION",
                            "abstention": abstention,
                        }
                    )
                traces.append(_trace(stage, perspective, call, {"proposals": proposals_raw, "abstentions": abstentions}))
            except ModelOutputError as exc:
                status = "partial"
                traces.append(_trace(stage, perspective, call, error=str(exc)))

        proposal_dicts = [proposal.to_dict() for proposal in all_proposals]
        eligible_proposal_dicts: list[dict[str, object]] = []
        for proposal in all_proposals:
            if proposal.proposal_confidence < self.min_confidence:
                human_review_queue.append(_review_queue_item(video_id, "below_min_confidence", proposal.to_dict()))
                continue
            eligible_proposal_dicts.append(proposal.to_dict())
        system, user = build_challenger_prompt(video_id, eligible_proposal_dicts, [unit.to_dict() for unit in evidence_units])
        challenge_call = self.llm_client.call_text_only(system, user, response_format="json_object")
        if not challenge_call.success:
            traces.append(_trace("challenge", "gold_challenger", challenge_call, error=challenge_call.error))
            return self._review_only_result(video_id, evidence_units, proposal_dicts, [], traces, "partial", "challenge_failed")
        try:
            review_raw = parse_review_response(challenge_call.raw_response)
            reviews = [parse_gold_review(review) for review in review_raw]
            traces.append(_trace("challenge", "gold_challenger", challenge_call, {"reviews": review_raw}))
        except (ModelOutputError, ValueError) as exc:
            traces.append(_trace("challenge", "gold_challenger", challenge_call, error=str(exc)))
            return self._review_only_result(video_id, evidence_units, proposal_dicts, [], traces, "partial", "challenge_parse_failed")

        eligible_proposal_ids = {
            clean_text(proposal.get("proposal_id"))
            for proposal in eligible_proposal_dicts
            if clean_text(proposal.get("proposal_id"))
        }
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
        }
        passed_proposals_by_id = {
            proposal.proposal_id: proposal
            for proposal in all_proposals
            if proposal.proposal_id in passed_proposal_ids
        }

        system, user = build_adjudicator_prompt(
            video_id,
            [
                proposal
                for proposal in eligible_proposal_dicts
                if clean_text(proposal.get("proposal_id")) in passed_proposal_ids
                and clean_text(proposal.get("task_type")).upper() != GoldTaskType.BP.value
            ],
            [review.to_dict() for review in reviews],
            [unit.to_dict() for unit in evidence_units],
        )
        adjudication_call = self.llm_client.call_text_only(system, user, response_format="json_object")
        if not adjudication_call.success:
            traces.append(_trace("adjudication", "gold_adjudicator", adjudication_call, error=adjudication_call.error))
            return self._review_only_result(video_id, evidence_units, proposal_dicts, [review.to_dict() for review in reviews], traces, "partial", "adjudication_failed")
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
            return self._review_only_result(video_id, evidence_units, proposal_dicts, [review.to_dict() for review in reviews], traces, "partial", "adjudication_parse_failed")

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
                local_payload.setdefault("review_status", "verified")
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
                adjudicated_source_ids.update(
                    clean_text(value)
                    for value in queue_payload.get("source_proposal_ids", []) or []
                    if clean_text(value)
                )
                queue_payload.setdefault("review_item_id", f"hr_{video_id}_adjudicator_{len(human_review_queue):03d}")
                queue_payload.setdefault("video_id", video_id)
                human_review_queue.append(queue_payload)

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
            issues = validate_gold_item(item, evidence_dict)
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
                "sampled_frame_count": len(frames_b64 or []),
                "known_limitations": [],
            },
        )
        return GoldBankResult(
            video_id=video_id,
            evidence_units=[unit.to_dict() for unit in evidence_units],
            gold_proposals=proposal_dicts,
            gold_reviews=[review.to_dict() for review in reviews],
            video_gold_record=record.to_dict(),
            human_review_queue=human_review_queue,
            agent_traces=traces,
            status=status,
        )

    def _review_only_result(
        self,
        video_id: str,
        evidence_units: list[EvidenceUnit],
        proposals: list[dict[str, object]],
        reviews: list[dict[str, object]],
        traces: list[dict[str, object]],
        status: str,
        reason: str,
    ) -> GoldBankResult:
        queue = [_review_queue_item(video_id, reason, proposal) for proposal in proposals]
        return GoldBankResult(
            video_id=video_id,
            evidence_units=[unit.to_dict() for unit in evidence_units],
            gold_proposals=proposals,
            gold_reviews=reviews,
            video_gold_record=None,
            human_review_queue=queue,
            agent_traces=traces,
            status=status,
        )
