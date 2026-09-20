"""Step 3: discover current-layer Wiki nodes."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from pydantic import ValidationError

from .config import WikiConfig
from .candidate_graph import (
    adaptive_stable_min_refs,
    cluster_cards_with_diagnostics,
    topic_label,
)
from .llm import WikiLLMRunner
from .prompts import (
    build_bottom_node_discovery_prompt,
    build_parent_node_discovery_prompt,
)
from .schemas import (
    DocumentCard,
    GeneratedNodeContext,
    SourceAssignmentItem,
    SourceAssignmentResponse,
    WikiBottomNodeDiscoveryResponse,
    WikiNode,
    WikiNodeDiscoveryItem,
    WikiParentNodeDiscoveryResponse,
)
from .uri import sanitize_node_id

logger = logging.getLogger(__name__)
MAX_VALIDATION_ATTEMPTS = 3
T = TypeVar("T")


@dataclass(frozen=True)
class BottomLayerDiscoveryResult:
    nodes: list[WikiNode]
    source_assignments: SourceAssignmentResponse


@dataclass(frozen=True)
class ParentLayerDiscoveryResult:
    nodes: list[WikiNode]
    source_assignments: SourceAssignmentResponse


class NodeDiscoveryRunner:
    def __init__(self, llm: WikiLLMRunner, config: WikiConfig):
        self.llm = llm
        self.config = config

    async def discover_bottom_layer(
        self,
        cards: list[DocumentCard],
        depth: int = 1,
    ) -> BottomLayerDiscoveryResult:
        if self.config.limits.use_candidate_graph_clustering:
            return self._discover_bottom_layer_from_candidate_graph(cards, depth)
        batch_size = max(1, int(self.config.limits.max_cards_per_discovery_batch))
        if len(cards) <= batch_size:
            return await self._discover_bottom_layer_batch(cards, depth)
        merged: dict[str, WikiNodeDiscoveryItem] = {}
        assignments: dict[str, set[str]] = {}
        unassigned: set[str] = set()
        for start in range(0, len(cards), batch_size):
            result = await self._discover_bottom_layer_batch(cards[start:start + batch_size], depth)
            for node, assignment in zip(result.nodes, result.source_assignments.assignments, strict=False):
                key = _canonical_topic_key(node.title)
                if key not in merged:
                    merged[key] = WikiNodeDiscoveryItem(title=node.title, scope=node.scope)
                elif len(node.scope) > len(merged[key].scope):
                    merged[key] = WikiNodeDiscoveryItem(title=merged[key].title, scope=node.scope)
                assignments.setdefault(key, set()).update(assignment.source_ids)
            unassigned.update(result.source_assignments.unassigned_source_ids)
        nodes = self._build_nodes(list(merged.values()), depth)
        return BottomLayerDiscoveryResult(
            nodes=nodes,
            source_assignments=SourceAssignmentResponse(assignments=[SourceAssignmentItem(node_id=n.node_id, source_ids=sorted(assignments.get(k, set())), support_scope=n.scope) for k, n in zip(merged, nodes, strict=False) if assignments.get(k)], unassigned_source_ids=sorted(unassigned)),
        )

    def _discover_bottom_layer_from_candidate_graph(
        self, cards: list[DocumentCard], depth: int
    ) -> BottomLayerDiscoveryResult:
        """Create full-coverage primary clusters before optional LLM refinement.

        Small components remain leaf topics instead of being dropped. The
        resulting source assignments are also the retrieval-time reverse map.
        """
        by_id = {card.doc_id: card for card in cards}
        graph_result = cluster_cards_with_diagnostics(
            cards,
            candidate_k=self.config.limits.candidate_graph_k,
            edge_threshold=self.config.limits.candidate_graph_edge_threshold,
        )
        groups = graph_result.groups
        stable_min_refs = max(
            1,
            adaptive_stable_min_refs(len(cards)),
            int(self.config.limits.min_refs_per_node) if self.config.limits.min_refs_per_node > 0 else 1,
        )
        if len(cards) <= 80:
            stable_min_refs = min(stable_min_refs, 2)
        logger.info(
            "[Wiki] Candidate graph: docs=%d candidate_k=%d edge_threshold=%.4f stable_min_refs=%d",
            len(cards), graph_result.candidate_k, graph_result.edge_threshold, stable_min_refs,
        )
        used_ids: set[str] = set()
        nodes: list[WikiNode] = []
        assignments: list[SourceAssignmentItem] = []
        for index, doc_ids in enumerate(groups, start=1):
            group_cards = [by_id[doc_id] for doc_id in doc_ids if doc_id in by_id]
            if not group_cards:
                continue
            title, scope = topic_label(group_cards)
            node_id = sanitize_node_id(title)
            suffix = 2
            existing = {node.node_id for node in nodes}
            while node_id in existing:
                node_id = f"{sanitize_node_id(title)}_{suffix}"
                suffix += 1
            cohesion = graph_result.group_cohesion.get("|".join(doc_ids), 0.0)
            kind = "stable_topic" if len(group_cards) >= stable_min_refs and cohesion > 0 else "leaf_topic"
            node = WikiNode(node_id=node_id, title=title, depth=depth, scope=scope, node_kind=kind)
            nodes.append(node)
            assignments.append(SourceAssignmentItem(node_id=node_id, source_ids=[card.doc_id for card in group_cards], support_scope=scope))
            used_ids.update(card.doc_id for card in group_cards)
        # Defensive full coverage if a future graph implementation omits docs.
        for card in cards:
            if card.doc_id in used_ids:
                continue
            node_id = sanitize_node_id(f"leaf_{card.doc_id}")
            scope = f"Leaf document topic for {card.title}; retained for retrieval coverage."
            nodes.append(WikiNode(node_id=node_id, title=card.title, depth=depth, scope=scope, node_kind="outlier"))
            assignments.append(SourceAssignmentItem(node_id=node_id, source_ids=[card.doc_id], support_scope=scope))
        return BottomLayerDiscoveryResult(
            nodes=nodes,
            source_assignments=SourceAssignmentResponse(assignments=assignments, unassigned_source_ids=[]),
        )

    async def _discover_bottom_layer_batch(self, cards: list[DocumentCard], depth: int) -> BottomLayerDiscoveryResult:
        prompt = build_bottom_node_discovery_prompt(cards, min_refs_per_node=self.config.limits.min_refs_per_node)
        return await _complete_with_validation_retry(
            self.llm, step="bottom_node_discovery", prompt=prompt,
            schema=WikiBottomNodeDiscoveryResponse.model_json_schema(),
            validate=lambda result: self._parse_bottom_layer_result(result, depth),
        )

    def _parse_bottom_layer_result(
        self,
        result: dict,
        depth: int,
    ) -> BottomLayerDiscoveryResult:
        response = WikiBottomNodeDiscoveryResponse.model_validate(result)
        nodes = self._build_nodes(response.nodes, depth)
        assignments = [
            SourceAssignmentItem(
                node_id=node.node_id,
                source_ids=item.supporting_doc_ids,
                support_scope=node.scope,
            )
            for node, item in zip(nodes, response.nodes, strict=False)
        ]
        return BottomLayerDiscoveryResult(
            nodes=nodes,
            source_assignments=SourceAssignmentResponse(
                assignments=assignments,
                unassigned_source_ids=response.unassigned_doc_ids,
            ),
        )

    async def discover_parent_layer(
        self,
        child_nodes: list[GeneratedNodeContext],
        depth: int,
    ) -> ParentLayerDiscoveryResult:
        title_to_node_id = _child_title_to_node_id(child_nodes)
        prompt = build_parent_node_discovery_prompt(
            child_nodes,
            min_child_nodes_per_parent=self.config.limits.min_child_nodes_per_parent,
        )
        return await _complete_with_validation_retry(
            self.llm,
            step="parent_node_discovery",
            prompt=prompt,
            schema=WikiParentNodeDiscoveryResponse.model_json_schema(),
            validate=lambda result: self._parse_parent_layer_result(
                result,
                depth,
                title_to_node_id,
            ),
        )

    def _parse_parent_layer_result(
        self,
        result: dict,
        depth: int,
        title_to_node_id: dict[str, str],
    ) -> ParentLayerDiscoveryResult:
        response = WikiParentNodeDiscoveryResponse.model_validate(result)
        nodes = self._build_nodes(response.nodes, depth)
        assignments = [
            SourceAssignmentItem(
                node_id=node.node_id,
                source_ids=_map_child_titles(item.supporting_child_titles, title_to_node_id, node.title),
                support_scope=node.scope,
            )
            for node, item in zip(nodes, response.nodes, strict=False)
        ]
        unassigned_source_ids = _map_child_titles(
            response.unassigned_child_titles,
            title_to_node_id,
            "unassigned_child_titles",
        )
        return ParentLayerDiscoveryResult(
            nodes=nodes,
            source_assignments=SourceAssignmentResponse(
                assignments=assignments,
                unassigned_source_ids=unassigned_source_ids,
            ),
        )

    def _build_nodes(
        self,
        discovered_nodes: list[WikiNodeDiscoveryItem],
        depth: int,
    ) -> list[WikiNode]:
        used_ids: set[str] = set()
        nodes: list[WikiNode] = []
        for discovered in discovered_nodes:
            base_id = sanitize_node_id(discovered.title)
            node_id = base_id
            suffix = 2
            while node_id in used_ids:
                node_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(node_id)
            nodes.append(
                WikiNode(
                    node_id=node_id,
                    title=discovered.title,
                    depth=depth,
                    scope=discovered.scope,
                )
            )
        return nodes


def _canonical_topic_key(title: str) -> str:
    """Reduce superficial batch-specific naming differences before LLM consolidation."""
    text = re.sub(r"[^a-z0-9\s]", " ", title.casefold())
    stop = {"the", "and", "of", "for", "in", "with", "based", "methods", "method", "approaches"}
    return " ".join(sorted({token for token in text.split() if token not in stop}))


def _child_title_to_node_id(child_nodes: list[GeneratedNodeContext]) -> dict[str, str]:
    title_to_node_id: dict[str, str] = {}
    for context in child_nodes:
        title = context.node.title
        if title in title_to_node_id:
            raise ValueError(f"duplicate child node title for parent discovery: {title}")
        title_to_node_id[title] = context.node.node_id
    return title_to_node_id


def _map_child_titles(
    titles: list[str],
    title_to_node_id: dict[str, str],
    field_name: str,
) -> list[str]:
    unknown_titles = [title for title in titles if title not in title_to_node_id]
    if unknown_titles:
        raise RuntimeError(f"{field_name} references unknown child titles: {unknown_titles}")
    return list(dict.fromkeys(title_to_node_id[title] for title in titles))



async def _complete_with_validation_retry(
    llm: WikiLLMRunner,
    *,
    step: str,
    prompt: str,
    schema: dict,
    validate: Callable[[dict], T],
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        try:
            result = await llm.complete_json(
                step=step,
                prompt=prompt,
                schema=schema,
            )
            return validate(result)
        except (RuntimeError, ValidationError) as exc:
            last_error = exc
            if attempt == MAX_VALIDATION_ATTEMPTS:
                break
            logger.info(
                "[Wiki] Retrying %s after validation failure attempt=%d/%d",
                step,
                attempt,
                MAX_VALIDATION_ATTEMPTS,
            )
    assert last_error is not None
    raise last_error
