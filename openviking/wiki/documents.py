"""Steps 5 and 6: generate node.md and node documents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from pydantic import ValidationError

from .llm import WikiLLMRunner
from .prompts import (
    build_node_documents_prompt,
    build_node_md_prompt,
    build_parent_node_documents_prompt,
)
from .schemas import (
    NodeDocument,
    NodeDocumentsResponse,
    NodeMarkdownResponse,
    WikiNode,
)

logger = logging.getLogger(__name__)
MAX_VALIDATION_ATTEMPTS = 3
T = TypeVar("T")


class NodeContentGenerator:
    def __init__(self, llm: WikiLLMRunner):
        self.llm = llm

    async def generate_node_md(self, node: WikiNode) -> str:
        return await _complete_with_validation_retry(
            self.llm,
            step="node_md",
            prompt=build_node_md_prompt(node),
            schema=NodeMarkdownResponse.model_json_schema(),
            node_id=node.node_id,
            validate=lambda result: self._parse_node_md_result(node, result),
        )

    def _parse_node_md_result(self, node: WikiNode, result: dict) -> str:
        response = NodeMarkdownResponse.model_validate(result)
        node_md = response.node_md.strip()
        if not node_md:
            raise RuntimeError(f"node_md for {node.node_id} is empty")
        return node_md

    async def generate_node_documents(
        self,
        node: WikiNode,
        source_documents: list[dict],
        *,
        max_input_chars: int = 60000,
        max_chars_per_source: int = 8000,
        microcluster_chars: int = 18000,
        max_evidence_cards: int = 8,
    ) -> list[NodeDocument]:
        if _source_char_count(source_documents) > max_input_chars:
            source_documents = await self._evidence_cards(
                node, source_documents, max_input_chars, max_chars_per_source, microcluster_chars, max_evidence_cards
            )
        source_documents = _pack_source_documents(
            source_documents,
            max_input_chars=max_input_chars,
            max_chars_per_source=max_chars_per_source,
        )
        prompt = build_node_documents_prompt(
            node,
            source_documents,
        )
        return await _complete_with_validation_retry(
            self.llm,
            step="node_documents",
            prompt=prompt,
            schema=NodeDocumentsResponse.model_json_schema(),
            node_id=node.node_id,
            validate=lambda result: self._parse_node_documents_result(
                node,
                result,
            ),
        )

    async def _evidence_cards(
        self, node: WikiNode, sources: list[dict], max_input_chars: int, max_chars_per_source: int,
        microcluster_chars: int, max_evidence_cards: int
    ) -> list[dict]:
        """Map large topics into topic-hinted, URI-preserving evidence cards."""
        groups: dict[str, list[dict]] = {}
        for source in sources:
            hints = sorted({str(hint).strip() for hint in (source.get("topic_hints") or []) if str(hint).strip()})
            key = " | ".join(hints[:2]) or "other evidence"
            groups.setdefault(key, []).append(source)
        cards: list[dict] = []
        card_limit = max(1, int(max_evidence_cards))
        ordered_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        selected_groups = [[hint, list(group)] for hint, group in ordered_groups[:card_limit]]
        # Preserve evidence coverage when a topic has more subgroups than cards:
        # distribute smaller residual groups instead of silently dropping them.
        for index, (_, group) in enumerate(ordered_groups[card_limit:]):
            selected_groups[index % len(selected_groups)][1].extend(group)
        per_group = max(4000, min(int(microcluster_chars), max_input_chars // max(1, len(selected_groups))))
        for index, (hint, group) in enumerate(selected_groups):
            packed = _pack_source_documents(group, max_input_chars=per_group, max_chars_per_source=max_chars_per_source)
            evidence_node = node.model_copy(update={"scope": f"Evidence card for subtopic {hint}. {node.scope}"})
            evidence = await _complete_with_validation_retry(
                self.llm, step="node_evidence_card", prompt=build_node_documents_prompt(evidence_node, packed),
                schema=NodeDocumentsResponse.model_json_schema(), node_id=node.node_id,
                validate=lambda result: self._parse_node_documents_result(node, result),
            )
            cards.append({"doc_id": f"evidence_card_{index}", "topic_hints": [hint], "sections": [
                {"section_uri": f"evidence://{node.node_id}/{index}", "content": doc.content} for doc in evidence
            ]})
        return cards or sources

    async def generate_parent_node_documents(
        self,
        node: WikiNode,
        child_nodes: list[dict],
        *,
        max_input_chars: int = 60000,
        max_chars_per_source: int = 8000,
    ) -> list[NodeDocument]:
        child_nodes = _pack_child_node_documents(
            child_nodes,
            max_input_chars=max_input_chars,
            max_chars_per_source=max_chars_per_source,
        )
        prompt = build_parent_node_documents_prompt(
            node,
            child_nodes,
        )
        return await _complete_with_validation_retry(
            self.llm,
            step="parent_node_documents",
            prompt=prompt,
            schema=NodeDocumentsResponse.model_json_schema(),
            node_id=node.node_id,
            validate=lambda result: self._parse_parent_node_documents_result(
                node,
                result,
            ),
        )

    def _parse_node_documents_result(
        self,
        node: WikiNode,
        result: dict,
    ) -> list[NodeDocument]:
        result = _normalize_node_documents_result(result)
        response = NodeDocumentsResponse.model_validate(result)
        documents = _build_node_documents(response.documents)
        if not documents:
            raise RuntimeError(f"node_documents for {node.node_id} is empty")
        return documents

    def _parse_parent_node_documents_result(
        self,
        node: WikiNode,
        result: dict,
    ) -> list[NodeDocument]:
        result = _normalize_node_documents_result(result)
        response = NodeDocumentsResponse.model_validate(result)
        documents = _build_node_documents(response.documents)
        if not documents:
            raise RuntimeError(f"node_documents for {node.node_id} is empty")
        return documents


def _normalize_node_documents_result(result: dict) -> dict:
    """Accept legacy LLM output where documents are plain content strings."""
    if not isinstance(result, dict):
        return result
    raw_documents = result.get("documents")
    if not isinstance(raw_documents, list):
        return result
    normalized = dict(result)
    normalized["documents"] = [
        {"content": item} if isinstance(item, str) else item
        for item in raw_documents
    ]
    return normalized


def _clip_evidence(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    if budget < 80:
        return text[:budget]
    return text[: budget - 16].rstrip() + "\n...(truncated)"


def _source_char_count(sources: list[dict]) -> int:
    return sum(len(str(section.get("content", ""))) for source in sources for section in source.get("sections", []))


def _pack_source_documents(
    source_documents: list[dict], *, max_input_chars: int, max_chars_per_source: int
) -> list[dict]:
    """Preserve broad source coverage while bounding a leaf-topic compile."""
    total_budget = max(4000, int(max_input_chars))
    per_source_budget = max(1000, int(max_chars_per_source))
    packed: list[dict] = []
    used = 0
    for source in source_documents:
        remaining = min(per_source_budget, total_budget - used)
        if remaining <= 0:
            break
        sections: list[dict] = []
        for section in source.get("sections", []):
            if remaining <= 0:
                break
            content = str(section.get("content", "")).strip()
            if not content:
                continue
            clipped = _clip_evidence(content, remaining)
            sections.append({**section, "content": clipped})
            used += len(clipped)
            remaining -= len(clipped)
        if sections:
            packed.append({**source, "sections": sections})
    return packed


def _pack_child_node_documents(
    child_nodes: list[dict], *, max_input_chars: int, max_chars_per_source: int
) -> list[dict]:
    """Bound parent compilation without letting a verbose child dominate."""
    total_budget = max(4000, int(max_input_chars))
    per_source_budget = max(1000, int(max_chars_per_source))
    packed: list[dict] = []
    used = 0
    for child in child_nodes:
        remaining = min(per_source_budget, total_budget - used)
        if remaining <= 0:
            break
        documents: list[dict] = []
        for document in child.get("documents", []):
            if remaining <= 0:
                break
            content = str(document.get("content", "")).strip()
            if not content:
                continue
            clipped = _clip_evidence(content, remaining)
            documents.append({**document, "content": clipped})
            used += len(clipped)
            remaining -= len(clipped)
        if documents:
            packed.append({**child, "documents": documents})
    return packed


def _build_node_documents(document_contents: list) -> list[NodeDocument]:
    return [
        NodeDocument.model_validate(
            {
                **document.model_dump(mode="json"),
                "document_id": f"{index:04d}",
            }
        )
        for index, document in enumerate(document_contents, start=1)
    ]


async def _complete_with_validation_retry(
    llm: WikiLLMRunner,
    *,
    step: str,
    prompt: str,
    schema: dict,
    node_id: str,
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
                "[Wiki] Retrying %s for node_id=%s after validation failure attempt=%d/%d",
                step,
                node_id,
                attempt,
                MAX_VALIDATION_ATTEMPTS,
            )
    assert last_error is not None
    raise last_error
