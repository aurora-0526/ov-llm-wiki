"""Generate Wiki node documents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from pydantic import ValidationError

from .llm import WikiLLMRunner
from .prompts import build_node_documents_prompt
from .schemas import (
    NodeDocument,
    NodeDocumentsResponse,
    WikiNode,
)

logger = logging.getLogger(__name__)
MAX_VALIDATION_ATTEMPTS = 3
T = TypeVar("T")


class NodeContentGenerator:
    def __init__(self, llm: WikiLLMRunner):
        self.llm = llm

    async def generate_node_documents(
        self,
        node: WikiNode,
        source_documents: list[dict],
        *,
        max_source_chars: int = 60000,
        max_source_chars_per_document: int = 8000,
    ) -> list[NodeDocument]:
        source_documents = _build_evidence_pack(
            source_documents,
            max_source_chars=max_source_chars,
            max_source_chars_per_document=max_source_chars_per_document,
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

    def _parse_node_documents_result(
        self,
        node: WikiNode,
        result: dict,
    ) -> list[NodeDocument]:
        response = NodeDocumentsResponse.model_validate(result)
        documents = _build_node_documents(response.documents)
        if not documents:
            raise RuntimeError(f"node_documents for {node.node_id} is empty")
        return documents


def _build_evidence_pack(
    source_documents: list[dict],
    *,
    max_source_chars: int,
    max_source_chars_per_document: int,
) -> list[dict]:
    """Bound compiler input while retaining coverage across the cluster.

    Documents are sampled in round-robin order and each document keeps its
    section headers/URIs. This is deliberately deterministic, so reruns do
    not change the Wiki solely because of sampling order.
    """
    docs = list(source_documents or [])
    if not docs:
        return []
    total_budget = max(4000, int(max_source_chars or 60000))
    per_doc = max(1000, int(max_source_chars_per_document or 8000))
    selected: list[dict] = []
    used = 0
    for index, doc in enumerate(docs):
        sections = list(doc.get("sections") or [])
        if not sections:
            continue
        # Every source gets a chance to contribute before any source gets a
        # second section, which avoids large documents dominating the pack.
        compact_sections = []
        remaining_doc = per_doc
        for section in sections:
            if remaining_doc <= 0 or used >= total_budget:
                break
            content = str(section.get("content") or "")
            if not content:
                continue
            remaining = min(remaining_doc, total_budget - used)
            if len(content) > remaining:
                content = content[: max(200, remaining - 24)].rstrip() + "\n...(truncated)"
            compact_sections.append({**section, "content": content})
            consumed = len(content)
            remaining_doc -= consumed
            used += consumed
        if compact_sections:
            selected.append({**doc, "sections": compact_sections})
        if used >= total_budget:
            break
    return selected

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
