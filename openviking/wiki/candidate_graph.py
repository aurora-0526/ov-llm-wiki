"""Dependency-free, data-adaptive candidate graph clustering for document cards."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .schemas import DocumentCard

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_+-]{1,}")
_STOP = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from", "in", "into", "is",
    "method", "methods", "model", "models", "of", "on", "or", "study", "studies", "the", "to",
    "using", "via", "with",
}


@dataclass(frozen=True)
class CandidateGraphResult:
    groups: list[list[str]]
    candidate_k: int
    edge_threshold: float
    group_cohesion: dict[str, float]


def cluster_cards(cards: list[DocumentCard], *, candidate_k: int = 0, edge_threshold: float = 0.0) -> list[list[str]]:
    """Backward-compatible access to adaptive graph communities."""
    return cluster_cards_with_diagnostics(cards, candidate_k=candidate_k, edge_threshold=edge_threshold).groups


def cluster_cards_with_diagnostics(
    cards: list[DocumentCard], *, candidate_k: int = 0, edge_threshold: float = 0.0
) -> CandidateGraphResult:
    """Build a sparse graph from corpus-relative candidate and edge budgets.

    Explicit positive values remain opt-in overrides. Zero selects a corpus-relative
    value, so a small focused collection and a broad large collection do not share
    a hidden ScholarQA-specific threshold.
    """
    if not cards:
        return CandidateGraphResult([], 0, 0.0, {})
    ids = [card.doc_id for card in cards]
    views = {card.doc_id: _views(card) for card in cards}
    postings: dict[str, set[str]] = defaultdict(set)
    for doc_id, value in views.items():
        for token in value["title"] | value["topics"] | value["terms"]:
            postings[token].add(doc_id)

    resolved_k = int(candidate_k) if int(candidate_k) > 0 else _adaptive_candidate_k(len(cards))
    pairs: dict[tuple[str, str], float] = {}
    for doc_id in ids:
        overlaps: Counter[str] = Counter()
        for token in views[doc_id]["title"] | views[doc_id]["topics"] | views[doc_id]["terms"]:
            for other in postings[token]:
                if other != doc_id:
                    overlaps[other] += 1
        for other, _ in overlaps.most_common(resolved_k):
            pair = tuple(sorted((doc_id, other)))
            if pair not in pairs:
                pairs[pair] = _similarity(views[pair[0]], views[pair[1]])

    scores = [score for score in pairs.values() if score > 0]
    threshold = float(edge_threshold) if float(edge_threshold) > 0 else _adaptive_edge_threshold(scores, len(cards))
    graph: dict[str, dict[str, float]] = {doc_id: {} for doc_id in ids}
    for (left, right), score in pairs.items():
        if score >= threshold:
            graph[left][right] = score
            graph[right][left] = score

    labels = {doc_id: doc_id for doc_id in ids}
    for _ in range(12):
        changed = False
        for doc_id in sorted(ids):
            weights: Counter[str] = Counter()
            for other, score in graph[doc_id].items():
                weights[labels[other]] += score
            if weights:
                best = min(((-weight, label) for label, weight in weights.items()))[1]
                if labels[doc_id] != best:
                    labels[doc_id] = best
                    changed = True
        if not changed:
            break

    grouped: dict[str, list[str]] = defaultdict(list)
    for doc_id, label in labels.items():
        grouped[label].append(doc_id)
    groups = [sorted(group) for group in grouped.values()]
    cohesion = {
        _group_key(group): _cohesion(group, graph)
        for group in groups
    }
    return CandidateGraphResult(groups, resolved_k, threshold, cohesion)


def adaptive_stable_min_refs(document_count: int) -> int:
    """Small corpora may compile coherent pairs; large corpora need more support."""
    if document_count <= 80:
        return 2
    if document_count <= 500:
        return 3
    return max(3, min(8, int(math.ceil(document_count * 0.005))))


def topic_label(cards: list[DocumentCard]) -> tuple[str, str]:
    counter: Counter[str] = Counter()
    for card in cards:
        view = _views(card)
        counter.update(view["topics"])
        counter.update(view["title"])
        counter.update(view["terms"])
    terms = [term for term, _ in counter.most_common(4)] or ["related documents"]
    title = " ".join(terms[:3]).title()
    scope = f"Documents centered on {'; '.join(terms[:4])}. Excludes documents without this local semantic evidence."
    return title, scope


def _adaptive_candidate_k(document_count: int) -> int:
    return max(12, min(64, int(math.ceil(4 * math.sqrt(max(1, document_count))))))


def _adaptive_edge_threshold(scores: list[float], document_count: int) -> float:
    if not scores:
        return 1.0
    # Small corpora are often intentionally heterogeneous: keep more supported
    # local edges. Larger corpora need a stricter percentile to avoid bridges.
    # A graph edge must represent a meaningfully strong local relation. The
    # percentile adapts to corpus density, while the floor prevents a broad
    # vocabulary from becoming one giant component through weak lexical links.
    percentile = 0.70 if document_count <= 100 else 0.75
    ordered = sorted(scores)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return max(0.10, min(0.22, ordered[index]))


def _group_key(group: list[str]) -> str:
    return "|".join(group)


def _cohesion(group: list[str], graph: dict[str, dict[str, float]]) -> float:
    if len(group) < 2:
        return 0.0
    values = [graph[left][right] for index, left in enumerate(group) for right in group[index + 1:] if right in graph[left]]
    return sum(values) / len(values) if values else 0.0


def _views(card: DocumentCard) -> dict[str, set[str]]:
    tokens = lambda text: {item for item in _TOKEN.findall(str(text).casefold()) if item not in _STOP}
    return {
        "title": tokens(card.title),
        "topics": tokens(" ".join(card.candidate_topics)),
        "terms": tokens(" ".join(card.important_terms)),
        "summary": tokens(card.summary),
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def _similarity(left: dict[str, set[str]], right: dict[str, set[str]]) -> float:
    return (
        0.40 * _jaccard(left["topics"], right["topics"])
        + 0.25 * _jaccard(left["title"], right["title"])
        + 0.20 * _jaccard(left["terms"], right["terms"])
        + 0.15 * _jaccard(left["summary"], right["summary"])
    )
