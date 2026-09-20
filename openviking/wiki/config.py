"""Wiki 生成管线配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WikiGenerationLimits:
    use_candidate_graph_clustering: bool = True
    # Zero selects a corpus-relative candidate budget / edge threshold. Positive
    # values are explicit overrides for controlled ablations.
    candidate_graph_k: int = 0
    candidate_graph_edge_threshold: float = 0.0
    max_cards_per_discovery_batch: int = 200
    max_parents_per_child: int = 4
    # 最多向上聚合多少层 Wiki 节点。
    max_depth: int = 6
    # 父节点至少要覆盖多少个子节点，否则不会保留。
    min_child_nodes_per_parent: int = 3
    # 底层节点至少要绑定多少个来源引用，否则会被拒绝。
    min_refs_per_node: int = 3
    # 同时发起多少个文档卡片生成请求。
    max_concurrent_cards: int = 10
    # 同时发起多少个节点内容生成请求。
    max_concurrent_nodes: int = 10
    # Bound one compile request. Large clusters are represented by a
    # coverage-preserving evidence pack rather than their entire raw corpus.
    max_compile_input_chars: int = 60000
    max_compile_chars_per_source: int = 8000
    compile_microcluster_chars: int = 18000
    max_compile_evidence_cards: int = 8
    # The adaptive compiler may expand a coherent medium topic up to this cap;
    # it still map-reduces large or diverse topics instead of consuming a model's
    # full context window.
    max_adaptive_compile_input_chars: int = 240000


@dataclass
class WikiConfig:
    # 写入产物中的管线版本标识。
    pipeline_version: str = "wiki_v2_doc_card"
    # 来源资源所在的根 URI，用来校验和记录引用来源。
    resource_root_uri: str = "viking://resources/"
    # Wiki 产物写入的根 URI。
    wiki_root_uri: str = "viking://wiki/"
    # 控制节点数量、层数、过滤阈值和并发量。
    limits: WikiGenerationLimits = field(default_factory=WikiGenerationLimits)
    # 传给底层 VLM/LLM 的模型配置。
    vlm_config: dict[str, Any] | None = None
