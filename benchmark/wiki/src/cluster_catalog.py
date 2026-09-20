"""Build a bounded query-specific directory from chunks and Wiki membership."""
from __future__ import annotations
import json
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

def build_cluster_catalog(question: str, config: dict, server_url: str, api_key: str) -> str:
    execution = config.get("execution", {})
    if str(execution.get("retrieval_strategy", "")).lower() != "cluster_catalog": return ""
    root = str(execution.get("wiki_root_uri", "")).rstrip("/")
    store = Path(str(config.get("paths", {}).get("vector_store", "")))
    path = store / "viking" / "default" / "wiki" / Path(*[x for x in root.replace("viking://wiki/", "").split("/") if x])
    try:
        assignments = json.loads((path / "source_assignments.json").read_text(encoding="utf-8"))
        nodes = {x["node_id"]: x for x in json.loads((path / "nodes.json").read_text(encoding="utf-8"))["nodes"]}
        cards = {x["doc_id"]: x for f in (path / "cards").glob("*.card.json") for x in [json.loads(f.read_text(encoding="utf-8"))]}
        budget = _catalog_budget(execution, len(cards), question)
        hits = _find(question, server_url, api_key, budget["seed_chunk_topk"])
    except Exception: return ""
    doc_nodes = {}
    for nid, refs in assignments.get("source_refs_by_node", {}).items():
        for ref in refs:
            if ref.get("ref_type") == "document": doc_nodes.setdefault(ref.get("doc_id"), []).append(nid)
    seeds = []
    for hit in hits:
        uri = str(hit.get("uri", "")); matches = [(len(str(c.get("resource_uri", ""))), did) for did,c in cards.items() if uri.startswith(str(c.get("resource_uri", "")))]
        if matches: seeds.append(max(matches)[1])
    seeds = list(dict.fromkeys(seeds)); node_ids = list(dict.fromkeys(n for d in seeds for n in doc_nodes.get(d, [])))[:budget["catalog_max_nodes"]]
    lines = [
        "QUERY-SCOPED WIKI CATALOG",
        f"Read budget: use at most {budget['max_raw_documents']} listed source documents and answer once evidence is sufficient.",
        "Seed chunks:",
    ]
    lines += [f"- URI: {x.get('uri','')} | {x.get('abstract') or x.get('overview') or ''}"[:budget["seed_chars"]] for x in hits]
    refs = assignments.get("source_refs_by_node", {})
    for nid in node_ids:
        node = nodes.get(nid, {}); lines.append(f"\nTOPIC: {node.get('title',nid)}\nSCOPE: {node.get('scope','')}")
        members = [r.get("doc_id") for r in refs.get(nid,[]) if r.get("ref_type")=="document"]
        ordered = list(dict.fromkeys([*[d for d in seeds if d in members], *members]))
        for did in ordered[:budget["catalog_docs_per_node"]]:
            c=cards.get(did)
            if c: lines.append(f"  DOC: {c.get('title','')}\n  URI: {c.get('resource_uri','')}\n  SUMMARY: {str(c.get('summary',''))[:budget['summary_chars']]}")
    return "\n".join(lines)[:budget["catalog_max_chars"]]


def _catalog_budget(execution: dict, document_count: int, question: str) -> dict[str, int]:
    """Derive a bounded catalog from corpus and question size.

    Adaptive is the default. Set ``retrieval_budget_mode: fixed`` only for a
    controlled reproduction of an older fixed-budget experiment.
    """
    if str(execution.get("retrieval_budget_mode", "adaptive")).lower() == "fixed":
        return {
            "seed_chunk_topk": max(1, int(execution.get("seed_chunk_topk", 8))),
            "catalog_max_nodes": max(1, int(execution.get("catalog_max_nodes", 4))),
            "catalog_docs_per_node": max(1, int(execution.get("catalog_docs_per_node", 12))),
            "catalog_max_chars": max(2000, int(execution.get("catalog_max_chars", 18000))),
            "seed_chars": 900,
            "summary_chars": 900,
            "max_raw_documents": max(2, int(execution.get("original_fallback_topk", 3))),
        }
    small_corpus = document_count <= 100
    catalog_chars = min(9000, max(5000, 5000 + len(question) * 2))
    return {
        "seed_chunk_topk": 5 if small_corpus else 8,
        "catalog_max_nodes": 2 if small_corpus else 3,
        "catalog_docs_per_node": min(10, max(4, int(document_count ** 0.5))),
        "catalog_max_chars": catalog_chars,
        "seed_chars": 550,
        "summary_chars": 650,
        "max_raw_documents": 2 if small_corpus else 3,
    }

def _find(query: str, url: str, key: str, limit: int) -> list[dict]:
    headers={"Content-Type":"application/json"}
    if key: headers["X-API-Key"]=key
    req=Request(f"{url}/api/v1/search/find",data=json.dumps({"query":query,"target_uri":"viking://resources","limit":max(1,limit),"level":[2]}).encode(),headers=headers,method="POST")
    with build_opener(ProxyHandler({})).open(req,timeout=15) as response: payload=json.loads(response.read().decode())
    result=payload.get("result",{}); return list(result.get("resources",[]) if isinstance(result,dict) else [])
