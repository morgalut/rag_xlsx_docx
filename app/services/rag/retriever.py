

from __future__ import annotations
import re
import itertools
from typing import Any, Dict, List, Tuple, Optional
from app.core.config import TOP_K
from app.services.embeddings.embeddings import SentenceTransformerEmbedder
from app.services.load.vector_store_mongo import MongoVectorStore


MIN_HITS = 3
MAX_K = max(TOP_K, 5)


def _domain_filter_for(question_or_domain: str | None) -> Optional[Dict[str, Any]]:
    """Heuristics for department filters based on domain terms."""
    if not question_or_domain:
        return None
    s = (question_or_domain or "").lower()
    if any(k in s for k in ["hr", "vacation", "leave", "holiday", "benefit", "sick"]):
        return {"meta.department": {"$regex": "HR|Human Resources", "$options": "i"}}
    if any(k in s for k in ["security", "password", "vpn", "login", "mfa"]):
        return {"meta.department": {"$regex": "IT|Information Technology", "$options": "i"}}
    if any(k in s for k in ["salary", "budget", "finance", "payroll", "invoice"]):
        return {"meta.department": {"$regex": "Finance|Accounting", "$options": "i"}}
    return None


def _keyword_regex(keywords: List[str]) -> str:
    safe = [re.escape(k) for k in keywords if k and isinstance(k, str)]
    if not safe:
        return ""
    return r"(" + r"|".join(safe) + r")"


def _rewrite_keywords(keywords: List[str]) -> List[str]:
    """Lightweight synonym expansion for broadened pass."""
    mapping = {
        "vacation": ["vacation", "annual leave", "paid leave", "PTO"],
        "leave": ["leave", "paid leave", "time off", "PTO"],
        "policy": ["policy", "guideline", "procedure", "rules"],
        "carry over": ["carry over", "rollover", "roll-over", "carry-forward"],
        "approval": ["approval", "approve", "authorized by", "manager approval", "HR approval"],
        "full-time": ["full-time", "full time", "FT", "regular employees"],
        "days": ["days", "entitlement", "allowance"]
    }
    expanded: List[str] = []
    for k in keywords:
        expanded.extend(mapping.get(k.lower(), [k]))
    # keep unique while preserving order
    dedup: List[str] = []
    for k in expanded:
        if k not in dedup:
            dedup.append(k)
    return dedup[:12]


class Retriever:
    """Agentic multipass retriever."""

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.embedder = SentenceTransformerEmbedder()
        self.store = MongoVectorStore()

    def embed_query(self, query: str) -> List[float]:
        return self.embedder.embed([query])[0]

    def _semantic(self, query: str, filter_: Optional[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        vec = self.embed_query(query)
        return self.store.query_knn(vec, k=k, filter_=filter_)

    def _keyword_scan(self, keywords: List[str], filter_: Optional[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Regex scan on raw text and common meta fields."""
        regex = _keyword_regex(keywords)
        if not regex:
            return []
        # Build $or conditions across text and structured meta
        or_terms = [
            {"text": {"$regex": regex, "$options": "i"}},
            {"meta.row": {"$regex": regex, "$options": "i"}},
            {"meta.columns": {"$elemMatch": {"$regex": regex, "$options": "i"}}},
            {"meta.sheet": {"$regex": regex, "$options": "i"}},
            {"meta.section": {"$regex": regex, "$options": "i"}},
            {"meta.file_name": {"$regex": regex, "$options": "i"}},
            {"meta.doc_type": {"$regex": regex, "$options": "i"}},
        ]
        query = {"$or": or_terms}
        if filter_:
            query = {"$and": [filter_, query]}

        # Plain find; score with simple term hit count (cheap heuristic)
        cursor = self.store.collection.find(query, {"text": 1, "doc_id": 1, "chunk_id": 1, "meta": 1}).limit(200)
        hits: List[Dict[str, Any]] = []
        pattern = re.compile(regex, re.IGNORECASE)
        for doc in cursor:
            txt = doc.get("text", "") or ""
            hit_count = len(pattern.findall(txt))
            score = float(min(1.0, 0.25 + 0.05 * hit_count))  # gentle cap
            hits.append({
                "text": txt,
                "doc_id": doc.get("doc_id"),
                "chunk_id": doc.get("chunk_id"),
                "meta": doc.get("meta", {}),
                "score": score
            })
        # sort by heuristic score, then take k
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]

    @staticmethod
    def _dedup(hits: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Deduplicate by (doc_id, chunk_id) keeping highest score."""
        best: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        for h in hits:
            key = (h.get("doc_id"), h.get("chunk_id"))
            if key not in best or h.get("score", 0) > best[key].get("score", 0):
                best[key] = h
        result = sorted(best.values(), key=lambda x: x.get("score", 0), reverse=True)
        return result[:k]

    def retrieve(self, question: str, analysis: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
        """
        Returns (diagnostics, context_texts, hits)
        diagnostics.status in {"exact", "closest_match", "insufficient_corpus"}
        """
        # ---- plan filters
        domain_hint = (analysis or {}).get("domain") or question
        filter_ = _domain_filter_for(domain_hint)
        keywords = (analysis or {}).get("keywords") or [question]
        k = max(self.top_k, MAX_K)

        # Pass 1: semantic + filter
        hits_1 = self._semantic(question, filter_, k)
        hits = list(hits_1)

        # Pass 2: keyword scan if needed
        if len(hits) < MIN_HITS:
            hits_2 = self._keyword_scan(keywords, filter_, k)
            hits.extend(hits_2)

        # Pass 3: broadened semantic (rewrites)
        if len(hits) < MIN_HITS:
            rew = " ".join(_rewrite_keywords(keywords))
            hits_3 = self._semantic(rew, filter_, k)
            hits.extend(hits_3)

        # Deduplicate and cap
        hits = self._dedup(hits, k=self.top_k)

        # Prepare context texts
        context = [h.get("text", "") for h in hits if h.get("text")]

        # Diagnose status
        status = "exact" if any(re.search(r"\b(vacation|paid\s+time\s+off|PTO|leave)\b", (h.get("text") or ""), re.I) for h in hits) else \
                 ("closest_match" if hits else "insufficient_corpus")

        diagnostics = {
            "status": status,
            "passes": {
                "semantic": len(hits_1),
                "keyword": 0 if 'hits_2' not in locals() else len(hits_2),
                "semantic_broadened": 0 if 'hits_3' not in locals() else len(hits_3),
            },
            "filter": filter_ or {},
            "keywords": keywords,
        }
        return diagnostics, context, hits
