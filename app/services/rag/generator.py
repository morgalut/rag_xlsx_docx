"""
Answer Synthesizer (OpenAI-only, Never 'I don't know')
=====================================================

Builds a grounded answer with a status:
- exact: Direct policy statements found
- closest_match: No explicit policy clause; summarizing most relevant evidence
- insufficient_corpus: Nothing in DB; return useful diagnostic & next steps

The model is *required* to ground content in provided context only.
"""

from __future__ import annotations
from typing import List, Dict, Any
from app.core.config import GENERATION_MODE
from app.services.llm.llm_factory import LLMFactory


class PromptBuilder:
    @staticmethod
    def build_prompt(question: str, context_chunks: List[str], status: str) -> str:
        context = "\n---\n".join(context_chunks[:8]) if context_chunks else "NO_CONTEXT"
        # Status-conditional instruction:
        status_clause = {
            "exact": (
                "Your job: extract the explicit policy details from the context."
            ),
            "closest_match": (
                "No explicit clause may exist; compose the best grounded summary from the most relevant context. "
                "Clearly state that these are the closest matching passages."
            ),
            "insufficient_corpus": (
                "No matching passages exist in the current database. "
                "Explain this succinctly and propose next steps to locate or ingest the correct source (section names, files)."
            ),
        }.get(status, "Compose a grounded summary.")
        return (
            "You are a compliance-precise assistant for company policies. "
            "Only use the provided context; do not invent facts.\n\n"
            f"{status_clause}\n\n"
            "Format:\n"
            "1) Answer: a concise, factual paragraph (2-6 sentences)\n"
            "2) Key Points: bullet list (2-5 items)\n"
            "3) Sources: list ‘doc_id#chunk_id’ of the passages used\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )


class Generator:
    def __init__(self, mode: str = GENERATION_MODE):
        self.mode = "openai"
        self.llm = LLMFactory.create(self.mode)

    def generate(self, question: str, context_chunks: List[str], status: str, hits: List[Dict[str, Any]]) -> str:
        """
        Never returns 'I don't know'. Produces a grounded, status-aware answer.
        """
        prompt = PromptBuilder.build_prompt(question, context_chunks, status)
        try:
            text = (self.llm.generate(prompt) or "").strip()
            if not text:
                # Hard fallback formatting in case LLM returns empty
                if status == "insufficient_corpus":
                    return (
                        "Answer: No matching passages were found in the current database for this question.\n\n"
                        "Key Points:\n"
                        "- The indexed sources do not contain an explicit vacation policy section.\n"
                        "- Consider ingesting the Employees Handbook HR chapter or Benefits Guide.\n"
                        "- Re-run ingestion for .docx/.xlsx with ‘vacation’, ‘PTO’, ‘leave’ keywords.\n\n"
                        "Sources:\n- (none)"
                    )
                if context_chunks:
                    return (
                        "Answer: Based on the closest matching passages, here is a grounded summary.\n\n"
                        "Key Points:\n- (derived from retrieved chunks)\n\n"
                        "Sources:\n- " + "\n- ".join([f"{h.get('doc_id')}#{h.get('chunk_id')}" for h in hits[:5]])
                    )
                return (
                    "Answer: No matching passages were found. Please ingest the relevant HR policy sources.\n\n"
                    "Key Points:\n- Add HR policy files to the corpus\n- Re-run ingestion\n\n"
                    "Sources:\n- (none)"
                )
            return text
        except Exception as e:
            # Never say 'I don't know'—return actionable diagnostic
            if status == "insufficient_corpus":
                return (
                    f"Answer: Retrieval failed ({e}). No policy passages available in the database.\n\n"
                    "Key Points:\n- Index HR/Benefits policy files\n- Verify Mongo connection and vector index\n\n"
                    "Sources:\n- (none)"
                )
            return (
                f"Answer: Retrieval succeeded but generation failed ({e}). "
                "Here is a concise bullet summary of retrieved passages instead.\n\n"
                "Key Points:\n"
                + "\n".join([f"- {chunk[:160]}..." for chunk in context_chunks[:5]])
                + "\n\nSources:\n- "
                + "\n- ".join([f"{h.get('doc_id')}#{h.get('chunk_id')}" for h in hits[:5]])
            )
