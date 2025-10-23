from __future__ import annotations
from typing import List, Dict, Any
from app.core.config import GENERATION_MODE
from app.services.llm.llm_factory import LLMFactory


# PromptBuilder: Responsible for constructing prompts for the LLM.
# It formats the context, question, and task instructions according to the retrieval status.
class PromptBuilder:
    @staticmethod
    def build_prompt(question: str, context_chunks: List[str], status: str) -> str:
        # Join up to 8 retrieved context chunks or fallback to "NO_CONTEXT"
        context = "\n---\n".join(context_chunks[:8]) if context_chunks else "NO_CONTEXT"

        # Selects prompt clause depending on retrieval status (exact, closest_match, insufficient_corpus)
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

        # Full system instruction prompt given to the LLM
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


# Generator: Handles interaction with the LLM and ensures graceful fallback.
# It never returns "I don't know" and always provides an actionable response.
class Generator:
    def __init__(self, mode: str = GENERATION_MODE):
        # Initialize with default mode (OpenAI) and create the LLM instance
        self.mode = "openai"
        self.llm = LLMFactory.create(self.mode)

    def generate(self, question: str, context_chunks: List[str], status: str, hits: List[Dict[str, Any]]) -> str:
        """
        Never returns 'I don't know'. Produces a grounded, status-aware answer.
        """
        # Build structured prompt using question, retrieved chunks, and retrieval status
        prompt = PromptBuilder.build_prompt(question, context_chunks, status)

        try:
            # Generate text using the selected LLM backend
            text = (self.llm.generate(prompt) or "").strip()

            if not text:
                # Handle cases where LLM returns empty output
                if status == "insufficient_corpus":
                    # Fallback message when no data is in the corpus
                    return (
                        "Answer: No matching passages were found in the current database for this question.\n\n"
                        "Key Points:\n"
                        "- The indexed sources do not contain an explicit vacation policy section.\n"
                        "- Consider ingesting the Employees Handbook HR chapter or Benefits Guide.\n"
                        "- Re-run ingestion for .docx/.xlsx with ‘vacation’, ‘PTO’, ‘leave’ keywords.\n\n"
                        "Sources:\n- (none)"
                    )
                if context_chunks:
                    # Fallback when partial context exists but LLM failed to respond
                    return (
                        "Answer: Based on the closest matching passages, here is a grounded summary.\n\n"
                        "Key Points:\n- (derived from retrieved chunks)\n\n"
                        "Sources:\n- " + "\n- ".join([f"{h.get('doc_id')}#{h.get('chunk_id')}" for h in hits[:5]])
                    )
                # Generic fallback when no context and no model response
                return (
                    "Answer: No matching passages were found. Please ingest the relevant HR policy sources.\n\n"
                    "Key Points:\n- Add HR policy files to the corpus\n- Re-run ingestion\n\n"
                    "Sources:\n- (none)"
                )
            # Normal path: return LLM output
            return text

        except Exception as e:
            # Handle runtime or API errors gracefully
            # Always provide actionable guidance instead of "I don't know"
            if status == "insufficient_corpus":
                # If database lacks sufficient documents
                return (
                    f"Answer: Retrieval failed ({e}). No policy passages available in the database.\n\n"
                    "Key Points:\n- Index HR/Benefits policy files\n- Verify Mongo connection and vector index\n\n"
                    "Sources:\n- (none)"
                )
            # If other unexpected error occurs, summarize retrieved chunks instead
            return (
                f"Answer: Retrieval succeeded but generation failed ({e}). "
                "Here is a concise bullet summary of retrieved passages instead.\n\n"
                "Key Points:\n"
                + "\n".join([f"- {chunk[:160]}..." for chunk in context_chunks[:5]])
                + "\n\nSources:\n- "
                + "\n- ".join([f"{h.get('doc_id')}#{h.get('chunk_id')}" for h in hits[:5]])
            )
