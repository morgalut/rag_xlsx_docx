
from __future__ import annotations
import json
import logging
from typing import Dict, Any

from app.core.config import OPENAI_API_KEY, OPENAI_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Dynamic OpenAI client loader (works with both SDK generations)
# ---------------------------------------------------------------------
try:
    # new SDK (≥1.0)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    _USE_NEW = True
    logger.debug(" Using new OpenAI SDK (>=1.0)")
except ImportError:
    import openai  # legacy SDK
    openai.api_key = OPENAI_API_KEY
    client = openai
    _USE_NEW = False
    logger.debug(" Using legacy OpenAI SDK (<1.0)")

# ---------------------------------------------------------------------
# Analyzer function
# ---------------------------------------------------------------------
def analyze_question(question: str) -> Dict[str, Any]:
    """
    Analyze a user question and return structured metadata.

    Args:
        question: The raw text query from the user.

    Returns:
        dict with keys: domain, intent, keywords
    """
    system_prompt = (
        "You are a precise AI system that classifies user questions "
        "for information retrieval and knowledge-base search.\n"
        "Return *only* valid JSON — no commentary."
    )

    user_prompt = (
        "Analyze this question and extract:\n"
        "- domain: e.g. HR, IT, Finance, Legal, Product\n"
        "- intent: one of [information, procedure, policy, definition, numeric, other]\n"
        "- keywords: 3-8 concise search terms\n\n"
        f"Question: {question}\n\n"
        "Respond in JSON only, for example:\n"
        '{"domain":"HR","intent":"policy","keywords":["vacation","policy","leave"]}'
    )

    try:
        if _USE_NEW:
            #  modern client
            response = client.chat.completions.create(
                model=OPENAI_MODEL or "gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
        else:
            #  legacy client
            response = client.ChatCompletion.create(
                model=OPENAI_MODEL or "gpt-4",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response["choices"][0]["message"]["content"]

        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("Response JSON not a dict")
        logger.debug(f" Question analysis result: {result}")
        return result

    except Exception as e:
        logger.warning(f" Question analysis failed: {e}")
        return {"domain": None, "intent": "unknown", "keywords": [question]}
