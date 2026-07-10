"""
LLM generation layer with structured JSON output and hallucination guard.

Why structured output instead of free-form markdown
----------------------------------------------------
Markdown format instructions ("structure your response EXACTLY as follows…")
break silently with small models like llama3.2 3B.  The model might miss a
section, add extra text, or use a slightly different heading.  The frontend
then has to parse brittle markdown.

Using Ollama's JSON mode (format="json") forces the model to emit valid JSON.
We include a concrete schema example in the system prompt so the model knows
the exact shape.  Pydantic validates and coerces the result.  On parse failure,
we fall back gracefully rather than crashing.

Hallucination guard
-------------------
Small models fabricate paper titles.  After parsing we cross-check every
title in related_papers against the actual retrieved documents.  Papers that
don't match any source title (by substring or Jaccard word overlap > 0.5) are
dropped.
"""

import json
import logging

import config
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from llm_provider import LLMProvider
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ── Response schema ───────────────────────────────────────────────────────────


class PaperReference(BaseModel):
    paper_id: str = ""
    title: str
    year: int | None = None
    authors: str = ""
    venue: str = ""
    relevance: str = ""


class PersonSuggestion(BaseModel):
    name: str
    role: str = ""
    department: str = ""
    relevant_work: str = ""


class ResearchLandscape(BaseModel):
    landscape_summary: str
    related_papers: list[PaperReference] = Field(default_factory=list)
    people_to_consult: list[PersonSuggestion] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    no_relevant_research: bool = False


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are TRACE, a research assistant for {institute}. A student has shared a research idea.
Analyse the provided context from institute papers and return a JSON object.

Return ONLY valid JSON with this exact structure — no prose, no markdown fences:
{{
  "landscape_summary": "2-3 sentences situating the idea within institute work",
  "related_papers": [
    {{"paper_id": "exact ID", "title": "exact title", "year": 2023, "authors": "Name1, Name2", "venue": "NeurIPS", "relevance": "one sentence"}}
  ],
  "people_to_consult": [
    {{"name": "Dr. X", "role": "faculty", "department": "CS", "relevant_work": "brief note"}}
  ],
  "next_steps": ["step 1", "step 2", "step 3"]
}}

Rules:
- Only include papers and people that explicitly appear in the context below.
- If no relevant research is found, return empty arrays and explain in landscape_summary.
- Keep landscape_summary concise (2-3 sentences max).

Context:
{context}"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_context(docs: list[Document]) -> str:
    """
    Format retrieved documents into a prompt context block.

    Abstract text is truncated to MAX_ABSTRACT_CHARS per chunk, which is
    computed dynamically from OLLAMA_NUM_CTX so the total prompt never
    overflows the model's context window.

    Default: (8192 - 800 system - 200 query - 1500 output) / 5 chunks * 4 chars/token
             ≈ 4600 chars per abstract — well within safe limits for most papers.
    """
    cap = config.MAX_ABSTRACT_CHARS
    parts = []
    for d in docs:
        m = d.metadata
        parts.append(
            f"PAPER: {m.get('paper_title', '')}\n"
            f"ID: {m.get('paper_id', '')}\n"
            f"Year: {m.get('year', '')}  Venue: {m.get('venue', '')}\n"
            f"All Authors: {m.get('authors', '')}\n"
            f"Institute Authors: {m.get('institute_authors', '')} "
            f"({m.get('institute_roles', '')})\n"
            f"Departments: {m.get('departments', '')}\n"
            f"Abstract: {d.page_content[:cap]}"
            + ("…" if len(d.page_content) > cap else "")
        )
    return "\n\n---\n\n".join(parts)


def _hallucination_guard(
    landscape: ResearchLandscape,
    source_docs: list[Document],
) -> ResearchLandscape:
    """Remove any cited paper whose title or ID cannot be matched to retrieved sources."""
    source_ids = {
        d.metadata.get("paper_id") for d in source_docs if d.metadata.get("paper_id")
    }
    source_titles = {d.metadata.get("paper_title", "").lower() for d in source_docs}

    def _is_grounded(paper: PaperReference) -> bool:
        """Check if paper matches a source ID, or fallback to fuzzy title match."""
        if paper.paper_id and paper.paper_id in source_ids:
            return True

        title_lower = paper.title.lower()
        title_words = set(title_lower.split())

        for source_title in source_titles:
            if title_lower in source_title or source_title in title_lower:
                return True

        for source_title in source_titles:
            source_words = set(source_title.split())
            if not (title_words | source_words):
                continue
            jaccard = len(title_words & source_words) / len(title_words | source_words)
            if jaccard > 0.75:
                return True

        return False

    valid: list[PaperReference] = []
    for paper in landscape.related_papers:
        if _is_grounded(paper):
            valid.append(paper)
        else:
            logger.warning(
                f"Hallucination dropped: '{paper.title}' (ID: '{paper.paper_id}')"
            )
    landscape.related_papers = valid
    return landscape


# ── Public API ────────────────────────────────────────────────────────────────


def generate_answer(query: str, docs: list[Document]) -> ResearchLandscape:
    """
    Call the LLM in JSON mode, parse the Pydantic schema, apply the
    hallucination guard.  Falls back to raw text if JSON parsing fails.
    """
    llm = LLMProvider.get_json_llm()
    context = _format_context(docs)
    system = _SYSTEM.format(institute=config.INSTITUTE_NAME, context=context)

    response = None
    try:
        response = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=query),
            ]
        )
        data = json.loads(response.content)
        landscape = ResearchLandscape(**data)
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parsing failed, falling back to raw text: {exc}")
        raw = response.content if response else str(exc)
        landscape = ResearchLandscape(
            landscape_summary=raw,
            related_papers=[],
            people_to_consult=[],
            next_steps=[],
        )
    except ValidationError as exc:
        logger.warning(f"Pydantic validation failed, falling back to raw text: {exc}")
        raw = response.content if response else str(exc)
        landscape = ResearchLandscape(
            landscape_summary=raw,
            related_papers=[],
            people_to_consult=[],
            next_steps=[],
        )
    except Exception as exc:
        logger.error(f"Unexpected LLM error: {exc}")
        landscape = ResearchLandscape(
            landscape_summary=str(exc),
            related_papers=[],
            people_to_consult=[],
            next_steps=[],
        )

    return _hallucination_guard(landscape, docs)
