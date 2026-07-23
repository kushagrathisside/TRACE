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
Small models fabricate paper titles *and* people.  After parsing we cross-check
every cited paper against the retrieved documents (exact ID, containment with a
length floor, or Jaccard word overlap > TITLE_JACCARD_THRESHOLD) and every
suggested person against the institute authors of those documents.

Grounding people matters at least as much as grounding papers: sending a student
to a professor who does not exist — or attributing work to the wrong person — is
the most damaging thing this system can do, and it is the claim users are least
able to verify themselves.

The guard returns counts alongside the answer so grounding rate is a metric, not
just a log line.
"""

import json
import logging
import re

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
    # Set when generation failed and the response is retrieval-only.  Exposed so
    # the schema-failure rate is measurable from traces instead of grep-able
    # from logs.
    generation_failed: bool = False
    failure_reason: str = ""


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
    {{"name": "Alex Chen", "role": "faculty", "department": "CS", "relevant_work": "brief note"}}
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


#: Word-overlap ratio above which a cited title counts as matching a source.
TITLE_JACCARD_THRESHOLD = 0.75

#: Containment matching ("the cited title appears inside a source title") is
#: only safe for reasonably specific strings.  Without a floor a fabricated
#: title of "Learning" matches nearly every paper in the corpus.
MIN_CONTAINMENT_CHARS = 12
MIN_CONTAINMENT_WORDS = 3


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _split_names(raw: str) -> set[str]:
    return {n.strip().lower() for n in (raw or "").split(",") if n.strip()}


def _hallucination_guard(
    landscape: ResearchLandscape,
    source_docs: list[Document],
) -> tuple[ResearchLandscape, dict]:
    """
    Drop cited papers and suggested people that cannot be traced to a source.

    Returns (landscape, stats) where stats feeds the grounding-rate metric:
    a rising drop rate is the earliest signal that generation quality has
    regressed, and it is measurable without any human labelling.
    """
    source_ids = {
        d.metadata.get("paper_id") for d in source_docs if d.metadata.get("paper_id")
    }
    source_titles = {_normalise(d.metadata.get("paper_title", "")) for d in source_docs}
    source_titles.discard("")

    # Everyone the retrieved papers can legitimately support a suggestion for.
    known_people: set[str] = set()
    for d in source_docs:
        known_people |= _split_names(d.metadata.get("institute_authors", ""))
        known_people |= _split_names(d.metadata.get("authors", ""))
    known_people.discard("")

    def _is_grounded(paper: PaperReference) -> bool:
        """Match a cited paper to a source by ID, containment, or word overlap."""
        if paper.paper_id and paper.paper_id in source_ids:
            return True

        title = _normalise(paper.title)
        if not title:
            return False
        title_words = set(title.split())

        specific_enough = (
            len(title) >= MIN_CONTAINMENT_CHARS
            and len(title_words) >= MIN_CONTAINMENT_WORDS
        )
        if specific_enough:
            for source_title in source_titles:
                if title in source_title or source_title in title:
                    return True

        for source_title in source_titles:
            source_words = set(source_title.split())
            union = title_words | source_words
            if not union:
                continue
            if len(title_words & source_words) / len(union) > TITLE_JACCARD_THRESHOLD:
                return True

        return False

    def _person_is_grounded(person: PersonSuggestion) -> bool:
        """
        A suggested person must appear as an author on a retrieved paper.

        Compared on normalised full names; a bare surname is not accepted,
        since a surname alone matches any number of real people.
        """
        name = _normalise(person.name)
        if not name or len(name.split()) < 2:
            return name in {_normalise(p) for p in known_people}
        return any(name == _normalise(known) for known in known_people)

    valid_papers: list[PaperReference] = []
    for paper in landscape.related_papers:
        if _is_grounded(paper):
            valid_papers.append(paper)
        else:
            logger.warning(
                f"Hallucination dropped (paper): '{paper.title}' "
                f"(ID: '{paper.paper_id}')"
            )

    valid_people: list[PersonSuggestion] = []
    for person in landscape.people_to_consult:
        if _person_is_grounded(person):
            valid_people.append(person)
        else:
            logger.warning(f"Hallucination dropped (person): '{person.name}'")

    stats = {
        "papers_cited": len(landscape.related_papers),
        "papers_dropped": len(landscape.related_papers) - len(valid_papers),
        "people_cited": len(landscape.people_to_consult),
        "people_dropped": len(landscape.people_to_consult) - len(valid_people),
    }

    landscape.related_papers = valid_papers
    landscape.people_to_consult = valid_people
    return landscape, stats


# ── Public API ────────────────────────────────────────────────────────────────


#: Shown when generation fails.  Never surface raw model output or exception
#: text to students: on a parse failure they used to see broken JSON, and on an
#: internal error they saw the exception message.
_FALLBACK_SUMMARY = (
    "The assistant could not produce a structured answer for this idea. "
    "The related papers below were retrieved successfully — please review them "
    "directly, or try rephrasing your idea."
)


def _fallback(reason: str, docs: list[Document]) -> ResearchLandscape:
    """
    Degrade to retrieval-only output.

    Retrieval succeeded even though generation did not, so the papers are still
    worth showing; only the synthesised prose is missing.
    """
    return ResearchLandscape(
        landscape_summary=_FALLBACK_SUMMARY,
        related_papers=[
            PaperReference(
                paper_id=d.metadata.get("paper_id", ""),
                title=d.metadata.get("paper_title", ""),
                year=d.metadata.get("year") or None,
                authors=d.metadata.get("authors", ""),
                venue=d.metadata.get("venue", ""),
                relevance="Retrieved as related work (summary unavailable).",
            )
            for d in docs
        ],
        people_to_consult=[],
        next_steps=[],
        generation_failed=True,
        failure_reason=reason,
    )


def generate_answer(query: str, docs: list[Document]) -> tuple[ResearchLandscape, dict]:
    """
    Call the LLM in JSON mode, parse the Pydantic schema, apply the
    hallucination guard.

    Returns (landscape, grounding_stats).  The stats are recorded in the query
    trace so grounding rate can be tracked over time.
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
        # Log the raw payload for debugging; never show it to the user.
        logger.warning(
            f"JSON parsing failed: {exc} | raw={(response.content if response else '')[:500]!r}"
        )
        landscape = _fallback("json_decode_error", docs)
    except ValidationError as exc:
        logger.warning(
            f"Pydantic validation failed: {exc} | "
            f"raw={(response.content if response else '')[:500]!r}"
        )
        landscape = _fallback("schema_validation_error", docs)
    except Exception as exc:
        logger.error(f"Unexpected LLM error: {exc}", exc_info=True)
        landscape = _fallback("llm_error", docs)

    return _hallucination_guard(landscape, docs)
