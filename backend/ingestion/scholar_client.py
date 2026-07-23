"""
Semantic Scholar Graph API client.

Incremental sync
----------------
`get_author_papers` accepts an optional `since_year`.  When provided it appends
a `year` range filter ("2024-") so only papers from that year onward are
returned.  The ingestor passes the year of the previous sync so repeat syncs
fetch only new publications instead of re-downloading every paper.

Rate limiting
-------------
Without an API key: 100 requests / 5 min (~0.33 req/s).
With an API key: 10 req/s.
We sleep 1 s between per-author calls regardless.  With a key, remove the sleep
or reduce it to 0.1 s for significantly faster bulk syncs.
"""

import time

import config
import httpx

_HEADERS: dict[str, str] = (
    {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY}
    if config.SEMANTIC_SCHOLAR_API_KEY
    else {}
)


#: Unauthenticated quota is 100 requests per 5 minutes, so a burst can only be
#: cleared by waiting out a window measured in minutes.  Backoff of 1-2-4s was
#: never going to recover from that; these waits are sized to the actual quota.
_RETRY_WAITS = (5, 15, 45, 90)
_MAX_RETRY_AFTER = 120


def _get(url: str, params: dict) -> dict:
    """GET with back-off on 429, honouring Retry-After when the server sends it."""
    resp = None
    for attempt, wait in enumerate(_RETRY_WAITS):
        resp = httpx.get(
            url,
            params=params,
            headers=_HEADERS,
            timeout=30,
        )
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                wait = min(int(retry_after), _MAX_RETRY_AFTER)
            if attempt < len(_RETRY_WAITS) - 1:
                time.sleep(wait)
                continue
            raise httpx.HTTPStatusError(
                f"Rate limited after {attempt + 1} retries. "
                f"Retry-After: {retry_after or 'unknown'} seconds. "
                "Set SEMANTIC_SCHOLAR_API_KEY to raise the quota to 10 req/s.",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        return resp.json()

    if resp:
        resp.raise_for_status()
    return {}


#: Semantic Scholar caps this endpoint at 1000 records per request.
_PAGE_SIZE = 100


def get_author_papers(
    author_id: str,
    since_year: int | None = None,
    max_papers: int | None = None,
) -> list[dict]:
    """
    Return published papers for a Semantic Scholar author ID.

    Pass since_year to restrict to papers from that year onward
    (e.g., since_year=2024 → "2024-" range filter).

    Results are paginated.  A single limit=1000 request silently truncated
    prolific authors — no error, no log, just a permanent recall ceiling for
    everyone with a long publication record.

    `max_papers` (config.MAX_PAPERS_PER_PERSON when unset) bounds how much is
    pulled per person, which bounds index size, embedding time and memory.
    Papers come back newest-first, so the cap keeps the most recent work.
    """
    limit = config.MAX_PAPERS_PER_PERSON if max_papers is None else max_papers
    url = f"{config.SEMANTIC_SCHOLAR_BASE_URL}/author/{author_id}/papers"

    papers: list[dict] = []
    offset = 0
    while True:
        page_size = _PAGE_SIZE if limit <= 0 else min(_PAGE_SIZE, limit - len(papers))
        if page_size <= 0:
            break
        params: dict = {
            "fields": config.PAPER_FIELDS,
            "limit": page_size,
            "offset": offset,
        }
        if since_year:
            params["year"] = f"{since_year}-"

        data = _get(url, params)
        batch = data.get("data", [])
        papers.extend(batch)

        next_offset = data.get("next")
        if not batch or next_offset is None:
            break
        offset = next_offset
        time.sleep(0.2)  # be polite between pages

    return papers


def search_author(name: str) -> list[dict]:
    """Search for authors by name — used by the admin panel's lookup widget."""
    url = f"{config.SEMANTIC_SCHOLAR_BASE_URL}/author/search"
    data = _get(
        url,
        {"query": name, "fields": "name,affiliations,paperCount", "limit": 10},
    )
    return data.get("data", [])
