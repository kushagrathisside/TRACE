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


def _get(url: str, params: dict) -> dict:
    """GET with exponential back-off on 429."""
    resp = None
    for attempt in range(4):
        resp = httpx.get(
            url,
            params=params,
            headers=_HEADERS,
            timeout=20,
        )
        if resp.status_code == 429:
            wait = 2**attempt
            if attempt < 3:
                time.sleep(wait)
                continue
            else:
                raise httpx.HTTPStatusError(
                    f"Rate limited after {attempt + 1} retries. "
                    f"Retry-After: {resp.headers.get('retry-after', 'unknown')} seconds",
                    request=resp.request,
                    response=resp,
                )
        resp.raise_for_status()
        return resp.json()

    if resp:
        resp.raise_for_status()
    return {}


def get_author_papers(author_id: str, since_year: int | None = None) -> list[dict]:
    """
    Return published papers for a Semantic Scholar author ID.
    Pass since_year to restrict to papers from that year onward
    (e.g., since_year=2024 → "2024-" range filter).
    """
    params: dict = {"fields": config.PAPER_FIELDS, "limit": 1000}
    if since_year:
        params["year"] = f"{since_year}-"
    url = f"{config.SEMANTIC_SCHOLAR_BASE_URL}/author/{author_id}/papers"
    data = _get(url, params)
    return data.get("data", [])


def search_author(name: str) -> list[dict]:
    """Search for authors by name — used by the admin panel's lookup widget."""
    url = f"{config.SEMANTIC_SCHOLAR_BASE_URL}/author/search"
    data = _get(
        url,
        {"query": name, "fields": "name,affiliations,paperCount", "limit": 10},
    )
    return data.get("data", [])
