import json
import logging
import sys
from pathlib import Path

# Adjust path to import from backend
sys.path.append("/home/kushagra/TRACE/backend")

from langchain_core.documents import Document
from rag.schemas import DocumentMetadata
from rag.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)

papers_file = Path(
    "/mnt/c/Users/Kushagra Srivastava/.gemini/antigravity/brain/19e0e384-268e-4394-8cc9-ef1580bb8ccb/scratch/papers.json"
)
with open(papers_file, "r", encoding="utf-8-sig") as f:
    data = json.load(f)

papers = data.get("data", [])
print(f"Loaded {len(papers)} papers from disk.")

vs = VectorStoreManager.get_or_create()

docs = []
doc_ids = []

for paper in papers:
    pid = paper.get("paperId")
    if not pid:
        continue

    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    if not title:
        continue

    content = (
        f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"
    )
    content = content[:1500]

    all_authors = ", ".join(a.get("name", "") for a in (paper.get("authors") or []))

    url = ""
    if paper.get("openAccessPdf"):
        url = (paper["openAccessPdf"] or {}).get("url", "")
    if not url:
        doi = (paper.get("externalIds") or {}).get("DOI", "")
        url = f"https://doi.org/{doi}" if doi else ""

    try:
        meta = DocumentMetadata(
            paper_id=pid,
            paper_title=title,
            year=paper.get("year") or 0,
            venue=paper.get("venue") or "",
            authors=all_authors,
            institute_authors="Prof. G. C. Nandi",
            institute_roles="Professor",
            departments="Information Technology",
            paper_url=url,
        )
    except Exception as e:
        print(f"Error parsing metadata for {pid}: {e}")
        continue

    docs.append(
        Document(
            page_content=content,
            metadata=meta.model_dump(),
        )
    )
    doc_ids.append(pid)

print(f"Upserting {len(docs)} documents into ChromaDB...")
batch_size = 10
total_upserted = 0
for i in range(0, len(docs), batch_size):
    batch_docs = docs[i : i + batch_size]
    batch_ids = doc_ids[i : i + batch_size]
    try:
        upserted = vs.upsert_documents(batch_docs, batch_ids)
        total_upserted += upserted or 0
    except Exception as e:
        print(f"Failed to upsert batch {i}: {e}")

print(f"Upserted {total_upserted} documents. Total in DB: {vs.count()}")
