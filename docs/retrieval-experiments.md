# TRACE — Retrieval Experiments: A Technical Report

**Subject:** Diagnosis, instrumentation and tuning of the TRACE hybrid retrieval pipeline
**Corpus:** 1,074 papers from 13 AI/ML researchers (Semantic Scholar)
**Period:** 2026-07-20 – 2026-07-21
**Status:** Retrieval defects resolved; ranking configuration provisional (see §8)

---

## 1. Summary

This report documents a sequence of seven measured experiments on the TRACE
retrieval pipeline, the defects they exposed, and the configuration changes
adopted as a result.

The headline findings:

1. **The evaluation harness could not measure anything.** Retrieval metrics
   compared paper *titles* against ground-truth paper *IDs*, so Hit Rate and
   nDCG evaluated to `0.0` on every run regardless of retrieval quality.
2. **The cross-encoder was not running in the deployed configuration**, and its
   absence was reported only by one startup log line.
3. **Two successive ground-truth generators leaked**, each in a different place,
   and each produced a confident result that reversed once the leak was removed.
4. **A cross-encoder regression on human-written queries survived a model swap**,
   which relocated the diagnosis from "wrong model" to "wrong objective".

Six code defects and five methodology defects were found. Three of the
methodology defects were introduced during this work and caught by
cross-checking; they are documented in §7 because the failure modes generalise.

**Net measured change on human-labelled queries** (`manual` slice, n=10,
nDCG@5): 0.519 under the configuration as-found-and-fixed → **0.764** under the
adopted configuration. On unambiguous title queries (n=40): 0.975 → 0.935, a
deliberate trade documented in §6.3.

---

## 2. System under test

| Component | Value |
|---|---|
| Generation | `llama3.2:3b` via Ollama |
| Embeddings | `all-minilm` via Ollama, 384-dim, cosine |
| Vector index | ChromaDB HNSW, M=48, ef_construction=200, search_ef=150 |
| Sparse index | `rank_bm25` Okapi, in-memory, rebuilt per sync |
| Fusion | Weighted Reciprocal Rank Fusion, k=60 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params) |
| Fetch depth | `RETRIEVAL_FETCH_K=20` |
| Result depth | `RETRIEVAL_K=5` |

**Host:** WSL2 on Windows, 7.6 GiB RAM, 24 vCPU, NVIDIA GPU via passthrough,
outbound HTTPS through a corporate proxy.

The environment is not incidental. Four experiments were invalidated or delayed
by host-level faults (§7.3), and two production code defects existed solely
because of proxy and mirror configuration.

---

## 3. Corpus construction

### 3.1 Why a real corpus

Synthetic abstracts are lexically clean and topically separable. Every retriever
scores well on them and an ablation across configurations shows no differences —
which is the opposite of what an ablation is for. Real publication records carry
the shared jargon, near-duplicate titles and stylistic variance that make hybrid
retrieval worth having.

### 3.2 Author resolution

13 well-known AI/ML researchers were registered as TRACE-Institute members.
Semantic Scholar author IDs were resolved at runtime rather than hardcoded.

**Defect found.** The first resolver filtered candidates by exact display-name
match, then took the highest paper count. Semantic Scholar files the canonical
record for "Geoffrey Hinton" under *"Geoffrey E. Hinton"*, so exact matching
found only thin duplicate profiles:

| Author | Exact-match resolver | Surname + initial resolver |
|---|---|---|
| Geoffrey Hinton | 5 papers | 12 papers |
| Fei-Fei Li | 7 | 15 |
| Pieter Abbeel | 22 | **596** |
| Sergey Levine | 18 | **551** |
| Ian Goodfellow | 9 | **268** |

Resolution now matches on (surname, first initial) and takes the profile with
the most papers, on the reasoning that duplicate profiles are always the thin
ones.

### 3.3 Result

| Metric | Value |
|---|---|
| Documents indexed | 1,074 |
| With substantive abstracts (>300 chars) | 720 |
| Cap applied | 120 papers/person |
| Ingestion wall time | 117.1 s |
| Ingestion errors | 0 |
| Self-retrieval Hit Rate@5 | **100.0%** (200 sampled) |

Self-retrieval at 100% confirms index integrity — right embedding model,
uncorrupted vectors, sane HNSW parameters. It is *not* evidence of search
quality; a document's own abstract retrieving that document is close to
trivially satisfiable.

### 3.4 Ingestion defects found

**Proxy environment destruction.** `llm_provider.py` deleted every environment
variable containing "proxy" at import time, to stop the local Ollama client
routing through a SOCKS proxy. Because `ingestor` transitively imports
`llm_provider`, this stripped the proxy configuration that Semantic Scholar
ingestion depends on. All 13 authors failed with `_ssl.c:1011: The handshake
operation timed out` — an error pointing nowhere near the cause. Fixed by
appending `localhost,127.0.0.1,::1` to `no_proxy` instead of deleting proxy
configuration. Note the host's pre-existing `no_proxy=<local>` is a Windows-style
value that Python's HTTP stack does not interpret, which is why explicit hosts
are required.

**Single-batch embedding.** LangChain's Chroma wrapper passes every document to
one embed call. At ~1,000 documents Ollama rejected the payload with
`the input length exceeds the context length (status 400)` and the entire sync
was lost. Fixed with bounded batching (`EMBED_BATCH_SIZE=64`) plus
per-document retry on batch failure, which also makes peak sync memory flat in
corpus size.

**Pagination.** `get_author_papers` issued a single `limit=1000` request with no
offset loop, silently truncating prolific authors. Now paginated at 100/page.

**Incremental metadata loss.** An incremental sync refetches only papers newer
than each person's `last_year`, so a paper co-authored by A and B can return
under A alone. The upsert then overwrote `institute_authors` with `"A"`,
dropping B permanently and degrading `people_to_consult` on every subsequent
sync. Existing institute authors are now merged before upsert, restricted to
people still in the registry so removals are not resurrected.

---

## 4. Evaluation methodology

### 4.1 Metrics

| Metric | Depth | Purpose |
|---|---|---|
| Recall@fetch_k | 20 | Candidate ceiling — nothing downstream can rank what retrieval never returned |
| Recall@k | 5 | What the user sees |
| Recall@k (single-label) | 5 | Recall restricted to queries with exactly one relevant paper |
| nDCG@k | 5 | Rank-aware quality; the metric a reranker is judged on |
| MRR@k | 5 | Rank of the first relevant result |
| retr p50 | — | Retrieval latency excluding all LLM calls |

**Recall@5 is not comparable across slices.** `author` queries carry a median of
109 relevant papers (min 10, max 150), so their Recall@5 is structurally capped
at 5/109 ≈ 0.046; single-label queries are effectively 0 or 1. A corpus average
of the two describes neither quantity. The single-label column exists for this
reason. nDCG and MRR are normalised and safe to compare.

**Recall@fetch_k must be computed over the pre-rerank candidate set.** Measured
over the truncated top-5 it silently collapses to Recall@5 (see §7.2).

### 4.2 Slices

| Slice | n | Query construction | Labels | Bias |
|---|---|---|---|---|
| `exact_title` | 40 | Paper's exact title | That paper | None; floor test |
| `author` | 13 | `"<name> research"` | All that author's papers | None |
| `topic` | 40 | Derived from the document | That paper | **Leaks — see §5.3** |
| `manual` | 10* | Hand-written research ideas | Hand-judged | None |

\* 12 written, 10 with ≥1 relevant paper. Two are genuine corpus gaps and score
zero relevant papers, which is a legitimate label exercising the no-results path.

### 4.3 Label provenance

Labels carry a `label_source` field. This is load-bearing, not bookkeeping: the
value of the `manual` slice is that it is *not* produced by the system under
test, so a model-labelled slice must never be reported as hand-labelled. The
final `manual` labels are human-verified (39 positives across 9 queries), judged
from query + title + abstract snippet only, without external knowledge or
full-text reading.

### 4.4 Candidate pooling

`manual` candidates were pooled from **dense + BM25 + hybrid with the reranker
disabled**, then judged. Labelling a single system's output would encode that
configuration's preferences as ground truth and guarantee it wins. This is
standard TREC pooling; it does not eliminate bias — a paper no retriever
surfaced is never judged, so recall is measured against the pool rather than the
corpus — but it removes the circularity of self-grading.

### 4.5 Evaluation hygiene

- **Cache bypassed.** Otherwise a second eval run reads back its own answers,
  reports cache latency as retrieval latency, and pollutes the production cache.
- **Retrieval-only path.** Ranking metrics depend only on which documents come
  back; generation costs 10–30 s/query and cannot change them. Running it made
  the ablation take hours for numbers it could not affect.
- **Missing data is `None`, never `0.0`.** An absent RAGAS install previously
  logged `faithfulness_mean = 0.0`, indistinguishable in a chart from a
  genuinely unfaithful system.

---

## 5. Experiment log

### 5.1 Experiment A — Baseline ablation (eval set v1)

**Configuration:** eval set v1 (topic slice generated by rewording titles);
reranker unavailable in all variants.

| variant | n | Recall@20 | Recall@5 | nDCG@5 | MRR@5 | retr p50 |
|---|---|---|---|---|---|---|
| dense | 92 | 0.742 | 0.742 | 0.699 | 0.698 | 29 ms |
| bm25 | 92 | 0.803 | 0.803 | 0.859 | 0.891 | 17 ms |
| hybrid | 92 | 0.789 | 0.789 | 0.714 | 0.693 | 32 ms |
| hybrid+ce | 92 | 0.789 | 0.789 | 0.714 | 0.693 | 32 ms |

**Two defects surfaced by this run, neither of them in the numbers themselves.**

The `hybrid+ce` row is identical to `hybrid` because the cross-encoder failed to
load: `config.py` defaulted `HF_ENDPOINT` to `https://hf-mirror.com`, which is
unreachable from this host. The run reported `reranker_active=False` honestly
rather than claiming a phantom gain — the loud-degradation work behaved as
designed. The mirror default is now opt-in.

`Recall@20 == Recall@5` in every row because the harness computed both over the
truncated top-5 result set (§7.2).

**Conclusion:** invalid as a quality measurement; valuable as a test of the
instrumentation.

### 5.2 Experiment B — After reranker restoration (eval set v1)

| variant | n | Recall@20 | Recall@5 | nDCG@5 | MRR@5 | retr p50 |
|---|---|---|---|---|---|---|
| dense | 92 | 0.760 | 0.742 | 0.699 | 0.698 | 29 ms |
| bm25 | 92 | 0.815 | 0.803 | 0.859 | 0.891 | 17 ms |
| hybrid | 92 | 0.816 | 0.789 | 0.714 | 0.693 | 33 ms |
| hybrid+ce | 92 | **0.816** | 0.807 | 0.830 | 0.841 | 51 ms |

nDCG@5 by slice:

| slice | n | dense | bm25 | hybrid | hybrid+ce |
|---|---|---|---|---|---|
| author | 13 | 0.284 | 0.577 | 0.464 | **0.213** |
| exact_title | 40 | 0.854 | 0.947 | 0.789 | **0.975** |
| topic | 39 | 0.677 | 0.862 | 0.720 | **0.888** |

**Structural invariant confirmed:** Recall@20 is identical for `hybrid` and
`hybrid+ce` (0.816). The cross-encoder reorders the same 20 candidates and
cannot change what is in the top 20. Any ablation reporting only
Recall@fetch_k will always show the reranker contributing nothing.

**Anomaly:** the cross-encoder *halved* nDCG on author queries (0.464 → 0.213)
while improving every other slice. Initially attributed to `ms-marco` being
unsuited to metadata lookups. That diagnosis was wrong — see §5.5.

### 5.3 Experiment C — RRF sparse-weight sweep (eval set v1)

| sparse weight | Recall@20 | nDCG@5 | MRR@5 | author slice |
|---|---|---|---|---|
| 1.0 | 0.816 | 0.830 | 0.841 | 0.213 |
| **2.0** | 0.817 | **0.885** | **0.913** | **0.602** |
| 3.0 | 0.817 | 0.885 | 0.913 | 0.602 |
| 5.0 | 0.817 | 0.885 | 0.913 | 0.602 |

An apparent +0.055 nDCG@5 for free, saturating at 2.0. **It was adopted, then
reverted.** The slice table shows why: `exact_title` (0.975) and `topic` (0.888)
scored *bit-identically at every weight*. The entire aggregate gain came from 13
author queries — whose scores were depressed by a code defect (§5.5), not by the
fusion weights.

**Generalisable rule:** *if a sweep moves the aggregate, check which slice moved.
If only one did, you are tuning that slice — and if that slice is depressed by a
bug, you are baking a workaround into a constant.*

### 5.4 Leakage audit of the `topic` slice

The v1 generator produced queries by stripping a title's subtitle and prefixing
"I want to work on". Measuring token overlap between each query and its source
document's title:

```
fraction of query tokens present verbatim in the source TITLE:
  median = 1.00   mean = 1.00
  queries at 100% overlap: 39 / 39
```

Every query. The slice was an exact-title test in disguise, making **86% of the
labelled set (40 exact_title + 39 topic) lexical title matching** — which alone
explains the result "BM25 beats hybrid".

The generator was rebuilt to sample abstract terms *absent* from the title.
Title overlap dropped to median 0.00. The follow-up check did not pass:

```
fraction of query tokens present verbatim in the target ABSTRACT:
  median = 1.00   mean = 1.00   n = 40
```

**The leak moved rather than disappearing.** A query built by sampling abstract
tokens is a verbatim bag-of-words from the target document: ideal for BM25 by
construction, and hostile to a bi-encoder, which must embed a disjointed keyword
list resembling no real query. Dense scoring 0.356 on this slice is an artifact
of the generator, not a property of dense retrieval.

**Conclusion:** any generator that derives a query from the document it must
retrieve leaks somewhere; the only question is where. A cleverer generator does
not fix this. Hand-written queries do.

### 5.5 Reranker input defect

The author-slice regression in Experiment B was attributed to model training
distribution. Inspecting what the model actually received:

```python
f"{d.metadata.get('paper_title','')} {d.page_content[:400]}"    # authors: absent
```

Author names live only in Chroma metadata. They had been added to the BM25
corpus and never to the cross-encoder input, so for `"Yoshua Bengio research"`
the model was scoring topical similarity against text containing **no author
names at all**. It could not see the field the query was about.

Fixed by including author, venue and department metadata in the reranked
document text. Effect on the same 13 queries with the same labels:

| | nDCG@5 (author slice) |
|---|---|
| Before fix | 0.213 |
| After fix | **0.662** |

**Generalisable rule:** *before concluding a model is unsuited to a query class,
print exactly what you fed it.*

### 5.6 Experiment D — Corrected harness, de-leaked topic slice

**Configuration:** eval set v2; reranker input fixed; host GPU passthrough
unavailable, so latency figures are CPU-bound and not comparable to other runs.

| variant | n | Recall@20 | R@5 1-lbl | nDCG@5 | MRR@5 |
|---|---|---|---|---|---|
| dense | 93 | 0.686 | 0.738 | 0.563 | 0.550 |
| bm25 | 93 | 0.817 | 0.938 | 0.863 | 0.892 |
| hybrid | 93 | 0.797 | 0.838 | 0.655 | 0.630 |
| hybrid+ce | 93 | **0.797** | 0.912 | **0.874** | **0.898** |

The cross-encoder now improves *every* slice (author +0.175, exact_title +0.181,
topic +0.271) at constant Recall@20. The anomaly is resolved.

BM25 still dominates dense, but per §5.4 this is unreadable: 80 of 93 labelled
queries reward verbatim lexical overlap.

### 5.7 Experiment E — Hand-written slice added

The `manual` slice was labelled (model draft, then human-verified) and included.

nDCG@5 by slice, human labels:

| slice | n | dense | bm25 | hybrid | hybrid+ce | labels |
|---|---|---|---|---|---|---|
| author | 13 | 0.284 | 0.577 | 0.464 | **0.649** | derived |
| exact_title | 40 | 0.854 | 0.947 | 0.789 | **0.975** | derived |
| topic | 40 | 0.359 | 0.872 | 0.593 | **0.866** | derived, leaky |
| **manual** | 10 | 0.561 | 0.495 | **0.728** | 0.519 | human |

**The hand-written slice inverts every conclusion the generated slices support:**

- BM25 is the **worst** single retriever (0.495), not the best. Its dominance
  elsewhere was the leakage artifact it appeared to be.
- Fusion earns its place — `hybrid` (0.728) beats both arms alone, the result
  hybrid retrieval is supposed to produce and which no generated slice showed.
- The cross-encoder **hurts** (0.728 → 0.519), reversing its behaviour
  everywhere else.

Per-query, the reranker regression is systematic rather than one outlier —
6 of 10 queries worse, deltas −0.32 to −0.50 (measured against the model-labelled
draft; the pattern persisted after human verification):

| query | #rel | hybrid | +ce | delta |
|---|---|---|---|---|
| manual-002 graph neural networks for molecular property… | 5 | 0.509 | 0.131 | −0.378 |
| manual-003 reinforcement learning for robotic manipulation… | 12 | 1.000 | 0.684 | −0.316 |
| manual-007 efficient fine-tuning of large models… | 6 | 0.723 | 0.339 | −0.384 |
| manual-008 using diffusion models for scientific data… | 4 | 0.637 | 0.246 | −0.390 |
| manual-009 evaluating factual grounding in RAG | 1 | 1.000 | 0.500 | −0.500 |
| manual-010 curriculum learning for sample-efficient RL | 5 | 1.000 | 0.509 | −0.491 |
| manual-005 self-supervised pretraining for video… | 7 | 0.485 | 0.723 | +0.237 |
| manual-006 detecting and mitigating social bias… | 3 | 0.765 | 1.000 | +0.235 |
| manual-011 multimodal models for robotics | 9 | 0.830 | 0.869 | +0.038 |

### 5.8 Experiment F — Reranker model swap

**Hypothesis:** `ms-marco-MiniLM` is trained on short search-log queries; TRACE
receives long natural-language research ideas. A model trained on longer, more
varied pairs should recover the regression.

| slice | n | hybrid | +ms-marco | +bge-reranker-base |
|---|---|---|---|---|
| author | 13 | 0.464 | **0.649** | **0.649** |
| exact_title | 40 | 0.789 | **0.975** | 0.966 |
| manual | 10 | **0.728** | 0.519 | 0.502 |
| topic | 40 | 0.593 | **0.866** | **0.866** |

Resource comparison:

| model | params | load | 3 pairs | peak RSS |
|---|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | 22 M | 8.9 s | 253 ms | 1.15 GB |
| `bge-reranker-base` | 278 M | 169.8 s | 55 ms | 2.67 GB |

**Hypothesis rejected.** Two independently trained cross-encoders regress
identically (0.519 vs 0.502). The cause is not one model's training
distribution. `ms-marco` retained: equal quality, 2.3× smaller memory
footprint. The `bge` variant is kept in `ablate.py` for future comparison.

### 5.9 Diagnosis — objective mismatch

Probing the single clearest failure. Query: *"evaluating factual grounding in
retrieval-augmented generation"*; correct document (the corpus's only RAG
paper): *Demonstrate-Search-Predict: Composing retrieval and language models for
knowledge-intensive NLP*.

| document text supplied to the reranker | score |
|---|---|
| current (title + authors + venue + 350 chars) | −2.92 |
| no metadata prefix, 350 chars | −2.55 |
| no metadata prefix, 2000 chars | **−1.25** |
| title + 2000 chars | −1.68 |

Input formatting accounts for part of it — truncation and the metadata prefix
cost ~1.7 points — but the correct paper still scores negative under every
variant.

**The reranker is optimising a different objective than the product needs.**
Cross-encoders are trained to answer *"does this passage answer this query"*.
By that criterion the score is correct: the paper proposes a RAG framework, it
does not evaluate factual grounding. TRACE needs *"is this the most useful thing
this institute has written"*, over a corpus where the honest answer is usually a
near-miss. The human labelling criterion — *would a student genuinely find this
useful* — is precisely the objective the reranker does not optimise.

This is a task mismatch, not a model defect, which is why swapping models
changed nothing.

### 5.10 Experiment G — Rank blending sweep

`RERANK_BLEND` combines cross-encoder and fusion order by **normalised rank**.
Rank rather than raw score because cross-encoder logits and RRF scores live on
incomparable scales, and logit ranges differ between reranker models — a
weighted sum of raw values would silently change meaning on a model swap.

nDCG@5 by slice, human labels:

| slice | n | 0.0 | 0.25 | **0.5** | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| author | 13 | 0.464 | 0.550 | 0.594 | 0.639 | **0.649** |
| exact_title | 40 | 0.789 | 0.830 | 0.935 | 0.966 | **0.975** |
| manual | 10 | 0.728 | 0.749 | **0.764** | 0.655 | 0.519 |
| topic | 40 | 0.593 | 0.642 | 0.745 | 0.844 | **0.866** |

Every derived slice increases monotonically toward 1.0. `manual` peaks at 0.5
and collapses beyond it. Note that 0.75 does **not** resolve the regression:
manual at 0.655 remains below no-reranking at 0.728.

---

## 6. Configuration adopted

### 6.1 Changed, with evidence

| Setting | From | To | Evidence |
|---|---|---|---|
| `RERANKER_MODEL_NAME` | *(empty)* | `ms-marco-MiniLM-L-6-v2` | Reranker was inactive in deployment |
| `HF_ENDPOINT` | `hf-mirror.com` | *(unset)* | Mirror unreachable → silent reranker failure (§5.1) |
| Reranker input | title + abstract | + author/venue/dept | Author slice 0.213 → 0.662 (§5.5) |
| `RERANK_BLEND` | *(n/a)* | `0.5` | §5.10 |
| `BM25_MIN_SCORE` | `>= 0` (no-op) | `> 0` | Zero-score noise outranked genuine dense hits |
| `MAX_PAPERS_PER_PERSON` | unbounded | `200` | Bounds index size and sync memory |
| `EMBED_BATCH_SIZE` | *(n/a)* | `64` | Single-batch embedding failed at ~1,000 docs |

### 6.2 Deliberately unchanged

`RRF_WEIGHT_SPARSE` remains `1.0`. A sweep favoured 2.0, but the gain came
entirely from 13 queries depressed by a separate defect (§5.3).

### 6.3 The trade being made

`RERANK_BLEND=0.5` costs **−0.040 nDCG@5 on `exact_title`** (n=40, unambiguous
labels) to gain **+0.245 on `manual`** (n=10, human labels) relative to pure
reranking.

The reasoning: `manual` is the only slice whose queries a human wrote, and it is
the shape production traffic takes — students describing a research idea, not
typing paper titles. `topic` pushes hardest toward 1.0 and is the least
trustworthy slice, so it is discounted.

This is a judgement about which slice represents production, not a fact
established by the data. It is reversible via one environment variable.

---

## 7. Defect catalogue

### 7.1 Production code

| # | Defect | Impact |
|---|---|---|
| 1 | Reranker disabled in deployment; failure logged once at startup | Ranking fell back to fusion order silently |
| 2 | `HF_ENDPOINT` defaulted to an unreachable mirror | Cross-encoder silently failed to download |
| 3 | BM25 filter `score >= 0` (a no-op) | Zero-score noise entered RRF and outranked genuine dense hits |
| 4 | Author names absent from both indexes | Named-entity queries could not match anything |
| 5 | Similarity guard applied only to the dense arm | BM25 keyword coincidences bypassed it into the LLM context |
| 6 | Reranker input excluded author metadata | Author-query nDCG halved |
| 7 | `people_to_consult` never grounded | Fabricated advisors reachable by students |
| 8 | Proxy variables deleted globally at import | All outbound HTTPS broken; ingestion failed with unrelated SSL errors |
| 9 | Single-batch embedding | Sync lost entirely at ~1,000 documents |
| 10 | Semantic Scholar pagination absent | Prolific authors truncated at 1,000 papers |
| 11 | Incremental sync overwrote co-author metadata | `people_to_consult` degraded on every sync |

### 7.2 Measurement

| # | Defect | Impact |
|---|---|---|
| 1 | Retrieval metrics compared titles to IDs | Hit Rate and nDCG were `0.0` on every run, logged to MLflow as measurements |
| 2 | Eval ran through the semantic cache | Second run measured the cache; polluted the production cache |
| 3 | RAGAS scored against empty context | Faithfulness was noise, not a measurement |
| 4 | `mean([])` returned `0.0` | "Not installed" indistinguishable from "scored zero" |
| 5 | Feedback carried no `query_id` | Ratings unattributable to what was retrieved |
| 6 | Recall@fetch_k computed over the truncated top-k | Candidate ceiling invisible; `Recall@20 == Recall@5` |
| 7 | Recall averaged across 1-label and ~109-label queries | Aggregate described neither quantity |
| 8 | Latency column labelled `total` on a retrieval-only path | Off by 10–30 s from user-facing latency |

Defects 6–8 were introduced during this work and caught by cross-checking.

### 7.3 Environment faults encountered

| Fault | Symptom | Resolution |
|---|---|---|
| Proxy intercepting localhost | `curl` returned an HTML error page **with exit code 0** — a false "service up" | `--noproxy '*'` for manual probes |
| Ollama runner wedged after client kill | 99.9% CPU, `keep_alive=-1`, unkillable (owned by user `ollama`) | Self-cleared after ~19 min |
| WSL2 GPU passthrough lost | `nvidia-smi`: "GPU access blocked by the operating system"; Ollama fell back to CPU | `wsl --shutdown` from Windows |
| CPU fallback | Query expansion 300 ms → 3,045 ms; embedding 30 ms → 753 ms | Quality metrics unaffected; latency columns invalidated |

**Operational note:** do not `kill -9` a client mid-generation. The Ollama runner
continues generating for the dead client and, under `keep_alive=-1`, does not
recycle.

### 7.4 Analyst error

The compact view built to read 148 pooled candidates used a title/year regex
that silently dropped entries, hiding 12 candidates for `manual-000` and 3 for
`manual-001`. Both were consequently judged on incomplete information —
`manual-000` was marked "no relevant papers" when the corpus does contain a
machine-translation paper. Caught by cross-checking applied label counts against
intent. The labelling script now verifies, per query, that what landed in the
file matches what was intended before applying.

---

## 8. Threats to validity

1. **`manual` is n=10.** The blend curve is not monotone
   (0.728 → 0.749 → 0.764 → 0.655 → 0.519); the peak at 0.5 could be two
   queries. This slice currently decides the ranking configuration.
2. **`topic` remains leaky.** Queries are verbatim bags of abstract tokens.
   Any dense-vs-BM25 comparison including this slice is uninterpretable.
3. **80 of 103 labelled queries reward verbatim lexical overlap.** The corpus
   aggregate is therefore biased toward sparse retrieval.
4. **Pool-limited recall.** `manual` labels were judged from a pooled candidate
   set; a relevant paper no retriever surfaced was never judged, so recall is
   measured against the pool, not the corpus.
5. **Single corpus, single domain.** 1,074 AI/ML papers from 13 authors. Nothing
   here has been tested on a different corpus or field.
6. **Latency figures span two hardware states.** Experiment D is CPU-bound;
   others are GPU. Only compare within a run.
7. **LLM-judge metrics are ungated.** RAGAS/DeepEval scores use the same small
   local model that generates the answers. Agreement with human labels has not
   been measured, so those metrics are reported and never used as gates.

---

## 9. Recommended next work

Ordered by value.

1. **Grow `manual` past 30 queries.** Everything downstream is currently decided
   by ten. This is the single highest-value action.
2. **Re-run the blend sweep** on the enlarged slice before treating `0.5` as
   settled.
3. **Route by query class.** Author-name queries want maximum reranking
   (0.649 at blend 1.0); research-idea queries want minimum (0.764 at 0.5). One
   global constant is serving two distinct distributions.
4. **Replace `topic` with hand-written queries**, or drop it from headline
   numbers. A generated slice cannot answer the question it is being asked.
5. **Calibrate the LLM judge.** Hand-label ~50 answers, compute Cohen's κ, and
   only then consider gating on RAGAS.
6. **Build a cache hard-negative set** — negations, entity swaps, scope changes
   — before tuning `CACHE_HIT_DISTANCE`. Without it a threshold sweep degenerates
   into "looser is better".
7. **Reranker input budget.** Full abstract scored 1.7 points higher than the
   350-char truncation on the probe in §5.9. Worth an ablation.

---

## 10. Reproduction

The corpus is **rebuilt from the Semantic Scholar API, not distributed**. No
archive is published: querying the API is what the API is for, whereas shipping
~1,100 publisher abstracts as a download is redistribution of third-party text.
Rebuilding is also more useful — it is always current, and it verifies the
ingestion path on the reader's machine rather than handing over an opaque index.

One input cannot be regenerated: the 39 hand-judged relevance labels. Those are
committed as `backend/data/eval_labels.json` (2.8 KB, query text → paper IDs, no
abstracts) and re-applied automatically whenever the eval set is rebuilt. So a
clean clone reproduces the numbers in §5 without any manual labelling.

```bash
# Prerequisites
ollama pull llama3.2:3b && ollama pull all-minilm
cp .env.example .env          # set ADMIN_PASSWORD, CORS_ORIGINS

# Corpus and ground truth
make seed                     # 13 researchers, ~1,100 papers, from the S2 API
make eval-set                 # exact_title / author / topic / manual
#                             committed labels are re-applied automatically

# Only if you want to add or revise judgements:
make label-pool               # pool candidates → data/manual_labels.md
#                             → tick relevant papers
make label-apply              # writes to eval_set.json AND eval_labels.json

# Experiments
make eval-fast                                        # metrics → MLflow
make ablate                                           # §5.6, §5.7
make sweep P=rerank_blend V=0,0.25,0.5,0.75,1         # §5.10
make sweep P=rrf_weight_sparse V=1,2,3,5              # §5.3
make sweep P=fetch_k V=5,10,20,40,80
```

Artifacts: `data/ablation_results.json`, `data/eval_last_run.json`,
`data/retrieval_traces.jsonl` (one record per query), `data/mlruns/`.

Determinism: the eval set is seeded (`--seed 13`); retrieval is deterministic
given a fixed index; LLM query expansion runs at `temperature=0` but is not
bit-reproducible across Ollama versions. Retrieval metrics are unaffected by
generation.

---

## 11. Closing observation

Three of the eleven measurement defects in this report were introduced during
the work that found the other eight, and each was caught the same way: by
checking a number against an independent derivation of it rather than against
expectation. The leaked slice was caught by measuring token overlap; the
reranker misdiagnosis by printing the model's actual input; the labelling error
by comparing applied counts against intent.

Every one of those checks took under a minute. Each invalidated a result that
looked entirely reasonable and would otherwise have been reported as a finding —
in two cases, as a configuration change.
