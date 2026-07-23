# TRACE — Evaluation Guide

## What this document is

A description of how TRACE is measured, and — more importantly — of the traps
that make RAG measurement produce confident numbers that mean nothing.

Earlier versions of this guide inlined full source listings. They drifted from
the code within weeks, and one of the listings contained a bug that was then
faithfully copied into the implementation (see *The title-vs-ID trap* below).
This version describes intent and method, and points at the modules that
implement them.

| Module | Purpose |
|---|---|
| [`eval/metrics.py`](../backend/eval/metrics.py) | Ranking metrics (Recall, nDCG, MRR, Precision, bootstrap CI) |
| [`eval/run_eval.py`](../backend/eval/run_eval.py) | Full eval set → metrics → MLflow, with slice breakdown |
| [`eval/ablate.py`](../backend/eval/ablate.py) | Stage-wise ablation and parameter sweeps |
| [`eval/build_eval_set.py`](../backend/eval/build_eval_set.py) | Generates the labelled query set |
| [`eval/seed_corpus.py`](../backend/eval/seed_corpus.py) | Builds a realistic corpus from real AI/ML authors |
| [`eval/self_retrieval.py`](../backend/eval/self_retrieval.py) | Index-integrity check, runs after every sync |
| [`eval/analyse_feedback.py`](../backend/eval/analyse_feedback.py) | Joins user ratings to traces |
| [`eval/ragas_scorer.py`](../backend/eval/ragas_scorer.py) | LLM-judge scoring (faithfulness, answer relevancy) |
| [`rag/trace.py`](../backend/rag/trace.py) | Per-query trace records — the substrate for all of the above |

> **See also:** [retrieval-experiments.md](retrieval-experiments.md) — the formal
> report of every experiment run against this pipeline, the eleven measurement
> defects found (three of them introduced during the work itself), and the
> evidence behind each adopted configuration value.

---

## 0. The measurement substrate

Nothing below works without this, so it comes first.

Every query emits one JSON record to `data/retrieval_traces.jsonl`:

```json
{
  "query_id": "3f2a…", "query": "graph neural networks for molecules",
  "cached": false, "expanded_query": "…",
  "dense":  [{"id": "a1b2", "distance": 0.41}, …],
  "bm25":   [{"id": "c3d4", "score": 8.12}, …],
  "fused_ids": ["a1b2", …], "guard_dropped": 3,
  "reranker_active": true,
  "reranked": [{"id": "a1b2", "score": 6.71}, …],
  "final_ids": ["a1b2", …],
  "grounding": {"papers_cited": 4, "papers_dropped": 0,
                "people_cited": 2, "people_dropped": 0},
  "stages": {"embed": 4.9, "dense_search": 17.8, "bm25_search": 3.1,
             "fusion": 0.4, "guard": 6.2, "rerank": 71.3, "generation": 9120.4},
  "retrieval_ms": 103.7, "latency_ms": 9231.2,
  "config": { … }
}
```

`query_id` is returned to the browser and sent back with any thumbs-up/down, so
a rating can be attributed to what was actually retrieved. Without that link a
thumbs-down tells you a query was bad and nothing else: not whether it was a
cache hit, whether the reranker was even running, or which documents were shown.

**`retrieval_ms` excludes LLM calls.** Total latency is dominated by generation,
which swamps every retrieval change you are trying to measure. Report them
apart or your tuning work will be invisible in the numbers.

---

## 1. Ground truth

`data/eval_set.json`, built by `eval/build_eval_set.py`, in four slices:

| Slice | Query | Label source | Bias |
|---|---|---|---|
| `exact_title` | A paper's exact title | That paper | None — a floor test |
| `author` | `"<name> research"` | That author's indexed papers | None |
| `topic` | Student-style idea derived from a title | The source paper | **Favours dense retrieval** |
| `manual` | Hand-written research ideas | *(you label these)* | None |

### The leakage problem in the `topic` slice

Generating a query from a document and then asking whether retrieval finds that
document rewards near-paraphrase matching — which is exactly what a dense
bi-encoder does best. Dense-only recall reads high, and the incremental gain
from BM25 and reranking reads low.

Two mitigations are applied: queries are derived from the **title only** while
the index holds title + abstract, and they are phrased as research ideas rather
than summaries. Neither removes the bias.

**So**: treat `topic` numbers as directional. When a decision actually rides on
the result, label a `manual` batch by hand and let it break the tie. If silver
and hand labels disagree in direction, the hand labels win.

### Why the corpus is real

`eval/seed_corpus.py` registers well-known AI/ML researchers and indexes their
actual publication records. Synthetic abstracts are lexically clean and
topically separable — every retriever scores well on them, and the ablation
shows no differences between configurations. Real records bring the shared
jargon, near-duplicate titles and stylistic variance that make hybrid retrieval
worth having in the first place.

Ingestion is bounded by `MAX_PAPERS_PER_PERSON` (default 200) and paginated, so
corpus size and sync memory stay predictable.

---

## 2. Retrieval metrics

Computed by `eval/metrics.py`, reported by `run_eval.py`.

| Metric | Depth | What it tells you |
|---|---|---|
| **Recall@fetch_k** | 20 | The ceiling. Nothing downstream can rank a document retrieval never returned |
| **Recall@k** | 5 | What the user can actually see |
| **nDCG@k** | 5 | Rank-aware quality — the metric a reranker is judged on |
| **MRR@k** | 5 | How high the *first* good result lands |
| **Precision@k** | 5 | Share of shown results that are relevant |
| **no_results_rate** | — | How often the guard returns nothing. Trades off against precision |

### The title-vs-ID trap

The original runner did this:

```python
retrieved = [s["title"] for s in result["sources"]]   # titles
relevant  = item["relevant_paper_ids"]                # IDs
```

No element could ever match, so Hit Rate and nDCG were `0.0` on every run,
regardless of retrieval quality — and were logged to MLflow as if they were
measurements.

`eval/metrics.py` now raises `ValueError` when one side looks like titles and
the other like IDs. **A metric that cannot fail is worse than no metric**, because
it looks like evidence. If you add a metric, add the case that makes it fail.

### Missing data is not zero

`metrics.mean([])` returns `None`, and `run_eval.py` drops `None` metrics before
logging. Previously an absent RAGAS install logged `faithfulness_mean = 0.0`,
indistinguishable in a chart from a genuinely unfaithful system.

### Always bypass the cache

`run_eval.py` calls `pipeline.run(..., bypass_cache=True)`. Otherwise the second
run of an eval reads back its own answers, reports cache latency as retrieval
latency, and pollutes the production cache with eval queries.

---

## 3. Stage-wise ablation

`python eval/ablate.py --markdown` produces an aggregate table and a per-slice
table. Corrected run (1,074 papers, 93 labelled queries, CPU-bound):

| variant | n | Recall@20 | R@5 1-lbl | nDCG@5 | MRR@5 | retr p50 |
|---|---|---|---|---|---|---|
| dense | 103 | 0.727 | 0.732 | 0.561 | 0.556 | 30 ms |
| bm25 | 103 | 0.800 | 0.927 | 0.828 | 0.875 | 17 ms |
| hybrid | 103 | 0.812 | 0.829 | 0.663 | 0.666 | 32 ms |
| hybrid+ce | 103 | **0.812** | 0.915 | **0.845** | **0.889** | 48 ms |

nDCG@5 by slice — and this is the whole point of slicing:

| slice | n | dense | bm25 | hybrid | hybrid+ce | labels |
|---|---|---|---|---|---|---|
| author | 13 | 0.284 | 0.577 | 0.464 | **0.649** | derived |
| exact_title | 40 | 0.854 | 0.947 | 0.789 | **0.975** | derived |
| topic | 40 | 0.359 | 0.872 | 0.593 | **0.866** | derived (leaky) |
| **manual** | 10 | 0.561 | 0.495 | **0.695** | 0.500 | model-labelled draft |

**The manual slice inverts every conclusion the generated slices support.**
On queries written by a human rather than derived from a document:

* BM25 is the **worst** single retriever (0.495), not the best. Its dominance
  everywhere else was the leakage artifact it looked like.
* Fusion earns its place: `hybrid` (0.695) beats both arms alone, which is the
  result hybrid retrieval is supposed to produce and which no generated slice
  showed.
* The cross-encoder **hurts** (0.695 → 0.500), reversing its behaviour on every
  derived slice.

Per-query, the reranker regression is systematic rather than one bad query —
6 of 10 got worse, with deltas from −0.32 to −0.50:

```
query        #rel   hybrid      +ce    delta
manual-002      5    0.509    0.131   -0.378   graph neural networks for molecular…
manual-003     12    1.000    0.684   -0.316   reinforcement learning for robotic…
manual-007      6    0.723    0.339   -0.384   efficient fine-tuning of large models…
manual-008      4    0.637    0.246   -0.390   using diffusion models for scientific…
manual-009      1    1.000    0.500   -0.500   evaluating factual grounding in RAG
manual-010      5    1.000    0.509   -0.491   curriculum learning for sample-efficient…
manual-005      7    0.485    0.723   +0.237   self-supervised pretraining for video…
manual-006      3    0.765    1.000   +0.235   detecting and mitigating social bias…
manual-011      9    0.830    0.869   +0.038   multimodal models for robotics
```

### Fixing it: two hypotheses, one survived

**Hypothesis 1 — wrong model.** `ms-marco-MiniLM` is trained on short
search-log queries; TRACE gets long natural-language research ideas. Swapping
in `BAAI/bge-reranker-base`, trained on longer and more varied pairs:

| slice | n | hybrid | +ms-marco | +bge |
|---|---|---|---|---|
| manual | 10 | **0.728** | 0.519 | 0.502 |
| exact_title | 40 | 0.789 | **0.975** | 0.966 |

No improvement — two independently trained cross-encoders regress identically.
The cause is not one model's training distribution. The bge variant is kept in
`ablate.py` for future comparison, but ms-marco stays: equal quality, 1.15 GB
peak RSS against 2.67 GB.

**Hypothesis 2 — task mismatch.** Cross-encoders are trained to answer "does
this passage answer this query". TRACE needs "is this the most useful thing
this institute has written", over a corpus where the honest answer is often a
near-miss. On *"evaluating factual grounding in retrieval-augmented
generation"* the reranker scores the corpus's only RAG paper at **-2.9** and
demotes it. By its own objective that is correct — the paper proposes a RAG
framework, it does not evaluate factual grounding. For the product it is wrong,
because that paper is the best answer available.

This survives. The fix is not to replace the reranker but to stop letting it
override fusion outright: `RERANK_BLEND` combines the two by normalised rank
(raw logits and RRF scores are incomparable, and logit ranges differ across
reranker models, so blending raw values would silently change meaning on a
model swap).

| slice | n | 0.0 | 0.25 | **0.5** | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| author | 13 | 0.464 | 0.550 | 0.594 | 0.639 | **0.649** |
| exact_title | 40 | 0.789 | 0.830 | 0.935 | 0.966 | **0.975** |
| manual | 10 | 0.728 | 0.749 | **0.764** | 0.655 | 0.519 |
| topic | 40 | 0.593 | 0.642 | 0.745 | 0.844 | **0.866** |

Default is now **0.5**. The derived slices all want 1.0; `manual` peaks at 0.5
and is the only slice whose queries a human wrote. Note that 0.75 does *not*
resolve the regression — manual at 0.655 is still below no-reranking at 0.728.

### How much to believe this

Not much yet, and the reasons are worth being precise about.

**n = 10.** The runner flags it, and it should be flagged: single-query swings
dominate.

**The labels are model-generated.** They are marked `label_source: "model"` in
`eval_set.json` and must not be reported as hand-labelled. A model judging
topical relevance shares priors with a semantic reranker, so the bias should
have *inflated* the cross-encoder here. That the regression appears anyway
makes it more interesting than a result confirming the bias — but it does not
make it sound.

**A competing explanation that fits the data.** These labels are inclusive
(up to 12 relevant papers for one query). nDCG@5 with many relevant documents
rewards filling the top-5 with any of them — which recall-shaped fusion order
does well and a precision-shaped reranker may not. Part of the regression could
be labelling style rather than reranker behaviour. Note it does not explain
everything: `manual-009` has exactly one relevant paper and still dropped from
1.000 to 0.500, meaning the reranker demoted the single correct answer.

**Therefore:** this is the highest-value lead in the eval, and it is a lead.
The next step is human verification of these 53 judgements, not a config change.
If the pattern survives human labels, routing decisions past the cross-encoder
— or replacing `ms-marco-MiniLM` with a model trained on longer, natural
queries — becomes the top retrieval priority.

### The topic slice is still not fit for purpose

After regenerating it from abstract terms rather than title terms, the
title-overlap check passed (median 1.00 → 0.00). But the obvious follow-up
check did not:

```
topic-slice query tokens appearing VERBATIM in the target ABSTRACT:
  median=1.00  mean=1.00  n=40
```

The leak moved; it did not go away. Queries built by sampling abstract tokens
are a verbatim bag-of-words drawn from the target document — ideal for BM25 by
construction, and actively hostile to a bi-encoder, which must embed a
disjointed keyword list that resembles no real query. Dense scoring 0.356 on
this slice is an artifact of the generator, not a property of dense retrieval.

So 80 of 93 labelled queries (exact_title + topic) reward verbatim lexical
overlap. **Do not read "BM25 beats dense" off this table.** Any generator that
derives a query from the document it must retrieve will leak somewhere; the
question is only where. The fix is not a cleverer generator — it is
hand-written queries.

**Consequence for tuning:** do not sweep `RRF_WEIGHT_SPARSE` or any
dense/sparse balance on this eval set. A sweep would faithfully optimise for
verbatim matching and hand back a config that is wrong for real student
queries. Label the `manual` queries first; that slice is small but it is the
only one with no generation bias, and it is the tiebreaker.

### Labelling the manual slice

```bash
make label-pool     # pools candidates from dense + bm25 + hybrid, reranker OFF
#                   → edit backend/data/manual_labels.md, tick relevant papers
make label-apply    # writes judgements back into eval_set.json
```

Two properties of this procedure matter:

* **Candidates are pooled across configurations**, not taken from one system's
  output. Labelling only what `hybrid+ce` returned would encode that
  configuration's preferences as ground truth and guarantee it wins.
* **The reranker is disabled while pooling**, so the component under test does
  not decide what gets judged.

This is standard TREC pooling. It does not eliminate bias — a paper no
retriever surfaced is never judged, so recall is measured against the pool
rather than the corpus — but it removes the circularity that makes a
self-graded eval meaningless.

When judging, prefer a false negative to a false positive: an incorrectly
marked paper rewards retrievers for finding the wrong thing. A query with zero
relevant papers is a legitimate label — it measures the no-results path.

### Methodology defects found while validating the first run

Every one produced a confident-looking number that was wrong. They are listed
because each is a general trap, not a TRACE quirk.


**1. The silver slice was 100% leaked.** `topic` queries were generated by
rewording paper titles. Measured afterwards, all 39 had *every* token present
verbatim in the source title — an exact-title test in disguise. That made 86%
of the labelled set lexical title matching, which single-handedly produced the
result "BM25 beats hybrid". Always measure query/document token overlap on a
generated slice before trusting it:

```python
overlap = len(set(tokenize(query)) & set(tokenize(title))) / len(set(tokenize(query)))
```

**2. A slice regression was blamed on the model instead of the code.** The
cross-encoder scored badly on author queries, and the first reading was "ms-marco
is not trained for metadata lookups". The actual cause: the reranker's document
text was `title + page_content`, and author names live only in metadata — the
model could not see the field the query was about. Before concluding a model is
unsuited to a query class, print exactly what you fed it.

**3. A global default was tuned on an artifact of that bug.** A sweep appeared
to show `RRF_WEIGHT_SPARSE=2.0` winning. But `exact_title` and `topic` scored
*bit-identically* at every weight — the entire gain came from 13 author queries
depressed by defect 2. Tuning a constant on 13 queries would have baked a
workaround for an unrelated bug into the config. **If a sweep moves the
aggregate, check which slice moved. If only one did, you are tuning that slice.**

**4. Recall was averaged across incomparable scales.** Author queries carry a
median of 109 relevant papers, so their Recall@5 is capped at 5/109 ≈ 0.046;
single-label queries are effectively 0 or 1. Averaging them gives a number that
describes neither. `ablate.py` now reports a single-label Recall column
alongside; nDCG and MRR are normalised and were always safe to compare.

A fifth, smaller one: the latency column read `total p50` but generation is not
run during retrieval-only evaluation, so ~350 ms was retrieval + query
expansion — off by 10-30 s from user-facing latency. It is now labelled
`+expand p50`.

### Sweeps

```bash
python eval/ablate.py --sweep fetch_k        --values 5,10,20,40,80
python eval/ablate.py --sweep search_ef      --values 10,50,100,150,300
python eval/ablate.py --sweep min_similarity --values 0.6,0.7,0.85,1.0
python eval/ablate.py --sweep cache_distance --values 0.04,0.06,0.08,0.10,0.12
```

Plot quality against `retrieval_p50_ms` and take the knee of the curve.

**Sweep order matters.** Sweeping `fetch_k` against an inactive reranker
measures the wrong system: more candidates with nothing to filter them *hurts*
Precision@5. Confirm `reranker_active=true` first — `ablate.py` prints it per
variant.

**For the cache-distance sweep you need hard negatives**, or the sweep
degenerates into "looser threshold = more hits = better". Build pairs
deliberately: paraphrases that *should* hit, plus negations, entity swaps and
scope changes ("RAG for medical imaging" vs "RAG for medical imaging
*excluding* radiology") that must not. Those sit at tiny embedding distance and
are the whole risk.

---

## 4. Slicing

A single average hides where search fails. Author-name queries, exact-title
lookups and broad topic queries fail for different reasons and are fixed by
different changes. `run_eval.py` prints nDCG per `query_type` with a bootstrap
95% CI.

**Watch the sample size.** At 20–30 queries per slice, a 0.05 nDCG gap between
slices is usually sampling noise. The runner flags slices under 30 and prints
the interval; if you need a slice to carry a decision, grow it to 150+ or accept
that you are reading variance.

A worked example of slicing paying off: the `author` slice reads near-zero on
any build where author names are absent from the index. That is a data-coverage
bug, not a ranking bug, and no aggregate number distinguishes the two.

---

## 5. Generation and grounding

The hallucination guard in [`rag/chain.py`](../backend/rag/chain.py) checks every
cited paper against the retrieved documents and every suggested person against
their authors, and returns counts. Those counts land in each trace, so:

| Metric | Target | Why |
|---|---|---|
| `paper_grounding_rate` | > 0.95 | Fraction of cited papers traceable to context |
| `person_grounding_rate` | > 0.95 | Same for suggested people |
| `generation_failure_rate` | < 0.01 | Schema/parse failures |

**Person grounding is the highest-severity metric here.** Sending a student to a
professor who does not exist — or attributing work to the wrong person — is the
most damaging output this system can produce, and the one a student is least
able to check. It requires no human labelling to measure.

### LLM-judge metrics

RAGAS faithfulness and answer relevancy run through the same small local model
that generated the answer. Before trusting either number, hand-label ~50 answers
and compute agreement (Cohen's κ). Until you have done that:

- Report judge metrics; do not gate on them.
- Gate CI on the deterministic ones: Recall, nDCG, grounding rate, schema-failure
  rate.

`ENABLE_RAGAS_SCORING=true` puts judge calls in the request path and adds
seconds of user-visible latency. It is for diagnosis, not steady state.

---

## 6. Online health

From `python eval/analyse_feedback.py`, which joins ratings to traces:

| Metric | Watch for |
|---|---|
| Thumbs-down rate, **attributed** | Split by cache hit / no-results / reranker inactive |
| `cache_hit_rate` | Meaningless without the false-hit rate beside it |
| `no_results_rate` | Rising usually means a stale index, not a model problem |
| `reranker_inactive_rate` | Should be 0. Anything else is a silent ranking regression |
| Latency p50/p95, **split** | `retrieval_ms` vs total |

A cache speedup quoted alone invites the obvious question. State it as: *N% of
queries served from cache at 8 ms p50 vs 2.6 s cold, measured false-hit rate M%.*
The bare ratio does not survive scrutiny.

---

## 7. Index guardrails

Run after every sync (`pipeline.post_sync_rebuild()` returns the rate; the API
marks the sync `degraded` below 0.90):

- **Self-retrieval Hit Rate@5** — each paper's own abstract must retrieve that
  paper. Detects embedding corruption and model mismatch. It is an *integrity*
  check, not a relevance metric; it is close to trivially satisfiable, so do not
  read a passing score as evidence of search quality.
- **Coverage** — share of registry members with ≥1 indexed paper; share of
  papers with a real abstract. Title-only documents are nearly invisible to
  dense retrieval.
- **Document-count delta per sync** — a sudden drop catches truncation and
  metadata-merge regressions.

---

## 8. Cadence

| When | What |
|---|---|
| Every query | Trace record; grounding counts |
| Every sync | Self-retrieval, coverage, doc-count delta |
| Every PR touching retrieval | `run_eval.py --no-ragas` + `ablate.py`; compare to baseline in MLflow |
| Weekly | `analyse_feedback.py`; review attributed thumbs-down |
| Before a config change ships | The relevant sweep, with the Pareto curve |

### Measure the baseline before you fix it

When fixing a retrieval bug, run the eval **before** the fix. That broken
baseline is the "before" column of every table worth showing; without it you
have one number and no narrative. It costs one extra eval run.

---

## 9. Installing eval dependencies

```bash
pip install mlflow                 # experiment tracking (required by run_eval)
pip install ragas datasets         # LLM-judge metrics (optional)
pip install deepeval               # pytest regression suite (optional)
```

```bash
# Full loop from an empty index
python eval/seed_corpus.py --ingest      # build a realistic corpus
python eval/build_eval_set.py            # label it
python eval/run_eval.py --no-ragas       # metrics → MLflow
python eval/ablate.py --markdown         # stage-wise table
make mlflow-ui                           # compare runs
```
