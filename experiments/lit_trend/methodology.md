# Methodology — Publication trend: runtime model/configuration selection in resource-constrained AI

## Database and query date

- Database: **OpenAlex** REST API (`api.openalex.org/works`), searched over titles and
  abstracts (`title_and_abstract.search` filter; OpenAlex applies stemming).
- Query date: **2026-08-18** (recorded in every cache file and figure caption).
- Retrieval: one boolean query per selection phrase, cursor pagination
  (`per-page=200`), polite-pool `mailto`, exponential backoff on HTTP 429. Raw
  responses cached under `cache/` (final run) and `cache_v1/` (first pass); rerunning
  statistics/plots does not re-download.

## Search definitions

Every query is `(selection phrase) AND (AI context group) AND (edge/resource context
group)`, restricted to `publication_year:2015-2025`. Full query strings:
`search_queries.txt`.

- **PRIMARY selection phrases**: "adaptive model selection", "dynamic model
  selection", "runtime model selection", "run-time model selection", "online model
  selection", "model switching", "model routing", "configuration selection".
- **BROAD (sensitivity)** adds: "adaptive inference", "dynamic inference", "runtime
  adaptation", "run-time adaptation", "adaptive DNN", "dynamic DNN". PRIMARY and
  BROAD are computed and reported separately, never mixed.
- **AI context (v2)**: "deep learning" OR "deep neural network" OR "neural network"
  OR DNN OR CNN OR "large language model" OR LLM.
- **Edge/resource context (v2)**: "edge computing" OR "edge AI" OR "edge device" OR
  "edge server" OR "embedded system" OR "embedded device" OR "embedded AI" OR
  "mobile device" OR "mobile edge" OR "on-device" OR "resource-constrained".

## Query refinement (v1 → v2)

The first pass (v1) used AI = {…, "inference"} and edge = {…, bare "embedded", bare
"mobile"}. A full manual census of all 78 v1 PRIMARY works measured precision at
**46% strict / 76% including borderline** — below the pre-registered 80% bar — with
two dominant false-positive modes: bare *inference* admitting statistical-inference
work (Bayesian model selection in transport modelling, gene circuits, econometrics),
and bare *mobile*/*embedded* admitting other-domain phrases ("mobile crane
configuration selection", aerospace configuration design). v2 therefore (a) removed
"inference", adding the unambiguous AI tokens CNN / LLM / "large language model"
that previously matched only through it, and (b) replaced bare mobile/embedded with
device phrases, adding "edge server". The change is applied identically to all years
and both definitions (a precision refinement, not curve tuning); v1 counts are
preserved in `stats_report_v1.txt` and show the same qualitative trend.

## Deduplication

Within each definition, works are unioned across the per-phrase queries and
deduplicated by (1) OpenAlex work ID, (2) DOI, (3) normalized title (lowercased,
punctuation stripped). Known residual: one arXiv/venue pair survived because the
arXiv title contains a literal "\n" sequence (noted below).

## Inclusion / exclusion criteria (validation labels)

A work is **Relevant** if it concerns runtime/adaptive selection or switching among
AI models or inference configurations on resource-constrained/edge platforms;
**Borderline** if adjacent (adaptive execution without discrete selection,
deployment-time rather than runtime selection, selection outside the strict edge
frame); **Irrelevant** otherwise (statistical model selection, other-domain
configuration selection, in-domain works without a selection/adaptation mechanism).
For BROAD, adaptive execution mechanisms (partitioning, early exit, dynamic
networks) count as Relevant by construction.

## Validation

- **PRIMARY: full census, not a sample** — all 55 v2 works (and all 78 v1 works)
  were individually labeled (`validation_sample.csv`, reasons included). Per-stratum
  pools were smaller than the pre-specified 20-per-stratum sample, so the census
  supersedes sampling. Result: **51% strict / 85% incl. borderline**.
- **BROAD: stratified random sample** (seed 0), 10 per stratum from the broad-only
  pool (≤2019, 2021, 2023, 2024, 2025; 49 total): **71% strict / 86% incl.
  borderline**. Label rates are roughly stable across strata (no evidence that
  precision decay manufactures the trend).
- **Manually verified PRIMARY trend** (Relevant labels only): 2015–2019: 0 ·
  2020: 2 · 2021: 5 · 2022: 3 · 2023: 3 · 2024: 2 · 2025: 13 (total 28). This
  verified series is the strongest defensible statement.

## Limitations

- Phrase search misses works that describe the concept without any indexed phrase;
  counts are lower bounds on the niche, and the PRIMARY niche is small (tens/year).
- Strict precision of the automated PRIMARY query is 51%: residual false positives
  are mostly in-domain-adjacent works (pruning, accelerators) that keyword filters
  cannot remove; the verified census corrects for this.
- 2025 may still be incompletely indexed (OpenAlex ingestion lag), which would bias
  the last point downward, not upward. 2026 is excluded (incomplete year).
- One known residual duplicate (Fast YOLO arXiv/venue pair) in BROAD; impact ≤1.
- OpenAlex venue metadata is noisy (arXiv/SSRN aggregations); topic labels are
  OpenAlex `primary_topic`, not hand-assigned.
- Labels were assigned by a single annotator (this analysis) from titles/abstract
  excerpts; borderline judgments are conservative but subjective.

## Reproduction

`~/.venvs/sc2bench/bin/python openalex_model_selection_analysis.py` (add
`--refetch` to ignore caches). Outputs: `primary_counts.csv`, `broad_counts.csv`,
`all_primary_papers.csv`, `all_broad_papers.csv`, `validation_sample.csv`,
`search_queries.txt`, `stats_report.txt`, figures (PNG 300 dpi + PDF).

## Tier 3 — runtime-adaptive inference ecosystem (added 2026-08-18, same query date)

Motivation: the PRIMARY lexical niche understates the field because the concept
lives under fragmented vocabulary (NestDNN-style multi-capacity models, Chameleon-
style configuration adaptation, INFaaS-style serving, early exit, partitioning).
Tier 3 unions the mechanism families — the PRIMARY and BROAD phrases plus: early
exit, multi-exit, model/inference cascade, DNN/model partitioning, split
computing/inference, collaborative inference, inference/computation offloading,
inference/model serving, anytime inference, input-adaptive, multi-capacity,
slimmable, configuration adaptation — under the SAME v2 AI and edge context groups
(`ecosystem_analysis.py`; outputs `ecosystem_counts.csv`, `all_ecosystem_papers.csv`,
`ecosystem_trend.{png,pdf}`).

Normalization: yearly ecosystem counts are divided by the yearly total of ALL works
matching (AI context AND edge context) — the edge-AI baseline, obtained via a single
`group_by=publication_year` call. This separates "the topic grew" from "everything
grew".

Result: 1,647 unique works; yearly counts increase monotonically 2015→2025
(2 → 480; 2019→2025 = 8.7×). The baseline itself grew 6,239 → 32,014 over the same
span; the ecosystem's share of edge-AI output rose from 0.22% (2015) and 0.88%
(2019) to ~1.5%, where it has held steady since 2021. Honest reading: absolute
interest rises every single year; relative to the (itself fast-growing) edge-AI
field, the topic expanded its share strongly through 2021 and has since grown at the
same pace as the field. Validation depth: tier 3 is a breadth indicator built from
mechanism-family phrases that are individually far less ambiguous than "model
selection"; it received spot checks, not the census/sample labeling applied to
PRIMARY/BROAD.
