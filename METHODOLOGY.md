# Methodology

This document explains the modeling choices made in the first full version
and the alternatives that were rejected or deferred.

## Scope decisions

- **English-only.** The raw corpus is 54 % German / 46 % English. A single
  model spanning both languages would either need a multilingual encoder
  (adds dependency, reduces interpretability) or would hide language as a
  silent confounder in subgroup analysis. Holding language fixed removes one
  large axis of variation.
- **Top-10 support departments only.** The `queue` column has 51 distinct
  values. Inspection shows two disjoint populations: 10 real support
  departments (e.g. `Technical Support`, `Billing and Payments`) and 41
  web-taxonomy topics (e.g. `Pets & Animals/Pet Services`). The latter only
  appear in the German subset. After the English filter, every row is
  already in a real department; the explicit whitelist is defensive.
- **Priority ∈ {low, medium, high}.** The English subset contains only
  these three priorities; `critical` and `very_low` only appear in German.

## Splits

- **Stratified on priority × department.** Standard stratified
  `train_test_split` in two stages (70 train → 15 val / 15 test). This
  preserves subgroup balance so bootstrap CIs on the test set are meaningful
  for every cell of the subgroup tables.

## Features

- **TF-IDF word 1–2gram ∪ char 3–5gram** (`src/features/tfidf.py`). Word
  n-grams capture content; character n-grams are robust to typos,
  morphological variation, and out-of-vocabulary product names. The union
  is implemented via `sklearn.pipeline.FeatureUnion` so that downstream
  models see one feature matrix.
- **Sublinear TF + min_df=2** to damp very frequent terms and drop hapaxes.

## Classification

- **Target 1 — priority** (3 classes). Imbalance: medium 41 %, high 39 %,
  low 20 %. Without class weights the classifier over-predicts medium and
  high. We use `class_weight="balanced"`.
- **Target 2 — department** (10 classes). Heavily imbalanced, with
  `General Inquiry` at ~400 rows and `Technical Support` at ~8 k.
- **Model.** TF-IDF + `LogisticRegression(solver="lbfgs",
  class_weight="balanced", max_iter=2000)`. This is the strongest
  interpretable linear baseline for sparse text at this scale.
- **Solver note.** Since scikit-learn 1.8, `liblinear` no longer supports
  multiclass directly; `lbfgs` is the natural multinomial replacement.

Rejected / deferred:
- **SVM.** Comparable performance to logistic regression on text at this
  scale, no probabilistic outputs; skipped to avoid redundancy.
- **DistilBERT fine-tune.** Requires `torch` + `transformers` (see
  `requirements.txt` optional block). Recommended as the next upgrade; the
  current score (priority macro-F1 0.65, department macro-F1 0.57) is the
  linear baseline that a transformer has to beat to justify the cost.

## Generation models

Three models, each serving a distinct interpretive role.

### 1. `constant` — worst-case floor

Always returns a single polite reply. Any meaningful generator must beat
this on content-sensitive metrics. Test ROUGE-L = 0.143 is the reference
floor.

### 2. `retrieval` — primary non-trivial baseline

Encodes each train ticket with the shared TF-IDF featurizer, builds a
sparse index, and returns the gold answer of the closest train ticket for
each test ticket. Rationale: many support corpora have strong
ticket-to-ticket redundancy — if similar tickets received similar
responses in training, nearest-neighbor retrieval is a strong lower bound
for a seq2seq model to beat. Test ROUGE-L = 0.696 makes that bound
explicit.

### 3. `group_template` — subgroup-conditioned baseline

For each (priority, department) cell, we pick the centroid-nearest
training answer as a single fixed template for that subgroup. This isolates
the "how much can I do with only subgroup identity and no ticket text?"
signal. It is also the closest lightweight analog of the "conditioned
generation" mitigation idea from the plan: if group-level conditioning
alone were enough, this would match retrieval. It does not (ROUGE-L 0.231),
which argues against group-identity-only conditioning as a fix.

Rejected / deferred:
- **FLAN-T5 / BART fine-tune.** Deferred to the transformer upgrade.
- **Full RAG.** The plan gates RAG on it serving a specific research
  question. Since the retrieval baseline already performs strongly, we
  introduce RAG only if a subsequent phase finds it closes subgroup gaps
  that a pure seq2seq model does not.

## Evaluation metrics

Per `CLAUDE.md` Risk 1, we use a mixed strategy rather than a single
surface metric.

**Per-example:**
- ROUGE-1 / ROUGE-2 / ROUGE-L (`rouge_score`, F-measure).
- BLEU and chrF (`sacrebleu`, sentence-level, normalized to [0, 1]).
- TF-IDF cosine between prediction and reference (`src/evaluation/semantic.py`)
  as a semantic-similarity proxy. BERTScore is deferred to the transformer
  upgrade and discussed in `EVALUATION.md`.
- Length ratio (pred / gold character length). Not a quality metric on its
  own but a diagnostic: very short or very long outputs correlate with
  specific failure modes.

**Aggregate:** bootstrap mean + 95 % CI from 1 000 resamples
(`src/evaluation/subgroup.py`). CIs make subgroup comparisons interpretable
at the smallest cells (e.g. General Inquiry, ~60 test rows).

## Subgroup analysis

For each generation model and each axis (priority, queue) we report:

1. Per-group mean + bootstrap 95 % CI and support size.
2. Gap = max − min over groups, with the argmax / argmin group.
3. Confounder-adjusted per-group means from a linear regression
   `metric ~ ticket_len + gold_len + ticket_ttr + one-hot(group)`. The
   adjusted mean adds the group-mean residual to the grand mean; comparing
   raw vs adjusted means shows how much of the gap is explained by simple
   covariates.

## Error analysis

- **Heuristic tagger** (`src/error_analysis/taxonomy.py`) flags four
  automatic categories: `too_short` (<80 chars), `too_long` (>2.5× gold),
  `generic` (matches stock phrases), `low_overlap` (pred ∩ gold tokens /
  |gold| < 0.15). These are pre-labels, not judgments — they focus manual
  review without claiming the taxonomy is complete.
- **Worst-K per subgroup**: the 10 lowest-ROUGE-L predictions in each
  priority / department cell are exported to `results/samples/` for
  inspection. Manual labeling of ~100 samples across subgroups is the
  next-phase task documented in `PLAN.md` Phase 7.
