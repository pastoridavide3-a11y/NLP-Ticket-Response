# Evaluation Protocol

## Principles

1. **No single metric decides anything.** Every aggregate claim is backed by
   at least a lexical metric (ROUGE / BLEU / chrF), a similarity proxy
   (TF-IDF cosine), and a sanity diagnostic (length ratio).
2. **Per-example first.** All metrics are computed per row and stored in
   `results/generations/{model}_{split}_scored.parquet`. Aggregates and
   subgroup breakdowns are recomputed from these files, so the evaluation
   is reproducible without re-running generation.
3. **Bootstrap CIs on every subgroup mean.** 1 000 resamples, 95 % interval
   (`src/evaluation/subgroup.py`). Reporting only the point estimate for a
   subgroup of size 60 would hide real uncertainty.

## Metrics

### Lexical
- **ROUGE-1 / ROUGE-2 / ROUGE-L (F-measure)** with stemming, via
  `rouge_score`.
- **BLEU (sentence-level)** via `sacrebleu.sentence_bleu`, rescaled to
  [0, 1].
- **chrF** via `sacrebleu.sentence_chrf`, rescaled to [0, 1].

### Semantic proxy
- **TF-IDF cosine** between prediction and reference. A shared TF-IDF
  vocabulary is fit per batch on predictions + references; we report the
  row-wise cosine.
- This is weaker than BERTScore. It still captures
  topic-and-vocabulary overlap at a level beyond unigram F1, and ranks
  models and subgroups consistently with ROUGE-L here.
- **Circularity caveat.** The retrieval baseline selects train neighbors
  with a TF-IDF cosine. Scoring its outputs with TF-IDF cosine uses the
  same feature family and therefore flatters retrieval relative to
  seq2seq systems that were not optimized in this space. We report
  TF-IDF cosine as a *diagnostic*, not a primary metric; the ROUGE /
  BLEU / chrF triad drives subgroup claims.
- **Next-phase upgrade:** BERTScore F1 with `roberta-large` (deferred to
  the transformer-upgrade phase, per `requirements.txt` optional block).

### Diagnostic
- **Length ratio** pred_chars / gold_chars. Not a quality metric, but
  correlates with specific failure modes (overly long generic answers,
  truncated responses).

## Subgroup reporting

For every generation model on the test split:

- `{model}_by_priority.csv` and `{model}_by_queue.csv` — per-group
  means, bootstrap 95 % CI bounds, support sizes, plus an `__overall__`
  row.
- `{model}_gaps_{priority|queue}.csv` — max, min, gap, argmax-group,
  argmin-group for each metric.
- `{model}_adjusted_{priority|queue}.csv` — confounder-adjusted per-group
  means for ROUGE-L (the primary metric).

## Confounder controls

Rationale: a department or priority gap could reflect systematic length or
richness differences rather than group-specific linguistic difficulty.

We fit
```
metric_i = β₀ + β₁·ticket_len_i + β₂·gold_len_i + β₃·ticket_ttr_i + ε_i
```
using ordinary least squares (`sklearn.linear_model.LinearRegression`) on
the covariates only (no group dummies), then report
`grand_mean + mean(residuals_in_group)` per subgroup. Including the group
dummies in the fit would force per-group residual means to zero by OLS
first-order conditions, making the adjustment a no-op. If adjusted means
preserve the raw ordering (as they do for retrieval on the department
axis in this dataset), the gap is not an artifact of the covariates we
controlled for.

This is a *partial* check. It does not rule out confounders we do not
measure (topical difficulty of IT-Support tickets, stylistic regularity in
Billing answers, etc.). `LIMITATIONS.md` names this explicitly.

## Qualitative rubric (deferred)

The full plan includes a 1–5 rubric on relevance / completeness / tone over
a stratified ~100-sample review. In this first full version we ship the
infrastructure (`results/samples/worst_{model}_by_{priority|queue}.csv`
plus heuristic tags) and leave the manual labelling pass for the next
phase. The samples are stratified so that every priority / department
cell is represented.

## Cross-model reporting

`results/tables/evaluation/cross_model_priority_rougeL.csv` and the
companion figure show ROUGE-L per priority across all three models on one
plot. This is the headline view used to argue whether the stronger model
closes subgroup gaps or only lifts the mean — the core question of the
project.
