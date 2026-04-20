# Unequal Replies in Automated Customer Support

First full version of the project specified in `CLAUDE.md` and `PLAN.md`.

**Research question.** Do semantically distinct categories of customer-support
tickets — defined by priority and department labels — induce systematic
differences in the quality of automatically generated responses?

Dataset: Tobi-Bueck customer-support tickets. Scope for this version:
English subset, top-10 support departments, priority ∈ {low, medium, high}.
Full scope decisions are documented in `LIMITATIONS.md`.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py        # scope filter + stratified splits
python scripts/run_eda.py             # EDA tables + figures
python scripts/train_classifier.py    # priority + department classifiers
python scripts/train_generator.py     # constant / retrieval / group_template
python scripts/run_eval.py            # lexical + semantic + subgroup tables
python scripts/make_error_samples.py  # worst-K per subgroup + heuristic tags
```

Re-running after `prepare_data.py` is deterministic (`src/utils/seeds.py`).

## Repository layout

```
src/                       # modular, reusable code
  config.py                # paths, seeds, scope constants
  data/                    # load, clean, stratified split
  features/                # TF-IDF word + char union
  classification/          # TF-IDF + LogReg + evaluation
  generation/              # constant, retrieval, group_template
  evaluation/              # lexical, semantic, subgroup, confounders
  error_analysis/          # heuristic tagger, worst-K sampler
  utils/                   # seeds, io, plotting
scripts/                   # end-to-end drivers (one per phase)
notebooks/                 # thin presentation layer over saved artifacts
results/
  tables/                  # CSV metrics (eda / classification / evaluation / error_analysis)
  figures/                 # PNG plots
  generations/             # model predictions on val + test (parquet)
  samples/                 # worst-K error samples per subgroup
  models/                  # trained joblib pipelines
data/processed/            # cleaned + split parquet files (reproducible)
tests/                     # smoke tests
```

## Where the results live

| Question | File |
|---|---|
| Overall generation metrics | `results/tables/evaluation/overall_metrics.csv` |
| Per-priority metrics + CIs | `results/tables/evaluation/{model}_by_priority.csv` |
| Per-department metrics + CIs | `results/tables/evaluation/{model}_by_queue.csv` |
| Subgroup gap summary | `results/tables/evaluation/{model}_gaps_{priority|queue}.csv` |
| Confounder-adjusted means | `results/tables/evaluation/{model}_adjusted_{priority|queue}.csv` |
| Cross-model comparison plot | `results/figures/evaluation/cross_model_priority_rougeL.png` |
| Classification metrics | `results/tables/classification/summary.csv` |
| Confusion matrices | `results/figures/classification/` |
| Worst-K error samples | `results/samples/worst_{model}_by_{priority|queue}.csv` |

## What this version does

- **EDA:** counts, length distributions, top n-grams per subgroup, TF-IDF+SVD
  projection, priority×department heatmap.
- **Classification:** TF-IDF (word 1–2gram ∪ char 3–5gram) + class-balanced
  logistic regression for priority and department. Evaluated on val + test
  with per-class metrics, confusion matrices, normalized heatmaps.
- **Generation:** three sensible baselines — constant reply, TF-IDF
  nearest-neighbor retrieval, subgroup-centroid template. Predictions saved
  to parquet for reproducible re-evaluation.
- **Evaluation:** ROUGE-1/2/L, BLEU, chrF, TF-IDF cosine, length ratio. All
  per-example so subgroup aggregates come with bootstrap 95 % CIs.
- **Subgroup analysis:** per-priority and per-department means + CIs, gap
  summary tables, confounder-adjusted means (regressing out ticket length,
  gold-response length, ticket type-token ratio).
- **Error analysis:** worst-K-by-ROUGE-L samples per subgroup with heuristic
  tags (too short / too long / generic / low overlap) and per-subgroup tag
  frequencies.

## What is deferred

BERTScore, transformer classifier (e.g. DistilBERT), and seq2seq generator
(FLAN-T5 / BART) are scaffolded conceptually in `PLAN.md` and flagged in
`requirements.txt` as optional. The current environment runs sklearn-only.
`LIMITATIONS.md` and `METHODOLOGY.md` explain why this is a principled
"first full version" rather than a toy run.
