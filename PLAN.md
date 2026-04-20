# Project Plan — Unequal Replies in Automated Customer Support

**Language:** Python 3.11+
**Dataset:** `data/customer_support_tickets.csv` (Tobi-Bueck, ~60K synthetic tickets)
**Core question:** Do semantically distinct ticket subgroups (priority, department) receive systematically worse generated responses?

---

## 0. Repository Layout

```
Ticket Response/
├── CLAUDE.md
├── PLAN.md
├── README.md
├── PROJECT_OVERVIEW.md
├── METHODOLOGY.md
├── EVALUATION.md
├── LIMITATIONS.md
├── requirements.txt
├── environment.yml                  # optional conda alternative
├── data/
│   ├── customer_support_tickets.csv
│   └── processed/                   # cleaned splits, cached features
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classification.ipynb
│   ├── 03_generation.ipynb
│   ├── 04_subgroup_analysis.ipynb
│   └── 05_error_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                    # paths, seeds, label maps
│   ├── data/
│   │   ├── load.py                  # load + schema check
│   │   ├── clean.py                 # text normalization
│   │   └── split.py                 # stratified train/val/test
│   ├── features/
│   │   ├── tfidf.py
│   │   └── embeddings.py            # sentence-transformer encoder cache
│   ├── classification/
│   │   ├── baseline_logreg.py
│   │   ├── transformer_clf.py       # DistilBERT/RoBERTa fine-tune
│   │   └── evaluate.py              # acc, F1-macro, per-class, confusion
│   ├── generation/
│   │   ├── retrieval_baseline.py    # nearest-neighbor over tickets
│   │   ├── seq2seq.py               # FLAN-T5 / BART fine-tune
│   │   ├── rag.py                   # optional RAG combiner
│   │   └── decode.py                # beam/sampling helpers
│   ├── evaluation/
│   │   ├── lexical.py               # ROUGE, BLEU, chrF
│   │   ├── semantic.py              # BERTScore, cos-sim, BLEURT (opt.)
│   │   ├── subgroup.py              # group-wise metrics + bootstrap CI
│   │   └── confounders.py           # length, vocab richness checks
│   ├── error_analysis/
│   │   ├── taxonomy.py              # failure category labels
│   │   └── sampler.py               # stratified error samples
│   └── utils/
│       ├── plotting.py
│       ├── io.py
│       └── seeds.py
├── scripts/
│   ├── prepare_data.py
│   ├── train_classifier.py
│   ├── train_generator.py
│   ├── run_eval.py
│   └── make_report_tables.py
├── results/
│   ├── tables/                      # CSV metrics
│   ├── figures/                     # PNG/PDF plots
│   └── samples/                     # qualitative examples
└── tests/
    └── test_data_loading.py
```

---

## 1. Environment & Tooling

- Python 3.11, virtualenv or conda
- Core libs: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- NLP: `transformers`, `datasets`, `sentence-transformers`, `tokenizers`, `torch`
- Eval: `evaluate`, `rouge-score`, `sacrebleu`, `bert-score`
- Optional: `faiss-cpu` (retrieval), `nltk`, `spacy` (linguistic features)
- Reproducibility: fix seeds in `src/utils/seeds.py`; pin versions in `requirements.txt`
- Hardware: assume single GPU (Colab/local). Fall back to distilled models if VRAM-limited.

---

## 2. Phase 1 — Data Exploration (`01_data_exploration.ipynb`)

Goals: understand schema, distributions, and confounders BEFORE modeling.

Steps:
1. Load CSV, inspect columns, dtypes, missing values.
2. Identify text fields (ticket body, subject, response) and label fields (priority, department, plus any others present).
3. Class distributions: bar plots for priority, department, joint priority×department.
4. Length distributions: token & character counts for ticket and response, per subgroup.
5. Lexical stats: type-token ratio, top n-grams per subgroup.
6. Semantic separability: sentence embeddings (e.g., `all-MiniLM-L6-v2`) + UMAP/t-SNE colored by labels.
7. Document EDA findings in markdown cells; surface confounders (e.g., one priority systematically longer).

Deliverables: plots in `results/figures/eda/`, summary tables in `results/tables/eda/`.

---

## 3. Phase 2 — Preprocessing & Splits (`src/data/`)

- Clean: strip HTML/markup, normalize whitespace, optional lowercase (keep cased version for transformers).
- Drop or flag rows with empty ticket or response.
- Stratified split by **(priority, department)** joint label: train 70 / val 15 / test 15.
- Persist splits to `data/processed/` as parquet for reproducibility.
- Save label encoders / mappings to `src/config.py`.

---

## 4. Phase 3 — Classification (`02_classification.ipynb`, `src/classification/`)

Purpose: confirm labels are recoverable from ticket text → labels carry textual signal that may also affect generation.

Two targets: **priority** and **department** (separate models).

Models:
- **Baseline:** TF-IDF (word 1–2grams, char 3–5grams) + Logistic Regression / Linear SVM.
- **Stronger:** fine-tuned `distilbert-base-uncased` (or `roberta-base` if GPU allows), 2–3 epochs, early stopping on val macro-F1.

Evaluation:
- Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.
- Report per-class support; flag minority classes.
- Calibration check (reliability diagram) — useful if predicted labels feed generation later.

Output: `results/tables/classification_metrics.csv`, confusion plots, brief error notes.

---

## 5. Phase 4 — Generation (`03_generation.ipynb`, `src/generation/`)

Models (at least one baseline + one stronger):

1. **Retrieval baseline (`retrieval_baseline.py`):** encode train tickets with sentence-transformer, FAISS index, return nearest train ticket's response.
2. **Seq2seq (`seq2seq.py`):** fine-tune `google/flan-t5-base` (fallback `t5-small` or `facebook/bart-base`) on (ticket → response). Max input 512, max output 256. AdamW, lr 3e-5, 2–3 epochs, beam=4.
3. **Optional RAG (`rag.py`):** prepend top-k retrieved (ticket, response) pairs to the seq2seq prompt. Only if it sharpens the research question (e.g., reduces subgroup gaps).

Decoding: beam search for primary results; report nucleus sampling sample for diversity inspection.

Output: generated responses on val and test, saved to `results/generations/{model}/{split}.parquet` (cols: id, ticket, gold, prediction, priority, department).

---

## 6. Phase 5 — Evaluation (`src/evaluation/`)

Mixed-metric strategy (per CLAUDE.md Risk 1):

- **Lexical:** ROUGE-1/2/L, BLEU, chrF.
- **Semantic:** BERTScore (F1, `roberta-large` or `microsoft/deberta-xlarge-mnli`), cosine similarity of sentence embeddings.
- **Auxiliary:** response length ratio (pred/gold), distinct-1/2 for diversity.
- **Optional rubric eval:** sample 50–100 generations, score on a 1–5 rubric (relevance, completeness, tone). Document protocol in `EVALUATION.md`.

Statistical care:
- Bootstrap 95% CIs (n=1000) for all aggregate metrics.
- Report subgroup support sizes alongside means.

---

## 7. Phase 6 — Subgroup Analysis (`04_subgroup_analysis.ipynb`)

Heart of the project.

For each generation model:
1. Compute every metric overall, by priority, by department, and by priority×department where support ≥ N (e.g., 50).
2. Tabulate gaps: max−min across groups, with bootstrap CIs.
3. Plot per-subgroup metric distributions (box/violin) plus mean ± CI bars.
4. **Confounder inspection (`confounders.py`):** within each subgroup, regress metric on ticket length, vocab richness, gold response length. Report whether subgroup gaps survive after controlling for these.
5. Compare models: does the stronger generator close subgroup gaps or just lift the mean?

Output: `results/tables/subgroup_metrics.csv`, gap plots, written interpretation.

---

## 8. Phase 7 — Error Analysis (`05_error_analysis.ipynb`)

1. Define failure taxonomy in `taxonomy.py`: generic, missing-detail, off-topic, wrong-tone, fluent-but-unhelpful, hallucinated-fact.
2. Stratified sampling (`sampler.py`): worst-K by metric within each subgroup.
3. Manually label ~100 samples; aggregate failure-type frequencies per subgroup.
4. Quote representative bad outputs in the notebook with anonymization where needed.
5. Tie findings back to the research question: do failure modes differ across subgroups?

---

## 9. Phase 8 — Optional Mitigation (only if Phases 1–7 are clean)

Pick **one** focused experiment:
- **Conditioned generation:** prepend `[priority=...] [department=...]` tokens to the input. Compare subgroup gaps before/after.
- **Retrieval augmentation:** evaluate whether RAG reduces gaps.
- **Reweighting / oversampling:** upweight underperforming subgroups during fine-tuning.

Report: did the gap shrink, stay, or move?

---

## 10. Documentation Tasks

Write alongside code, not after:

- `README.md` — setup, how to reproduce each phase.
- `PROJECT_OVERVIEW.md` — narrative summary of question, dataset, pipeline, headline findings.
- `METHODOLOGY.md` — model and metric choices with rationale and rejected alternatives.
- `EVALUATION.md` — exact protocol, metric definitions, rubric (if used), CI procedure.
- `LIMITATIONS.md` — synthetic-data caveats, metric weaknesses, subgroup imbalance, scope of claims.

---

## 11. Reproducibility & Quality Gates

- Fixed seeds (numpy, torch, transformers).
- Save model checkpoints + tokenizer configs under `results/models/{name}/`.
- All notebooks run top-to-bottom from a clean kernel before commit.
- `scripts/run_eval.py` regenerates every table in `results/tables/` from saved generations.
- Lightweight tests in `tests/` for data loading and metric wrappers.

---

## 12. Suggested Timeline (course-project scale)

| Week | Focus |
|------|-------|
| 1 | Env setup, EDA, splits (Phases 1–2) |
| 2 | Classification baseline + transformer (Phase 3) |
| 3 | Retrieval baseline + seq2seq fine-tune (Phase 4) |
| 4 | Evaluation pipeline + subgroup analysis (Phases 5–6) |
| 5 | Error analysis + optional mitigation (Phases 7–8) |
| 6 | Documentation polish, final report, reproducibility check |

---

## 13. Risk Register (mirrors CLAUDE.md)

- **Weak gen eval** → mixed lexical+semantic+rubric; never single-metric claims.
- **Confounders** → explicit length / richness controls in subgroup analysis.
- **Synthetic-data overclaim** → hedge every conclusion, document in `LIMITATIONS.md`.
- **Project sprawl** → keep main thread = subgroup disparities in generation; gate optional phases on core completion.

---

## 14. Definition of Done

- All five notebooks run end-to-end on the processed splits.
- `results/tables/subgroup_metrics.csv` exists with overall + per-priority + per-department metrics, with CIs and support.
- Error analysis notebook contains ≥100 manually labeled failure samples.
- `PROJECT_OVERVIEW.md` answers the core research question with appropriately hedged claims and a limitations section.
