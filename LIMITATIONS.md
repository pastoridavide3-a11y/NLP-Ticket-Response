# Limitations

## Synthetic data

The Tobi-Bueck corpus is generated, not scraped from real support
transcripts. Observed patterns — including the strong retrieval baseline
(ROUGE-L ≈ 0.70) and the Billing-vs-IT-Support gap — may partly reflect the
generation procedure used to build the dataset (common templates, shared
opening / closing phrases, topic-specific stylistic regularities). Any
claim in `PROJECT_OVERVIEW.md` is qualified with "on this dataset under
this setup."

## Scope filters

- **English only.** The raw corpus is 54 % German / 46 % English. We drop
  the German subset to remove a large silent confounder (different
  tokenization and answer style) and to isolate the clean department
  taxonomy, which only covers English rows.
- **Top-10 support departments only.** The remaining 41 `queue` values are
  web-taxonomy topic labels (e.g. `Pets & Animals/Pet Services`) and do not
  belong to the support-department population the research question is
  about. They appear only in the German subset; after the English filter
  the whitelist is a no-op in practice but is kept explicit so that data
  refreshes cannot silently break the scope.
- **Priorities `critical` and `very_low` are excluded.** They do not appear
  in the English subset. This means the subgroup analysis cannot speak to
  the tail priorities where, in principle, disparities might be largest.
- **Rows with missing gold answers (~21 %) are dropped.** We verified that
  missing `answer` almost always coincides with missing `type` (13 178 out
  of 13 189 rows), which suggests a systematic unlabeled batch rather than
  random missingness. Dropped rows are not used anywhere; their absence
  does not affect the subgroup comparisons among labeled rows, but it
  limits claims about full-corpus performance.

## Metric limitations

- **Retrieval-friendly corpus.** When training tickets have near-duplicates
  in the test set, retrieval looks extremely strong. A direct leakage
  diagnostic (`results/tables/evaluation/retrieval_leakage_diagnostic.csv`)
  shows ~22 % of test tickets have TF-IDF cosine ≥ 0.99 to some train
  ticket. On those, the retrieval baseline copies the paired train
  answer. Its ROUGE-L of 0.70 is therefore a memorization-inflated upper
  bound, not a generic ceiling, and gap comparisons between retrieval and
  a future seq2seq model should be read with that in mind.
- **TF-IDF cosine ↔ TF-IDF retrieval circularity.** Scoring retrieval
  outputs with TF-IDF cosine uses the same feature family the retriever
  optimized over, flattering retrieval. We keep it as a diagnostic, not a
  headline metric; see `EVALUATION.md`.
- **TF-IDF cosine is a weak semantic proxy.** It rewards topical / lexical
  overlap, not paraphrased equivalence. Two answers that are semantically
  equivalent but phrased differently will score low. BERTScore is the
  natural upgrade and is listed in `requirements.txt` as optional.
- **No contemporary LLM reference.** We do not compare against an
  instruction-tuned model. The scope here is classical baselines; this
  version deliberately does not position itself relative to GPT-/Claude-
  class systems.

## Subgroup-analysis limitations

- **Controls are partial.** The confounder regression only handles ticket
  length, gold-response length, and ticket type-token ratio. Deeper
  topical-difficulty covariates (number of named entities, question-answer
  matching, technical vocabulary density) are not modeled.
- **Small cells.** `General Inquiry` has ~60 test rows; bootstrap CIs are
  wide for subgroups of that size. Gap claims are driven by the larger
  cells.
- **Multiple testing.** We report gaps across many metrics × many
  subgroups. The largest single gap is not corrected for multiple
  comparisons; the broader pattern (IT Support consistently lowest across
  5 content metrics) is the evidence, not any single cell.

## Modeling limitations

- **No transformer classifier in this version.** The TF-IDF + LogReg
  baseline is strong (priority macro-F1 ≈ 0.65, department ≈ 0.57) but a
  fine-tuned DistilBERT would likely improve both. We explicitly defer
  this to the next phase; the code path is ready for it (`PLAN.md` §4).
- **No seq2seq generator in this version.** The retrieval and constant
  baselines plus the subgroup-conditioned template cover the research
  question — is quality homogeneous across subgroups? — without requiring
  a fine-tuned seq2seq model. The main unanswered follow-up is whether a
  trained generator closes the department gap that retrieval exhibits.

## Interpretation boundaries

- We do **not** claim that priority or department "causes" worse generation.
  The analysis shows associations between subgroup identity and metric
  level, with partial confounder controls.
- We do **not** claim that results transfer to production support systems.
  The dataset is synthetic; subgroup gaps here are reasonable hypotheses
  for what to measure in real data, not statements about real data.
- A discovered gap ≠ a fairness violation. Some subgroups may be genuinely
  harder (rarer vocabulary, more diverse answer structures). The project
  measures gaps and asks whether modeling choices close them; it does not
  prescribe parity.
