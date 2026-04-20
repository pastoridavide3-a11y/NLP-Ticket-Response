# Project Overview — Unequal Replies in Automated Customer Support

## Research question

Do semantically distinct categories of customer-support tickets — operationalized
through priority and department labels — receive systematically worse generated
responses, even when overall metrics look reasonable?

## Dataset and scope

- **Source.** `data/customer_support_tickets.csv`, the Tobi-Bueck synthetic
  customer-support corpus (~61 k rows).
- **Scope (first full version).** English subset, top-10 real support
  departments, priorities ∈ {low, medium, high}. After filtering and
  removing rows with missing gold answers, **28 254 rows**. Stratified
  70/15/15 split on priority × department → train 19 777, val 4 238, test 4 239.
- **Why this scope.** The German subset (and 42 web-taxonomy "queues" that
  appear only inside it) mixes two distinct populations: real support
  departments and generic Google-topic labels. Restricting to English lets
  us treat `queue` as a clean department variable. Further justification in
  `LIMITATIONS.md`.

## Pipeline

1. **Data prep** (`scripts/prepare_data.py`) — apply scope, normalize
   whitespace, build a combined `subject + body` ticket field, stratified
   split, persist to parquet.
2. **EDA** (`scripts/run_eda.py`) — counts, length distributions, top
   n-grams, TF-IDF → SVD scatter colored by priority / department.
3. **Classification** (`scripts/train_classifier.py`) — TF-IDF (word 1–2g ∪
   char 3–5g) + class-balanced logistic regression for priority and
   department, evaluated with macro-F1 and per-class metrics.
4. **Generation** (`scripts/train_generator.py`) — three baselines:
   - `constant`: single fixed polite response (lower bound).
   - `retrieval`: TF-IDF nearest-neighbor train-answer lookup.
   - `group_template`: for each (priority, department) pair the
     centroid-nearest train answer becomes a fixed group template.
5. **Evaluation** (`scripts/run_eval.py`) — per-example ROUGE-1/2/L, BLEU,
   chrF, TF-IDF cosine, length ratio. Bootstrap 95 % CIs on all aggregate
   means. Per-priority and per-department tables plus gap summaries.
   Confounder-adjusted per-group means for the primary metric.
6. **Error analysis** (`scripts/make_error_samples.py`) — worst-K-by-ROUGE-L
   predictions per subgroup with heuristic tags; per-subgroup tag
   frequencies.

## Headline numbers (test split)

### Classification
| Target     | Accuracy | Macro-F1 |
|------------|----------|----------|
| priority   | 0.659    | 0.649    |
| department | 0.556    | 0.573    |

Labels are well above majority-class baselines (~0.41 for priority, ~0.29
for department), confirming that priority and department correspond to
recoverable textual signal and can be treated as proxies for semantically
distinct ticket types.

### Generation (test, aggregate)
| Model          | ROUGE-L | chrF | TF-IDF cos | Length ratio |
|----------------|---------|------|------------|--------------|
| constant       | 0.143   | 0.205 | 0.035     | 0.73         |
| group_template | 0.231   | 0.369 | 0.069     | 1.96         |
| retrieval      | **0.696** | **0.725** | **0.597** | 1.12   |

Retrieval dominates. The constant floor confirms metrics discriminate; the
group_template sits in between — conditioning on labels alone, without the
ticket text, is not enough.

**Caveat — near-duplicate leakage.** ~22 % of test tickets have a TF-IDF
cosine ≥ 0.99 against some train ticket (see
`results/tables/evaluation/retrieval_leakage_diagnostic.csv`). On those, the
retrieval baseline effectively copies the paired train answer, inflating its
overall score. The retrieval number should therefore be read as an upper
bound "what a nearest-neighbor can do on this synthetic corpus", not as a
generic ceiling. The subgroup *ordering* is still informative since every
department benefits from leakage, but absolute levels are optimistic.

### Subgroup gaps (retrieval, test)
| Subgroup axis | ROUGE-L max − min | max group | min group |
|---|---|---|---|
| priority   | 0.024 | medium              | low        |
| department | 0.045 | Billing & Payments  | IT Support |

- **Priority gaps are small** on this synthetic English subset. Low
  scores slightly below medium/high (ROUGE-L 0.680 vs 0.704 / 0.697). The
  low-vs-medium 95 % per-group bootstrap CIs overlap, but only marginally
  (overlap width ≈ 0.009 ROUGE-L), so "within noise" is not obvious from
  per-group CIs alone; a paired bootstrap over the difference would be
  needed to settle it. The effect size (~0.02 ROUGE-L) is small relative
  to the near-duplicate inflation noted above either way.
- **Department gaps are larger.** IT Support is consistently the hardest
  across ROUGE-1/2/L, BLEU, chrF, and TF-IDF cosine. Billing & Payments is
  consistently easiest — it's a narrow, formulaic domain with repetitive
  answer templates that retrieval exploits.
- **Length ratio varies more than content metrics** (Human Resources 1.40 vs
  Service Outages 1.09). Retrieval outputs can be noticeably longer than the
  gold response in some departments.

### Confounder check

`results/tables/evaluation/retrieval_adjusted_{priority,queue}.csv` reports
per-group means before and after linearly regressing out ticket length, gold
response length, and ticket type-token ratio. The IT-Support-lowest /
Billing-highest ordering persists after adjustment, suggesting the
department gap is not purely a length artifact.

## What this version concludes (hedged)

- On this English, top-10-department subset of the Tobi-Bueck dataset:
  - The retrieval baseline reaches high surface-level quality (ROUGE-L
    ~0.70) because many answer templates recur across similar tickets.
  - **Priority-defined subgroup gaps are small (~0.02 ROUGE-L). Per-group
    95 % bootstrap CIs overlap marginally for low-vs-medium — not a clean
    null, not a clean signal.**
  - **Department-defined subgroup gaps are larger and robust to basic
    length / richness controls.** IT Support is the hardest department for
    the retrieval model; Billing & Payments is the easiest.
  - A pure label-conditioned model (`group_template`) cannot match
    retrieval, indicating that group-level structure alone is insufficient
    — ticket-level text features carry most of the usable signal.

These findings are tentative because the dataset is synthetic (`CLAUDE.md`
Risk 3). They describe model behavior *on this dataset under this setup*,
not a universal fairness claim.

## Next steps

1. Plug in a transformer encoder (DistilBERT) for classification — this is
   the natural stronger model listed in `requirements.txt`.
2. Fine-tune FLAN-T5-base as a seq2seq generator. The current retrieval
   baseline already performs well, so the bar is high: we expect the biggest
   gains in subgroups where retrieval fails (short IT-Support tickets whose
   nearest neighbor answers a superficially similar but topically different
   issue).
3. Add BERTScore to decouple semantic similarity from lexical overlap.
4. Optional mitigation: conditioned generation (prepend
   `[priority=...][department=...]`) and measure whether the department gap
   shrinks.
