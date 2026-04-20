# Project Guidance: Unequal Replies in Automated Customer Support

## Project Context

This project studies automated customer support with two connected NLP components:

1. a **classification component**, which assigns labels such as ticket priority and department;
2. a **generation component**, which produces a response to the customer ticket.

The broader motivation is practical and industrial: in a realistic support pipeline, both components may coexist. The classification model helps route or characterize the request, while the generation model produces a response. Even if the generation model is trained on the full ticket distribution, it may not perform equally well across all kinds of requests.

The central concern of this project is **response quality consistency**. Ideally, a response generator should maintain a comparable level of usefulness and adequacy across semantically different types of tickets. If some kinds of requests systematically receive worse generated responses, that is a meaningful operational weakness.

This project should be implemented in a way that fits a standard NLP course project: exploratory data analysis, multiple models, strong baselines, clear evaluation, and explicit error analysis. The final output should include code, notebooks, and documentation that explain both the implementation and the reasoning behind the choices.

---

## Core Research Question

The core research question is:

**Do semantically distinct categories of customer support tickets induce systematic differences in the quality of automatically generated responses?**

A more operational version is:

**Does response generation quality vary across ticket subgroups defined by priority and department labels?**

These labels are not treated as arbitrary metadata. They are treated as **proxies for meaningful differences in the input text**. The working hypothesis is that the same linguistic properties that make a ticket likely to belong to a given urgency level or department may also influence how difficult it is for a generation model to produce a good response.

This does **not** mean that classification causes generation quality differences. The project should avoid causal claims unless they are strongly justified. The safer interpretation is:

- tickets differ in linguistic and semantic structure;
- these differences are partially reflected in labels such as priority and department;
- the generation model may perform unevenly across these subgroups;
- such unevenness should be measured, analyzed, and possibly mitigated.

---

## Intended Contribution

This project is **not** just about maximizing overall generation quality.

The intended contribution is to go beyond average performance and ask whether the model behaves **uniformly** across different input categories.

The project should therefore aim to answer questions such as:

- Is generation quality homogeneous across ticket types?
- Are some priorities or departments systematically harder for the generator?
- Do aggregate metrics hide important subgroup failures?
- Can these disparities be reduced through better modeling or conditioning?

This framing is relevant both:
- **industrially**, because a support system should be reliable across ticket types, especially critical ones;
- **academically**, because it relates to robustness, subgroup analysis, and heterogeneous model performance across semantically distinct inputs.

---

## Dataset Assumptions

The project is based on the **Tobi-Bueck Customer Support Tickets** dataset, which contains roughly 60K synthetic support tickets and includes:

- ticket text
- response text
- priority label
- department label

Important note: because the dataset is synthetic, all conclusions must be phrased carefully. Any discovered asymmetry may reflect:
- genuine modeling difficulty,
- dataset construction artifacts,
- label distribution effects,
- stylistic regularities in the synthetic responses.

The project should therefore avoid overstating its conclusions. It is acceptable to conclude that a generation model exhibits disparities **on this dataset and under this setup**, but not that the phenomenon universally holds in all real customer-support settings.

---

## What the Project Should Do

The project should include at least the following broad components.

### 1. Data Exploration
Perform a careful exploratory analysis before modeling. This should include, when possible:

- class distributions for priority and department
- ticket and response length distributions
- imbalance analysis
- lexical and semantic differences across groups
- examples of representative tickets by class
- possible correlations between labels and textual properties

This stage is important because later subgroup comparisons are only meaningful if the structure of the dataset is understood.

### 2. Classification Analysis
Build one or more models that predict:

- priority from ticket text
- department from ticket text

The purpose is not only to get a decent classifier, but also to establish that the labels are in fact recoverable from the language of the ticket. This supports the idea that the labels capture meaningful textual variation.

The project should include:
- at least one simple baseline
- at least one stronger model
- standard evaluation metrics
- confusion/error analysis

### 3. Response Generation Analysis
Build one or more models that generate a response from the ticket text.

The project should compare:
- simple and interpretable baselines
- stronger generative approaches
- possibly retrieval-based or retrieval-augmented approaches if justified

The key goal is not just to report one overall score, but to measure generation quality **across subgroups**.

### 4. Subgroup Evaluation
This is the heart of the project.

Evaluate generation quality:
- overall
- by priority
- by department
- optionally by priority × department intersections if sample sizes allow it

The analysis should determine whether some subgroups receive systematically worse generated responses.

### 5. Error Analysis
Include a qualitative and quantitative analysis of failure modes. This should answer:
- when does the model fail?
- how does it fail?
- are failures different across subgroups?

Examples of failure categories may include:
- overly generic responses
- missing key details from the ticket
- semantically off-target replies
- correct tone but low usefulness
- fluent but unhelpful responses

---

## What the Project Should Not Do

The project should avoid becoming too broad or too speculative.

In particular, avoid the following unless they are clearly justified and feasible:

- making strong causal claims about semantics causing quality differences
- using too many large models without clear comparison logic
- relying on a single weak evaluation metric
- introducing many loosely connected components that weaken the main story
- overcomplicating the pipeline before establishing strong baselines

The project should prefer a **clean, interpretable, well-evaluated design** over an overly ambitious but shallow implementation.

---

## Methodological Philosophy

Claude should treat this project as a **carefully scoped empirical NLP study**, not as a vague “use LLMs somewhere” task.

Key principles:

1. **Start simple, then deepen**
   - establish baselines first
   - only add complexity if it clearly improves the analysis

2. **Prefer interpretable comparisons**
   - each modeling choice should answer a question
   - avoid adding components that do not support the research question

3. **Do not rely only on average metrics**
   - subgroup-level analysis is essential

4. **Be conservative in claims**
   - if a result is ambiguous, say so
   - document limitations explicitly

5. **Justify choices**
   - every major step should be explained in prose, not just implemented in code

---

## Suggested but Not Mandatory Modeling Directions

This section is intentionally flexible. Claude may choose the most appropriate stack depending on data format, available compute, and implementation practicality.

### Classification
Reasonable options include:
- bag-of-words or TF-IDF + logistic regression / linear SVM
- pretrained encoder fine-tuning
- sentence embedding based classifiers

A simple baseline is strongly encouraged, and at least one stronger model should be considered.

### Generation
Reasonable options include:
- nearest-neighbor or retrieval-based baselines
- sequence-to-sequence transformer models
- instruction-tuned text-to-text models
- retrieval-augmented generation, if it serves the research question

Generation should not be limited to a single model. At least one baseline and one stronger approach are desirable.

### Retrieval / RAG
RAG is allowed, but it should not be included just because it is fashionable.

It is useful only if it supports a clear research question such as:
- does retrieval improve response quality?
- does retrieval reduce subgroup disparities?
- does using similar historical tickets make generation more consistent?

If retrieval is used, Claude should clearly distinguish:
- pure retrieval baseline
- pure generation baseline
- retrieval-augmented generation

This helps isolate where any gains come from.

---

## Critical Risks and How to Handle Them

### Risk 1: Weak evaluation of generation
A common failure mode is evaluating generated responses only against the reference response using a surface-overlap metric. This is insufficient because multiple responses may be acceptable.

Claude should therefore prefer a **mixed evaluation strategy**, for example:
- lexical overlap metrics
- semantic similarity metrics
- targeted qualitative evaluation
- optional rubric-based manual review on a sample

The gold response should be treated as an important reference, but not as the only possible valid response.

### Risk 2: Confounding variables
If one ticket category is longer, more technical, rarer, or more ambiguous than another, subgroup performance differences may be due to those factors rather than subgroup identity itself.

Claude should explicitly inspect possible confounders such as:
- text length
- vocabulary richness
- label imbalance
- rarity of examples
- similarity to training examples
- response length and style

Where possible, the analysis should discuss whether observed disparities may be partially explained by these factors.

### Risk 3: Overclaiming from synthetic data
The dataset is synthetic, so results may reflect synthetic generation rules. Claude must document this limitation clearly.

### Risk 4: Project sprawl
The project should not simultaneously become:
- a classification project,
- a generation project,
- a RAG project,
- a fairness project,
- a causal inference project,
- and a human evaluation framework project.

Claude should maintain a coherent main thread:
**generation quality disparities across ticket subgroups**, supported by classification and error analysis.

---

## Preferred Structure of the Analysis

Claude should organize the work roughly along these lines:

1. **Problem framing**
   - explain the pipeline setting
   - define the research question
   - motivate why average performance is not enough

2. **Dataset understanding**
   - inspect schema
   - analyze distributions
   - discuss limitations of synthetic data

3. **Preliminary linguistic analysis**
   - characterize subgroup differences
   - investigate whether labels correspond to meaningful textual differences

4. **Classification experiments**
   - train and compare models for priority and department prediction
   - show which labels are learnable from the text

5. **Generation experiments**
   - build at least one simple baseline
   - build one or more stronger approaches
   - evaluate overall quality

6. **Subgroup evaluation**
   - compare performance by priority and department
   - highlight disparities and uncertainty

7. **Error analysis**
   - provide examples and failure categorization
   - identify practical weaknesses

8. **Optional mitigation analysis**
   - try one focused mitigation if feasible
   - examples: conditioning on labels, retrieval, balancing, reweighting

9. **Conclusions**
   - summarize findings
   - distinguish strong findings from tentative observations
   - document limitations and next steps

---

## Deliverables Claude Should Produce

Claude should generate a complete project workspace, not just isolated code.

Expected outputs include:

### A. Executable notebooks
Claude should create clearly structured notebooks with explanatory markdown and code comments. Suggested notebooks may include:

- `01_data_exploration.ipynb`
- `02_classification.ipynb`
- `03_generation.ipynb`
- `04_subgroup_analysis.ipynb`
- `05_error_analysis.ipynb`

These can be merged or reorganized if a better structure emerges.

### B. Reusable source code
Claude may also create a `src/` directory for modular code, for example:
- data loading and preprocessing
- feature extraction
- training utilities
- evaluation utilities
- plotting and reporting utilities

### C. Documentation files
Claude should create documentation files that explain both the implementation and the rationale behind the choices. Suggested documents:

- `PROJECT_OVERVIEW.md`
  - full project summary
  - research question
  - dataset
  - pipeline
  - main findings

- `METHODOLOGY.md`
  - detailed explanation of modeling choices
  - why specific baselines and metrics were selected
  - trade-offs and rejected alternatives

- `EVALUATION.md`
  - exact evaluation protocol
  - subgroup metrics
  - caveats of generation evaluation

- `LIMITATIONS.md`
  - synthetic data concerns
  - metric limitations
  - subgroup imbalance
  - interpretation boundaries

- `README.md`
  - setup instructions
  - how to run notebooks/scripts
  - expected outputs

### D. Results artifacts
Claude should save tables, plots, and possibly CSV summaries of metrics to a `results/` or `artifacts/` folder.

---

## Expectations for Explanations

Claude should not only write code, but also explain the reasoning behind each major decision.

For each substantial choice, Claude should answer questions like:
- Why is this model included?
- What role does it serve: baseline, strong benchmark, interpretability, or mitigation?
- Why is this metric appropriate here?
- What are the limitations of this comparison?
- How do the results relate back to the research question?

The written explanations should be detailed enough that a reader can understand the scientific logic of the project without reverse-engineering the code.

---

## Evaluation Strategy Guidance

Claude should not assume that one metric is enough.

A robust evaluation strategy for generation should ideally combine:
- one or more lexical similarity metrics
- one or more semantic similarity metrics
- qualitative error analysis
- optional small-scale human or rubric-based evaluation if feasible

For subgroup analysis, Claude should compute metrics separately for:
- each priority class
- each department class
- optionally intersections if data is sufficient

Claude should also report support sizes so that subgroup comparisons are interpretable.

---

## Optional Extension Ideas

If the main pipeline is stable and well evaluated, Claude may explore one of the following extensions:

1. **Conditioned generation**
   - explicitly include predicted or gold priority/department labels in the generation input
   - test whether this improves consistency across groups

2. **Retrieval augmentation**
   - retrieve similar tickets or ticket-response pairs from the training set
   - test whether this improves quality or reduces subgroup disparities

3. **Difficulty analysis**
   - investigate whether subgroup disparities correlate with linguistic difficulty indicators

4. **Mitigation**
   - rebalance training
   - targeted prompting
   - subgroup-aware conditioning

Extensions should only be pursued if the core project is already coherent and working.

---

## Writing Style and Scientific Caution

Claude should write in a style suitable for an academic project report:
- clear
- precise
- modest in claims
- explicit about assumptions and limitations

Avoid exaggerated statements such as:
- “this proves”
- “this shows causality”
- “this generalizes to all support systems”

Prefer statements such as:
- “within this dataset and setup”
- “the results suggest”
- “the evidence is consistent with”
- “a plausible interpretation is”

---

## Final Objective

The final project should answer, as rigorously as possible within a course-project scope:

**Are automated customer-support generators equally good across different kinds of tickets, or do some semantically distinct request types receive systematically worse responses?**

Everything in the project should support that objective.

The project should balance:
- ambition
- clarity
- methodological discipline
- practical feasibility

When uncertain, Claude should prefer:
- stronger baselines,
- cleaner analysis,
- better documentation,
over unnecessary architectural complexity.