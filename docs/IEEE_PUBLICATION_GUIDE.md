# IEEE Paper Publication Guide for AutoDS-Agent

This guide describes how to turn AutoDS-Agent into a defensible research paper and submit it to a suitable IEEE conference or journal.

> **Important:** A working application or cloud deployment is not enough for publication. The paper must make an original, evidence-backed research contribution. Do not invent results, claim unsupported novelty, submit the same paper to more than one venue at once, or use private user datasets without permission.

## 1. What IEEE publication requires

Your submission should demonstrate all of the following.

| Requirement | What it means for AutoDS-Agent |
|---|---|
| Original contribution | State exactly what is new beyond existing AutoML, conversational analytics, and agent frameworks. |
| Technical method | Explain architecture, trusted execution boundary, adaptive-decision rules, retrieval, memory, benchmark policy, and security constraints. |
| Empirical evidence | Run repeatable experiments on public datasets and compare against clear baselines. |
| Accurate reporting | Report all datasets, splits, seeds, metrics, failures, runtime/cost measurements, and limitations honestly. |
| Reproducibility | Share code, setup instructions, versions, configuration template, and lawful dataset links. |
| Ethics and originality | Cite all borrowed ideas/data/code, disclose AI-generated paper content where required, and avoid simultaneous submissions. |

IEEE requires original research that has not been published before and is not under review elsewhere. IEEE also requires accurate reporting and proper citation. See the [IEEE ethical requirements](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/ethical-requirements/).

## 2. Define a realistic research claim

Do **not** write: “AutoDS-Agent is the best AutoML system.”

Write a testable claim, for example:

> “AutoDS-Agent combines a constrained natural-language planner with verified local analytics and adaptive, evidence-based pipeline decisions. We evaluate whether these controls improve analysis reliability, transparency, and resource-aware model selection relative to fixed baseline workflows.”

Your paper should distinguish:

| Component | Claim you can make only if verified in code and experiments |
|---|---|
| Ask Q&A | Natural-language questions are converted to structured, allowlisted plans; local code calculates every numeric answer. |
| Adaptive planning | Dataset characteristics deterministically affect the safe pipeline configuration. |
| Benchmarking | Candidate models are compared using persisted validation metrics and local timing measurements. |
| Explainability | Explanations are grounded in verified results; LLM text does not calculate metrics. |
| RAG | Local curated papers give advisory context only and do not alter metrics or selected models. |
| Memory | Persisted compatible experiment records can provide historical context but do not guarantee future performance. |

If a capability is partial, say so clearly. For example, “offline curated research retrieval” is not the same as live scholarly search.

## 3. Choose a paper type and venue

### Option A: IEEE conference paper

Best for a student project with a completed prototype and focused experimental results.

1. Find conferences in AI, data science, machine learning, software engineering, or intelligent systems that match the paper scope.
2. Read the conference call for papers carefully: topic fit, page limit, anonymization requirement, deadlines, presentation obligation, and indexing details vary by event.
3. Verify that the event is genuinely IEEE-sponsored before paying or submitting.

### Option B: IEEE journal paper

Best only after you have a stronger evaluation, deeper novelty, expanded related work, and more complete analysis than a normal conference paper.

Use IEEE’s publication tools to identify a suitable venue and template: [IEEE Author Tools](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/).

### Do not submit to multiple venues at once

Choose one venue, submit once, and wait for a decision. If rejected, improve the paper before submitting it elsewhere. If a later journal version expands a conference paper, cite the conference paper and explain the substantial additional contribution.

## 4. Build a publishable evaluation plan

### 4.1 Research questions

Use focused questions such as:

| ID | Research question |
|---|---|
| RQ1 | Can structured, allowlisted plans answer supported dataset questions correctly and reject unsupported requests safely? |
| RQ2 | Do adaptive pipeline decisions change appropriately for missingness, imbalance, cardinality, outliers, and dataset size? |
| RQ3 | How does the proposed model-selection workflow compare with fixed baseline pipelines on predictive performance and training time? |
| RQ4 | Does the decision audit improve explanation completeness without introducing unverified numerical claims? |
| RQ5 | What are the limits of the system: unsupported questions, small datasets, resource limits, and free-cloud constraints? |

### 4.2 Public datasets

Use lawful, publicly available datasets with clear licenses. A balanced study should include:

- Binary classification: Titanic, customer churn, breast cancer, credit-risk-style public data.
- Multiclass classification: Iris, Wine, Digits, or other public UCI/OpenML data.
- Regression: California Housing, Ames Housing, or other public housing datasets.
- Optional time data: a public time-series dataset only if your time-aware capability is evaluated.

For every dataset, report source, license, rows, features, target, task, missingness, class distribution, and any preprocessing performed.

Do not include private user uploads, credentials, raw personally identifiable information, or proprietary company data in the paper/repository without proper permission.

### 4.3 Baselines

Compare against simple, reproducible baselines. Examples:

| Proposed capability | Fair baseline |
|---|---|
| Adaptive pipeline | Same candidate models with a fixed/default preprocessing plan. |
| Candidate recommendation | Same allowlisted candidate pool without context-aware filtering. |
| Benchmark table | Selection by the configured primary validation metric only. |
| Ask Q&A safety | Manual SQL/Python is **not** a safety-equivalent baseline; instead measure supported-answer correctness and safe refusal. |

Never compare different train/test splits and call the result a fair model comparison.

### 4.4 Metrics

Use task-appropriate metrics and define the primary selection metric before running experiments.

| Task | Report |
|---|---|
| Binary / multiclass classification | Accuracy, precision, recall, F1, ROC-AUC where applicable, confusion matrix. |
| Regression | MAE, RMSE, R². MAE/RMSE are target units, not percentages. |
| System cost | Training time, prediction time per row if measured, candidate count, optimization trials, and hardware/VM details. |
| Ask Q&A | Exact answer correctness on a labelled question set, plan validity, refusal correctness, and latency. |
| Explainability | Evidence coverage and unsupported-claim rate from a manually checked evaluation set. |

### 4.5 Experimental rigor

- Freeze software versions and random seeds.
- Use the same data split for all methods in one comparison.
- Keep an untouched test set; do not select models on final test performance.
- Run repeated seeds or cross-validation if computationally practical, then report mean and variation.
- Record failed runs instead of silently removing them.
- State your hardware/VM size, operating system, Python version, Docker version, and package versions.
- Do not use a free-cloud VM as a performance benchmark unless you disclose its constraints and repeatability limitations.

## 5. Required AutoDS-Agent evidence package

Create an `experiments/` folder or a separate reproducibility repository containing:

```text
experiments/
  README.md                 # exact reproduction steps
  datasets.md               # sources, licenses, and download instructions
  questions.csv             # Ask Q&A benchmark questions + expected verified result
  configs/                  # safe non-secret experiment configurations
  results/                  # aggregate tables only; no private uploads
  scripts/                  # repeatable commands if available
  environment.md            # hardware, Docker, package, and OS versions
```

Keep secrets out of this package. The repository must use `.env.example`, not `.env`.

IEEE encourages authors to share code, data, and research outputs so others can reproduce the reported conclusions. See [IEEE research reproducibility guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/).

## 6. Recommended paper structure

Use the exact template and page limit from the selected venue. A common structure is:

1. **Title** — specific, not marketing language.
2. **Abstract** — problem, method, evaluation, main measured result, and limitation.
3. **Keywords** — AutoML, trustworthy AI, data science automation, multi-agent systems, explainable ML.
4. **Introduction** — motivation, problem, gaps in existing methods, contributions.
5. **Related Work** — AutoML, conversational data analysis, trustworthy AI, RAG/memory, explainability, resource-aware ML.
6. **System Design** — architecture diagram, data flow, agents, trusted boundary, threat model.
7. **Methodology** — adaptive decisions, plan validation, analytics executor, benchmark policy, reflection/RAG rules.
8. **Experimental Setup** — datasets, baselines, splits, metrics, environment, research questions.
9. **Results and Discussion** — tables/plots, failure analysis, comparisons, qualitative examples.
10. **Limitations and Threats to Validity** — free-tier constraints, dataset scope, LLM dependence, no causal claims, incomplete chart coverage, etc.
11. **Conclusion and Future Work** — concise, evidence-backed.
12. **References** — every dataset, paper, library, and reused concept cited correctly.
13. **Acknowledgment / AI disclosure** — if applicable.

### Suggested contribution list

Only retain items that your experiments prove:

1. A constrained architecture where LLMs plan/interpret but allowlisted local code performs analytics, training, and evaluation.
2. Deterministic context-aware pipeline decisions based on verified dataset properties.
3. A benchmark and decision-audit interface that exposes validation performance, timing, complexity, and generalization evidence.
4. A context-aware assistant that separates dataset calculations from persisted experiment retrieval and rejects unsupported operations.

## 7. Figures and tables to prepare

| Item | Purpose |
|---|---|
| Architecture diagram | Show browser, frontend, API, database, Redis, Celery worker, trusted engine, LLM boundary, RAG, and storage. |
| Trusted-execution flow | Question → router → Pydantic plan → allowlisted operation → verified response. |
| Adaptive-decision table | Dataset evidence → decision → allowlisted action. |
| Benchmark table | Model, validation metric, test metric, train time, inference time, complexity, selected status. |
| Performance/cost chart | Plot measured validation performance against measured train time. |
| Ask Q&A evaluation table | Question type, valid plan rate, correct answer rate, safe refusal rate. |
| Ablation table | Full system vs fixed/no-adaptive/no-memory/no-RAG-advice configurations where implemented. |
| Failure/limitation table | Unsupported question types, known risks, mitigation, and future work. |

Every number in a table/figure must be traceable to a recorded experiment. Never use a visually attractive chart with invented data.

## 8. Write the manuscript

1. Create an ORCID for every author if the selected venue requires or recommends it.
2. Download the exact Word or LaTeX template for the selected conference/journal.
3. Use the venue’s page limit; do not shrink fonts/margins to bypass it.
4. Write the methodology before polishing the introduction—this exposes unsupported claims early.
5. Create tables from saved experiment outputs, not manually typed values.
6. Have every author check the final version, contributions, affiliation, funding, conflicts, and references.

IEEE provides official Word/LaTeX template guidance: [conference templates](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/) and [journal templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/).

## 9. AI use and disclosure

AutoDS-Agent is itself an AI-related system, but that does not remove the authors’ responsibility for the paper.

- Authors are responsible for every sentence, citation, table, figure, and result.
- Check every reference manually; do not let an LLM invent citations.
- Do not upload confidential reviewer comments or an under-review manuscript to public AI tools.
- If AI generated substantive text, figures, images, or code used in the submitted article, disclose the AI system, affected sections, and level of use in the acknowledgments according to the selected IEEE venue’s policy.
- Grammar-only assistance is treated differently, but disclosure is recommended when uncertain.

IEEE’s current author policy describes disclosure expectations for AI-generated article content. Read it before submission: [IEEE submission and AI-content policy](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/).

Example disclosure to adapt only if true:

> **Acknowledgment:** An AI language tool was used for grammar and readability suggestions in Sections I and V. The authors reviewed, revised, and take full responsibility for all content, citations, experimental results, and conclusions.

Do not state that AI was used for an activity if it was not used.

## 10. Pre-submission checklist

### Research

- [ ] The paper has one clear research problem and testable research questions.
- [ ] Claims match implemented code and measured outcomes.
- [ ] All experiments use recorded configurations, seeds, datasets, and splits.
- [ ] Baselines are fair and run on comparable data splits.
- [ ] Every metric is correctly formatted and explained.
- [ ] Failed runs and limitations are reported honestly.

### Ethics and reproducibility

- [ ] No secrets, private datasets, model artifacts, user accounts, or personal data are in the repository or paper.
- [ ] Dataset licenses and sources are cited.
- [ ] Related work, algorithms, libraries, and reused figures are cited.
- [ ] The work is not submitted to another venue at the same time.
- [ ] AI assistance is disclosed if required.
- [ ] Code, Docker Compose setup, `.env.example`, and reproduction instructions are available.

### Formatting

- [ ] Exact venue template and page limit are used.
- [ ] Title, author list, affiliations, and email addresses are correct.
- [ ] Anonymous-review rules are followed if the venue requires double-blind review.
- [ ] Figures are readable in grayscale and have descriptive captions.
- [ ] References are complete and consistently formatted.
- [ ] The final PDF passes the venue’s PDF requirements.

## 11. Submission process

1. Select one legitimate IEEE venue that fits the scope and deadline.
2. Read the Call for Papers, author kit, template, ethics policy, and submission instructions.
3. Create author accounts in the stated submission system.
4. Upload the manuscript PDF and any required source/metadata files.
5. Enter all authors exactly as they appear in the paper.
6. Declare conflicts of interest and prior related publications if the venue asks.
7. Submit once and save the submission confirmation.
8. Do not submit the same manuscript elsewhere while it is under review.

For conferences that require it after acceptance, use IEEE PDF eXpress or the designated PDF checker. IEEE explains the PDF checks here: [IEEE Xplore PDF requirements](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/meet-ieee-xplore-requirements/).

## 12. After review

### If you receive “minor revision” or “major revision”

1. Create a response-to-reviewers document.
2. Quote each reviewer comment briefly.
3. State exactly what changed and where: page, section, table, or figure.
4. If you disagree, respond respectfully with evidence.
5. Re-run experiments if a reviewer questions validity; do not change results silently.
6. Upload the revised manuscript and response before the deadline.

### If accepted

1. Make only the requested final edits.
2. Recheck author names, references, figures, and permissions.
3. Complete copyright/electronic forms required by the venue.
4. Run the final PDF compliance check if requested.
5. Register and present if the conference requires presentation for publication.
6. Tag the exact reproducibility release in GitHub, for example `paper-v1.0`.
7. Archive the release with a DOI service such as Zenodo if appropriate.

## 13. Practical timeline

| Week | Deliverable |
|---|---|
| 1 | Freeze the research questions, contribution statement, datasets, and baselines. |
| 2 | Build evaluation scripts/configuration and run pilot experiments. |
| 3–4 | Run final experiments; collect tables, figures, failures, and timing evidence. |
| 5 | Write methodology, experimental setup, and results. |
| 6 | Write introduction, related work, limitations, and conclusion. |
| 7 | Internal review, reproducibility check, citation audit, and template/PDF check. |
| 8 | Submit to one selected venue. |

## 14. Final recommendation for this project

The strongest first paper is likely a **conference-style systems/evaluation paper**, not a broad claim of a new general-purpose AutoML theory. Focus the contribution on trustworthy, constrained automation:

> **AutoDS-Agent: A Verified Multi-Agent Framework for Context-Aware Data Science Automation**

Support that claim with reproducible public-dataset evidence, explicit baselines, and honest limitations. The live cloud VM is useful for a demo, but the GitHub repository, Docker Compose configuration, experiment evidence, and paper methodology are what make the work reviewable.
