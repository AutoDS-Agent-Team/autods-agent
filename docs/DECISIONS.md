# Architecture decisions

1. **Monorepo:** Keep frontend, backend, infrastructure, documentation, and future worker code versioned together.
2. **Primary LLM:** Use the Gemini API as the primary language-model provider.
3. **Fallback LLM:** Use Ollama with `qwen3:8b` when Gemini fails, times out, is unavailable, is rate-limited, or emits invalid structured output.
4. **Validated plans:** Require structured JSON plans and validate them with Pydantic before execution.
5. **Trusted execution:** Execute only explicitly registered, trusted Python functions; never execute arbitrary LLM-generated code.
6. **Initial data scope:** Begin with structured CSV datasets.
7. **Initial ML scope:** Support binary classification, multiclass classification, and regression.
8. **Persistent data:** Use PostgreSQL for durable application and experiment data.
9. **Background work:** Use Redis and Celery for future asynchronous jobs.
10. **Development orchestration:** Use Docker Compose for a reproducible local stack.
11. **Dataset storage:** Store CSV bytes in configured artifact storage under generated UUID filenames; store only metadata in PostgreSQL.
12. **Deterministic profiling:** Use Pandas-based trusted code for profiling. Profiling does not require or call an LLM.
13. **Conservative CSV headers:** Persist deterministic header detection. When evidence is ambiguous, preserve the first row as data and generate positional column names rather than risk silent data loss.
14. **Two-step experiment definition:** Persist deterministic objective analysis separately from confirmation so a future LLM interpreter can plug into the same backend-validated contract.
15. **Explicit ambiguity:** Do not infer a definitive task for low-cardinality integer targets; expose classification and regression as compatible choices and require confirmation.
16. **Authoritative confirmation:** User confirmation may override a suggestion, but only supported task enums and targets satisfying backend dataset/task validation can become ready for planning.
17. **Provider isolation:** Planning depends on an `LLMProvider` abstraction. Gemini is primary and Ollama is fallback; provider failures never fabricate a plan.
18. **Metadata-only prompts:** Send profile metadata but no CSV rows or categorical top values to an LLM.
19. **Local plan authority:** Always revalidate provider output with strict Pydantic enums and confirmed task/target context before persistence or execution.
20. **Allowlisted execution:** Convert plan values to implementations only through explicit registries; never resolve arbitrary imports, class names, or generated code.
21. **Leakage prevention:** Split before preprocessing and fit the `ColumnTransformer` only on training data. Validation and test splits are transform-only.
22. **Model isolation:** Persist each requested baseline as its own model run so one failure does not discard successful runs.
23. **Deferred comparison:** Train baselines without evaluating, ranking, or selecting a winner until Milestone 6.
24. **Validation-first selection:** Compare only successful trusted artifacts on validation metrics. The primary metric direction is explicit; ties sort by model name then generated run ID. Test data is evaluated only after selection and never changes it.
25. **Bounded trusted optimization:** Optuna tunes only the initially selected baseline with code-defined search spaces, 1–20 trials, a timeout, and train/validation data only.
26. **Portable explainability:** Use native feature importances and linear coefficients for the currently supported models, avoiding a heavyweight SHAP runtime while returning real model-derived explanations.
27. **Artifact persistence:** Store generated prediction CSVs and HTML reports under configured confined roots and retain only identifiers/metadata in PostgreSQL.
