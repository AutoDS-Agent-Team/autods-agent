"""Offline retrieval over a small, curated set of research summaries.

The corpus/index boundary is intentionally provider-neutral so a future source
adapter can be added without giving retrieval any authority over ML execution.
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "curated_research.json"
_MIN_SIMILARITY = 0.025
_SINGLE_TERM_ALLOWLIST = {"automl", "autosklearn", "tpot", "meta-learning", "pipeline"}


@lru_cache(maxsize=1)
def _index() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, Any]:
    corpus = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    papers = corpus["papers"]
    chunks = [
        {"paper": paper, **chunk}
        for paper in papers
        for chunk in paper.get("chunks", [])
    ]
    documents = [" ".join(chunk.get("topics", [])) + " " + chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(documents)
    analyzer = vectorizer.build_analyzer()
    for chunk, document in zip(chunks, documents):
        chunk["_terms"] = set(analyzer(document))
    return papers, chunks, vectorizer, matrix


def retrieve_research(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Return locally indexed, source-backed advisory evidence; never fetches URLs."""
    if not query or not re.search(r"[a-zA-Z]", query):
        return []
    _, chunks, vectorizer, matrix = _index()
    query_vector = vectorizer.transform([query])
    query_terms = set(vectorizer.build_analyzer()(query))
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    results, seen = [], set()
    for index, score in ranked:
        chunk = chunks[index]
        paper = chunk["paper"]
        matched_terms = query_terms & chunk["_terms"]
        if score < _MIN_SIMILARITY or (len(matched_terms) < 2 and not (matched_terms & _SINGLE_TERM_ALLOWLIST)) or paper["id"] in seen:
            continue
        seen.add(paper["id"])
        results.append({
            "title": paper["title"], "authors": paper.get("authors", []),
            "year": paper.get("year"), "venue": paper.get("venue"),
            "source_url": paper["source_url"], "evidence_summary": chunk["text"],
            "topics": chunk.get("topics", []),
            "relevance_score": round(float(score), 4),
        })
        if len(results) >= max(1, min(limit, 5)):
            break
    return results
