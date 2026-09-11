"""
grants_rag.py — RAG retrieval module for GrantPilot AI.
Loads grants.csv, builds a TF-IDF index, and returns ranked grant matches
for a given startup profile.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_GRANTS_CSV = os.path.join(os.path.dirname(__file__), "grants.csv")


def _load_grants(path: str = _GRANTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["grant_name"])
    return df


def _build_corpus(df: pd.DataFrame) -> list[str]:
    """Combine key columns into a single searchable text string per grant."""
    docs = []
    for _, row in df.iterrows():
        parts = [
            str(row.get("grant_name", "")),
            str(row.get("organization", "")),
            str(row.get("domain", "")),
            str(row.get("location", "")),
            str(row.get("startup_stage", "")),
            str(row.get("eligibility", "")),
            str(row.get("description", "")),
        ]
        docs.append(" ".join(parts))
    return docs


def _build_query(startup: dict) -> str:
    """Build a free-text query from the startup profile dict."""
    return (
        f"{startup.get('name', '')} "
        f"{startup.get('domain', '')} "
        f"{startup.get('stage', '')} "
        f"{startup.get('location', '')} "
        f"{startup.get('description', '')}"
    )


def retrieve_grants(startup: dict, top_k: int = 5) -> list[dict]:
    """
    Retrieve and rank the top_k most relevant grants for a startup profile.

    Parameters
    ----------
    startup : dict
        Keys: name, domain, stage, location, funding_required, description
    top_k : int
        Number of top grants to return

    Returns
    -------
    list[dict]
        Sorted list of grant records (as dicts) with an added 'score' key.
    """
    df = _load_grants()
    corpus = _build_corpus(df)
    query = _build_query(startup)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        record = df.iloc[idx].to_dict()
        record["score"] = round(float(scores[idx]), 4)
        results.append(record)

    return results


def format_grants_for_prompt(grants: list[dict]) -> str:
    """Serialise ranked grants into a compact readable block for the LLM prompt."""
    lines = []
    for i, g in enumerate(grants, 1):
        lines.append(
            f"[Grant {i}]\n"
            f"Name: {g.get('grant_name')}\n"
            f"Organization: {g.get('organization')}\n"
            f"Domain: {g.get('domain')}\n"
            f"Location: {g.get('location')}\n"
            f"Stage: {g.get('startup_stage')}\n"
            f"Funding Range: {g.get('funding_min', 'N/A')} - {g.get('funding_max', 'N/A')}\n"
            f"Deadline: {g.get('deadline')}\n"
            f"Eligibility: {g.get('eligibility')}\n"
            f"Description: {g.get('description')}\n"
            f"Apply: {g.get('application_url')}\n"
            f"Relevance Score: {g.get('score')}\n"
        )
    return "\n".join(lines)


def get_kb_summary() -> dict:
    """Return metadata about the knowledge base for display in the UI."""
    df = _load_grants()
    return {
        "total_grants": len(df),
        "domains": sorted(df["domain"].dropna().unique().tolist()),
    }


def score_grant_dimensions(grant: dict, startup: dict) -> dict:
    """
    Compute simple binary fit scores for individual dimensions.
    Returns a dict with keys: domain_fit, location_fit, stage_fit, funding_fit.
    Each value is True / False / None (None = cannot determine).
    """
    # Domain fit
    grant_domain = str(grant.get("domain", "")).lower()
    startup_domain = str(startup.get("domain", "")).lower()
    domain_fit = startup_domain in grant_domain or grant_domain in startup_domain

    # Location fit
    grant_location = str(grant.get("location", "")).lower()
    startup_location = str(startup.get("location", "")).lower()
    location_fit = (
        "global" in grant_location
        or any(w in grant_location for w in startup_location.split(",") if w.strip())
        or any(w in startup_location for w in grant_location.split(",") if w.strip())
    )

    # Stage fit
    grant_stage = str(grant.get("startup_stage", "")).lower()
    startup_stage = str(startup.get("stage", "")).lower()
    stage_fit = startup_stage.lower() in grant_stage

    # Funding fit — startup ask within grant range
    try:
        ask = int(str(startup.get("funding_required", "0")).replace(",", "").replace("₹", "").strip())
        fmin = int(grant.get("funding_min", 0))
        fmax = int(grant.get("funding_max", 0))
        funding_fit = fmin <= ask <= fmax
    except (ValueError, TypeError):
        funding_fit = None

    return {
        "domain_fit": domain_fit,
        "location_fit": location_fit,
        "stage_fit": stage_fit,
        "funding_fit": funding_fit,
    }
