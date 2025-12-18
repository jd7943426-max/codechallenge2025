# src/codechallenge2025/participant_solution.py
"""
Easy Participant Template for #codechallenge2025

You ONLY need to implement the function: match_single

The find_matches function is provided for you — no need to change it!
"""

import pandas as pd
from typing import List, Dict, Any, Optional


def _parse_alleles(value: Any) -> Optional[List[float]]:
    """Parse a locus value into a list of alleles.

    Returns None for missing values.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text == "-":
        return None

    alleles: List[float] = []
    for allele in text.split(","):
        allele = allele.strip()
        if not allele:
            continue
        try:
            alleles.append(float(allele))
        except ValueError:
            # Non-numeric microvariants are unlikely here; skip if unparseable
            continue

    return alleles or None


def _evaluate_locus(query_alleles: Optional[List[float]], candidate_alleles: Optional[List[float]]):
    """Classify a single locus comparison.

    Returns tuple (consistent, mutated, inconclusive, mismatch_score)
    where mismatch_score penalises hard mismatches when computing CLR-like value.
    """

    if query_alleles is None or candidate_alleles is None:
        return 0, 0, 1, 0

    shared = set(query_alleles).intersection(candidate_alleles)
    if shared:
        return 1, 0, 0, 0

    # Allow a ±1 repeat mutation tolerance when alleles are numeric
    mutated = any(abs(a - b) == 1 for a in query_alleles for b in candidate_alleles)
    if mutated:
        return 0, 1, 0, 0.5

    return 0, 0, 0, 1  # hard mismatch


def match_single(
    query_profile: Dict[str, Any], database_df: pd.DataFrame
) -> List[Dict]:
    """
    Find the top 10 candidate matches for a SINGLE query profile.

    Args:
        query_profile: dict with 'PersonID' and locus columns (e.g. {'PersonID': 'Q001', 'TH01': '9,9.3', ...})
        database_df: Full database as pandas DataFrame (500k rows)

    Returns:
        List of up to 10 candidate dicts, sorted by strength (best first):
        [
            {
                "person_id": "P000123",
                "clr": 1e15,                    # Combined Likelihood Ratio
                "posterior": 0.99999,           # Optional: posterior probability
                "consistent_loci": 20,
                "mutated_loci": 1,
                "inconclusive_loci": 0
            },
            ...
        ]
    """
    candidates: List[Dict[str, Any]] = []
    query_id = query_profile["PersonID"]

    # Pre-parse query alleles once
    query_alleles_by_locus = {
        locus: _parse_alleles(value)
        for locus, value in query_profile.items()
        if locus != "PersonID"
    }

    loci = [col for col in database_df.columns if col != "PersonID"]

    for _, candidate in database_df.iterrows():
        if candidate["PersonID"] == query_id:
            continue  # skip self comparisons

        consistent = mutated = inconclusive = 0
        mismatch_penalty = 0.0

        for locus in loci:
            q_alleles = query_alleles_by_locus.get(locus)
            c_alleles = _parse_alleles(candidate[locus])
            c, m, inc, penalty = _evaluate_locus(q_alleles, c_alleles)
            consistent += c
            mutated += m
            inconclusive += inc
            mismatch_penalty += penalty

        # Simple CLR-like score favouring consistent loci and tolerating limited mutations
        score = max(consistent * 2 + mutated - mismatch_penalty, 0.0) + 1e-6
        posterior = score / (score + 1)  # assume flat prior of 0.5

        candidates.append(
            {
                "person_id": candidate["PersonID"],
                "clr": score,
                "posterior": posterior,
                "consistent_loci": consistent,
                "mutated_loci": mutated,
                "inconclusive_loci": inconclusive,
            }
        )

    candidates.sort(key=lambda x: x["clr"], reverse=True)
    return candidates[:10]


# ============================================================
# DO NOT MODIFY BELOW THIS LINE — This runs your function!
# ============================================================


def find_matches(database_path: str, queries_path: str) -> List[Dict]:
    """
    Main entry point — automatically tested by CI.
    Loads data and calls your match_single for each query.
    """
    print("Loading database and queries...")
    database_df = pd.read_csv(database_path)
    queries_df = pd.read_csv(queries_path)

    results = []

    print(f"Processing {len(queries_df)} queries...")
    for _, query_row in queries_df.iterrows():
        query_id = query_row["PersonID"]
        query_profile = query_row.to_dict()

        print(f"  Matching query {query_id}...")
        top_candidates = match_single(query_profile, database_df)

        results.append(
            {
                "query_id": query_id,
                "top_candidates": top_candidates[:10],  # Ensure max 10
            }
        )

    print("All queries processed.")
    return results
