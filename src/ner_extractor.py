"""
ner_extractor.py
----------------
Named Entity Recognition (NER) for extracting organisations
mentioned in legitimate (ham) emails using spaCy.
"""

from __future__ import annotations

import spacy

from .preprocessing import remove_punctuation, stemming_word


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_organisations(
    texts,
    nlp,
    exclude_word: str = "subject",
) -> set[str]:
    """
    Extract unique organisation names from a collection of email texts.

    Each email is pre-processed (punctuation removal + stemming) before
    running the spaCy NER pipeline.

    Parameters
    ----------
    texts        : iterable of cleaned ham email strings
    nlp          : spaCy Language model (must include NER component)
    exclude_word : token to remove from text before NER (e.g. "subject")

    Returns
    -------
    set[str] : unique organisation names found across all emails
    """
    org_dataset: set[str] = set()
    for email in texts:
        cleaned = remove_punctuation(stemming_word(email.replace(exclude_word, "")))
        doc = nlp(cleaned)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_dataset.add(ent.text)
    return org_dataset


def organisations_to_list(org_set: set[str]) -> list[str]:
    """Return a sorted list from the organisations set."""
    return sorted(org_set)


def print_organisations(org_set: set[str]) -> None:
    """Pretty-print extracted organisations."""
    orgs = organisations_to_list(org_set)
    print(f"\n=== Organisations found in HAM emails ({len(orgs)}) ===")
    for org in orgs:
        print(f"  • {org}")
