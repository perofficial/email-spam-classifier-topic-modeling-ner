"""
semantic_distance.py
--------------------
Cosine similarity analysis between spam and ham email vectors
using pre-trained GloVe embeddings (glove-wiki-gigaword-300).
"""

from __future__ import annotations

import numpy as np
import gensim
import gensim.downloader
from scipy.spatial import distance


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_glove(model_name: str = "glove-wiki-gigaword-300"):
    """
    Download (or load from cache) a pre-trained GloVe word embedding model.

    Parameters
    ----------
    model_name : gensim downloader model identifier

    Returns
    -------
    gensim KeyedVectors
    """
    print(f"Loading GloVe embeddings: {model_name} ...")
    glove_vectors = gensim.downloader.load(model_name)
    print("GloVe loaded.")
    return glove_vectors


# ---------------------------------------------------------------------------
# Vector computation
# ---------------------------------------------------------------------------

def avg_vector(text_string: str, glove_vectors, dims: int = 300) -> np.ndarray:
    """
    Compute the mean GloVe vector for all known words in a string.

    Parameters
    ----------
    text_string   : input text
    glove_vectors : gensim KeyedVectors
    dims          : embedding dimensionality

    Returns
    -------
    np.ndarray of shape (dims,) — zero vector if no known words found.
    """
    words = gensim.utils.simple_preprocess(text_string, deacc=True)
    vector = np.zeros(dims)
    valid = 0
    for word in words:
        if word in glove_vectors.key_to_index:
            vector += glove_vectors[word]
            valid += 1
    return vector / valid if valid > 0 else np.zeros(dims)


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is all-zero (undefined similarity).
    """
    if np.all(vec_a == 0) or np.all(vec_b == 0):
        return 0.0
    return float(1 - distance.cosine(vec_a, vec_b))


def topic_cosine_similarity(
    spam_topic_string: str,
    ham_topic_string: str,
    glove_vectors,
) -> float:
    """
    Compute cosine similarity between the aggregated topic vectors
    of spam and ham corpora.

    Parameters
    ----------
    spam_topic_string : concatenated spam topic keywords
    ham_topic_string  : concatenated ham topic keywords
    glove_vectors     : gensim KeyedVectors

    Returns
    -------
    float : cosine similarity score in [-1, 1]
    """
    spam_vec = avg_vector(spam_topic_string, glove_vectors)
    ham_vec = avg_vector(ham_topic_string, glove_vectors)
    sim = cosine_similarity(spam_vec, ham_vec)
    print(f"Topic cosine similarity (spam vs ham): {sim:.4f}")
    return sim


def pairwise_mean_cosine_similarity(
    spam_texts,
    ham_texts,
    glove_vectors,
    exclude_word: str = "subject",
) -> float:
    """
    Compute the mean pairwise cosine similarity between all spam and ham emails.

    This gives a corpus-level measure of semantic overlap between the two classes.

    Parameters
    ----------
    spam_texts   : iterable of cleaned spam email strings
    ham_texts    : iterable of cleaned ham email strings
    glove_vectors: gensim KeyedVectors
    exclude_word : token to strip before vectorising

    Returns
    -------
    float : mean cosine similarity
    """
    spam_vectors = [
        avg_vector(t.replace(exclude_word, ""), glove_vectors)
        for t in spam_texts
    ]
    ham_vectors = [
        avg_vector(t.replace(exclude_word, ""), glove_vectors)
        for t in ham_texts
    ]

    similarities = [
        cosine_similarity(sv, hv)
        for sv in spam_vectors
        for hv in ham_vectors
    ]

    mean_sim = float(np.mean(similarities))
    print(f"Average pairwise cosine similarity (spam vs ham): {mean_sim:.6f}")
    return mean_sim
