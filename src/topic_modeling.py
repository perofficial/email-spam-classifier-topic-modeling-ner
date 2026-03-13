"""
topic_modeling.py
-----------------
LDA topic modeling for spam and ham email corpora using Gensim.
"""

from __future__ import annotations

import string

import gensim
import gensim.models
from gensim import corpora

from .preprocessing import sent_to_words, remove_punctuation, clean_text, stemming_word


# ---------------------------------------------------------------------------
# Corpus building
# ---------------------------------------------------------------------------

def build_lda_corpus(
    texts,
    min_word_len: int = 3,
    exclude_words: list[str] | None = None,
):
    """
    Tokenise texts and build gensim Dictionary + BoW corpus for LDA.

    Parameters
    ----------
    texts          : iterable of cleaned email strings
    min_word_len   : minimum token length to keep
    exclude_words  : additional tokens to drop (e.g. ["subject"])

    Returns
    -------
    (data_words, id2word, corpus)
    """
    if exclude_words is None:
        exclude_words = ["subject"]

    data_words = list(sent_to_words(texts))
    id2word = corpora.Dictionary(data_words)
    corpus = [
        id2word.doc2bow(
            [w for w in doc if len(w) > min_word_len and w not in exclude_words]
        )
        for doc in data_words
    ]
    return data_words, id2word, corpus


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lda(
    corpus,
    id2word,
    num_topics: int = 15,
    passes: int = 3,
    workers: int = 2,
    random_state: int = 42,
) -> gensim.models.LdaMulticore:
    """
    Train an LDA Multicore model.

    Parameters
    ----------
    corpus      : gensim BoW corpus
    id2word     : gensim Dictionary
    num_topics  : number of topics to extract
    passes      : training passes over the corpus
    workers     : parallel workers
    random_state: seed for reproducibility

    Returns
    -------
    Fitted LdaMulticore model
    """
    model = gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        passes=passes,
        workers=workers,
        random_state=random_state,
    )
    return model


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

def extract_topic_string(
    lda_model: gensim.models.LdaMulticore,
    num_topics: int = 15,
    num_words: int = 20,
) -> str:
    """
    Flatten all topic keywords into a single space-separated string.
    Useful for downstream vector averaging and cosine distance computation.

    Parameters
    ----------
    lda_model  : fitted LdaMulticore
    num_topics : number of topics to retrieve
    num_words  : keywords per topic

    Returns
    -------
    A single string of concatenated topic keywords.
    """
    sentence = ""
    topics = lda_model.show_topics(num_topics, num_words)
    for _, topic_str in topics:
        tokens = topic_str.split(" + ")
        for token in tokens:
            word = token.split("*")[1].replace('"', "").strip()
            sentence += word + " "
        sentence += " "
    return sentence.strip()


# ---------------------------------------------------------------------------
# Preprocessing helpers for topic modeling
# ---------------------------------------------------------------------------

def prepare_texts_for_lda(texts, nlp, stop_words: set):
    """
    Apply full cleaning + punctuation removal pipeline to a text series
    before feeding into LDA.

    Parameters
    ----------
    texts      : pd.Series or list of raw email strings
    nlp        : spaCy Language model
    stop_words : stopwords set

    Returns
    -------
    pd.Series / list of cleaned strings
    """
    return [
        remove_punctuation(clean_text(t, nlp, stop_words))
        for t in texts
    ]
