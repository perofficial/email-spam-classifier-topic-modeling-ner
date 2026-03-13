"""
preprocessing.py
----------------
Text cleaning and preprocessing utilities for email spam classification.
"""

import re
import string

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import gensim


def download_resources() -> None:
    """Download required NLTK resources."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def load_nlp_models():
    """
    Load NLP models (spaCy pipeline + NLTK stopwords).

    Returns
    -------
    nlp : spacy.Language
    stop_words : set[str]
    """
    download_resources()
    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words("english"))
    return nlp, stop_words


def clean_text(sentence: str, nlp, stop_words: set) -> str:
    """
    Full text-cleaning pipeline:
    - Lowercase
    - Remove punctuation
    - Lemmatization via spaCy
    - Remove stopwords
    - Remove digits

    Parameters
    ----------
    sentence   : raw email text
    nlp        : spaCy Language model
    stop_words : set of stopwords to filter

    Returns
    -------
    Cleaned sentence as a string.
    """
    sentence = sentence.lower()
    for c in string.punctuation:
        sentence = sentence.replace(c, " ")
    doc = nlp(sentence)
    sentence = " ".join([token.lemma_ for token in doc])
    sentence = " ".join(word for word in sentence.split() if word not in stop_words)
    sentence = re.sub(r"\d", "", sentence)
    return sentence


def stemming_word(word: str) -> str:
    """Apply Porter stemming to a single word."""
    stemmer = nltk.stem.PorterStemmer()
    return stemmer.stem(word)


def remove_punctuation(sentence: str) -> str:
    """Remove punctuation from a sentence (used as pre-step for LDA tokenisation)."""
    for c in string.punctuation:
        sentence = sentence.replace(c, " ")
    return sentence


def sent_to_words(sentences):
    """
    Tokenise a list of sentences using gensim simple_preprocess.

    Yields
    ------
    list[str] : tokens for each sentence
    """
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)
