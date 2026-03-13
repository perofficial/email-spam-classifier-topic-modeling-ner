"""
email-spam-classifier — src package
"""

from .preprocessing import clean_text, load_nlp_models
from .classifier import train, predict_email, save_artifacts, load_artifacts
from .topic_modeling import build_lda_corpus, train_lda, extract_topic_string
from .semantic_distance import load_glove, topic_cosine_similarity, pairwise_mean_cosine_similarity
from .ner_extractor import extract_organisations
from .pipeline import run_pipeline

__all__ = [
    "clean_text",
    "load_nlp_models",
    "train",
    "predict_email",
    "save_artifacts",
    "load_artifacts",
    "build_lda_corpus",
    "train_lda",
    "extract_topic_string",
    "load_glove",
    "topic_cosine_similarity",
    "pairwise_mean_cosine_similarity",
    "extract_organisations",
    "run_pipeline",
]
