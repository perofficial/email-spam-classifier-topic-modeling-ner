"""
pipeline.py
-----------
End-to-end orchestration pipeline for the email spam analysis project.

Steps
-----
1. Load & preprocess dataset
2. Train spam/ham classifier + save joblib artifacts
3. LDA topic modeling on spam and ham
4. Cosine semantic distance between spam/ham topic vectors
5. NER extraction of organisations from ham emails
"""

from __future__ import annotations

import os

import pandas as pd

from .preprocessing import load_nlp_models, clean_text
from .classifier import train as train_classifier, save_artifacts, predict_email
from .topic_modeling import (
    prepare_texts_for_lda,
    build_lda_corpus,
    train_lda,
    extract_topic_string,
)
from .semantic_distance import (
    load_glove,
    topic_cosine_similarity,
    pairwise_mean_cosine_similarity,
)
from .ner_extractor import extract_organisations, print_organisations
from .visualizations import plot_all


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "data/spam_dataset.csv"
DEFAULT_MODEL_PATH = "models/mlp_classifier_model.joblib"
DEFAULT_VECTORIZER_PATH = "models/vectorizer.joblib"
NUM_TOPICS = 15
NUM_WORDS = 20
LDA_PASSES = 3


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
    num_topics: int = NUM_TOPICS,
    skip_glove: bool = False,
    output_dir: str = "reports/figures",
) -> dict:
    """
    Execute the full analysis pipeline.

    Parameters
    ----------
    data_path       : path to the raw CSV dataset
    model_path      : output path for the joblib classifier
    vectorizer_path : output path for the joblib vectorizer
    num_topics      : number of LDA topics per corpus
    skip_glove      : if True, skip the GloVe download / cosine step
                      (useful for quick smoke-tests)

    Returns
    -------
    dict with keys:
        model, vectorizer, lda_spam, lda_ham,
        topic_similarity (float or None),
        mean_pairwise_similarity (float or None),
        organisations (set[str])
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n[1/6] Loading dataset ...")
    df_raw = pd.read_csv(data_path)
    print(f"      Shape: {df_raw.shape}")

    # ------------------------------------------------------------------
    # 2. Preprocessing + classifier training
    # ------------------------------------------------------------------
    print("\n[2/6] Preprocessing + training classifier ...")
    nlp, stop_words = load_nlp_models()

    cleaned_texts = [clean_text(t, nlp, stop_words) for t in df_raw["text"]]
    clean_df = pd.DataFrame({"data": cleaned_texts, "target": df_raw["label_num"]})

    model, vectorizer, report, X_test, y_test, y_pred = train_classifier(clean_df)
    save_artifacts(model, vectorizer, model_path, vectorizer_path)

    # ------------------------------------------------------------------
    # 3. Topic modeling
    # ------------------------------------------------------------------
    print("\n[3/6] Topic modeling (LDA) ...")
    df_spam = df_raw[df_raw["label_num"] == 1].copy()
    df_ham = df_raw[df_raw["label_num"] == 0].copy()

    clean_spam = prepare_texts_for_lda(df_spam["text"], nlp, stop_words)
    clean_ham = prepare_texts_for_lda(df_ham["text"], nlp, stop_words)

    _, id2word_spam, corpus_spam = build_lda_corpus(clean_spam)
    _, id2word_ham, corpus_ham = build_lda_corpus(clean_ham)

    lda_spam = train_lda(corpus_spam, id2word_spam, num_topics=num_topics, passes=LDA_PASSES)
    lda_ham = train_lda(corpus_ham, id2word_ham, num_topics=num_topics, passes=LDA_PASSES)

    sentence_spam = extract_topic_string(lda_spam, num_topics, NUM_WORDS)
    sentence_ham = extract_topic_string(lda_ham, num_topics, NUM_WORDS)

    print(f"\nSpam topic keywords (sample): {sentence_spam[:200]} ...")
    print(f"\nHam topic keywords (sample):  {sentence_ham[:200]} ...")

    # ------------------------------------------------------------------
    # 4. Cosine semantic distance
    # ------------------------------------------------------------------
    topic_sim = None
    mean_sim = None
    if not skip_glove:
        print("\n[4/6] Computing cosine semantic distances ...")
        glove = load_glove()
        topic_sim = topic_cosine_similarity(sentence_spam, sentence_ham, glove)
        mean_sim = pairwise_mean_cosine_similarity(clean_spam, clean_ham, glove)
    else:
        print("\n[4/6] Skipping GloVe / cosine step (skip_glove=True).")

    # ------------------------------------------------------------------
    # 5. NER — organisations from ham emails
    # ------------------------------------------------------------------
    print("\n[5/6] Extracting organisations from HAM emails ...")
    organisations = extract_organisations(clean_ham, nlp)
    print_organisations(organisations)

    # ------------------------------------------------------------------
    # 6. Visualizations
    # ------------------------------------------------------------------
    print("\n[6/6] Generating performance visualizations ...")
    viz_paths = plot_all(
        df_raw=df_raw,
        model=model,
        vectorizer=vectorizer,
        classification_report=report,
        lda_spam=lda_spam,
        lda_ham=lda_ham,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        topic_similarity=topic_sim,
        mean_pairwise_similarity=mean_sim,
        output_dir=output_dir,
    )

    print("\n✅  Pipeline complete.")
    return {
        "model": model,
        "vectorizer": vectorizer,
        "classification_report": report,
        "lda_spam": lda_spam,
        "lda_ham": lda_ham,
        "topic_similarity": topic_sim,
        "mean_pairwise_similarity": mean_sim,
        "organisations": organisations,
        "viz_paths": viz_paths,
    }
