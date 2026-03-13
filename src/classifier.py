"""
classifier.py
-------------
TF-IDF vectorisation + MLP spam classifier.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .preprocessing import clean_text


# ---------------------------------------------------------------------------
# Vectoriser
# ---------------------------------------------------------------------------

def create_bow(
    dataset,
    vectorizer: TfidfVectorizer | None = None,
):
    """
    Build or reuse a TF-IDF Bag-of-Words matrix.

    Parameters
    ----------
    dataset    : iterable of strings
    vectorizer : fitted TfidfVectorizer or None (fit a new one)

    Returns
    -------
    (np.ndarray, TfidfVectorizer)
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(dataset)
    else:
        x = vectorizer.transform(dataset)
    return x.toarray(), vectorizer


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model() -> MLPClassifier:
    """
    Instantiate the MLP classifier with the project's default hyper-parameters.

    Architecture
    ------------
    - Activation : logistic (sigmoid)
    - Solver     : Adam
    - Hidden layer: 1 × 100 neurons
    - Max iterations: 1 000
    - Tolerance  : 0.0005 (early-stopping proxy)
    """
    return MLPClassifier(
        activation="logistic",
        solver="adam",
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=42,
        tol=0.0005,
    )


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    text_col: str = "data",
    label_col: str = "target",
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[MLPClassifier, TfidfVectorizer, dict]:
    """
    Full training pipeline: split → vectorise → fit → evaluate.

    Parameters
    ----------
    df           : cleaned DataFrame with text and label columns
    text_col     : column name for email text
    label_col    : column name for binary label (1 = spam, 0 = ham)
    test_size    : fraction held out for evaluation
    random_state : reproducibility seed

    Returns
    -------
    (model, vectorizer, report_dict)
    """
    sent_train, sent_test, label_train, label_test = train_test_split(
        df[text_col],
        df[label_col],
        stratify=df[label_col],
        test_size=test_size,
        random_state=random_state,
    )

    X_train, vectorizer = create_bow(sent_train, None)
    X_test, vectorizer = create_bow(sent_test, vectorizer)

    model = build_model()
    model.fit(X_train, label_train)

    predictions = model.predict(X_test)
    report = classification_report(label_test, predictions, output_dict=True)
    print(classification_report(label_test, predictions))

    return model, vectorizer, report


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_email(
    email_text: str,
    model: MLPClassifier,
    vectorizer: TfidfVectorizer,
    nlp,
    stop_words: set,
) -> dict:
    """
    Classify a single raw email.

    Parameters
    ----------
    email_text : raw email string
    model      : fitted MLPClassifier
    vectorizer : fitted TfidfVectorizer
    nlp        : spaCy Language model
    stop_words : stopwords set

    Returns
    -------
    dict with keys: 'label' (int), 'class' (str), 'probabilities' (np.ndarray)
    """
    cleaned = clean_text(email_text, nlp, stop_words)
    bow, _ = create_bow([cleaned], vectorizer)
    label = int(model.predict(bow)[0])
    probs = model.predict_proba(bow)[0]
    return {
        "label": label,
        "class": "SPAM" if label == 1 else "HAM",
        "probabilities": probs,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    model: MLPClassifier,
    vectorizer: TfidfVectorizer,
    model_path: str = "models/mlp_classifier_model.joblib",
    vectorizer_path: str = "models/vectorizer.joblib",
) -> None:
    """Persist model and vectorizer to disk using joblib."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved  → {model_path}")
    print(f"Vectorizer   → {vectorizer_path}")


def load_artifacts(
    model_path: str = "models/mlp_classifier_model.joblib",
    vectorizer_path: str = "models/vectorizer.joblib",
) -> tuple[MLPClassifier, TfidfVectorizer]:
    """Load model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Artifacts loaded successfully.")
    return model, vectorizer
