"""
visualizations.py
-----------------
Performance and analysis visualizations for the email spam classifier pipeline.

Charts produced
---------------
1. plot_class_distribution     — spam vs ham class balance (bar chart)
2. plot_confusion_matrix       — classifier confusion matrix (heatmap)
3. plot_classification_report  — precision / recall / f1 per class (grouped bars)
4. plot_roc_curve              — ROC curve with AUC score
5. plot_loss_curve             — MLP training loss curve
6. plot_topic_words            — top-N keywords per LDA topic (horizontal bars)
7. plot_cosine_similarity      — spam/ham topic cosine similarity (gauge-style bar)
8. plot_all                    — convenience wrapper that generates all charts
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

SPAM_COLOR = "#E74C3C"
HAM_COLOR  = "#2ECC71"
ACCENT     = "#3498DB"
DARK_BG    = "#2C3E50"
GRID_COLOR = "#ECF0F1"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})

OUTPUT_DIR = "reports/figures"


def _save(fig: plt.Figure, filename: str, output_dir: str) -> str:
    """Save figure to disk and return the full path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  ✓ Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label_num",
    output_dir: str = OUTPUT_DIR,
    show: bool = False,
) -> str:
    """
    Bar chart showing the spam / ham class balance.

    Parameters
    ----------
    df        : raw DataFrame with a numeric label column
    label_col : column containing 0 (ham) / 1 (spam)
    output_dir: directory where the PNG is saved
    show      : if True, call plt.show() after saving

    Returns
    -------
    Path to the saved figure.
    """
    counts = df[label_col].value_counts().sort_index()
    labels = ["HAM (0)", "SPAM (1)"]
    colors = [HAM_COLOR, SPAM_COLOR]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontweight="bold",
        )

    total = counts.sum()
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of emails")
    ax.set_xlabel("")
    ax.set_ylim(0, counts.max() * 1.15)

    pct_spam = counts.get(1, 0) / total * 100
    pct_ham  = counts.get(0, 0) / total * 100
    fig.text(
        0.5, -0.04,
        f"HAM {pct_ham:.1f}%  |  SPAM {pct_spam:.1f}%  (total: {total:,})",
        ha="center", fontsize=10, color="#666666",
    )

    if show:
        plt.show()
    path = _save(fig, "class_distribution.png", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    output_dir: str = OUTPUT_DIR,
    show: bool = False,
) -> str:
    """
    Heatmap confusion matrix for the classifier predictions.

    Parameters
    ----------
    y_true     : ground-truth labels
    y_pred     : predicted labels
    output_dir : directory where the PNG is saved
    show       : if True, call plt.show()

    Returns
    -------
    Path to the saved figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HAM", "SPAM"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")

    if show:
        plt.show()
    path = _save(fig, "confusion_matrix.png", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Classification report
# ---------------------------------------------------------------------------

def plot_classification_report(
    report: dict,
    output_dir: str = OUTPUT_DIR,
    show: bool = False,
) -> str:
    """
    Grouped bar chart for precision, recall and F1-score per class.

    Parameters
    ----------
    report     : dict returned by sklearn.metrics.classification_report(..., output_dict=True)
    output_dir : directory where the PNG is saved
    show       : if True, call plt.show()

    Returns
    -------
    Path to the saved figure.
    """
    classes = ["0", "1"]
    labels  = ["HAM", "SPAM"]
    metrics = ["precision", "recall", "f1-score"]
    colors  = [ACCENT, SPAM_COLOR, HAM_COLOR]

    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [report[c][metric] for c in classes]
        bars = ax.bar(x + (i - 1) * width, values, width, label=metric.capitalize(), color=color, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Classification Report — Precision / Recall / F1")
    ax.legend(loc="lower right")

    support_text = "  |  ".join(
        f"{lbl}: {int(report[c]['support'])} samples"
        for lbl, c in zip(labels, classes)
    )
    fig.text(0.5, -0.04, support_text, ha="center", fontsize=9, color="#666666")

    if show:
        plt.show()
    path = _save(fig, "classification_report.png", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    model: MLPClassifier,
    X_test: np.ndarray,
    y_test,
    output_dir: str = OUTPUT_DIR,
    show: bool = False,
) -> str:
    """
    ROC curve with AUC score.

    Parameters
    ----------
    model      : fitted MLPClassifier
    X_test     : vectorised test features
    y_test     : ground-truth test labels
    output_dir : directory where the PNG is saved
    show       : if True, call plt.show()

    Returns
    -------
    Path to the saved figure.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=SPAM_COLOR, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle="--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=SPAM_COLOR)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    if show:
        plt.show()
    path = _save(fig, "roc_curve.png", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. MLP training loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(
    model: MLPClassifier,
    output_dir: str = OUTPUT_DIR,
    show: bool = False,
) -> str:
    """
    Plot the MLP training loss curve over iterations.

    Parameters
    ----------
    model      : fitted MLPClassifier (must have loss_curve_ attribute)
    output_dir : directory where the PNG is saved
    show       : if True, call plt.show()

    Returns
    -------
    Path to the saved figure.
    """
    if not hasattr(model, "loss_curve_"):
        raise AttributeError("Model has no loss_curve_. Ensure it was trained with warm_start=False.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model.loss_curve_, color=ACCENT, lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("MLP Training Loss Curve")

    min_loss  = min(model.loss_curve_)
    min_iter  = model.loss_curve_.index(min_loss)
    ax.annotate(
        f"min loss: {min_loss:.4f}",
        xy=(min_iter, min_loss),
        xytext=(min_iter + len(model.loss_curve_) * 0.05, min_loss + 0.02),
        arrowprops=dict(arrowstyle="->", color=DARK_BG),
        fontsize=9,
    )

    if show:
        plt.show()
    path = _save(fig, "training_loss_curve.png", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. LDA topic words
# ---------------------------------------------------------------------------

def plot_topic_words(
    lda_model,
    title: str = "Top Keywords per Topic",
    num_topics: int = 10,
    num_words: int = 8,
    output_dir: str = OUTPUT_DIR,
    show: bool = False,
) -> str:
    """
    Horizontal bar chart of the top-N keywords for each LDA topic.

    Parameters
    ----------
    lda_model  : fitted gensim LdaMulticore model
    title      : chart title (e.g. "Spam Topics" / "Ham Topics")
    num_topics : how many topics to plot
    num_words  : keywords per topic
    output_dir : directory where the PNG is saved
    show       : if True, call plt.show()

    Returns
    -------
    Path to the saved figure.
    """
    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
    n_cols = 2
    n_rows = (len(topics) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.8))
    axes = axes.flatten()

    for idx, (topic_id, word_weights) in enumerate(topics):
        words   = [w for w, _ in word_weights][::-1]
        weights = [p for _, p in word_weights][::-1]
        color   = SPAM_COLOR if "spam" in title.lower() else HAM_COLOR

        axes[idx].barh(words, weights, color=color, edgecolor="white")
        axes[idx].set_title(f"Topic {topic_id + 1}", fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Weight", fontsize=8)
        axes[idx].tick_params(labelsize=8)
        axes[idx].grid(axis="x")

    # Hide unused subplots
    for ax in axes[len(topics):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    safe_title = title.lower().replace(" ", "_")
    if show:
        plt.show()
    path = _save(fig, f"topics_{safe_title}.png", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7. Cosine similarity gauge
# ---------------------------------------------------------------------------

def plot_cosine_similarity(
    topic_similarity: Optional[float] = None,
    mean_pairwise_similarity: Optional[float] = None,
    output_dir: str = OUTPUT_DIR,
    show: bool = False,
) -> str:
    """
    Horizontal bar chart comparing the two cosine similarity scores
    (topic-level and pairwise mean).

    Parameters
    ----------
    topic_similarity         : cosine similarity between aggregated topic vectors
    mean_pairwise_similarity : mean pairwise cosine similarity across all email pairs
    output_dir               : directory where the PNG is saved
    show                     : if True, call plt.show()

    Returns
    -------
    Path to the saved figure.
    """
    scores = {}
    if topic_similarity is not None:
        scores["Topic vector\nsimilarity"] = topic_similarity
    if mean_pairwise_similarity is not None:
        scores["Mean pairwise\nsimilarity"] = mean_pairwise_similarity

    if not scores:
        raise ValueError("At least one similarity score must be provided.")

    labels = list(scores.keys())
    values = list(scores.values())

    def _bar_color(v: float) -> str:
        if v < 0.3:
            return HAM_COLOR    # good separation
        if v < 0.6:
            return "#F39C12"    # moderate
        return SPAM_COLOR       # high overlap

    colors = [_bar_color(v) for v in values]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(labels, values, color=colors, height=0.4, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontweight="bold",
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Cosine Similarity  (0 = orthogonal, 1 = identical)")
    ax.set_title("Semantic Distance — SPAM vs HAM")
    ax.axvline(0.5, color="#AAAAAA", linestyle="--", lw=1, label="0.5 threshold")
    ax.legend(fontsize=9)

    legend_patches = [
        mpatches.Patch(color=HAM_COLOR,   label="Good separation  (< 0.30)"),
        mpatches.Patch(color="#F39C12",   label="Moderate overlap (0.30 – 0.60)"),
        mpatches.Patch(color=SPAM_COLOR,  label="High overlap     (> 0.60)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    if show:
        plt.show()
    path = _save(fig, "cosine_similarity.png", output_dir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 8. Convenience wrapper
# ---------------------------------------------------------------------------

def plot_all(
    df_raw: pd.DataFrame,
    model: MLPClassifier,
    vectorizer: TfidfVectorizer,
    classification_report: dict,
    lda_spam,
    lda_ham,
    X_test: np.ndarray,
    y_test,
    y_pred,
    topic_similarity: Optional[float] = None,
    mean_pairwise_similarity: Optional[float] = None,
    output_dir: str = OUTPUT_DIR,
    show: bool = False,
) -> dict[str, str]:
    """
    Generate and save all performance visualizations in one call.

    Parameters
    ----------
    df_raw                   : original raw DataFrame
    model                    : fitted MLPClassifier
    vectorizer               : fitted TfidfVectorizer
    classification_report    : dict from sklearn classification_report
    lda_spam                 : fitted LDA model for spam
    lda_ham                  : fitted LDA model for ham
    X_test                   : vectorised test set features
    y_test                   : ground-truth test labels
    y_pred                   : classifier predictions on test set
    topic_similarity         : cosine similarity of topic vectors (optional)
    mean_pairwise_similarity : mean pairwise cosine similarity (optional)
    output_dir               : directory where all PNGs are saved
    show                     : if True, display each chart interactively

    Returns
    -------
    dict mapping chart name → file path
    """
    print(f"\n📊  Generating visualizations → {output_dir}")
    paths = {}

    paths["class_distribution"]    = plot_class_distribution(df_raw, output_dir=output_dir, show=show)
    paths["confusion_matrix"]       = plot_confusion_matrix(y_test, y_pred, output_dir=output_dir, show=show)
    paths["classification_report"]  = plot_classification_report(classification_report, output_dir=output_dir, show=show)
    paths["roc_curve"]              = plot_roc_curve(model, X_test, y_test, output_dir=output_dir, show=show)
    paths["loss_curve"]             = plot_loss_curve(model, output_dir=output_dir, show=show)
    paths["topics_spam"]            = plot_topic_words(lda_spam, title="Spam Topics", output_dir=output_dir, show=show)
    paths["topics_ham"]             = plot_topic_words(lda_ham,  title="Ham Topics",  output_dir=output_dir, show=show)

    if topic_similarity is not None or mean_pairwise_similarity is not None:
        paths["cosine_similarity"]  = plot_cosine_similarity(
            topic_similarity, mean_pairwise_similarity,
            output_dir=output_dir, show=show,
        )

    print(f"\n✅  All figures saved to '{output_dir}/'")
    return paths
