"""
main.py
-------
CLI entry point for the email spam classifier pipeline.

Usage
-----
# Run the full pipeline (trains, saves models, extracts topics and orgs)
python main.py

# Run without downloading GloVe (faster smoke-test)
python main.py --skip-glove

# Classify a single email from a text file
python main.py --predict path/to/email.txt
"""

from __future__ import annotations

import argparse
import sys

from src.pipeline import run_pipeline, DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH
from src.classifier import load_artifacts, predict_email
from src.preprocessing import load_nlp_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="Email Spam Classifier — ProfessionAI",
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA_PATH,
        help="Path to the CSV dataset (default: data/spam_dataset.csv)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Output / input path for the classifier joblib",
    )
    parser.add_argument(
        "--vectorizer", default=DEFAULT_VECTORIZER_PATH,
        help="Output / input path for the vectorizer joblib",
    )
    parser.add_argument(
        "--skip-glove", action="store_true",
        help="Skip GloVe download and cosine distance step",
    )
    parser.add_argument(
        "--predict", metavar="EMAIL_FILE",
        help="Path to a .txt file containing a raw email to classify",
    )
    parser.add_argument(
        "--num-topics", type=int, default=15,
        help="Number of LDA topics (default: 15)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.predict:
        # ---- Inference mode ----
        try:
            with open(args.predict, "r", encoding="utf-8") as f:
                email_text = f.read()
        except FileNotFoundError:
            print(f"[ERROR] File not found: {args.predict}")
            sys.exit(1)

        model, vectorizer = load_artifacts(args.model, args.vectorizer)
        nlp, stop_words = load_nlp_models()
        result = predict_email(email_text, model, vectorizer, nlp, stop_words)

        print(f"\nPrediction : {result['class']}")
        print(f"Label      : {result['label']}")
        print(f"P(HAM)     : {result['probabilities'][0]:.4f}")
        print(f"P(SPAM)    : {result['probabilities'][1]:.4f}")
    else:
        # ---- Full pipeline mode ----
        run_pipeline(
            data_path=args.data,
            model_path=args.model,
            vectorizer_path=args.vectorizer,
            num_topics=args.num_topics,
            skip_glove=args.skip_glove,
        )


if __name__ == "__main__":
    main()
