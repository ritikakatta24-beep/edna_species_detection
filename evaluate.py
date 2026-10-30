"""
Reproducible evaluation for the BioTrace k-NN species classifier.

Uses the same cleaning, k-mer encoding, split (80/20, stratified, seed 42) and
model settings as train.py, then reports metrics beyond plain accuracy.

Usage:
    python evaluate.py
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from train import (
    INPUT_CSV, K, RANDOM_STATE, TEST_SIZE,
    build_kmer_vocab, is_placeholder_species, sequence_to_kmer_vector,
)


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[~df["species"].apply(is_placeholder_species)].reset_index(drop=True)

    kmer_index = {kmer: i for i, kmer in enumerate(build_kmer_vocab(K))}
    X = np.stack([sequence_to_kmer_vector(s, K, kmer_index) for s in df["sequence"]])
    y = df["species"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine").fit(X_tr, y_tr)
    pred = knn.predict(X_te)
    proba = knn.predict_proba(X_te)
    classes = knn.classes_

    counts = df["species"].value_counts()
    out = {
        "sequences": int(len(df)),
        "species": int(df["species"].nunique()),
        "train": int(len(X_tr)),
        "test": int(len(X_te)),
        "majority_class_baseline": round(float(counts.iloc[0] / len(df)), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "macro_f1": round(float(f1_score(y_te, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_te, pred, average="weighted", zero_division=0)), 4),
    }

    top5 = np.argsort(-proba, axis=1)[:, :5]
    out["top5_accuracy"] = round(float(np.mean([y_te[i] in classes[top5[i]] for i in range(len(y_te))])), 4)

    genus = lambda a: np.array([s.split()[0] for s in a])
    out["genus_accuracy"] = round(float((genus(pred) == genus(y_te)).mean()), 4)

    top_conf = proba.max(axis=1)
    out["confidence_tradeoff"] = []
    for th in (0.3, 0.6, 0.8, 1.0):
        m = top_conf >= th
        out["confidence_tradeoff"].append({
            "threshold": th,
            "coverage": round(float(m.mean()), 3),
            "accuracy_on_covered": round(float((pred[m] == y_te[m]).mean()), 4),
        })

    train_support = Counter(y_tr)
    sup = np.array([train_support[s] for s in y_te])
    out["accuracy_by_train_support"] = {}
    for name, lo, hi in (("1-4", 1, 4), ("5-9", 5, 9), ("10+", 10, 10**9)):
        m = (sup >= lo) & (sup <= hi)
        out["accuracy_by_train_support"][name] = {
            "test_samples": int(m.sum()),
            "accuracy": round(float((pred[m] == y_te[m]).mean()), 4),
        }

    print(json.dumps(out, indent=2))
    with open("eval_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()