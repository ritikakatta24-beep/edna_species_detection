"""
Step 2 of the ML pipeline: k-mer encoding + k-NN training.

Takes amphibia_dataset.csv (record_id, species, sequence) and:
1. Cleans out placeholder/unresolved species labels (e.g. Genus_X, Family_XX)
2. Converts each DNA sequence into a fixed-length k-mer frequency vector
3. Splits into train/test sets
4. Trains a k-NN classifier
5. Reports accuracy + shows which species are most confused with each other

Usage:
    python train_knn.py
"""

import re
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT_CSV = "amphibia_dataset.csv"
K = 4  # k-mer length -- 4 is a good default (256 possible k-mers)
TEST_SIZE = 0.2
RANDOM_STATE = 42


def is_placeholder_species(name: str) -> bool:
    """Returns True for BOLD's unresolved-taxonomy labels like 'Hylidae_XX' or 'Limnonectes_X'."""
    if not isinstance(name, str):
        return True
    return bool(re.search(r"_X{1,2}$", name.strip()))


def build_kmer_vocab(k: int):
    bases = ["A", "T", "C", "G"]
    return ["".join(p) for p in itertools.product(bases, repeat=k)]


def sequence_to_kmer_vector(seq: str, k: int, kmer_index: dict) -> np.ndarray:
    vec = np.zeros(len(kmer_index), dtype=np.float32)
    seq = seq.upper()
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer in kmer_index:  # skips k-mers with ambiguous bases like N
            vec[kmer_index[kmer]] += 1
            total += 1
    if total > 0:
        vec /= total  # normalize to frequencies, so sequence length doesn't skew results
    return vec


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows, {df['species'].nunique()} unique species labels.")

    print("\nRemoving placeholder/unresolved species labels (e.g. Genus_X, Family_XX)...")
    before = len(df)
    df = df[~df["species"].apply(is_placeholder_species)]
    print(f"Removed {before - len(df)} rows. {len(df)} rows remain, {df['species'].nunique()} real species.")

    print(f"\nBuilding k-mer vocabulary for k={K}...")
    kmer_vocab = build_kmer_vocab(K)
    kmer_index = {kmer: i for i, kmer in enumerate(kmer_vocab)}
    print(f"Vocabulary size: {len(kmer_vocab)} possible k-mers.")

    print("\nConverting sequences to k-mer frequency vectors (this may take a minute)...")
    X = np.stack([sequence_to_kmer_vector(seq, K, kmer_index) for seq in df["sequence"]])
    y = df["species"].values
    print(f"Feature matrix shape: {X.shape}")

    print("\nSplitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    print("\nTraining k-NN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(X_train, y_train)

    print("\nEvaluating on test set...")
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall accuracy: {acc:.2%}")

    print("\nPer-species performance (precision/recall/f1) for top species:")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    report_df = report_df.sort_values("support", ascending=False)
    print(report_df.head(15))

    print("\nSaving model + vocabulary for later use in the app...")
    import joblib
    joblib.dump({"model": knn, "kmer_index": kmer_index, "k": K}, "knn_species_model.joblib")
    print("Saved to knn_species_model.joblib")


if __name__ == "__main__":
    main()