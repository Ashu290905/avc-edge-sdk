#!/usr/bin/env python3
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from feature_extraction import parse_ir_string, extract_features

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)  # create models folder if it doesn't exist

TRAIN_CSV    = "train_super.csv"   # produced by build_super_csvs.py
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET_RATIO = 0.40   # upsample minorities to 40% of majority (tune)
VERBOSE      = True

def capped_smote(Xs, y, ratio=TARGET_RATIO):
    counts = np.bincount(y)
    maj = counts.max()
    target = int(max(2, maj * ratio))

    strategy = {c: target for c, cnt in enumerate(counts) if 0 < cnt < target}
    if not strategy:
        return Xs, y

    k = max(1, min(5, counts[counts > 0].min() - 1))
    sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=RANDOM_STATE)
    return sm.fit_resample(Xs, y)

def main():
    # Load
    df = pd.read_csv(TRAIN_CSV)
    X_list, y = [], []
    for _, row in df.iterrows():
        prof = parse_ir_string(row["binary_string"])
        feats = extract_features(prof)
        if feats is not None:
            X_list.append(feats)
            y.append(row["label"])

    if not X_list:
        raise RuntimeError("No features extracted.")

    X = np.asarray(X_list, dtype=np.float32)
    print(f"▶ Features: {X.shape[1]}  Samples: {X.shape[0]}  Superclasses: {len(set(y))}")

    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    Xb, yb = capped_smote(Xs, y_enc)
    print(f"▶ After capped SMOTE: {Xb.shape[0]} samples")

    X_tr, X_va, y_tr, y_va = train_test_split(
        Xb, yb, test_size=TEST_SIZE, stratify=yb, random_state=RANDOM_STATE
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.01,                 
        learning_rate_init=3e-4,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        verbose=VERBOSE,
        random_state=RANDOM_STATE,
    )

    mlp.fit(X_tr, y_tr)

    y_pred = mlp.predict(X_va)
    print("\n=== Validation (superclasses) ===")
    print(classification_report(y_va, y_pred, target_names=le.classes_, digits=4))

    # Plot recalls
    report = classification_report(
        y_va, y_pred, target_names=le.classes_, digits=4, output_dict=True
    )
    recalls = [report[c]["recall"] for c in le.classes_]
    plt.figure(figsize=(14, 6))
    plt.bar(le.classes_, recalls)
    plt.xticks(rotation=90)
    plt.ylabel("Recall")
    plt.title("Per-Superclass Recall (MLP)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "train_per_superclass_recall.png"), dpi=300)
    plt.show()

    # Save artifacts in models/ folder
    joblib.dump(mlp,    os.path.join(MODELS_DIR, "mlp_super_model.pkl"))
    joblib.dump(le,     os.path.join(MODELS_DIR, "mlp_super_label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "mlp_super_scaler.pkl"))

    print(f"\n✔ Saved models to: {MODELS_DIR}")

if __name__ == "__main__":
    main()
