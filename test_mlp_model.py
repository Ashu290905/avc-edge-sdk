#!/usr/bin/env python3
import time
import datetime as dt
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report

from feature_extraction import parse_ir_string, extract_features

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TEST_CSV = "test_super.csv"   # produced by build_super_csvs.py

def main():
    # Load artefacts from models/ folder
    mlp_path    = os.path.join(MODELS_DIR, "mlp_super_model.pkl")
    le_path     = os.path.join(MODELS_DIR, "mlp_super_label_encoder.pkl")
    scaler_path = os.path.join(MODELS_DIR, "mlp_super_scaler.pkl")

    mlp    = joblib.load(mlp_path)
    le     = joblib.load(le_path)
    scaler = joblib.load(scaler_path)

    expected_dim = scaler.mean_.shape[0]
    print(f"Model expects feature dimension: {expected_dim}")

    df = pd.read_csv(TEST_CSV)

    X, y_super, y_fine = [], [], []
    t_feat = 0.0
    for _, row in df.iterrows():
        t0 = time.perf_counter()
        prof  = parse_ir_string(row["binary_string"])
        feats = extract_features(prof)
        t_feat += (time.perf_counter() - t0)
        if feats is not None:
            X.append(feats)
            y_super.append(row["label"])       # superclass
            y_fine.append(row.get("fine_label", ""))  # original fine label if present

    if not X:
        print("No valid samples.")
        return

    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] != expected_dim:
        raise ValueError(f"Feature dim {X.shape[1]} != expected {expected_dim}. Retrain after feature changes.")

    t_model = 0.0
    t0 = time.perf_counter()
    Xs = scaler.transform(X)
    t_model += (time.perf_counter() - t0)

    t0 = time.perf_counter()
    y_pred_enc = mlp.predict(Xs)
    t_model += (time.perf_counter() - t0)

    y_true_enc = le.transform(y_super)
    y_pred = le.inverse_transform(y_pred_enc)

    present_idx   = sorted(np.unique(y_true_enc))
    present_names = le.inverse_transform(present_idx)

    print("\n=== Test report (superclasses) ===")
    print(classification_report(
        y_true_enc, y_pred_enc,
        labels=present_idx,
        target_names=present_names,
        digits=4
    ))

    report = classification_report(
        y_true_enc, y_pred_enc,
        labels=present_idx,
        target_names=present_names,
        digits=4,
        output_dict=True
    )

    rows = []
    for name in present_names:
        rows.append({
            "superclass": name,
            "precision": report[name]["precision"],
            "recall":    report[name]["recall"],
            "f1":        report[name]["f1-score"],
            "support":   int(report[name]["support"]),
        })
    per_super_df = pd.DataFrame(rows).sort_values("superclass")

    n = len(X)
    feat_ms  = (t_feat  / n) * 1000.0
    model_ms = (t_model / n) * 1000.0
    total_ms = feat_ms + model_ms

    print("\n=== Timing ===")
    print(f"Avg feature extraction / sample: {feat_ms:.3f} ms")
    print(f"Avg scale+predict   / sample: {model_ms:.3f} ms")
    print(f"Avg end-to-end      / sample: {total_ms:.3f} ms")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    preds_path   = f"test_super_predictions_{ts}.csv"
    metrics_path = f"test_super_metrics_{ts}.csv"

    pd.DataFrame({
        "fine_label":       y_fine,
        "true_superclass":  y_super,
        "pred_superclass":  y_pred,
    }).to_csv(preds_path, index=False)

    per_super_df.to_csv(metrics_path, index=False)

    print("\nSaved:")
    print(f"  {preds_path}")
    print(f"  {metrics_path}")

if __name__ == "__main__":
    main()
