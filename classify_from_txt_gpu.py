#!/usr/bin/env python3
import sys, time, joblib, torch
import torch.nn as nn
import numpy as np
from feature_extraction import parse_ir_string, extract_features

MODEL_PT  = "models/mlp_model.pt"
SCALER_PK = "models/mlp_scaler.pkl"
ENC_PK    = "models/mlp_label_encoder.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load artefacts ---
checkpoint = torch.load(MODEL_PT, map_location=device)
layer_sizes = checkpoint["layer_sizes"]
layers = []
for i in range(len(layer_sizes) - 2):
    layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.ReLU()]
layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
model = nn.Sequential(*layers).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

scaler = joblib.load(SCALER_PK)
le     = joblib.load(ENC_PK)

def classify(binary_str: str):
    profile = parse_ir_string(binary_str)
    feats = extract_features(profile)
    if feats is None:
        raise ValueError("Invalid / empty sensor string")

    x = scaler.transform([feats]).astype(np.float32)
    with torch.no_grad():
        inp = torch.from_numpy(x).to(device)
        t0 = time.perf_counter()
        logits = model(inp)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pred_idx = torch.argmax(logits, dim=1).cpu().item()
    return le.inverse_transform([pred_idx])[0], elapsed_ms

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <binary_string.txt>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        binary_str = f.read().strip()

    label, t_ms = classify(binary_str)
    print(label)
    print(f"Time: {t_ms:.3f} ms")
