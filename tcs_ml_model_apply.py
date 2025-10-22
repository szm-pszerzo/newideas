#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for TCS classification.

Inputs:
  - newdata.csv (columns: oe, gyto; tcs optional)
  - tcs_artifacts/ created by the training export step:
      model.joblib
      le_tcs.joblib
      le_gyto.joblib
      le_prefix2.joblib
      le_prefix3.joblib
      le_suffix2.joblib
      le_suffix3.joblib
      meta.joblib

Output:
  - predicted_newdata.csv (adds tcs_pred, tcs_top3, confidence)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib
import gc  # <-- needed for the safe loader

ARTIFACTS_DIR = Path("tcs_artifacts")
INPUT_FILE    = Path("newdata.csv")
OUTPUT_FILE   = Path("predicted_newdata.csv")

# --------------------------- Safe Joblib Loader ---------------------------
def safe_joblib_load(path):
    """Load large joblib artifacts with mmap, fallback to normal load on MemoryError."""
    try:
        return joblib.load(path, mmap_mode="r")
    except MemoryError:
        gc.collect()
        return joblib.load(path, mmap_mode=None)

# --------------------------- Heuristics ---------------------------
def heuristic_guess(oe: str, gyto: str):
    oe_u = str(oe).upper()
    gyto_u = str(gyto).upper()
    if "41" in oe_u or oe_u.startswith(("1K0", "1J0")):
        return "Suspension"
    if "86" in oe_u and "TOYOTA" in gyto_u:
        return "Engine"
    if "LAMP" in oe_u or "LIGHT" in oe_u or "VALEO" in gyto_u:
        return "Lighting"
    if "BOSCH" in gyto_u and re.search(r"\d{3,}", oe_u):
        return "Electrical"
    return None

# --------------------------- Feature Extraction ---------------------------
def extract_features(oe: str) -> pd.Series:
    oe = str(oe)
    n_digits = sum(c.isdigit() for c in oe)
    n_letters = sum(c.isalpha() for c in oe)
    L = len(oe) if len(oe) > 0 else 1
    return pd.Series({
        "len_oe": len(oe),
        "n_digits": n_digits,
        "n_letters": n_letters,
        "has_dash": int("-" in oe),
        "has_space": int(" " in oe),
        "has_slash": int("/" in oe),
        "digit_ratio": n_digits / L,
        "prefix2": oe[:2],
        "prefix3": oe[:3],
        "suffix2": oe[-2:],
        "suffix3": oe[-3:],
    })

# Safe transform helpers for encoders
def safe_transform_fragment(le, value: str) -> int:
    """Map unseen fragment tokens to '_OTHER_' class, which must exist in training."""
    value = str(value)
    classes = set(le.classes_)
    if value not in classes:
        value = "_OTHER_"
        if value not in classes:
            value = le.classes_[0]
    return int(np.where(le.classes_ == value)[0][0])

def safe_transform_gyto(le, value: str, fallback: str | None) -> int:
    value = str(value)
    classes = set(le.classes_)
    if value not in classes:
        value = fallback if fallback in classes else le.classes_[0]
    return int(np.where(le.classes_ == value)[0][0])

def build_feature_matrix(df: pd.DataFrame, encs: dict, meta: dict) -> np.ndarray:
    # extract base features
    feat = df["oe"].apply(extract_features)
    df_feat = pd.concat([df.copy(), feat], axis=1)

    # encode fragments and gyto using saved encoders
    df_feat["prefix2_enc"] = [safe_transform_fragment(encs["le_prefix2"], v) for v in df_feat["prefix2"].astype(str)]
    df_feat["prefix3_enc"] = [safe_transform_fragment(encs["le_prefix3"], v) for v in df_feat["prefix3"].astype(str)]
    df_feat["suffix2_enc"] = [safe_transform_fragment(encs["le_suffix2"], v) for v in df_feat["suffix2"].astype(str)]
    df_feat["suffix3_enc"] = [safe_transform_fragment(encs["le_suffix3"], v) for v in df_feat["suffix3"].astype(str)]

    df_feat["gyto_enc"] = [safe_transform_gyto(encs["le_gyto"], v, meta.get("gyto_fallback")) for v in df_feat["gyto"].astype(str)]

    # align to training feature order
    df_feat["prefix2"] = df_feat["prefix2_enc"]
    df_feat["prefix3"] = df_feat["prefix3_enc"]
    df_feat["suffix2"] = df_feat["suffix2_enc"]
    df_feat["suffix3"] = df_feat["suffix3_enc"]

    feature_cols = meta["feature_cols"]
    X = df_feat[feature_cols].astype(np.float32).values
    return X, df_feat

# --------------------------- Load Artifacts ---------------------------
print("🔹 Loading artifacts…")
model = safe_joblib_load(ARTIFACTS_DIR / "model.joblib")
le_tcs = safe_joblib_load(ARTIFACTS_DIR / "le_tcs.joblib")
le_gyto = safe_joblib_load(ARTIFACTS_DIR / "le_gyto.joblib")
le_prefix2 = safe_joblib_load(ARTIFACTS_DIR / "le_prefix2.joblib")
le_prefix3 = safe_joblib_load(ARTIFACTS_DIR / "le_prefix3.joblib")
le_suffix2 = safe_joblib_load(ARTIFACTS_DIR / "le_suffix2.joblib")
le_suffix3 = safe_joblib_load(ARTIFACTS_DIR / "le_suffix3.joblib")
meta = safe_joblib_load(ARTIFACTS_DIR / "meta.joblib")

encoders = {
    "le_gyto": le_gyto,
    "le_prefix2": le_prefix2,
    "le_prefix3": le_prefix3,
    "le_suffix2": le_suffix2,
    "le_suffix3": le_suffix3,
}

# --------------------------- Load New Data ---------------------------
print("🔹 Reading newdata.csv…")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE.resolve()}")

df_new = pd.read_csv(INPUT_FILE)
if "oe" not in df_new.columns or "gyto" not in df_new.columns:
    raise ValueError("newdata.csv must contain at least columns: 'oe', 'gyto'")

df_new["oe"] = df_new["oe"].astype(str).str.upper().str.strip()
df_new["gyto"] = df_new["gyto"].astype(str).str.upper().str.strip()

# --------------------------- Build Features ---------------------------
X_new, df_feat = build_feature_matrix(df_new, encoders, meta)

# --------------------------- Predict ---------------------------
print("🔹 Predicting (heuristics first, then model)…")
tcs_pred, tcs_top3, confidence = [], [], []

for i, row in df_new.iterrows():
    h = heuristic_guess(row["oe"], row["gyto"])
    if h is not None:
        tcs_pred.append(h)
        tcs_top3.append(h)
        confidence.append(1.0)
        continue

    probs = model.predict_proba(X_new[i:i+1])[0]
    idx = np.argsort(probs)[::-1][:3]
    labels = le_tcs.inverse_transform(idx)
    tcs_pred.append(labels[0])
    tcs_top3.append(", ".join(labels))
    confidence.append(float(probs[idx[0]]))

df_out = df_new.copy()
df_out["tcs_pred"] = tcs_pred
df_out["tcs_top3"] = tcs_top3
df_out["confidence"] = confidence

# --------------------------- Save ---------------------------
df_out.to_csv(OUTPUT_FILE, index=False)
print(f"🎉 Done. Wrote: {OUTPUT_FILE.resolve()}")
