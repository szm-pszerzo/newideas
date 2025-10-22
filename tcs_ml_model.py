#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict product category (tcs) from oe and gyto using
pattern features + memory-safe RandomForest + heuristics.

Input : rawdata.csv  (columns: oe, gyto, tcs) — in same folder
Output: predicted_tcs.csv (adds tcs_pred, tcs_top3, confidence)
Also writes artifacts to ./tcs_artifacts for inference.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
import joblib

# --------------------------- Config ---------------------------
INPUT_FILE   = Path("rawdata.csv")
OUTPUT_FILE  = Path("predicted_tcs.csv")
ARTIFACTS_DIR = Path("tcs_artifacts")

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Memory-safe RF settings
RF_PARAMS = dict(
    n_estimators=120,
    max_depth=22,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=1,  # critical to avoid RAM spikes
)

# Cap vocab sizes for string fragments (prevents huge cardinality)
TOPK_PREFIX2 = 200
TOPK_PREFIX3 = 300
TOPK_SUFFIX2 = 200
TOPK_SUFFIX3 = 300

# --------------------------- Helpers ---------------------------
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

def cap_by_frequency(s: pd.Series, top_k: int, other_token: str = "_OTHER_") -> pd.Series:
    """Keep top_k frequent tokens, map the rest to OTHER (controls feature cardinality)."""
    vc = s.value_counts(dropna=False)
    keep = set(vc.head(top_k).index)
    return s.where(s.isin(keep), other_token)

def heuristic_guess(oe: str, gyto: str):
    """Return a heuristic tcs if pattern detected, else None."""
    oe_u = str(oe).upper()
    gyto_u = str(gyto).upper()

    # Examples — extend with your domain knowledge
    if "41" in oe_u or oe_u.startswith(("1K0", "1J0")):
        return "Suspension"
    if "86" in oe_u and "TOYOTA" in gyto_u:
        return "Engine"
    if "LAMP" in oe_u or "LIGHT" in oe_u or "VALEO" in gyto_u:
        return "Lighting"
    if "BOSCH" in gyto_u and re.search(r"\d{3,}", oe_u):
        return "Electrical"
    return None

# --------------------------- Load & Clean ---------------------------
print("🔹 Loading data…")
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE.resolve()}")

df_all = pd.read_csv(INPUT_FILE)

required_cols = {"oe", "gyto", "tcs"}
missing = required_cols - set(df_all.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df_all["oe"] = df_all["oe"].astype(str).str.upper().str.strip()
df_all["gyto"] = df_all["gyto"].astype(str).str.upper().str.strip()
# keep tcs as-is; some rows may be NaN (unlabeled)

print(f"✅ Loaded {len(df_all)} rows.")

# --------------------------- Feature Engineering (ALL rows) ---------------------------
print("🔹 Extracting OE features…")
feat_all = df_all["oe"].apply(extract_features)
df_all = pd.concat([df_all, feat_all], axis=1)

# Cap high-cardinality string fragments BEFORE encoding (ensures '_OTHER_' exists)
df_all["prefix2"] = cap_by_frequency(df_all["prefix2"].astype(str), TOPK_PREFIX2)
df_all["prefix3"] = cap_by_frequency(df_all["prefix3"].astype(str), TOPK_PREFIX3)
df_all["suffix2"] = cap_by_frequency(df_all["suffix2"].astype(str), TOPK_SUFFIX2)
df_all["suffix3"] = cap_by_frequency(df_all["suffix3"].astype(str), TOPK_SUFFIX3)

# --- Fit & STORE each fragment encoder explicitly ---
prefix2_encoder = LabelEncoder()
prefix3_encoder = LabelEncoder()
suffix2_encoder = LabelEncoder()
suffix3_encoder = LabelEncoder()

df_all["prefix2"] = prefix2_encoder.fit_transform(df_all["prefix2"].astype(str))
df_all["prefix3"] = prefix3_encoder.fit_transform(df_all["prefix3"].astype(str))
df_all["suffix2"] = suffix2_encoder.fit_transform(df_all["suffix2"].astype(str))
df_all["suffix3"] = suffix3_encoder.fit_transform(df_all["suffix3"].astype(str))

# Manufacturer encoder (store it too)
le_gyto = LabelEncoder()
df_all["gyto_enc"] = le_gyto.fit_transform(df_all["gyto"].astype(str))

feature_cols = [
    "len_oe", "n_digits", "n_letters", "has_dash",
    "has_space", "has_slash", "digit_ratio",
    "prefix2", "prefix3", "suffix2", "suffix3",
    "gyto_enc",
]

# --------------------------- Build Train Set ---------------------------
print("🔹 Preparing training dataset…")
df_train = df_all.dropna(subset=["tcs"]).copy()

# Remove ultra-rare classes so stratify works (min 2 samples per class)
tcs_counts = df_train["tcs"].value_counts()
rare_classes = tcs_counts[tcs_counts < 2].index
if len(rare_classes) > 0:
    print(f"⚠️ Removing {len(rare_classes)} rare tcs classes with <2 samples from training only.")
    df_train = df_train[~df_train["tcs"].isin(rare_classes)].copy()

if df_train["tcs"].nunique() < 2:
    raise ValueError("Need at least 2 classes in 'tcs' after filtering to train a classifier.")

# Cast to compact dtypes for memory safety (on numeric feature columns)
for col in feature_cols:
    if df_all[col].dtype == bool:
        df_all[col] = df_all[col].astype(np.int8)
    elif pd.api.types.is_integer_dtype(df_all[col]):
        df_all[col] = df_all[col].astype(np.int32)
    else:
        df_all[col] = df_all[col].astype(np.float32)

X_train_mat = df_train[feature_cols].astype(np.float32).values
le_tcs = LabelEncoder()
y_train_vec = le_tcs.fit_transform(df_train["tcs"].values)

# Stratified split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_train_mat, y_train_vec,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_vec
)

print(f"🧮 Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
print(f"   Train mem ~ {X_tr.nbytes/1e6:.1f} MB, Test mem ~ {X_te.nbytes/1e6:.1f} MB")

# --------------------------- Train Model (RandomForest) ---------------------------
print("🔹 Training RandomForest (memory-safe settings)…")
model = RandomForestClassifier(**RF_PARAMS)
model.fit(X_tr, y_tr)

y_pred = model.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"✅ Accuracy: {acc:.2%}")

# Match labels to what's present in y_te ∪ y_pred
labels_eval = np.union1d(y_te, y_pred)
target_names_eval = le_tcs.inverse_transform(labels_eval)
print("🔹 Classification report (only labels present in test/pred):")
print(classification_report(
    y_te, y_pred,
    labels=labels_eval,
    target_names=target_names_eval,
    zero_division=0
))

# Aggregate metrics
p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
    y_te, y_pred, average="macro", zero_division=0
)
p_w, r_w, f1_w, _ = precision_recall_fscore_support(
    y_te, y_pred, average="weighted", zero_division=0
)
print(f"📊 Macro F1: {f1_macro:.4f} | Weighted F1: {f1_w:.4f}")

# # --------------------------- Predict ALL rows (optional output) ---------------------------
# print("🔹 Generating predictions for all rows (heuristics + model)…")
# def row_vector(row) -> np.ndarray:
#     return row[feature_cols].astype(np.float32).to_numpy().reshape(1, -1)
#
# tcs_pred = []
# tcs_top3 = []
# confidence = []
#
# for _, row in df_all.iterrows():
#     # Heuristic first
#     h = heuristic_guess(row["oe"], row["gyto"])
#     if h is not None:
#         tcs_pred.append(h)
#         tcs_top3.append(h)
#         confidence.append(1.0)
#         continue
#
#     # Model path
#     x = row_vector(row)
#     probs = model.predict_proba(x)[0]
#     idx = np.argsort(probs)[::-1][:3]
#     labels = le_tcs.inverse_transform(idx)
#     tcs_pred.append(labels[0])
#     tcs_top3.append(", ".join(labels))
#     confidence.append(float(probs[idx[0]]))
#
# df_out = df_all.copy()
# df_out["tcs_pred"] = tcs_pred
# df_out["tcs_top3"] = tcs_top3
# df_out["confidence"] = confidence
#
# print("🔹 Saving results CSV…")
# df_out.to_csv(OUTPUT_FILE, index=False)
# print(f"🎉 Wrote predictions: {OUTPUT_FILE.resolve()}")

# --------------------------- Export artifacts for inference ---------------------------
print("🔹 Saving artifacts for inference…")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Fallback for unseen GYTO at inference: training mode (most frequent)
try:
    gyto_fallback = df_all.assign(_gyto_raw=le_gyto.inverse_transform(df_all["gyto_enc"]))["_gyto_raw"].mode()[0]
except Exception:
    gyto_fallback = le_gyto.classes_[0] if hasattr(le_gyto, "classes_") else None

meta = {
    "feature_cols": feature_cols,
    "rf_params": RF_PARAMS,
    "gyto_fallback": gyto_fallback,
    "topk": {
        "prefix2": TOPK_PREFIX2,
        "prefix3": TOPK_PREFIX3,
        "suffix2": TOPK_SUFFIX2,
        "suffix3": TOPK_SUFFIX3,
    },
    "_note": "Fragments were capped with _OTHER_ before encoding; unseen tokens at inference should map to _OTHER_.",
}

joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
joblib.dump(le_tcs, ARTIFACTS_DIR / "le_tcs.joblib")
joblib.dump(le_gyto, ARTIFACTS_DIR / "le_gyto.joblib")
joblib.dump(prefix2_encoder, ARTIFACTS_DIR / "le_prefix2.joblib")
joblib.dump(prefix3_encoder, ARTIFACTS_DIR / "le_prefix3.joblib")
joblib.dump(suffix2_encoder, ARTIFACTS_DIR / "le_suffix2.joblib")
joblib.dump(suffix3_encoder, ARTIFACTS_DIR / "le_suffix3.joblib")
joblib.dump(meta, ARTIFACTS_DIR / "meta.joblib")

print("✅ Artifacts saved under ./tcs_artifacts/")
