"""
deocker_services.py
--------------------
D-CHSS Dengue Detection modèles এর prediction service।

এই সংস্করণটি 13.ipynb থেকে export করা-trained sklearn pipeline artifact
(dengue_dchss_model.pkl) ব্যবহার করে। প্রি-প্রসেসিং (D-ChSS feature builder,
imputation, scaling ও মডেল) সবই artifact এর ভেতরে ব accepted থাকে,
যা make_prediction() (dchss_model.py) এর মাধ্যমে predict করে।

Django শুধু artifact লোড করে, একটি CBC record পাঠায় এবং ফলাফল ফেরত পায়।
dchss_score, risk_group এবং top_feature_explanations extraordinaire field গুলো
artifact মেটাডাটা থেকে এখানে compute করা হয়, যাতে Django view / model schema
এবং frontend API response কনট్రాక్టর একই থাকে (কোনো ব্রেকিং চেঞ্জ নেই)।
"""

import os
import sys
import pickle
import threading
from django.conf import settings

# ---------------------------------------------------------------------------
# Repo root ও ml_model ডিরেক্টরি কে import path এ যোগ করা, যাতে
# dchss_model.py (dengues/ml_model এ থাকা) যে কোনো working directory থেকে
# import করা যায়।
# ---------------------------------------------------------------------------
_PROJECT_ROOT = settings.BASE_DIR.parent  # /app (repo root)
_ML_MODEL_DIR = os.path.join(str(_PROJECT_ROOT), "dengues", "ml_model")
if str(_ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_MODEL_DIR))

from dchss_model import make_prediction  # noqa: E402


class ModelNotLoadedError(Exception):
    """Model artifact (pkl) খুঁজে না পেলে / load করতে সমস্যা হলে এই exception raise হবে"""
    pass


# ---------------------------------------------------------------------------
# ⭐ Model artifact path - এখানেই .pkl ফাইলটা রাখতে হবে
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(str(_ML_MODEL_DIR), "dengue_dchss_model.pkl")

EPS = 1e-6

_artifact = None
_lock = threading.Lock()


def load_artifact():
    """
    Singleton loader - pickle file একবারই load হয়ে memory তে cache থাকে।
    Thread-safe (Gunicorn multi-worker এ সমস্যা হবে না)।
    """
    global _artifact
    if _artifact is None:
        with _lock:
            if _artifact is None:
                if not os.path.exists(MODEL_PATH):
                    raise ModelNotLoadedError(
                        f"Model artifact পাওয়া যায়নি: {MODEL_PATH}. "
                        f"13.ipynb থেকে generate করা 'dengue_dchss_model.pkl' "
                        f"file টা এই path এ কপি করতে হবে।"
                    )
                with open(MODEL_PATH, "rb") as f:
                    _artifact = pickle.load(f)
    return _artifact


# ---------------------------------------------------------------------------
# D-ChSS score ও risk group helpers (artifact মেটাডাটা থেকে compute)
# dchss_model.DChSSFeatureBuilder এর "dchss_only" মোডের লজciens mirror করে।
# ---------------------------------------------------------------------------
def _get_interval(artifact, feature, gender):
    """Gender-specific clinical reference range বের করা (না থাকলে default)"""
    interval = artifact["reference_ranges"][feature]
    if isinstance(interval, dict):
        if gender == 1:
            return interval["male"]
        if gender == 0:
            return interval["female"]
        return interval["default"]
    return interval  # tuple (low, high)


def _compute_deviation(value, low, high, max_dev):
    """
    Reference range এর বাউন্ডারি থেকে signed deviation বের করা এবং max_dev 에 cap করা।
    dchss_model.py এর transform()-এর সাথে একই সংজ্ঞা।
    """
    if value is None or low is None or high is None:
        return 0.0
    if value < low:
        dev = (value - low) / low
    elif value > high:
        dev = (value - high) / high
    else:
        dev = 0.0
    return float(max(-max_dev, min(max_dev, dev)))


def compute_dchss_score(raw_input, artifact):
    """
    D-ChSS score বের করা artifact এর reference_ranges, dchss_weights এবং
    ts/ls/hs features ব্যবহার করে। (TS, LS, HS উপাদানও রিটার্ন করে)
    """
    reference_ranges = artifact["reference_ranges"]
    weights = artifact["dchss_weights"]  # {'TS':.., 'LS':.., 'HS':..}
    cbc_features = artifact["cbc_features"]
    ts_features = artifact["ts_features"]
    ls_features = artifact["ls_features"]
    hs_features = artifact["hs_features"]

    gender = str(raw_input.get("gender", "other")).strip().lower()
    if gender in {"male", "m", "man", "1"}:
        gender_enc = 1
    elif gender in {"female", "f", "woman", "0"}:
        gender_enc = 0
    else:
        gender_enc = -1

    max_dev = 1.0
    signed = {}
    for feature in cbc_features:
        value = raw_input.get(feature)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        low, high = _get_interval(artifact, feature, gender_enc)
        signed[feature] = _compute_deviation(value, low, high, max_dev)

    absolute = {f: abs(v) for f, v in signed.items()}

    def mean_dev(feats):
        vals = [absolute[f] for f in feats if f in absolute]
        return sum(vals) / len(vals) if vals else 0.0

    ts = mean_dev(ts_features)
    ls = mean_dev(ls_features)
    hs = mean_dev(hs_features)

    dchss = weights["TS"] * ts + weights["LS"] * ls + weights["HS"] * hs
    return round(dchss, 4), {"TS": ts, "LS": ls, "HS": hs, "signed": signed}


def assign_risk_group(dchss_score, risk_thresholds=None):
    """
    D-ChSS score কে Low / Moderate / High এ ভাগ করা।
    artifact এ risk_thresholds না থাকলে default threshold ব্যবহার করা হয়।
    """
    if risk_thresholds is None:
        risk_thresholds = (0.25, 0.50)
    low_thresh, high_thresh = risk_thresholds
    if dchss_score < low_thresh:
        return "Low"
    elif dchss_score <= high_thresh:
        return "Moderate"
    else:
        return "High"


def build_top_feature_explanations(raw_input, artifact, top_n=8):
    """
    প্রতিটি CBC feature এর deviation magnitude অনুযায়ী top_N ফিচার কে
    explanation হিসেবে রিটার্ন করা (SHAP এর বদলে lightweight approximation)।
    effect মানে 'dengue risk বাড়াচ্ছে' বা 'dengue risk কমাচ্ছে'।
    """
    reference_ranges = artifact["reference_ranges"]
    cbc_features = artifact["cbc_features"]

    gender = str(raw_input.get("gender", "other")).strip().lower()
    if gender in {"male", "m", "man", "1"}:
        gender_enc = 1
    elif gender in {"female", "f", "woman", "0"}:
        gender_enc = 0
    else:
        gender_enc = -1

    contributions = []
    for feature in cbc_features:
        value = raw_input.get(feature)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        low, high = _get_interval(artifact, feature, gender_enc)
        if low is None or high is None:
            continue
        dev = _compute_deviation(value, low, high, 1.0)
        if abs(dev) < EPS:
            continue
        effect = "dengue risk বাড়াচ্ছে" if dev > 0 else "dengue risk কমাচ্ছে"
        contributions.append((feature, round(dev, 4), effect))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:top_n]

    explanations = []
    for feat, contrib, effect in top:
        explanations.append({"feature": feat, "contribution": contrib, "effect": effect})
    return explanations


# ---------------------------------------------------------------------------
# ⭐ Public API - Django view থেকে এই ফাংশনটাই call হবে
# ---------------------------------------------------------------------------
def predict_dengue(raw_input: dict) -> dict:
    """
    raw_input: {
        'gender': 'male'|'female'|'other', 'age': float,
        'hemoglobin':.., 'neutrophils':.., 'lymphocytes':.., 'monocytes':..,
        'eosinophils':.., 'rbc':.., 'hct':.., 'mcv':.., 'mch':.., 'mchc':..,
        'rdw_cv':.., 'platelet':.., 'mpv':.., 'pdw':.., 'pct':.., 'wbc':..
    }

    রিটার্ন ডিকশনারি keys গুলো views.py ও frontend (DenguePredictionResult)-এর
    সাথে একই রাখা হয়েছে, তাই কোনো ব্রেকিং চেঞ্জ নেই।
    """
    raw_input = dict(raw_input)
    raw_input["gender"] = str(raw_input.get("gender", "other")).lower()

    artifact = load_artifact()

    # Core probability / prediction trained sklearn pipeline- এর মাধ্যমে
    result = make_prediction(artifact, raw_input)

    probability = float(result["probability"])
    threshold = float(result["threshold"])
    prediction_int = int(result["prediction"])
    prediction_label = "Positive" if prediction_int == 1 else "Negative"

    probability_percent = round(probability * 100, 2)
    threshold_used_percent = round(threshold * 100, 2)

    # D-ChSS score ও risk group artifact মেটাডাটা থেকে compute
    dchss_score, _dchss_parts = compute_dchss_score(raw_input, artifact)
    risk_group = assign_risk_group(dchss_score)

    # Top feature explanations
    explanations = build_top_feature_explanations(raw_input, artifact, top_n=8)

    return {
        "probability_percent": probability_percent,
        "prediction": prediction_label,
        "risk_group": risk_group,
        "dchss_score": dchss_score,
        "threshold_used_percent": threshold_used_percent,
        "top_feature_explanations": explanations,
        "label": result["label"],
        "disclaimer": "Research demonstration only - NOT a medical diagnosis tool.",
    }