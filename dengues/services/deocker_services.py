import os
import sys
import pickle
import threading
from django.conf import settings

_PROJECT_ROOT = settings.BASE_DIR
_ML_MODEL_DIR = os.path.join(str(_PROJECT_ROOT), "dengues", "ml_model")
if str(_ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_MODEL_DIR))

from dchss_model import make_prediction  # noqa: E402


class ModelNotLoadedError(Exception):
    pass


MODEL_PATH = os.path.join(str(_ML_MODEL_DIR), "dengue_dchss_model.pkl")

EPS = 1e-6

_artifact = None
_lock = threading.Lock()


def _get_key(artifact, *names, default=None):
    """Artifact থেকে możliwo কী-নেমের jouk প্রথম পাওয়া ভ্যালু রিটার্ন করে।"""
    for name in names:
        if name in artifact:
            return artifact[name]
    return default


def load_artifact():
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


def _get_interval(artifact, feature, gender):
    reference_ranges = _get_key(
        artifact,
        "reference_ranges",
        "reference_range",
        "ranges",
        default={},
    )
    interval = reference_ranges.get(feature)
    if interval is None:
        return None, None
    if isinstance(interval, dict):
        if gender == 1:
            val = interval.get("male")
        elif gender == 0:
            val = interval.get("female")
        else:
            val = interval.get("default")
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return val[0], val[1]
        return None, None
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        return interval[0], interval[1]
    return None, None


def _compute_deviation(value, low, high, max_dev):
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
    reference_ranges = _get_key(
        artifact,
        "reference_ranges",
        "reference_range",
        "ranges",
        default={},
    )
    weights = _get_key(
        artifact,
        "dchss_weights",
        "dchss_weight",
        "weights",
        "dchss",
        default={"TS": 1.0, "LS": 1.0, "HS": 1.0},
    )
    cbc_features = _get_key(
        artifact,
        "cbc_features",
        "cbc_feature_names",
        "features",
        default=[],
    )
    ts_features = _get_key(
        artifact,
        "ts_features",
        "ts_feature_names",
        "ts",
        default=[],
    )
    ls_features = _get_key(
        artifact,
        "ls_features",
        "ls_feature_names",
        "ls",
        default=[],
    )
    hs_features = _get_key(
        artifact,
        "hs_features",
        "hs_feature_names",
        "hs",
        default=[],
    )

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

    dchss = (
        weights.get("TS", 1.0) * ts
        + weights.get("LS", 1.0) * ls
        + weights.get("HS", 1.0) * hs
    )
    return round(dchss, 4), {"TS": ts, "LS": ls, "HS": hs, "signed": signed}


def assign_risk_group(dchss_score, risk_thresholds=None):
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
    reference_ranges = _get_key(
        artifact,
        "reference_ranges",
        "reference_range",
        "ranges",
        default={},
    )
    cbc_features = _get_key(
        artifact,
        "cbc_features",
        "cbc_feature_names",
        "features",
        default=[],
    )

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


def predict_dengue(raw_input: dict) -> dict:
    raw_input = dict(raw_input)
    raw_input["gender"] = str(raw_input.get("gender", "other")).lower()

    try:
        rdw_cv_val = float(raw_input.get("rdw_cv", 0) or 0)
        platelet_val = float(raw_input.get("platelet", 0) or 0)
        if platelet_val > 0:
            raw_input["rdw_platelet_ratio"] = (rdw_cv_val / platelet_val) * 1000.0
        else:
            raw_input["rdw_platelet_ratio"] = 0.0
    except (TypeError, ValueError):
        raw_input["rdw_platelet_ratio"] = 0.0

    artifact = load_artifact()

    result = make_prediction(artifact, raw_input)

    probability = float(result["probability"])
    threshold = float(result["threshold"])
    prediction_int = int(result["prediction"])
    prediction_label = "Positive" if prediction_int == 1 else "Negative"

    probability_percent = round(probability * 100, 2)
    threshold_used_percent = round(threshold * 100, 2)

    dchss_score, _dchss_parts = compute_dchss_score(raw_input, artifact)
    risk_group = assign_risk_group(dchss_score)

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