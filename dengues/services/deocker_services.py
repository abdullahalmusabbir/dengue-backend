"""
deocker_services.py
--------------------
Pure-Python (ZERO external ML library) prediction service for the
D-CHSS Dengue Detection model.

Model artifact location:
    backend/dengue/dengues/ml_model/dchss_model_pure_python.pkl

Uses only: os, math, pickle, threading (builtin) + Django settings.
NO pandas / numpy / scikit-learn / scipy / xgboost / shap anywhere.
"""

import os
import math
import pickle
import threading
from django.conf import settings


class ModelNotLoadedError(Exception):
    """Model artifact (pkl) খুঁজে না পেলে / load করতে সমস্যা হলে এই exception raise হবে"""
    pass


# -----------------------------------------------------------------
# ⭐ Model artifact path - এখানেই .pkl ফাইলটা রাখতে হবে
# -----------------------------------------------------------------
MODEL_PATH = os.path.join(
    settings.BASE_DIR, 'dengues', 'ml_model', 'dchss_model_pure_python.pkl'
)

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
                        f"notebook থেকে generate করা 'dchss_model_pure_python.pkl' "
                        f"file টা ঐ path এ কপি করতে হবে।"
                    )
                with open(MODEL_PATH, 'rb') as f:
                    _artifact = pickle.load(f)
    return _artifact


# -----------------------------------------------------------------
# Pure-python math helpers
# -----------------------------------------------------------------
def sigmoid(z):
    """Numerically stable sigmoid - overflow avoid করার জন্য branch করা হয়েছে"""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def get_range(param, gender, reference_ranges):
    """Gender-specific clinical reference range বের করা (না থাকলে default)"""
    rng = reference_ranges.get(param)
    if rng is None:
        return None
    if isinstance(rng, dict):
        if gender == 'male':
            return rng.get('male', rng.get('default'))
        else:
            return rng.get('female', rng.get('default'))
    return rng  # tuple (low, high)


def _mean_std(stats):
    """
    Scaler-stats নਿਰ্বল unterschiedlichen formato এ থাকতে পারে:
      - dict: {'mean': ..., 'std': ...}
      - tuple: (mean, std)
    এই ফাংশনটি দুikt scenarioই handle করে এবং (mean, std) রিটার্ন করে।
    """
    if isinstance(stats, dict):
        return stats['mean'], stats['std']
    # tuple (mean, std)
    return stats[0], stats[1]


def compute_deviation(value, low, high, max_dev):
    """
    Normal range এর মধ্যমান (mid) থেকে signed deviation বের করা,
    half-range দিয়ে normalize করা এবং max_dev এ cap করা।
    """
    if value is None or low is None or high is None:
        return 0.0
    mid = (low + high) / 2.0
    half_range = (high - low) / 2.0
    if half_range == 0:
        return 0.0
    dev = (value - mid) / half_range
    if dev > max_dev:
        dev = max_dev
    elif dev < -max_dev:
        dev = -max_dev
    return dev


def compute_state(value, low, high):
    """Explanation এর জন্য human-readable state: low / normal / high"""
    if value is None:
        return 'unknown'
    if value < low:
        return 'low'
    if value > high:
        return 'high'
    return 'normal'


def calibrate_prob(raw_prob, cal_x, cal_y):
    """
    PAVA (Isotonic Regression) থেকে পাওয়া knot points (cal_x, cal_y) দিয়ে
    linear interpolation করে raw probability কে calibrated probability তে রূপান্তর করা।
    """
    if raw_prob <= cal_x[0]:
        return cal_y[0]
    if raw_prob >= cal_x[-1]:
        return cal_y[-1]
    for i in range(1, len(cal_x)):
        if raw_prob <= cal_x[i]:
            x0, x1 = cal_x[i - 1], cal_x[i]
            y0, y1 = cal_y[i - 1], cal_y[i]
            if x1 == x0:
                return y1
            t = (raw_prob - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return cal_y[-1]


# -----------------------------------------------------------------
# Feature engineering pipeline (notebook cells 8,9,10,12,13 এর mirror)
# -----------------------------------------------------------------
def build_feature_row(raw_input, artifact):
    reference_ranges = artifact['reference_ranges']
    max_dev = artifact['max_dev']
    cbc_features = artifact['cbc_features']
    thrombo_params = artifact['thrombo_params']
    leuko_params = artifact['leuko_params']
    hemo_params = artifact['hemo_params']
    comp_scaler = artifact['component_scaler']
    dchss_weights = artifact['dchss_weights']  # (w_ts, w_ls, w_hs)

    gender = str(raw_input.get('gender', 'other')).lower()

    row = {}
    for feat in cbc_features:
        row[feat] = float(raw_input[feat])

    # ---- deviation features ----
    dev_values = {}
    for feat in cbc_features:
        rng = get_range(feat, gender, reference_ranges)
        if rng is None:
            dev_values[feat] = 0.0
            continue
        low, high = rng
        dev_values[feat] = compute_deviation(row[feat], low, high, max_dev)
        row[f'{feat}_dev'] = dev_values[feat]

    # ---- ratio features ----
    row['nlr'] = row['neutrophils'] / (row['lymphocytes'] + EPS)
    row['plr'] = row['platelet'] / (row['lymphocytes'] + EPS)
    row['lmr'] = row['lymphocytes'] / (row['monocytes'] + EPS)
    row['pdw_mpv_ratio'] = row['pdw'] / (row['mpv'] + EPS)
    row['rdw_platelet_ratio'] = row['rdw_cv'] / (row['platelet'] + EPS)

    # ---- D-CHSS component raw scores ----
    ts_raw = sum(abs(dev_values[p]) for p in thrombo_params) / len(thrombo_params)
    ls_raw = sum(abs(dev_values[p]) for p in leuko_params) / len(leuko_params)
    hs_raw = sum(abs(dev_values[p]) for p in hemo_params) / len(hemo_params)

    def std_comp(name, value):
        stats = comp_scaler[name]
        mean, std = _mean_std(stats)
        std = std if std != 0 else 1.0
        return (value - mean) / std

    ts_std = std_comp('TS_raw', ts_raw)
    ls_std = std_comp('LS_raw', ls_raw)
    hs_std = std_comp('HS_raw', hs_raw)

    w_ts, w_ls, w_hs = dchss_weights
    dchss = w_ts * ts_std + w_ls * ls_std + w_hs * hs_std

    row['TS_std'] = ts_std
    row['LS_std'] = ls_std
    row['HS_std'] = hs_std
    row['DCHSS'] = dchss

    return row, dev_values


def standardize_row(row, artifact):
    """Full feature vector কে train-time fitted mean/std দিয়ে standardize করা"""
    full_features = artifact['full_features']
    full_scaler = artifact['full_scaler']
    z = []
    for feat in full_features:
        stats = full_scaler[feat]
        mean, std = _mean_std(stats)
        std = std if std != 0 else 1.0
        val = row.get(feat, 0.0)
        z.append((val - mean) / std)
    return z


def compute_logit(z_vector, artifact):
    weights = artifact['model_weights']
    bias = artifact['model_bias']
    logit = bias
    for w, zv in zip(weights, z_vector):
        logit += w * zv
    return logit


def explain_contributions(z_vector, artifact, top_n=8):
    """
    SHAP এর বদলে exact linear decomposition: contribution[i] = weight[i] * z[i]
    Logistic regression হওয়ায় এটা পুরোপুরি accurate/exact explanation (approximation না)।
    """
    weights = artifact['model_weights']
    full_features = artifact['full_features']
    contributions = []
    for feat, w, zv in zip(full_features, weights, z_vector):
        contributions.append((feat, w * zv))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:top_n]

    explanations = []
    for feat, contrib in top:
        effect = 'dengue risk বাড়াচ্ছে' if contrib > 0 else 'dengue risk কমাচ্ছে'
        explanations.append({
            'feature': feat,
            'contribution': round(contrib, 4),
            'effect': effect,
        })
    return explanations


def assign_risk_group(dchss_score, risk_thresholds):
    low_thresh, high_thresh = risk_thresholds
    if dchss_score < low_thresh:
        return 'Low'
    elif dchss_score <= high_thresh:
        return 'Moderate'
    else:
        return 'High'


# -----------------------------------------------------------------
# ⭐ Public API - Django view থেকে এই ফাংশনটাই call হবে
# -----------------------------------------------------------------
def predict_dengue(raw_input: dict) -> dict:
    """
    raw_input: {
        'gender': 'male'|'female'|'other', 'age': float,
        'hemoglobin':.., 'neutrophils':.., 'lymphocytes':.., 'monocytes':..,
        'eosinophils':.., 'rbc':.., 'hct':.., 'mcv':.., 'mch':.., 'mchc':..,
        'rdw_cv':.., 'platelet':.., 'mpv':.., 'pdw':.., 'pct':.., 'wbc':..
    }
    """
    # Frontend কখনো কখনো 'Male'/'Female' (capital) পাঠায়; serializer Lowercase চায়,
    # তাই নিরাপদের জন্য এখানে নরমালাইজ করছি।
    raw_input = dict(raw_input)
    raw_input['gender'] = str(raw_input.get('gender', 'other')).lower()

    artifact = load_artifact()

    row, _dev_values = build_feature_row(raw_input, artifact)
    z_vector = standardize_row(row, artifact)

    raw_logit = compute_logit(z_vector, artifact)
    raw_prob = sigmoid(raw_logit)

    calibrated_prob = calibrate_prob(raw_prob, artifact['cal_x'], artifact['cal_y'])
    probability_percent = round(calibrated_prob * 100, 2)

    threshold = artifact['threshold']
    prediction = 'Positive' if calibrated_prob >= threshold else 'Negative'

    risk_group = assign_risk_group(row['DCHSS'], artifact['risk_thresholds'])

    top_feature_explanations = explain_contributions(z_vector, artifact, top_n=8)

    return {
        'probability_percent': probability_percent,
        'prediction': prediction,
        'risk_group': risk_group,
        'dchss_score': round(row['DCHSS'], 4),
        'threshold_used_percent': round(threshold * 100, 2),
        'top_feature_explanations': top_feature_explanations,
        'disclaimer': 'Research demonstration only - NOT a medical diagnosis tool.',
    }