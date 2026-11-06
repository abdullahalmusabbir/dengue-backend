"""Reusable D-CHSS preprocessing and prediction helpers for the saved model."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DCHSSFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build raw, D-CHSS, signed or directional CBC features."""

    def __init__(
        self, mode="raw", raw_features=None, numeric_features=None,
        cbc_features=None, reference_ranges=None, ts_features=None,
        ls_features=None, hs_features=None, weights=None, max_deviation=1.0
    ):
        self.mode = mode
        self.raw_features = raw_features
        self.numeric_features = numeric_features
        self.cbc_features = cbc_features
        self.reference_ranges = reference_ranges
        self.ts_features = ts_features
        self.ls_features = ls_features
        self.hs_features = hs_features
        self.weights = weights
        self.max_deviation = max_deviation

    def fit(self, X, y=None):
        frame = pd.DataFrame(X, columns=self.raw_features)
        self.medians_ = (
            frame[self.numeric_features]
            .apply(pd.to_numeric, errors="coerce")
            .median()
            .fillna(0.0)
        )
        return self

    def _get_interval(self, feature, gender):
        interval = self.reference_ranges[feature]
        if isinstance(interval, dict):
            if gender == 1:
                return interval["male"]
            if gender == 0:
                return interval["female"]
            return interval["default"]
        return interval

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.raw_features).copy()

        for column in self.numeric_features:
            frame[column] = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(self.medians_[column])
            )

        frame["gender_enc"] = (
            pd.to_numeric(frame["gender_enc"], errors="coerce").fillna(-1)
        )

        signed = pd.DataFrame(index=frame.index)
        for feature in self.cbc_features:
            deviations = []
            for value, gender in zip(
                frame[feature].to_numpy(),
                frame["gender_enc"].to_numpy()
            ):
                sex = int(gender) if gender in (0, 1) else -1
                low, high = self._get_interval(feature, sex)

                if value < low:
                    deviation = (value - low) / low
                elif value > high:
                    deviation = (value - high) / high
                else:
                    deviation = 0.0

                deviations.append(
                    float(np.clip(
                        deviation, -self.max_deviation, self.max_deviation
                    ))
                )

            signed[f"{feature}_dev"] = deviations

        directional = pd.DataFrame(index=frame.index)
        for feature in self.cbc_features:
            deviation = signed[f"{feature}_dev"]
            directional[f"{feature}_low_dev"] = np.maximum(-deviation, 0)
            directional[f"{feature}_high_dev"] = np.maximum(deviation, 0)

        absolute = signed.abs()
        frame["TS"] = absolute[
            [f"{feature}_dev" for feature in self.ts_features]
        ].mean(axis=1)
        frame["LS"] = absolute[
            [f"{feature}_dev" for feature in self.ls_features]
        ].mean(axis=1)
        frame["HS"] = absolute[
            [f"{feature}_dev" for feature in self.hs_features]
        ].mean(axis=1)
        frame["DCHSS"] = sum(
            self.weights[name] * frame[name] for name in self.weights
        )

        if self.mode == "raw":
            result = frame[self.raw_features]
        elif self.mode == "dchss_only":
            result = frame[["TS", "LS", "HS", "DCHSS"]]
        elif self.mode == "combined_signed":
            result = pd.concat([frame[self.raw_features], signed], axis=1)
        elif self.mode == "combined_directional":
            result = pd.concat([frame[self.raw_features], directional], axis=1)
        else:
            raise ValueError(f"Unknown feature mode: {self.mode}")

        self.feature_names_out_ = np.asarray(result.columns, dtype=object)
        return result.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def make_prediction(model_artifact, payload):
    """
    Accept one CBC record as a dictionary and return a prediction.

    Expected keys are the canonical feature names stored in artifact['raw_features'].
    Use 'gender' or 'sex' as Male/Female or 1/0; gender_enc is derived here.
    """
    raw_features = model_artifact["raw_features"]
    aliases = {
        "sex": "gender", "haemoglobin": "hemoglobin", "hb": "hemoglobin",
        "mcv(fl)": "mcv", "mch(pg)": "mch", "mpv(fl)": "mpv",
        "platelet_count": "platelet", "total_platelet_count": "platelet",
        "wbc_count": "wbc", "total_wbc_count": "wbc",
        "hematocrit": "hct", "haematocrit": "hct", "pcv": "hct",
        "rdw": "rdw_cv"
    }

    cleaned = {}
    for key, value in payload.items():
        normalized = str(key).strip().lower()
        cleaned[aliases.get(normalized, normalized)] = value

    gender = str(cleaned.get("gender", "")).strip().lower()
    if gender in {"male", "m", "man", "1"}:
        gender_enc = 1
    elif gender in {"female", "f", "woman", "0"}:
        gender_enc = 0
    else:
        gender_enc = cleaned.get("gender_enc", np.nan)

    row = {}
    for feature in raw_features:
        if feature == "gender_enc":
            row[feature] = gender_enc
        else:
            row[feature] = cleaned.get(feature, np.nan)

    input_frame = pd.DataFrame([row], columns=raw_features)
    probability = float(model_artifact["pipeline"].predict_proba(input_frame)[0, 1])
    threshold = float(model_artifact["decision_threshold"])
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "label": "Dengue positive" if prediction == 1 else "Dengue negative",
        "probability": probability,
        "threshold": threshold
    }