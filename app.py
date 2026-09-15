from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ---- Model files ship inside the deployment itself (same repo as app.py), so no
# Hugging Face Hub download / token is needed. Set MODELS_DIR if you keep the
# .joblib files in a subfolder instead of next to app.py.
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.dirname(os.path.abspath(__file__)))


def load_model(filename):
    local_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Model file not found: {local_path}. Make sure {filename} was "
            f"uploaded alongside app.py (or set MODELS_DIR to where it lives)."
        )
    return joblib.load(local_path)


rf_model = load_model("risk_model.joblib")
scaler = load_model("scaler.joblib")
kmeans_model = load_model("kmeans_model.joblib")
knn_index = load_model("similarity_index.joblib")
feature_cols = load_model("feature_cols.joblib")

# reference_data is loaded from CSV rather than the .joblib pickle: pickled pandas
# DataFrames embed the exact internal binary format of the pandas version that saved
# them, so a datetime64[us] column saved by a newer pandas can fail to unpickle on an
# older one with a cryptic NotImplementedError. CSV has no such version coupling.
reference_data = pd.read_csv(os.path.join(MODELS_DIR, "reference_data.csv"))

# Must exactly match what was used to fit the scaler/kmeans during training (notebook [17]-[19])
CLUSTER_FEATURES = ["PM10", "SO2", "NO2", "CO", "O3"]

# ---- OpenAQ-compatible input contract ----
# No "wd", no "station", no weather fields (TEMP/PRES/DEWP/RAIN/WSPM): OpenAQ doesn't
# reliably provide any of those, so the model was retrained without them. Node only
# needs to hand us pollutant concentrations plus a timestamp.
REQUIRED_FIELDS = ["PM10", "SO2", "NO2", "CO", "O3", "hour", "month"]

# Human-readable label for the cluster id, purely for the API response — the model
# itself just uses the numeric id. Update this if you re-run notebook [19] and the
# cluster order/meaning shifts (check the per-cluster pollutant means it prints).
CLUSTER_LABELS = {
    0: "cluster_0",
    1: "cluster_1",
    2: "cluster_2",
}


def get_time_period(hour):
    if hour < 6:
        return "night"
    elif hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"


def build_feature_vector(data):
    """Turn a raw OpenAQ-shaped reading into a single-row DataFrame matching feature_cols exactly."""

    # Start every column at 0 — handles the one-hot time_period columns automatically:
    # whichever category doesn't apply just stays 0.
    row = {col: 0 for col in feature_cols}

    # --- Direct pollutant fields ---
    # Note: PM2.5 is NOT in feature_cols (it was excluded from training as leakage,
    # since risk_level is derived from it) and is not read here at all — the whole
    # point of this endpoint is to estimate risk without already knowing PM2.5.
    direct_fields = ["PM10", "SO2", "NO2", "CO", "O3"]
    for field in direct_fields:
        if field in row:
            row[field] = float(data[field])

    # --- Pollutant-ratio features (see notebook [14]) ---
    pm10 = float(data["PM10"])
    if "NO2_PM10_ratio" in row:
        row["NO2_PM10_ratio"] = float(data["NO2"]) / (pm10 + 1)
    if "SO2_PM10_ratio" in row:
        row["SO2_PM10_ratio"] = float(data["SO2"]) / (pm10 + 1)
    if "CO_PM10_ratio" in row:
        row["CO_PM10_ratio"] = float(data["CO"]) / (pm10 + 1)
    if "O3_PM10_ratio" in row:
        row["O3_PM10_ratio"] = float(data["O3"]) / (pm10 + 1)

    # --- Cyclical time features ---
    hour = int(data["hour"])
    month = int(data["month"])
    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    row["month_sin"] = np.sin(2 * np.pi * month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month / 12)

    # --- time_period one-hot ---
    period_col = f"time_period_{get_time_period(hour)}"
    if period_col in row:
        row[period_col] = 1
    # if the period matches whatever category was dropped during training (drop_first=True),
    # no column gets set — that's correct, it mirrors how training data represented it.

    # --- pollution_cluster: must run the SAME scaler + SAME kmeans model used in training ---
    pollutant_row = pd.DataFrame([[data[c] for c in CLUSTER_FEATURES]], columns=CLUSTER_FEATURES)
    scaled_pollutants = scaler.transform(pollutant_row)
    cluster_label = int(kmeans_model.predict(scaled_pollutants)[0])

    if "pollution_cluster" in row:
        row["pollution_cluster"] = cluster_label

    # Build the final row, and force column ORDER to exactly match training —
    # sklearn models care about column order, not just column names.
    X = pd.DataFrame([row])[feature_cols]

    if list(X.columns) != list(feature_cols):
        raise ValueError("Generated feature columns do not match training features")

    return X, cluster_label


@app.route("/health", methods=["GET"])
def health():
    """Simple endpoint to confirm the server is up and the model files loaded correctly."""
    return jsonify({"status": "ok", "n_features": len(feature_cols)})


@app.route("/risk", methods=["POST"])
def risk():
    """The ML service's main endpoint. Node sends pollutant readings (already pulled
    from OpenAQ for the user's location) plus hour/month; we return a risk level,
    per-class probabilities, and the pollution-regime cluster."""
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        X, cluster_label = build_feature_vector(data)
    except (ValueError, TypeError, KeyError) as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400

    prediction = rf_model.predict(X)[0]
    probs = rf_model.predict_proba(X)[0]
    classes = rf_model.classes_

    response = {
        "risk": prediction,
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(classes, probs)},
        "pollution_cluster": cluster_label,
        "pollution_profile": CLUSTER_LABELS.get(cluster_label, f"cluster_{cluster_label}"),
    }

    return jsonify(response)


@app.route("/recommend-window", methods=["POST"])
def recommend_window():
    """Given a current pollutant snapshot (assumed roughly stable over the next few
    hours — true intraday forecasting would need a live OpenAQ time series, which is
    a future enhancement), scan every hour of the day and return the windows with the
    lowest predicted high-risk probability. `month` defaults to the current month if
    not supplied."""
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in CLUSTER_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        month = int(data.get("month", pd.Timestamp.utcnow().month))
        n_windows = int(data.get("n_windows", 3))
    except (ValueError, TypeError):
        return jsonify({"error": "month and n_windows must be integers"}), 400

    if n_windows < 1:
        return jsonify({"error": "n_windows must be at least 1"}), 400

    high_idx = None
    hourly_risk = []

    for hour in range(24):
        hour_data = {**data, "hour": hour, "month": month}
        try:
            X, cluster_label = build_feature_vector(hour_data)
        except (ValueError, TypeError, KeyError) as e:
            return jsonify({"error": f"Invalid input values: {str(e)}"}), 400

        probs = rf_model.predict_proba(X)[0]
        classes = list(rf_model.classes_)
        if high_idx is None:
            high_idx = classes.index("high")

        hourly_risk.append({
            "hour": hour,
            "high_risk_probability": round(float(probs[high_idx]), 4),
            "predicted_risk": rf_model.predict(X)[0],
        })

    best_hours = sorted(hourly_risk, key=lambda h: h["high_risk_probability"])[:n_windows]
    best_hours = sorted(best_hours, key=lambda h: h["hour"])  # chronological order for display

    windows = [
        {
            "start": f"{h['hour']:02d}:00",
            "end": f"{(h['hour'] + 1) % 24:02d}:00",
            "risk": h["high_risk_probability"],
            "predicted_risk": h["predicted_risk"],
        }
        for h in best_hours
    ]

    return jsonify({"recommended_windows": windows, "hourly_detail": hourly_risk})


@app.route("/similar", methods=["POST"])
def similar():
    """Given raw pollutant readings, find historical conditions with a similar
    pollutant profile that ended up LOW risk — the recommendation-layer endpoint."""
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in CLUSTER_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        pollutant_row = pd.DataFrame([[float(data[c]) for c in CLUSTER_FEATURES]], columns=CLUSTER_FEATURES)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400

    scaled_query = scaler.transform(pollutant_row)

    # pull more neighbors than we need, since we'll filter down to low-risk ones only
    try:
        n_requested = int(data.get("n_results", 5))
    except (ValueError, TypeError):
        return jsonify({"error": "n_results must be an integer"}), 400

    if n_requested < 1:
        return jsonify({"error": "n_results must be at least 1"}), 400

    n_neighbors = min(200, len(reference_data))
    distances, neighbor_idx = knn_index.kneighbors(
        scaled_query, n_neighbors=n_neighbors
    )

    neighbors = reference_data.iloc[neighbor_idx[0]].copy()
    neighbors["distance"] = distances[0]

    low_risk_matches = neighbors[neighbors["risk_level"] == "low"].head(n_requested)

    if low_risk_matches.empty:
        return jsonify({"matches": [], "note": "No similar low-risk historical conditions found nearby."})

    matches = []
    for _, row in low_risk_matches.iterrows():
        matches.append({
            "datetime": str(row["datetime"]),
            "distance": round(float(row["distance"]), 4),
            **{c: round(float(row[c]), 2) for c in CLUSTER_FEATURES}
        })

    return jsonify({"matches": matches})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)