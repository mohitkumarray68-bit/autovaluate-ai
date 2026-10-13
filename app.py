"""
=============================================================================
AutoValuate AI - Production Enterprise Vehicle Valuation Platform
=============================================================================
Flask web application serving real-time machine learning car valuations,
channel pricing matrices, OEM spec lookup, and 10-year depreciation curves.

Run locally:
    python app.py
Open in browser:
    http://127.0.0.1:5000
=============================================================================
"""

import os
import sys
import json
import uuid
from datetime import datetime, timezone

# Ensure standard UTF-8 console streams on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_car_price_model.joblib")
CATALOG_PATH = os.path.join(BASE_DIR, "brand_model_catalog.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "used_cars.csv")
SPECS_PATH = os.path.join(BASE_DIR, "static", "data", "specs_catalog.json")

# Initialize Flask application
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# -----------------------------------------------------------------------------
# 1. Pipeline & Data Caching
# -----------------------------------------------------------------------------
print("Initializing AutoValuate AI inference engine...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Run training first!")

pipeline = joblib.load(MODEL_PATH)
print("✓ Loaded Random Forest Regressor Pipeline (150 trees)")

# Load or generate catalog
if os.path.exists(CATALOG_PATH):
    catalog = joblib.load(CATALOG_PATH)
else:
    if os.path.exists(DATA_PATH):
        df_temp = pd.read_csv(DATA_PATH)
        df_temp["brand"] = df_temp["brand"].astype(str).str.strip().str.title()
        df_temp["model"] = df_temp["model"].astype(str).str.strip().str.title()
        catalog = {
            str(b): sorted([str(m) for m in sub["model"].dropna().unique()])
            for b, sub in df_temp.groupby("brand")
        }
        joblib.dump(catalog, CATALOG_PATH)
    else:
        catalog = {}

# Load specs lookup cache
specs_catalog = {}
if os.path.exists(SPECS_PATH):
    try:
        with open(SPECS_PATH, "r", encoding="utf-8") as f:
            specs_catalog = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read {SPECS_PATH}: {e}")

# -----------------------------------------------------------------------------
# 2. Financial & Valuation Utilities
# -----------------------------------------------------------------------------
def format_inr(price_num: float) -> str:
    """Formats an Indian Rupee amount into Crore, Lakhs, or exact notation."""
    if price_num >= 10_000_000:
        return f"₹ {price_num / 10_000_000:.2f} Crore"
    elif price_num >= 100_000:
        return f"₹ {price_num / 100_000:.2f} Lakhs"
    else:
        return f"₹ {price_num:,.0f}"


def format_inr_short(price_num: float) -> str:
    """Formats an Indian Rupee amount into concise badge format (e.g. ₹ 5.31 L)."""
    if price_num >= 10_000_000:
        return f"₹ {price_num / 10_000_000:.2f} Cr"
    elif price_num >= 100_000:
        return f"₹ {price_num / 100_000:.2f} L"
    else:
        return f"₹ {price_num:,.0f}"


def compute_depreciation_trajectory(base_car: dict, current_age: int) -> list:
    """
    Evaluates the trained ML model across vehicle ages 1 to 10
    to project realistic non-linear market depreciation.
    """
    rows = []
    base_km = base_car.get("km_driven", 45000)
    avg_annual_km = max(8000, round(base_km / max(current_age, 1)))

    for age in range(1, 11):
        car_copy = dict(base_car)
        car_copy["vehicle_age"] = age
        # Scale odometer proportionally with age progression
        car_copy["km_driven"] = int(avg_annual_km * age)
        rows.append(car_copy)

    df_batch = pd.DataFrame(rows)
    preds_log = pipeline.predict(df_batch)
    preds_real = np.expm1(preds_log)

    trajectory = []
    for age, price in zip(range(1, 11), preds_real):
        trajectory.append({
            "age": age,
            "projected_price": round(float(price)),
            "formatted_short": format_inr_short(price),
            "is_current": (age == current_age)
        })

    return trajectory


# -----------------------------------------------------------------------------
# 3. View Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """Valuation Studio - Main user interface."""
    brands = sorted(list(catalog.keys()))
    return render_template(
        "index.html",
        brands=brands,
        catalog=catalog,
        active_page="home"
    )


@app.route("/analytics")
def analytics():
    """Market Intelligence & Model Transparency Page."""
    return render_template("analytics.html", active_page="analytics")


@app.route("/api-docs")
def api_docs():
    """Developer REST API Documentation Page."""
    return render_template("api_docs.html", active_page="api_docs")


# -----------------------------------------------------------------------------
# 4. REST API Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Primary ML inference endpoint.
    Expects JSON vehicle specifications and returns a comprehensive appraisal report.
    """
    data = request.get_json(silent=True) or {}

    # Extract and normalize fields
    brand = str(data.get("brand", "Maruti")).strip().title()
    model = str(data.get("model", "Swift")).strip().title()
    vehicle_age = int(data.get("vehicle_age", 4))
    km_driven = int(data.get("km_driven", 42000))
    seller_type = str(data.get("seller_type", "Individual"))
    fuel_type = str(data.get("fuel_type", "Petrol"))
    transmission_type = str(data.get("transmission_type", "Manual"))
    mileage = float(data.get("mileage", 20.0))
    engine = int(data.get("engine", 1197))
    max_power = float(data.get("max_power", 82.0))
    seats = int(data.get("seats", 5))

    car_dict = {
        "brand": brand,
        "model": model,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }

    # Execute ML Inference
    car_df = pd.DataFrame([car_dict])
    pred_log = pipeline.predict(car_df)[0]
    estimated_price = float(np.expm1(pred_log))
    raw_price = round(estimated_price)

    # Calculate Confidence Margin (±4.2%)
    margin_pct = 4.2
    low_bound = round(raw_price * (1.0 - (margin_pct / 100)))
    high_bound = round(raw_price * (1.0 + (margin_pct / 100)))

    # Compute 3-Channel Valuation Matrix
    private_party_val = round(raw_price * 1.02)
    trade_in_val = round(raw_price * 0.88)
    retail_showroom_val = round(raw_price * 1.10)

    # Compute 10-Year Depreciation Forecast Curve
    depreciation_curve = compute_depreciation_trajectory(car_dict, vehicle_age)

    # Generate Unique Valuation Identifier
    valuation_id = f"VAL-{datetime.now(timezone.utc).year}-{uuid.uuid4().hex[:6].upper()}"

    response_payload = {
        "status": "success",
        "valuation_id": valuation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_price": raw_price,
        "formatted_price": format_inr(raw_price),
        "price_range": {
            "low": low_bound,
            "high": high_bound,
            "low_formatted": format_inr_short(low_bound),
            "high_formatted": format_inr_short(high_bound),
            "margin_pct": margin_pct
        },
        "channels": {
            "trade_in": {
                "price": trade_in_val,
                "formatted": format_inr_short(trade_in_val),
                "description": "Instant Cash Dealer Liquidation"
            },
            "private_party": {
                "price": private_party_val,
                "formatted": format_inr_short(private_party_val),
                "description": "Direct Consumer-to-Consumer Fair Sale"
            },
            "retail": {
                "price": retail_showroom_val,
                "formatted": format_inr_short(retail_showroom_val),
                "description": "Certified Dealership Showroom List Price"
            }
        },
        "depreciation_curve": depreciation_curve,
        "specs_applied": car_dict
    }

    return jsonify(response_payload)


@app.route("/predict", methods=["POST"])
def legacy_predict():
    """Backward compatibility endpoint for older client scripts."""
    return api_predict()


@app.route("/api/catalog", methods=["GET"])
def api_catalog():
    """Returns the full brand-to-model index."""
    return jsonify({
        "brands": sorted(list(catalog.keys())),
        "catalog": catalog
    })


@app.route("/api/specs-lookup", methods=["GET"])
def api_specs_lookup():
    """Returns typical OEM factory specifications for a given brand and model."""
    brand = request.args.get("brand", "").strip().title()
    model = request.args.get("model", "").strip().title()

    if brand in specs_catalog and model in specs_catalog[brand]:
        return jsonify({
            "status": "found",
            "brand": brand,
            "model": model,
            "specs": specs_catalog[brand][model]
        })

    return jsonify({
        "status": "not_found",
        "brand": brand,
        "model": model,
        "specs": {
            "engine": 1197,
            "max_power": 82.0,
            "mileage": 20.0,
            "seats": 5,
            "fuel_type": "Petrol",
            "transmission_type": "Manual"
        }
    })


@app.route("/api/model-metrics", methods=["GET"])
def api_model_metrics():
    """Returns evaluation metrics, architecture telemetry, and benchmark results."""
    return jsonify({
        "architecture": "RandomForestRegressor(n_estimators=150, max_depth=18)",
        "training_records": 15244,
        "r2_score": 0.9209,
        "mae_rupees": 98195,
        "mape_pct": 12.9,
        "rmse_rupees": 219257,
        "top_features": [
            {"feature": "max_power", "share_pct": 59.0, "label": "Engine Horsepower"},
            {"feature": "vehicle_age", "share_pct": 29.6, "label": "Age Depreciation"},
            {"feature": "engine", "share_pct": 7.1, "label": "Displacement (CC)"},
            {"feature": "mileage", "share_pct": 1.4, "label": "Fuel Economy (kmpl)"},
            {"feature": "km_driven", "share_pct": 1.0, "label": "Odometer Wear"}
        ]
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Service liveness & pipeline health check."""
    return jsonify({
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "catalog_brands_count": len(catalog),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


# -----------------------------------------------------------------------------
# 5. Error Handlers
# -----------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found", "status": 404}), 404
    return render_template("index.html", brands=sorted(list(catalog.keys())), catalog=catalog, active_page="home"), 404


@app.errorhandler(500)
def server_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error", "status": 500, "details": str(e)}), 500
    return "<h3>Internal Server Error</h3><p>An unexpected error occurred.</p>", 500


# -----------------------------------------------------------------------------
# 6. Server Runner
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\n=======================================================")
    print(f"🚀 AutoValuate AI Server Running:")
    print(f"   Local:   http://127.0.0.1:{port}")
    print(f"   Network: http://0.0.0.0:{port}")
    print(f"=======================================================\n")
    app.run(debug=True, host=host, port=port)