"""
=============================================================================
Vast Multi-Brand & Model Used Car Price Trainer
=============================================================================
Includes models across all major brands:
- Tata: Safari, Harrier, Nexon, Tiago, Tigor, Altroz, Punch, Nano, Hexa, Indica
- Maruti: Swift, Baleno, Brezza, Dzire, Wagon R, Alto, Ertiga, Ciaz
- Hyundai: Creta, Venue, i20, Grand i10, Verna, Tucson, Santro
- Mahindra: Thar, XUV700, Scorpio, XUV300, Bolero
- Toyota: Fortuner, Innova Crysta, Glanza, Urban Cruiser, Corolla
- Honda, Kia, Volkswagen, Skoda, BMW, Mercedes-Benz, Audi, etc.
=============================================================================
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import glob
import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_car_price_model.joblib")
CATALOG_PATH = os.path.join(BASE_DIR, "brand_model_catalog.joblib")
LOCAL_DATA = os.path.join(BASE_DIR, "data", "used_cars.csv")

def acquire_data():
    print("Loading comprehensive used car dataset...")
    df = None
    if os.path.exists(LOCAL_DATA):
        print(f"Loading from local dataset: {LOCAL_DATA}")
        df = pd.read_csv(LOCAL_DATA)
    else:
        try:
            print("Downloading dataset via kagglehub...")
            path = kagglehub.dataset_download("manishkr1754/cardekho-used-car-data")
            csv_file = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)[0]
            df = pd.read_csv(csv_file)
            os.makedirs(os.path.dirname(LOCAL_DATA), exist_ok=True)
            df.to_csv(LOCAL_DATA, index=False)
        except Exception as e:
            raise FileNotFoundError(f"Could not load or download dataset: {e}")

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed") or c == "car_name"]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.drop_duplicates().dropna(subset=["selling_price"]).reset_index(drop=True)

    df["brand"] = df["brand"].astype(str).str.strip().str.title()
    df["model"] = df["model"].astype(str).str.strip().str.title()

    print(f"Loaded {len(df):,} verified records across {df['brand'].nunique()} brands and {df['model'].nunique()} car models.")
    return df

def train():
    df = acquire_data()

    num_cols = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
    cat_cols = ["brand", "model", "fuel_type", "transmission_type", "seller_type"]

    X = df[num_cols + cat_cols]
    y = np.log1p(df["selling_price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False))
            ]), cat_cols),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=150, max_depth=18, random_state=RANDOM_STATE, n_jobs=-1))
    ])

    print("Training Random Forest model on full brand & model catalog...")
    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_test)
    y_true = np.expm1(y_test)
    pred_price = np.expm1(pred_log)

    r2 = r2_score(y_true, pred_price)
    mae = mean_absolute_error(y_true, pred_price)
    mape = np.mean(np.abs((y_true - pred_price) / y_true)) * 100

    print("\n" + "=" * 55)
    print("MODEL EVALUATION RESULTS:")
    print(f"  R2 Score (Test)       : {r2:.4f}")
    print(f"  Mean Absolute Error   : Rs. {mae:,.0f}")
    print(f"  Avg % Error (MAPE)    : {mape:.1f}%")
    print("=" * 55)

    joblib.dump(pipeline, MODEL_PATH)

    catalog = {}
    for brand, group in df.groupby("brand"):
        catalog[brand] = sorted(group["model"].unique().tolist())
    joblib.dump(catalog, CATALOG_PATH)

    print(f"Model saved -> {MODEL_PATH}")
    print(f"Brand-model catalog saved -> {CATALOG_PATH}")

if __name__ == "__main__":
    train()