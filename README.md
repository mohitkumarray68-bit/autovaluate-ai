# Used Car Price Prediction & Regression Comparison

An end-to-end Machine Learning project that estimates the selling price of used cars based on specifications (brand, age, odometer mileage, fuel type, transmission, engine displacement, horsepower, and seating capacity).

The project compares **5 regression approaches**, identifies the **key drivers of car depreciation and pricing**, and provides a ready-to-run prediction script.

---

## 1. Quick Start

Ensure Python is available, then run:

### Train and compare all models:
```bash
python train_and_compare.py
```
*Loads data, trains 5 models, displays performance tables, calculates feature importance, and exports `best_car_price_model.joblib`.*

### Run the Interactive Web Platform:
```bash
python app.py
```
*Launches the enterprise automotive valuation platform at **http://127.0.0.1:5000** (and local network **http://0.0.0.0:5000**) with:*
- **Valuation Studio (`/`)**: Live vehicle configurator with OEM spec auto-fill across 31 brands, 10-year depreciation forecast curves, 3-channel market pricing (Trade-In, Private Party, Retail), dynamic auto loan EMI calculator, and printable appraisal certificates.
- **Market Intelligence (`/analytics`)**: Regression comparison leaderboard, permutation feature importance, actual vs predicted plots, and logarithmic scaling methodology.
- **Developer REST API (`/api-docs`)**: Full documentation with cURL examples for programmatic valuation integration.

### Deploy to Production (Render / Cloud / Docker):
```bash
# Production WSGI server
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --threads 4

# Or build & run Docker container
docker build -t autovaluate-ai .
docker run -p 5000:5000 autovaluate-ai
```

### Predict prices via CLI:
```bash
python predict.py
```
*Loads the trained model and predicts selling prices in Indian Rupees (Lakhs).*

---

## 2. Dataset Overview

The dataset contains **15,244 unique used cars** across India with the following features:

| Feature | Type | Example | What It Measures |
| :--- | :--- | :--- | :--- |
| **`brand`** | Categorical | Maruti, Hyundai, BMW | Car manufacturer brand |
| **`vehicle_age`** | Numerical | 5 | Age of the vehicle in years (2026 - registration year) |
| **`km_driven`** | Numerical | 50,000 | Total kilometers traveled on the odometer |
| **`fuel_type`** | Categorical | Petrol, Diesel, CNG | Fuel consumed |
| **`transmission_type`** | Categorical | Manual, Automatic | Gearbox type |
| **`seller_type`** | Categorical | Individual, Dealer | Who is selling the car |
| **`mileage`** | Numerical | 21.2 | Fuel efficiency in km/litre |
| **`engine`** | Numerical | 1197 | Engine displacement capacity in CC |
| **`max_power`** | Numerical | 82.0 | Engine output power in brake horsepower (bhp) |
| **`seats`** | Numerical | 5 | Number of passenger seats |
| **`selling_price`** | **Target** | ₹5,50,000 | Actual market selling price (in ₹ Rupees) |

---

## 3. Data Preprocessing (Explained Simply)

1. **Handling Missing Values & Clean-up**:
   - Duplicate rows were dropped (167 records).
   - Numerical missing values are replaced by the **median** (robust to outliers).
   - Categorical missing values are replaced by the **most frequent category** (mode).

2. **Categorical Encoding (One-Hot Encoding)**:
   - Text features like `brand` and `fuel_type` are converted into numerical binary flags (0 or 1) so algorithms can perform math on them. Rare brands (fewer than 15 occurrences) are grouped into an "infrequent" bucket to prevent overfitting.

3. **Feature Scaling (StandardScaler)**:
   - Numerical columns like `km_driven` (thousands) and `vehicle_age` (single digits) are standardized to zero mean and unit variance so no single large number dominates linear equations.

4. **The Target Log Transformation (`np.log1p`)**:
   - **Why?** Used car prices are heavily right-skewed (most cars cost ₹3-8 Lakhs, while a few luxury cars cost ₹50+ Lakhs).
   - Training directly on raw rupees causes models to obsess over luxury cars and make large relative errors on normal cars.
   - Taking the natural logarithm $\log(1 + \text{price})$ makes the price distribution normal (bell-shaped), stabilizing variance. Predictions are converted back using $\exp(\hat{y}) - 1$.

---

## 4. Regression Models Comparison

We compared 5 classic regression algorithms on a held-out **20% test dataset (3,049 cars)**:

| Model | $R^2$ Score (Higher is Better) | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Average % Error (MAPE) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.9209** | **Rs. 98,195** | **Rs. 2,19,257** | **12.9%** |
| **Gradient Boosting Regressor** | 0.9146 | Rs. 1,04,405 | Rs. 2,27,902 | 13.4% |
| **Decision Tree Regressor** | 0.8540 | Rs. 1,21,563 | Rs. 2,97,951 | 15.4% |
| **Linear Regression** | 0.6231 | Rs. 1,43,595 | Rs. 4,78,710 | 17.4% |
| **Ridge Regression ($L_2$)** | 0.6141 | Rs. 1,43,695 | Rs. 4,84,404 | 17.4% |

![Model Comparison](static/images/model_comparison.png)

### Why Ensemble Tree Models Won:
- **Linear & Ridge Regression** assume price scales in straight lines with features ($y = w_1 x_1 + \dots$). However, car pricing is deeply non-linear (e.g., luxury brands hold value differently than budget brands; high horsepower adds non-linear value).
- **Random Forest** combines 150 decision trees using bagging (bootstrap aggregation). It captures complex non-linear interactions without overfitting, explaining **92.1% of all price variance** with an average error of only **12.9%**.

![Actual vs Predicted](static/images/actual_vs_predicted.png)

---

## 5. Which Features Have the Strongest Effect on Price?

We measured **Permutation Feature Importance** on the held-out test data. This technique shuffles each feature column individually and records the resulting drop in model accuracy ($R^2$). A feature that causes a massive drop when scrambled is critical.

| Rank | Feature | Relative Importance Share (%) | Real-World Meaning |
| :---: | :--- | :---: | :--- |
| **1** | **`max_power`** | **59.0%** | **Engine Horsepower**: By far the #1 driver. High-bhp engines belong to premium cars and sports trims. |
| **2** | **`vehicle_age`** | **29.6%** | **Age Depreciation**: Each passing year steadily erodes the car's resale value. |
| **3** | **`engine`** | **7.1%** | **Engine Displacement (CC)**: Bigger engine capacity strongly correlates with higher price tiers. |
| **4** | **`mileage`** | **1.4%** | **Fuel Economy (kmpl)**: High-mileage economy cars cost less upfront than low-mileage luxury cars. |
| **5** | **`km_driven`** | **1.0%** | **Odometer Wear**: Secondary to age, but higher odometer readings reduce price. |
| **6** | **`brand`** | **0.8%** | **Badge / Manufacturer**: Premium brands command a premium (mostly already captured via horsepower & engine). |
| **7** | **`transmission_type`** | **0.3%** | **Automatic vs Manual**: Automatics cost slightly more. |
| **8** | **`fuel_type`** | **0.3%** | **Diesel / Petrol / CNG**: Diesel historically retained slightly more value. |
| **9** | **`seller_type`** | **0.3%** | **Dealer vs Individual**: Certified dealers charge small margins. |
| **10** | **`seats`** | **0.2%** | **Seat Capacity**: Minimal direct impact across passenger cars. |

![Feature Importance](static/images/feature_importance.png)

> **Key Takeaway**: Over **88%** of a used car's valuation is determined by just two factors:
> 1. **Engine Performance Class** (`max_power` + `engine`) &mdash; determines the price ceiling.
> 2. **Age & Usage** (`vehicle_age` + `km_driven`) &mdash; determines the depreciation curve.

---

## 6. How to Explain This Project in an Interview or Presentation (Cheat Sheet)

If someone asks you to walk through the project, use this simple 4-step structure:

1. **The Problem**:
   > *"Used car pricing is subjective and tricky because buyers and sellers don't know the fair market value. I built a machine learning pipeline that accurately predicts selling price based on 10 vehicle attributes."*

2. **The Data & Preprocessing**:
   > *"I trained on 15,244 real used cars. The most important preprocessing step was applying a log transformation `log(1 + price)` to the target. Because car prices span from ₹50,000 to over ₹50 Lakhs, log transformation prevents luxury cars from biasing the entire model."*

3. **Model Selection**:
   > *"I compared 5 regression models: Linear Regression, Ridge, Decision Tree, Random Forest, and Gradient Boosting. Random Forest performed best with an $R^2$ of **0.92** and an average error of only **12.9%**, because it naturally handles non-linear relationships between power, age, and brand."*

4. **The Core Insight**:
   > *"Using permutation importance, I discovered that **Engine Max Power (59%)** and **Vehicle Age (30%)** together account for nearly 90% of price variance. Features like fuel type and transmission have only minor marginal effects."*

---

## 7. Project File Structure

```
ML Model/
├── data/
│   └── used_cars.csv            # Clean local dataset (15,244 cars)
├── static/
│   ├── css/styles.css           # Bespoke automotive design system (dark/light themes)
│   ├── js/main.js               # Reactive client controller, Chart.js, EMI, & modal
│   ├── images/                  # Benchmark plots & app icon assets
│   └── data/specs_catalog.json  # Precomputed OEM specifications across 31 brands
├── templates/
│   ├── base.html                # App layout, header, footer & appraisal certificate
│   ├── index.html               # Valuation Studio & live configurator
│   ├── analytics.html           # Model transparency & benchmark leaderboard
│   └── api_docs.html            # Developer REST API integration guide
├── best_car_price_model.joblib  # Serialized Random Forest pipeline (150 trees)
├── brand_model_catalog.joblib   # Brand-to-model index
├── app.py                       # Production Flask web application
├── train_and_compare.py         # Full training & benchmarking script
├── train_vast_model.py          # High-performance cloud training script
├── predict.py                   # CLI inference tool
├── requirements.txt             # Python production dependencies
├── Procfile                     # Web process configuration for cloud deployment
├── render.yaml                  # 1-click cloud deployment spec
├── Dockerfile                   # Containerized deployment spec
├── push_to_github.bat           # 1-click script to push project to GitHub
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation & viva guide
```
