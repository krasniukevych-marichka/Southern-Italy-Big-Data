# Southern Italy Big Data — Airbnb Market Analysis & Price Prediction

Presentation: https://canva.link/9lol1uvpl439ofq

A Databricks-based end-to-end analytics project using **Inside Airbnb** listing snapshots to compare **Rome (A)** and **Sicily (B)**, build a reusable **Bronze → Silver → Gold** data pipeline, train an explainable nightly-price prediction model, and investigate its generalisation to **Naples (C)**.

> **Geographical scope:** Rome is a city in central Italy; Sicily is an island/region in southern Italy. Their listing totals and neighbourhood counts represent different geographic scales and should not be interpreted as like-for-like market density measures.

## Data and snapshot dates

| Dataset | Role | Inside Airbnb source path | Snapshot date | Silver listings |
|---|---|---|---|---:|
| Rome | City A — training and evaluation | `italy/lazio/rome` | **2026-09-17** | **37,451** |
| Sicily | Area B — training and evaluation | `italy/sicilia/sicily` | **2026-09-25** | **56,783** |
| Naples | City C — out-of-domain transfer test | See `06_score_city.ipynb` / run parameters | **2026-09-19** | Not included in the Rome/Sicily dashboard totals |

**Source:** [Inside Airbnb](https://insideairbnb.com/get-the-data/). Each row in the underlying listing snapshot represents a listing, identified by `id`. Listed nightly prices are analysed in **EUR**. These are advertised prices from a point-in-time snapshot, not observed booking transactions. The two core datasets contain **94,234 listings** in total after the team's Silver processing.

## Objectives

1. Examine the distribution, missingness, extreme values and explanatory features of Airbnb listings in markets A and B.
2. Build a parameterised Databricks pipeline instead of duplicating ETL notebooks for each city.
3. Predict nightly prices without target leakage and compare the model against a room-type/capacity median baseline.
4. Test the unchanged pipeline/model on city C, explain domain shift and document any subsequent changes.
5. Deliver a three-page Databricks AI/BI dashboard and a tested Genie analytics agent.

## Medallion architecture

```text
Inside Airbnb CSV snapshots
          |
          v
BRONZE — workspace.bronze.listings
  Raw source columns, city/snapshot metadata, Delta storage
          |
          v
SILVER — workspace.silver.listings
  Explicit data types, cleaning, duplicate checks, missingness
  and outlier classification
          |
          +-----------------------------+
          |                             |
          v                             v
GOLD — workspace.gold.listing_features  GOLD — dashboard aggregates
  Model-ready listing attributes         city_overview
  and eligibility flags                  room_type_metrics
          |                              neighbourhood_metrics
          v                             |
  Train/test split + baseline            v
  + price prediction model        Databricks AI/BI Dashboard
          |
          v
  workspace.gold.model_metrics
  workspace.gold.price_predictions
  workspace.gold.model_feature_importance
          |
          v
   Model evaluation + Genie Agent
```

**Pipeline:** `job_city_pipeline.yml` defines notebook tasks and a `city` parameter. The reusable stages are ingestion (`02_bronze.ipynb`), cleaning (`03_silver.ipynb`), feature engineering (`04_gold_features.ipynb`) and scoring (`06_score_city.ipynb`). The training step (`05_train_model.ipynb`) builds and stores a versioned model based on A+B data. Pipeline deployment paths and parameters need to be adjusted to the target Databricks workspace.

## Repository contents

| File | Purpose |
|---|---|
| `01_eda.ipynb` | EDA, data profiling, missingness, outliers, leakage review and cross-market interpretation |
| `02_bronze.ipynb` | Ingest Inside Airbnb snapshots into Bronze Delta tables |
| `03_silver.ipynb` | Clean, type and classify listing records |
| `04_gold_features.ipynb` | Build model-ready Gold features by city |
| `05_train_model.ipynb` | Train the baseline and versioned ML models on A+B, and write predictions and metrics |
| `06_score_city.ipynb` | Score the transfer city using the saved model |
| `07_gold_dashboard.ipynb` | Create city, room-type and neighbourhood aggregate tables for the AI/BI dashboard |
| `model_utils.py` | Shared model and feature-engineering utilities |
| `job_city_pipeline.yml` | Parameterised Databricks job definition |
| `MODEL.md` | Detailed model design, A/B metrics, Naples diagnosis and v1/v2 comparison |

## Data quality and leakage controls

- Preserve snapshot metadata and the listing identifier; check for duplicate or malformed identifiers.
- Parse types explicitly, including large integer IDs and advertised price fields; retain missing-price records for market counts but exclude them from supervised price modelling.
- Classify missing/normal/tail/suspicious-price records rather than blindly removing every high-priced villa.
- Exclude columns computed from the target price, such as `estimated_revenue_l365d`, `price_quote_*` and `price_per_guest`.
- Do not use post-price outcomes (such as occupancy/reviews/availability) as predictors in the documented price model.
- Fit training imputations and category handling on the training data, avoiding information leakage from held-out observations.

## Price model and evaluation

- **Holdout:** 20% of each training market, grouped by `host_id` so listings from the same host do not cross the split.
- **Baseline:** median nightly price by `room_type` and `accommodates`, with documented fallback rules.
- **Model:** scikit-learn `HistGradientBoostingRegressor`, trained on a logarithmic target and interpretable listing/amenity/density features.
- **Primary metric:** MAE in EUR; lower values are better.

### Verified v1 held-out results (Rome and Sicily)

| Evaluation group | Test listings | Baseline MAE | v1 model MAE | MAE reduction |
|---|---:|---:|---:|---:|
| Rome | 6,777 | €84.02 | €63.99 | 23.8% |
| Sicily | 10,129 | €73.30 | €59.38 | 19.0% |
| **A+B combined** | **16,906** | **€77.59** | **€61.23** | **21.1%** |

These are **held-out test results**, not training errors. The model improves on the baseline in both original markets. Full model methodology and the v2 design are documented in [`MODEL.md`](MODEL.md).

### City C — transfer validation

The initial v1 model, trained on Rome and Sicily, performed poorly when transferred to Naples (City C), achieving an MAE of €74.12 compared with the local baseline of €58.39.

After implementing model improvements (v2), the Naples MAE decreased to €54.45, representing a **26.54% reduction in prediction error** compared with v1 and a **6.75% improvement over the baseline**.

Both model versions were evaluated on the same held-out test set of **2,157 Naples listings** (snapshot: September 19, 2026).

The improvements addressed differences in pricing patterns and listing density between the training markets and Naples, demonstrating better cross-city generalisation.

## Databricks AI/BI Dashboard

**Dashboard:** `Southern Italy | Airbnb Market & Price Analytics`

**Dashboard URL:** https://dbc-58c2959d-d992.cloud.databricks.com/dashboardsv3/01f1de27a9e211f29825b8cffbf4333c/published?o=7474650174797440

| Page | Main insight | Main visuals |
|---|---|---|
| **1 — City overview** | Rome's median listed price is about 64% above Sicily's | Market KPIs, median nightly price, annual availability, room-type mix and top-priced areas |
| **2 — A/B comparison** | Rome's price premium persists within major room types | Side-by-side median prices, premium percentage and listing counts by room type |
| **3 — Model performance** | v2 reduces Naples prediction MAE by 26.5% compared with v1 | Rome/Sicily model evaluation, actual vs predicted prices, Naples baseline vs v1 vs v2, MAE by room type, and extreme prediction errors |

**Illustrative market findings:**

| Metric | Rome | Sicily |
|---|---:|---:|
| Listings | 37,451 | 56,783 |
| Unique hosts within each market | 19,779 | 29,025 |
| Median nightly price | €188.50 | €115.00 |
| Average available days in the next 365 | 207.4 | 227.0 |
| Mean review rating (rated listings only) | 4.77 | 4.75 |
| Entire home/apt listing share | 74.46% | 79.51% |

**Interpretation limits:** Availability is *not* occupancy; listing counts are *not* booking volume. Rome is a city and Sicily is a much larger geographic region. Summed city-level unique-host counts should not be presented as unique hosts across both places unless cross-city duplicates are removed.

## Genie analytics agent

**Agent:** `Southern Italy — Airbnb Market & Price Insights`

**Genie URL:** https://dbc-58c2959d-d992.cloud.databricks.com/genie/rooms/01f1de3b9d1c1c91a8bdc05d2a76931d?o=7474650174797440

The agent is configured on these seven Gold tables:

- `workspace.gold.city_overview`
- `workspace.gold.room_type_metrics`
- `workspace.gold.neighbourhood_metrics`
- `workspace.gold.model_metrics`
- `workspace.gold.price_predictions`
- `workspace.gold.model_metrics_all`
- `workspace.gold.price_predictions_naples`

Instructions define EUR units, city scope, median vs mean, model version and evaluation filters, prediction error interpretation, and limitations of geographic comparisons. Additional instructions support Naples transfer evaluation, distinguishing v1 (`first_run`) from v2 (`after_fix`) and requiring dynamic retrieval of MAE values from Gold tables instead of hardcoded results.

**Benchmarks:** **4/4 Good** across all four evaluation questions. Checked questions:

1. What is the median nightly Airbnb price in Rome and Sicily?
2. What are the baseline and v1 model MAEs for Rome, Sicily and both cities combined?
3. Which room type has the largest share of Airbnb listings in Sicily?
4. How did the Airbnb price prediction model perform on Naples before and after the changes? Compare baseline, v1, and v2 MAE and calculate the percentage improvement.

The third benchmark initially returned extra rows; it was fixed by teaching Genie to use `ORDER BY ... DESC LIMIT 1` for a singular “largest” question. Preserve benchmark screenshots in the submission evidence.

## How to reproduce in Databricks

1. Import or clone this repository into a Databricks workspace with access to a writable Unity Catalog and Serverless compute.
2. Confirm the snapshot URL parameters (`city`, `snapshot`, `source_path`). Rome uses `2026-09-17` and Sicily `2026-09-25` in the source EDA configuration.
3. Run the Bronze and Silver notebooks for **Rome** and **Sicily**; verify row counts and city/snapshot fields.
4. Run `04_gold_features.ipynb` for each training market. Run `05_train_model.ipynb` **once** with `train_cities=rome,sicily`, `model_version=v1`, `model_spec=v1`, and `overwrite_model=no` for the original baseline-model comparison.
5. Run `07_gold_dashboard.ipynb` to create BI aggregates. Re-create or import the AI/BI dashboard from the relevant Databricks workspace.
6. Configure Genie with the seven Gold sources, SQL examples and benchmark questions. Review its SQL and benchmark outputs.
7. For City C (Naples), the v1 first-run and v2 after-fix evaluation results were generated in the team's primary Databricks workspace. The resulting CSV exports were imported into `workspace.gold.model_metrics_all` and `workspace.gold.price_predictions_naples` in a separate workspace used for the AI/BI Dashboard and Genie Agent. Both model versions were validated against the imported prediction data.

## Acknowledgements

Data: [Inside Airbnb](https://insideairbnb.com/get-the-data/). Development: Databricks, Apache Spark/PySpark, Delta Lake, SQL, Python, scikit-learn, AI/BI Dashboards and Genie.
