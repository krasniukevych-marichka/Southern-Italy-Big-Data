# Price model (Person 3)

Files: `04_gold_features.ipynb`, `05_train_model.ipynb`, `06_score_city.ipynb`, `model_utils.py` (shared code, lives next to the notebooks).

## Data flow

```
silver.listings ──04 (city)──▶ gold.listing_features ──05 (A+B, once per version)──▶ price_model_v1.joblib, price_model_v2.joblib
                                                      │                      │
                                                      └──06 (city, model vN)─┴──▶ gold.price_predictions
                                                                                  gold.model_metrics
                                                                                  gold.model_feature_importance (from 05)
```

## Decisions

| What | How | Why |
|---|---|---|
| Rows for training and evaluation | `has_price` and `outlier_class ∈ {normal, tail}` | rows without a price have no target; placeholder prices ≥ €5,000 would distort MAE (EDA, sections 5.1, 9) |
| Split | 20% of the listings of **each city** go to test, assigned by whole hosts (`host_id`), seed 42 | a third of listings have a "twin" from the same host — a random split would give an over-optimistic MAE (EDA, item 18) |
| Baseline | median price by (`room_type`, `accommodates`) on the training split **of the same city**; fallback → `room_type` → city | required by the assignment; the fallback covers rare combinations (hotel / shared rooms) |
| Target | `log(price)`, prediction = `exp()` | price skew is 30 / 113; log price skew is 0.8 / 1.1 |
| Model | `HistGradientBoostingRegressor` (scikit-learn) | captures non-linearities (in Sicily price only rises from 6+ guests) and handles missing values natively |
| Features | size (`accommodates`, `bedrooms`, `beds`, `bathrooms`), `room_type`, `property_type` (< 50 listings in train → infrequent), amenities (pool, sea, air conditioning, elevator…), `amenities_count`, host profile, `long_stay`, licence, **density of listings within ~1.5 km and ~6 km** | all known for a new listing and easy to explain |
| Not used | `estimated_revenue_l365d`, `price_quote_*`, `price_per_guest` (leakage); reviews, availability, occupancy (consequences of price); ids / URLs; coordinates, neighbourhood names and the city | coordinates and neighbourhoods are tied to one city's geography and do not transfer to C (on A/B they improve MAE by only €0.6) |
| Imputation | train-split median + `is_missing` flag inside the pipeline | no leakage from test |

## Test-set results on A and B

| City | n test | Median price, € | Baseline MAE, € | v1 MAE, € | v2 MAE, € | v2 vs baseline |
|---|---|---|---|---|---|---|
| Rome | 6,777 | 190 | 84.02 | 63.99 | **64.19** | −23.6% |
| Sicily | 10,129 | 115 | 73.30 | 59.38 | **58.99** | −19.5% |
| Both | 16,906 | 142 | 77.59 | 61.23 | **61.07** | −21.3% |

All numbers are from the Databricks runs (`05_train_model`, `gold.model_metrics`). The v2 changes did not cost anything on A+B:
the combined MAE is even slightly lower (61.07 vs 61.23), so making the model transferable came for free.

The median absolute error is about €31 (MAE is pulled up by expensive villas). On average the model slightly under-prices (bias about −€19):
a prediction made via `exp(log)` is closer to the median than to the mean, which is exactly what minimises MAE.

Most important v1 features (MAE increase when the feature is shuffled): density within ~6 km (+€26), bathrooms (+7), guests (+5),
density within ~1.5 km (+4), pool (+4), bedrooms (+4). In v2 the relative density within ~6 km drops to +€8 (still the top feature, but no longer standing in for the city), followed by guests (+6.5), bathrooms (+5.7), pool (+4.6) and bedrooms (+4.5).

## City C: Naples

### First run (v1, unchanged pipeline and model)

The pipeline and the scoring notebook ran without errors. Results on the Naples test split (2,157 listings, median price €112):

| | MAE, € |
|---|---|
| Baseline on Naples (own train split) | 58.39 |
| A+B baseline rule applied to Naples | 60.92 |
| Model v1 on A+B (test) | 61.23 |
| **Model v1 on Naples (test)** | **74.12** |

On Naples the model is 27% **worse** than the simple baseline, and worse even than the A+B baseline rule, which has never seen Naples either.
So the problem is not just that the own-city baseline knows Naples prices: the model itself does not transfer.

### Why (diagnosis from the first-run output)

- **The model over-prices Naples:** bias **+€28** (on A+B it was −€19), MAPE 63% vs 31% on A+B.
- **Naples is cheaper but denser than the training cities.** Median price €111 vs €141 on A+B (Rome €190). Median log density within ~1.5 km is
  7.68 in Naples vs 5.94 on A+B; mean 7.20 vs 6.77 in Rome. Density within ~6 km was the strongest v1 feature (+€26), and on A+B it
  effectively meant "dense, expensive Rome vs scattered, cheaper Sicily". For Naples it said "Rome", and the model priced it like Rome.
- **Not the cause:** unseen categories (`room_type` 0%, `property_type` 0.7% of Naples listings), missing values (same rates as A+B).
  Smaller shifts that do not explain a +€28 bias: fewer pools (0.4% vs 10%) and less free parking (16% vs 56%).

### Model changes (v2)

| # | Change | Kind | Reason |
|---|---|---|---|
| 1 | Target becomes `log(price) − log(median price of the city's train split)`; for a new city the level is the median price of **that city's own training split** | model change — a price level the model never saw | v1 had no way to know that Naples is cheaper than Rome; it inferred the level from density and got it wrong (+€28 bias) |
| 2 | Density features become relative to the median density of the same city (`log_density_*_rel`) | model change — a feature that means something different in city C | absolute density meant "Rome-like price level" on A+B; Naples is denser than Rome. Relative density keeps the useful part ("central vs outskirts within the city") |

The Naples test split is not touched by either change: the price level uses only Naples' training split — the same information the own-city
baseline uses, so the comparison with that baseline stays fair. The model itself is still trained on Rome and Sicily only.

**Evidence before the Naples run (leave-one-city-out on A/B):** trained on one city, tested on the other.

| | Rome → Sicily | Sicily → Rome |
|---|---|---|
| Own-city baseline | 73.3 | 84.0 |
| v1 approach | 76.8 | 104.4 |
| v2 approach | **71.4** | **79.5** |
| only relative density, no price level | 124.7 | 117.6 |

v1 transfers worse than the baseline in both directions; v2 beats the baseline in both. Relative density alone makes things worse —
the price level is the essential change, and relative density stops density from standing in for the city.

### After the changes (v2)

Both runs are evaluated on the same held-out Naples test split (2,157 listings, snapshot 2026-09-19).

| | MAE, € |
|---|---|
| Baseline on Naples (own train split) | 58.39 |
| Model v1 on A+B (test) | 61.23 |
| Model v2 on A+B (test) | 61.07 |
| Model v1 on Naples — first run | 74.12 |
| **Model v2 on Naples — after the changes** | **54.45** |

- v2 cuts the Naples error by **26.5%** compared with v1 (74.12 → 54.45).
- v2 now **beats the Naples baseline by 6.7%** (58.39 → 54.45); v1 was 27% worse than it.

### The remaining gap

In euros, the Naples MAE (54.45) is even lower than on A+B (61.07), but only because Naples is cheaper. Relative to the price level the model is still
weaker on Naples: MAE is about 49% of the median price in Naples (54.45 / 111.9) vs about 43% on A+B (61.07 / 142.3), and the gain over the baseline
is 6.7% on Naples vs 21% on A+B. Likely reasons:
- the model still learns *how* features change price only from Rome and Sicily; in Naples amenities are distributed differently
  (pools 0.4% vs 10%, free parking 16% vs 56%), so the relative effects it learned fit Naples less well;
- the city price level is a single number (the median of the Naples training split); it does not capture that the gap between
  room types or between central and outer areas may be different in Naples;
- the model has never seen a Naples listing; the only Naples information it uses is the price level and the city's own density distribution.

A natural next step (not done, to keep the transfer test honest) would be to add the Naples training split to the training data and retrain.

### Summary of changes for city C

| Change | Kind | Effect on Naples MAE |
|---|---|---|
| None needed in the model part of the pipeline (scoring ran without errors) | pipeline fix | — |
| Price relative to the city's price level (from the city's own train split) | model change | together: 74.12 → 54.45 |
| Density relative to the city's own median density | model change | |
