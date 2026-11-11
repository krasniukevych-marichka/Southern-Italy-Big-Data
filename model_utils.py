"""Shared price-model code: features, split, baseline, pipeline, metrics.

Imported by 05_train_model and 06_score_city (lives in the same repository folder).
Nothing here is specific to Rome or Sicily: everything city-dependent is computed from that city's own data.

Two model specs live side by side so that the frozen v1 file keeps working:
  v1 — week 1 model (absolute density, absolute log price target);
  v2 — after the city C run: density relative to the city, price relative to the city's price level.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
TEST_FRAC = 0.2

# ---- features (see 01_eda, sections 10–12) ------------------------------------
NUM_BASE = [
    "accommodates", "bedrooms", "beds", "bathrooms",             # size of the place
    "amenities_count",                                            # equipment
    "calculated_host_listings_count_entire_homes",                # professional host
    "calculated_host_listings_count_private_rooms",
    "host_tenure_months",
]
BOOL = [
    "bathrooms_shared", "long_stay", "min_nights_missing", "has_license",
    "am_pool", "am_sea", "am_ac", "am_hot_tub", "am_elevator",
    "am_free_parking", "am_dishwasher", "am_workspace",
]
CAT = ["room_type", "property_type"]

DENSITY = ["log_density_1km", "log_density_5km"]
DENSITY_REL = [c + "_rel" for c in DENSITY]

SPECS = {
    # v1: location = absolute density of listings around (location without coordinates)
    "v1": {"num": NUM_BASE + DENSITY, "city_level": False},
    # v2 (model changes after the city C first run):
    #  1) density relative to the median density of the same city — in v1 "dense" also meant
    #     "Rome-like price level", and Naples is denser than Rome but much cheaper;
    #  2) target = log(price) − log(median price of the city's train split): the model learns
    #     how much a listing deviates from its city's price level, and the level of a new city
    #     is estimated from that city's own training split.
    "v2": {"num": NUM_BASE + DENSITY_REL, "city_level": True},
}

# Backwards-compatible names (v1)
NUM = SPECS["v1"]["num"]
FEATURES = NUM + BOOL + CAT

# Deliberately NOT used:
#  - leakage: estimated_revenue_l365d, price_quote_*, price_per_guest, log_price;
#  - consequences of price / unknown for a new listing: number_of_reviews*, review_scores_*,
#    reviews_per_month, availability_*, estimated_occupancy_l365d;
#  - id, host_id, name, URLs, snapshot constants;
#  - latitude/longitude and neighbourhood_cleansed: tied to one city's geography,
#    do not transfer to a new city (on A/B they improve MAE by only 0.6 EUR);
#  - city: the model must work on a city it has never seen.


def spec_of(art) -> str:
    """Spec name of a saved artifact; files saved before v2 existed are v1."""
    return art.get("spec", "v1") if isinstance(art, dict) else "v1"


def features_of(spec: str) -> list:
    return SPECS[spec]["num"] + BOOL + CAT


def in_scope(df: pd.DataFrame) -> pd.Series:
    """Rows used for training and evaluation: price present and not an error/placeholder (01_eda, item 16)."""
    return df["has_price"].fillna(False).astype(bool) & df["outlier_class"].isin(["normal", "tail"])


def add_relative_density(df: pd.DataFrame) -> pd.DataFrame:
    """log density minus the median log density of the same city.

    Pass ALL listings of a city (not only in-scope rows): the reference is the city's supply,
    and price is not used.
    """
    df = df.copy()
    for c in DENSITY:
        df[c + "_rel"] = df[c] - df.groupby("city")[c].transform("median")
    return df


def host_split(df: pd.DataFrame, seed: int = SEED, frac: float = TEST_FRAC) -> pd.Series:
    """80/20 split within each city, grouped by host_id (01_eda, item 18).

    Hosts are shuffled with a fixed seed and whole hosts go to test until ~20% of the city's
    listings are reached. The result does not depend on row order.
    """
    out = pd.Series("train", index=df.index, dtype="object")
    for _, g in df.groupby("city"):
        hosts = g.groupby("host_id").size().sort_index()
        order = np.random.default_rng(seed).permutation(hosts.index.values)
        cum = hosts.loc[order].cumsum().values / hosts.sum()
        test_hosts = set(order[cum <= frac])
        out.loc[g.index[g["host_id"].isin(test_hosts)]] = "test"
    return out


# ---- baseline: median price by (room_type, accommodates) on train -------------------
def fit_baseline(train: pd.DataFrame) -> dict:
    """Medians with fallback: (room_type, accommodates) → room_type → whole sample."""
    return {
        "cell": train.groupby(["room_type", "accommodates"])["price"].median().to_dict(),
        "room": train.groupby("room_type")["price"].median().to_dict(),
        "all": float(train["price"].median()),
    }


def predict_baseline(b: dict, df: pd.DataFrame) -> np.ndarray:
    return np.array([
        b["cell"].get((r, a), b["room"].get(r, b["all"]))
        for r, a in zip(df["room_type"], df["accommodates"])
    ], dtype=float)


# ---- city price level (v2) ------------------------------------------------------
def fit_city_level(train: pd.DataFrame) -> dict:
    """{city: log(median price)} computed on the training split of each city."""
    return {c: float(np.log(g["price"].astype(float).median())) for c, g in train.groupby("city")}


def level_offset(df: pd.DataFrame, levels: dict) -> np.ndarray:
    missing = set(df["city"].unique()) - set(levels)
    if missing:
        raise ValueError(f"No price level for {missing}: compute it with fit_city_level on that city's train split")
    return df["city"].map(levels).to_numpy(float)


# ---- model --------------------------------------------------------------------
def to_X(df: pd.DataFrame, spec: str = "v1") -> pd.DataFrame:
    num = SPECS[spec]["num"]
    X = df.reindex(columns=num + BOOL + CAT).copy()
    for c in num + BOOL:
        X[c] = pd.to_numeric(X[c].astype("float64"), errors="coerce")
    for c in CAT:
        X[c] = X[c].astype("object").where(X[c].notna(), "missing")
    return X


def make_model(spec: str = "v1") -> Pipeline:
    pre = ColumnTransformer([
        # median is learned on train only; add_indicator = is_missing flag (01_eda, item 24)
        ("num", SimpleImputer(strategy="median", add_indicator=True), SPECS[spec]["num"] + BOOL),
        # rare property_type (< 50 listings in train) → one "infrequent" category (item 23)
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), CAT),
    ])
    gbm = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, random_state=SEED)
    return Pipeline([("pre", pre), ("gbm", gbm)])


def fit_model(train: pd.DataFrame, spec: str = "v1", levels: dict = None) -> Pipeline:
    m = make_model(spec)
    y = np.log(train["price"].astype(float)).to_numpy()            # target: log(price) (item 17)
    if SPECS[spec]["city_level"]:
        y = y - level_offset(train, levels)                        # v2: relative to the city level
    m.fit(to_X(train, spec), y)
    return m


def predict_model(m: Pipeline, df: pd.DataFrame, spec: str = "v1", levels: dict = None) -> np.ndarray:
    z = m.predict(to_X(df, spec))
    if SPECS[spec]["city_level"]:
        z = z + level_offset(df, levels)
    return np.exp(z)                                               # back to EUR


# ---- metrics ------------------------------------------------------------------
def metrics_rows(df: pd.DataFrame, group_label=None) -> dict:
    """df: test rows with columns price, baseline_pred, baseline_ab_pred, model_pred."""
    y = df["price"].to_numpy(float)
    err = lambda col: np.abs(df[col].to_numpy(float) - y)
    mae_b = float(err("baseline_pred").mean())
    mae_m = float(err("model_pred").mean())
    return {
        "n_test": int(len(df)),
        "median_price": float(np.median(y)),
        "mae_baseline": round(mae_b, 2),
        "mae_baseline_ab": round(float(err("baseline_ab_pred").mean()), 2),
        "mae_model": round(mae_m, 2),
        "improvement_vs_baseline_pct": round(100 * (mae_b - mae_m) / mae_b, 1),
        "median_ae_model": round(float(np.median(err("model_pred"))), 2),
        "bias_model": round(float((df["model_pred"].to_numpy(float) - y).mean()), 2),
        "mape_model_pct": round(float(100 * np.mean(err("model_pred") / y)), 1),
    }
