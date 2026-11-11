"""Спільний код моделі ціни: ознаки, split, baseline, пайплайн, метрики.

Імпортується з 05_train_model і 06_score_city (лежить у тій самій папці репозиторію).
Тут немає нічого специфічного для Rome чи Sicily: усе, що залежить від міста, рахується з даних цього міста.
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

# ---- ознаки (див. 01_eda, розділи 10–12) -----------------------------------
NUM = [
    "accommodates", "bedrooms", "beds", "bathrooms",             # розмір житла
    "amenities_count",                                            # оснащення
    "calculated_host_listings_count_entire_homes",                # професійний хост
    "calculated_host_listings_count_private_rooms",
    "host_tenure_months",
    "log_density_1km", "log_density_5km",                         # локація без координат
]
BOOL = [
    "bathrooms_shared", "long_stay", "min_nights_missing", "has_license",
    "am_pool", "am_sea", "am_ac", "am_hot_tub", "am_elevator",
    "am_free_parking", "am_dishwasher", "am_workspace",
]
CAT = ["room_type", "property_type"]
FEATURES = NUM + BOOL + CAT

# Свідомо НЕ використовуємо:
#  - leakage: estimated_revenue_l365d, price_quote_*, price_per_guest, log_price;
#  - наслідки ціни / невідомі для нового оголошення: number_of_reviews*, review_scores_*,
#    reviews_per_month, availability_*, estimated_occupancy_l365d;
#  - id, host_id, name, URL-и, snapshot-константи;
#  - latitude/longitude і neighbourhood_cleansed: прив'язані до географії конкретного міста,
#    не переносяться на нове місто (на A/B дають лише −0.6 EUR MAE);
#  - city: модель має працювати на місті, якого не бачила.


def in_scope(df: pd.DataFrame) -> pd.Series:
    """Рядки, на яких вчимо й оцінюємо: є ціна і це не помилка/заглушка (01_eda, п. 16)."""
    return df["has_price"].fillna(False).astype(bool) & df["outlier_class"].isin(["normal", "tail"])


def host_split(df: pd.DataFrame, seed: int = SEED, frac: float = TEST_FRAC) -> pd.Series:
    """80/20 окремо в кожному місті, згрупований за host_id (01_eda, п. 18).

    Хости перемішуються з фіксованим seed, у test ідуть цілі хости, поки не набереться ~20%
    оголошень міста. Результат не залежить від порядку рядків.
    """
    out = pd.Series("train", index=df.index, dtype="object")
    for _, g in df.groupby("city"):
        hosts = g.groupby("host_id").size().sort_index()
        order = np.random.default_rng(seed).permutation(hosts.index.values)
        cum = hosts.loc[order].cumsum().values / hosts.sum()
        test_hosts = set(order[cum <= frac])
        out.loc[g.index[g["host_id"].isin(test_hosts)]] = "test"
    return out


# ---- baseline: медіана ціни за (room_type, accommodates) на train -----------------
def fit_baseline(train: pd.DataFrame) -> dict:
    """Медіани з fallback: (room_type, accommodates) → room_type → уся вибірка."""
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


# ---- модель -------------------------------------------------------------------
def to_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.reindex(columns=FEATURES).copy()
    for c in NUM + BOOL:
        X[c] = pd.to_numeric(X[c].astype("float64"), errors="coerce")
    for c in CAT:
        X[c] = X[c].astype("object").where(X[c].notna(), "missing")
    return X


def make_model() -> Pipeline:
    pre = ColumnTransformer([
        # медіана рахується лише на train; add_indicator = флаг is_missing (01_eda, п. 24)
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUM + BOOL),
        # рідкісні property_type (< 50 оголошень у train) → одна категорія «infrequent» (п. 23)
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), CAT),
    ])
    gbm = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, random_state=SEED)
    return Pipeline([("pre", pre), ("gbm", gbm)])


def fit_model(train: pd.DataFrame) -> Pipeline:
    m = make_model()
    m.fit(to_X(train), np.log(train["price"].astype(float)))   # таргет: log(price) (п. 17)
    return m


def predict_model(m: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return np.exp(m.predict(to_X(df)))                          # назад у EUR


# ---- метрики ------------------------------------------------------------------
def metrics_rows(df: pd.DataFrame, group_label=None) -> dict:
    """df: тестові рядки з колонками price, baseline_pred, baseline_ab_pred, model_pred."""
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
