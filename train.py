"""
Train and compare models that predict metaverse land prices.

Pipeline: load CSV -> one-hot encode categories -> log-transform target
-> compare Linear Regression vs Random Forest vs Gradient Boosting
-> save the best model + charts.

Usage:
    python train.py --data land_parcels.csv
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["world", "district"]
NUMERIC = ["x", "y", "size_parcels", "dist_to_center", "dist_to_plaza",
           "next_to_road", "has_building", "market_hype"]
TARGET = "price_usd"


def build_pipeline(regressor) -> TransformedTargetRegressor:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    # Prices are skewed, so learning log(price) works much better.
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", regressor)]),
        func=np.log1p, inverse_func=np.expm1,
    )


def main(data_path: str) -> None:
    df = pd.read_csv(data_path)
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
    }

    results, fitted = {}, {}
    for name, reg in candidates.items():
        model = build_pipeline(reg).fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {"MAE_usd": round(mean_absolute_error(y_test, pred), 2),
                         "R2": round(r2_score(y_test, pred), 4)}
        fitted[name] = model
        print(f"{name:<18}  MAE ${results[name]['MAE_usd']:>10,.2f}   R² {results[name]['R2']:.4f}")

    best_name = max(results, key=lambda n: results[n]["R2"])
    best = fitted[best_name]
    print(f"\n🏆 Best model: {best_name}")

    joblib.dump(best, "model.joblib")
    with open("metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    # ---- Chart 1: predicted vs actual ----
    pred = best.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, s=8, alpha=0.4, color="#7c5cff")
    lim = [y_test.min(), y_test.max()]
    ax.plot(lim, lim, "--", color="#ff5fa2", lw=1.5, label="Perfect prediction")
    ax.set(xscale="log", yscale="log", xlabel="Actual price (USD)", ylabel="Predicted price (USD)",
           title=f"{best_name}: R² = {results[best_name]['R2']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("pred_vs_actual.svg")

    # ---- Chart 2: feature importance (tree models) ----
    inner = best.regressor_.named_steps["model"]
    if hasattr(inner, "feature_importances_"):
        names = best.regressor_.named_steps["pre"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        imp = pd.Series(inner.feature_importances_, index=names).sort_values().tail(10)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        imp.plot.barh(ax=ax, color="#00c9ff")
        ax.set(title="Top 10 price drivers", xlabel="Importance")
        fig.tight_layout()
        fig.savefig("feature_importance.svg")

    print("Saved model.joblib, metrics.json, pred_vs_actual.svg, feature_importance.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="land_parcels.csv")
    main(parser.parse_args().data)
