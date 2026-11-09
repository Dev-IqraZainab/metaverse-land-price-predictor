"""
Generate a synthetic dataset of metaverse land parcels.

Real marketplace data is noisy and behind APIs, so this script simulates
realistic patterns (location, size, roads, plazas, market hype) to make the
ML pipeline easy to run and reproduce. Swap in real data later with the same columns.

Usage:
    python generate_data.py --rows 5000 --out land_parcels.csv
"""
import argparse

import numpy as np
import pandas as pd

WORLDS = {"Decentraland": 1.00, "Sandbox": 1.25, "Otherside": 1.60, "Somnium": 0.70}
DISTRICTS = ["Fashion", "Gaming", "Crypto Valley", "Music", "Residential", "Art"]


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    world = rng.choice(list(WORLDS), size=rows, p=[0.40, 0.30, 0.15, 0.15])
    district = rng.choice(DISTRICTS, size=rows)
    x = rng.integers(-150, 151, size=rows)
    y = rng.integers(-150, 151, size=rows)
    size = rng.choice([1, 2, 4, 9, 16], size=rows, p=[0.55, 0.20, 0.13, 0.08, 0.04])
    dist_center = np.sqrt(x**2 + y**2).round(1)
    dist_plaza = rng.gamma(shape=2.0, scale=12.0, size=rows).round(1)
    next_to_road = rng.random(rows) < 0.30
    has_building = rng.random(rows) < 0.35
    hype = rng.uniform(0.6, 1.6, size=rows).round(2)  # market sentiment index

    district_mult = pd.Series(district).map({
        "Fashion": 1.35, "Gaming": 1.25, "Crypto Valley": 1.40,
        "Music": 1.10, "Residential": 0.85, "Art": 1.00,
    }).to_numpy()
    world_mult = pd.Series(world).map(WORLDS).to_numpy()

    base = 2500  # USD for a 1x1 parcel far from everything
    price = (
        base
        * size ** 0.9
        * world_mult
        * district_mult
        * hype
        * (1 + 2.5 * np.exp(-dist_center / 40))
        * (1 + 0.8 * np.exp(-dist_plaza / 10))
        * np.where(next_to_road, 1.20, 1.0)
        * np.where(has_building, 1.15, 1.0)
        * rng.lognormal(mean=0.0, sigma=0.15, size=rows)  # market noise
    ).round(2)

    return pd.DataFrame({
        "world": world, "district": district, "x": x, "y": y, "size_parcels": size,
        "dist_to_center": dist_center, "dist_to_plaza": dist_plaza,
        "next_to_road": next_to_road.astype(int), "has_building": has_building.astype(int),
        "market_hype": hype, "price_usd": price,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="land_parcels.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} parcels to {args.out}")
    print(df.describe().round(2).T[["mean", "min", "max"]])
