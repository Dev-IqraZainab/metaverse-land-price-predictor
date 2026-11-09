"""
Predict the price of a single metaverse land parcel with the trained model.

Usage:
    python predict.py --world Sandbox --district Gaming --x 10 --y -5 --size 4 --road 1
"""
import argparse
import math

import joblib
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", default="Decentraland",
                   choices=["Decentraland", "Sandbox", "Otherside", "Somnium"])
    p.add_argument("--district", default="Gaming",
                   choices=["Fashion", "Gaming", "Crypto Valley", "Music", "Residential", "Art"])
    p.add_argument("--x", type=int, default=0)
    p.add_argument("--y", type=int, default=0)
    p.add_argument("--size", type=int, default=1, help="number of 1x1 parcels in the estate")
    p.add_argument("--plaza", type=float, default=20.0, help="distance to nearest plaza")
    p.add_argument("--road", type=int, choices=[0, 1], default=0)
    p.add_argument("--building", type=int, choices=[0, 1], default=0)
    p.add_argument("--hype", type=float, default=1.0, help="market sentiment 0.6 to 1.6")
    a = p.parse_args()

    model = joblib.load("model.joblib")
    parcel = pd.DataFrame([{
        "world": a.world, "district": a.district, "x": a.x, "y": a.y,
        "size_parcels": a.size, "dist_to_center": round(math.hypot(a.x, a.y), 1),
        "dist_to_plaza": a.plaza, "next_to_road": a.road,
        "has_building": a.building, "market_hype": a.hype,
    }])
    price = model.predict(parcel)[0]
    print(f"🏝️  {a.world} · {a.district} · ({a.x}, {a.y}) · {a.size} parcel(s)")
    print(f"💰 Estimated price: ${price:,.0f}")


if __name__ == "__main__":
    main()
