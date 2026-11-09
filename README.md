<h1 align="center">🏝️ Metaverse Land Price Predictor</h1>

<p align="center">
  <b>A machine-learning pipeline that estimates the price of virtual land parcels from location, size, district and market hype.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/R²-0.957-00C9FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-7C5CFF?style=for-the-badge" />
</p>

---

## 🎯 The Problem

In worlds like **Decentraland**, **The Sandbox** and **Otherside**, land parcels sell for anywhere from a few hundred to hundreds of thousands of dollars. This project builds a model that answers one question: **what is this parcel worth?**

## 📊 Results

Three models were trained on the same 80/20 split (log-transformed target):

| Model | MAE (USD) | R² |
|---|---:|---:|
| Linear Regression | $4,894 | 0.462 |
| Random Forest | $2,556 | 0.906 |
| **Gradient Boosting** 🏆 | **$1,779** | **0.957** |

<p align="center">
  <img src="pred_vs_actual.png" width="45%" />
  <img src="feature_importance.png" width="52%" />
</p>

**Key insights:**
- 📐 **Estate size** is by far the biggest price driver
- 📈 **Market hype** (sentiment) comes second, so timing matters
- 📍 Distance to the **map center** and **plazas** adds a clear location premium, as in real estate

## 🧱 Features Used

| Feature | Description |
|---|---|
| `world` | Decentraland, Sandbox, Otherside, Somnium |
| `district` | Fashion, Gaming, Crypto Valley, Music, Residential, Art |
| `x`, `y` | Parcel coordinates |
| `size_parcels` | Estate size (1, 2, 4, 9 or 16 parcels) |
| `dist_to_center`, `dist_to_plaza` | Location features |
| `next_to_road`, `has_building` | Binary amenities |
| `market_hype` | Market sentiment index (0.6 to 1.6) |

> ℹ️ **About the data:** `generate_data.py` creates a **synthetic** dataset that mimics real marketplace patterns, so the project runs offline and is fully reproducible. The pipeline uses plain CSV columns, so you can swap in real data (e.g. from a marketplace API) without changing the training code.

## 🚀 Quick Start

```bash
git clone https://github.com/Dev-IqraZainab/metaverse-land-price-predictor.git
cd metaverse-land-price-predictor
pip install -r requirements.txt

python generate_data.py --rows 5000     # 1. create dataset
python train.py                          # 2. train + compare models, save charts
python predict.py --world Sandbox --district Gaming --x 10 --y -5 --size 4 --road 1
# 🏝️  Sandbox · Gaming · (10, -5) · 4 parcel(s)
# 💰 Estimated price: $42,509
```

## 📁 Project Structure

```
metaverse-land-price-predictor/
├── generate_data.py        # Synthetic dataset generator
├── train.py                # Preprocessing, model comparison, charts, saves best model
├── predict.py              # CLI: price a single parcel
├── metrics.json            # Latest training results
├── pred_vs_actual.png      # Evaluation chart
├── feature_importance.png  # What drives price
└── requirements.txt
```

## 🧠 Approach

1. **Preprocessing:** one-hot encode `world` and `district` with a `ColumnTransformer`
2. **Target transform:** train on `log(price)` with `TransformedTargetRegressor`, since prices are heavily skewed
3. **Model comparison:** Linear Regression (baseline) vs Random Forest vs Gradient Boosting
4. **Selection:** keep the best model by R² and save it with `joblib`

## 🗺️ Roadmap

- [ ] Plug in real sales data from marketplace APIs
- [ ] Add XGBoost / LightGBM and hyperparameter tuning
- [ ] Streamlit web app with an interactive map
- [ ] SHAP explanations per prediction

## 👩‍💻 Author

**Iqra Zainab**: AI & Unity Game Developer | Metaverse & Web3

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github)](https://github.com/Dev-IqraZainab)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/iqra-zainab/)

📄 Licensed under the [MIT License](LICENSE).
