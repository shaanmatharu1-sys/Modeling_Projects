import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import ta

# Establish Tickers for Industries (ETFs)
tickers = {
    "Semiconductors": "SOXX",
    "Software": "IGV",
    "Banks": "KBE",
    "Insurance": "KIE",
    "Oil and Gas": "XLE",
    "Utilities": "XLU",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Healthcare": "XLV",
    "Real Estate": "XLRE",
    "Materials": "XLB"
}

# Pull and Calculate Data
data = yf.download(list(tickers.values()), period="5y", auto_adjust=True)["Close"]  # type: ignore
returns = data.pct_change().dropna()

# Feature Engineering
features = pd.DataFrame(index=returns.index)

for col in returns.columns:
    features[f'{col}_mean_5'] = returns[col].rolling(5).mean()
    features[f'{col}_mean_21'] = returns[col].rolling(21).mean()
    features[f'{col}_vol_5'] = returns[col].rolling(5).std()
    features[f'{col}_vol_21'] = returns[col].rolling(21).std()
    features[f'{col}_corr_SOXX'] = returns[col].rolling(60).corr(returns["SOXX"])  # type: ignore

target = pd.DataFrame(index=returns.index)
for sector, ticker in tickers.items():
    target[sector] = (data[ticker].shift(-126) / data[ticker]) - 1

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler

predictions = {}

for sector, ticker in tickers.items():
    # Select features for this ticker
    X = features.filter(like=ticker).dropna()
    y = target[sector].loc[X.index]

    # Combine X and y to remove any remaining NaNs
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined.drop(columns=[sector])
    y = combined[sector]

    if X.empty or y.empty:
        print(f"Skipping {sector} because features or target are empty.")
        continue

    # === Standardize features here ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # <-- scaled features for training

    model = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
    model.fit(X_scaled, y)

    # Scale the latest features the same way before prediction
    latest_features_scaled = scaler.transform(X.iloc[-1:].values)
    future_pred = model.predict(latest_features_scaled)[0]

    # Optional: clip extreme predictions to realistic range
    future_pred = np.clip(future_pred, -0.2, 0.2)  # -20% to +20%
    predictions[sector] = future_pred * 100




pred_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted 6-Month Growth (%)'])
pred_df = pred_df.sort_values(by='Predicted 6-Month Growth (%)', ascending=False)

print("\n=== Expected Growth by May 2026 ===\n")
print(pred_df)

plt.figure(figsize=(10,6))
plt.bar(pred_df.index, pred_df['Predicted 6-Month Growth (%)'], color='green')
plt.title("Predicted 6-Month Growth by Sector (to ~May 2026)")
plt.ylabel("Predicted Return (%)")
plt.xticks(rotation=45)
plt.axhline(0, color='red', linestyle='--')
plt.tight_layout()
plt.show()
