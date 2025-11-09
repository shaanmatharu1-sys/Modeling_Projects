import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


ticker = input("Enter a stock ticker symbol (e.g., AAPL, MSFT, SPY): ").strip().upper()
data = yf.download(ticker, period="5y")['Close']

if data.empty:
    raise ValueError("No data found for the given ticker symbol.")

S0 = float(data.iloc[-1])
log_returns = np.log(1 + data.pct_change())
mu = float(log_returns.mean())
sigma = float(log_returns.std())


print(f"--- {ticker} Historical Summary ---")
print(f"Current Price: ${S0:.2f}")
print(f"Estimated Annual Return (μ): {mu:.4f}")
print(f"Estimated Volatility (σ): {sigma:.4f}")


T = 1.0      # years to simulate
N = 252      # steps per year
M = 10000    # number of simulated paths
dt = T / N

np.random.seed(42)
Z = np.random.randn(M, N)
increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
log_paths = np.cumsum(increments, axis=1)
S = S0 * np.exp(np.column_stack([np.zeros(M), log_paths]))
final_prices = S[:, -1]
expected_final = np.mean(final_prices)
ci_95 = np.percentile(final_prices, [2.5, 97.5])

print("\n--- Monte Carlo Results ---")
print(f"Expected Price in 1 Year: ${expected_final:.2f}")
print(f"95% Confidence Interval: ${ci_95[0]:.2f} - ${ci_95[1]:.2f}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(50):
    plt.plot(S[i, :], lw=1, alpha=0.5)
plt.title(f"{ticker} Monte Carlo Price Paths")
plt.xlabel("Days")
plt.ylabel("Price")

plt.subplot(1, 2, 2)
plt.hist(final_prices, bins=50, density=True, color='lightblue', edgecolor='k')
plt.axvline(expected_final, color='r', linestyle='--', label='Expected')
plt.title("Distribution of Final Prices")
plt.xlabel("Price After 1 Year")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
