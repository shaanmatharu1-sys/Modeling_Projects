import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="Momentum Dashboard", layout="wide")
st.title("📈 Momentum & Beta Dashboard")

# --- Inputs ---
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock symbol (e.g. AAPL, TSLA, NVDA):", value="AAPL").upper()

# Default dates: 1 year of data
default_end = datetime.now()
default_start = default_end - timedelta(days=365)

start = st.sidebar.date_input("Start date", value=default_start)
end = st.sidebar.date_input("End date", value=default_end)
run_button = st.sidebar.button("Run Analysis")

# --- RSI Function ---
def compute_RSI(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))

# --- Rolling Beta Function (FIXED) ---
def compute_rolling_beta(stock_df, market_df, window=20):
    """
    Compute rolling beta between stock and market.
    Properly aligns data and handles NaN values.
    """
    # Create aligned dataframe with both series
    combined = pd.DataFrame({
        'stock': stock_df["Close"],
        'market': market_df["Close"]
    })
    
    # Calculate returns
    stock_returns = combined['stock'].pct_change()
    market_returns = combined['market'].pct_change()
    
    # Rolling covariance and variance
    cov = stock_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    
    # Beta = Cov(stock, market) / Var(market)
    beta = cov / var
    
    return beta

# --- Run Analysis ---
if run_button:
    if not ticker:
        st.warning("Please enter a valid stock symbol.")
    else:
        with st.spinner(f"Fetching data for {ticker} and S&P 500..."):
            try:
                df_stock = yf.download(ticker, start=start, end=end, progress=False)
                df_spy = yf.download("^GSPC", start=start, end=end, progress=False)
                
                # Fix MultiIndex columns issue
                if isinstance(df_stock.columns, pd.MultiIndex):
                    df_stock.columns = df_stock.columns.get_level_values(0)
                if isinstance(df_spy.columns, pd.MultiIndex):
                    df_spy.columns = df_spy.columns.get_level_values(0)
                
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                st.stop()

        if df_stock.empty or df_spy.empty:
            st.error("No data found for that symbol and date range.")
            st.stop()

        # --- Compute Indicators ---
        df_stock['RSI'] = compute_RSI(df_stock)
        short_ema = df_stock['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df_stock['Close'].ewm(span=26, adjust=False).mean()
        df_stock['MACD'] = short_ema - long_ema
        df_stock['Signal'] = df_stock['MACD'].ewm(span=9, adjust=False).mean()
        df_stock['Hist'] = df_stock['MACD'] - df_stock['Signal']
        df_stock['ROC'] = ((df_stock['Close'] - df_stock['Close'].shift(12)) / df_stock['Close'].shift(12)) * 100
        df_stock["Beta"] = compute_rolling_beta(df_stock, df_spy, window=20)

        # --- Display Latest Values ---
        col1, col2, col3, col4 = st.columns(4)
        latest = df_stock.iloc[-1]
        
        with col1:
            st.metric("Current Price", f"${latest['Close']:.2f}")
        with col2:
            st.metric("RSI", f"{latest['RSI']:.1f}")
        with col3:
            st.metric("Beta (20d)", f"{latest['Beta']:.2f}" if not pd.isna(latest['Beta']) else "N/A")
        with col4:
            st.metric("ROC (12d)", f"{latest['ROC']:.1f}%")

        # --- Plot Indicators ---
        fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
        fig.suptitle(f"{ticker} Momentum Indicators + Beta vs S&P 500", fontsize=14)

        axes[0].plot(df_stock.index, df_stock['Close'], color='steelblue')
        axes[0].set_ylabel('Price ($)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(df_stock.index, df_stock['RSI'], color='orange')
        axes[1].axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        axes[1].axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        axes[1].set_ylabel('RSI')
        axes[1].set_ylim(0, 100)
        axes[1].legend(loc='upper left', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(df_stock.index, df_stock['MACD'], label='MACD', color='blue')
        axes[2].plot(df_stock.index, df_stock['Signal'], label='Signal', color='red')
        axes[2].bar(df_stock.index, df_stock['Hist'], color='gray', alpha=0.4, label='Histogram')
        axes[2].legend(loc='upper left', fontsize=8)
        axes[2].set_ylabel('MACD')
        axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(df_stock.index, df_stock['ROC'], label='ROC 12', color='purple')
        axes[3].axhline(0, color='gray', linestyle='--', alpha=0.7)
        axes[3].set_ylabel('ROC (%)')
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(df_stock.index, df_stock['Beta'], color='darkgreen', label='Rolling Beta (20d)', linewidth=1.5)
        axes[4].axhline(1, color='red', linestyle='--', alpha=0.7, label='Market Beta')
        axes[4].legend(loc='upper left', fontsize=8)
        axes[4].set_ylabel('Beta')
        axes[4].set_xlabel('Date')
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # --- Summary Stats ---
        with st.expander("📊 View Summary Statistics"):
            st.subheader("Latest Indicator Values")
            summary_data = {
                'Indicator': ['Price', 'RSI', 'MACD', 'Signal', 'ROC (12d)', 'Beta (20d)'],
                'Value': [
                    f"${latest['Close']:.2f}",
                    f"{latest['RSI']:.2f}",
                    f"{latest['MACD']:.4f}",
                    f"{latest['Signal']:.4f}",
                    f"{latest['ROC']:.2f}%",
                    f"{latest['Beta']:.3f}" if not pd.isna(latest['Beta']) else "N/A"
                ]
            }
            st.table(pd.DataFrame(summary_data))

            st.subheader("MACD Interpretation")
            if latest['MACD'] > latest['Signal']:
                st.info(f"MACD = {latest['MACD']:.4f}, Signal = {latest['Signal']:.4f} → The MACD line is above the Signal line, indicating a potential **bullish** trend.")
            elif latest['MACD'] < latest['Signal']:
                st.info(f"MACD = {latest['MACD']:.4f}, Signal = {latest['Signal']:.4f} → The MACD line is below the Signal line, indicating a potential **bearish** trend.")
            else:
                st.info(f"MACD = {latest['MACD']:.4f}, Signal = {latest['Signal']:.4f} → The MACD line is equal to the Signal line, indicating a **neutral** trend.")

            st.subheader("RSI Interpretation")
            if latest['RSI'] > 70:
                st.info(f"RSI = {latest['RSI']:.2f} → The current RSI indicates that this stock may be **overbought**. A potential price correction may occur.")
            elif latest['RSI'] < 30:
                st.info(f"RSI = {latest['RSI']:.2f} → The current RSI indicates that this stock may be **oversold**. A potential price increase may occur.")
            else:
                st.info(f"RSI = {latest['RSI']:.2f} → The current RSI indicates that this stock is in a **neutral** state.")

            st.subheader("Beta Interpretation")
            if not pd.isna(latest['Beta']):
                beta_val = latest['Beta']
                if beta_val > 1:
                    st.info(f"📈 Beta = {beta_val:.2f} → Stock is **more volatile** than the market")
                elif beta_val < 1 and beta_val > 0:
                    st.info(f"📉 Beta = {beta_val:.2f} → Stock is **less volatile** than the market")
                elif beta_val < 0:
                    st.info(f"🔄 Beta = {beta_val:.2f} → Stock moves **inversely** to the market")
                else:
                    st.info(f"➡️ Beta = {beta_val:.2f} → Stock moves **with** the market")

            st.subheader(f"🗞️ Latest News for {ticker}")

            try:
                stock = yf.Ticker(ticker)
                news_items = stock.news

                if news_items and isinstance(news_items, list):
                    for article in news_items[:5]:  # Show up to 5 latest articles
                        title = article.get("title") or article.get("content", {}).get("title", "Untitled")
                        link = article.get("link") or article.get("content", {}).get("canonicalUrl", "")
                        publisher = article.get("publisher") or article.get("content", {}).get("provider", "Unknown")
                        publish_time = article.get("providerPublishTime")

                        if title and link:
                            st.markdown(f"**[{title}]({link})**")
                            st.caption(f"📰 {publisher} — {datetime.fromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M') if publish_time else ''}")
                            st.write("---")
                        else:
                            st.write("⚠️ Could not parse one news article properly.")
                else:
                    st.info("No recent news available for this stock.")

            except Exception as e:
                st.warning(f"Could not load news: {e}")


            st.subheader("Where to go with these indicators?")
            st.markdown("""
            It is important to take into consideration that no indicator is perfect and should not be used in isolation.
            Overbought/Oversold conditions can be prolonged in strong trends, and Beta values can change over time.
            As always, take a holistic approach to your movement, a fundamental analysis of qualitative data is also important.
            1. **Combine Indicators**: Use multiple indicators to confirm signals.
            2. **Set Stop-Losses**: Always manage risk with stop-loss orders
            3. **Stay Informed**: Keep up with market news and events that may impact stock prices.
            """)

        st.success("✅ Analysis complete.")
        st.caption("Developed by Shaan Matharu — educational purposes only, not financial advice.")
        st.caption("This code was generated with GitHub Copilot, OpenAI, and Streamlit.")
        st.caption("Data provided by Yahoo Finance.")