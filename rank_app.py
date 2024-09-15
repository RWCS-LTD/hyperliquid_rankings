import streamlit as st
import requests
from datetime import datetime, timezone, timedelta
import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils.constants import MAINNET_API_URL
import numpy as np
import matplotlib.pyplot as plt
import io
import time

# Set global rate limit parameters
RATE_LIMIT = 100  # Example: 100 requests per minute
RATE_LIMIT_WINDOW = 60  # Window in seconds
REQUESTS_MADE = 0  # Counter to keep track of requests made

def enforce_rate_limit():
    global REQUESTS_MADE
    REQUESTS_MADE += 1
    # Enforce rate limit by sleeping if necessary
    if REQUESTS_MADE >= RATE_LIMIT:
        st.write(f"Please continue to wait. Pimpin ain't easy...")
        time.sleep(RATE_LIMIT_WINDOW)
        REQUESTS_MADE = 0  # Reset the counter after sleeping

def fetch_with_retries(info, asset_name, interval, start_time, end_time, max_retries=7):
    attempt = 0
    while attempt < max_retries:
        try:
            enforce_rate_limit()  # Check the rate limit before making a request
            historical_data = info.candles_snapshot(
                asset_name,
                interval=interval,
                startTime=start_time,
                endTime=end_time
            )
            return historical_data
        except Exception as e:
            wait_time = min(2 ** attempt, 60)  # Cap the wait time at 60 seconds
            time.sleep(wait_time)
            attempt += 1

    return None

def calculate_omega_ratio(returns, threshold=0):
    gains = returns[returns > threshold]
    losses = returns[returns <= threshold]
    if len(losses) == 0 or len(gains) == 0:
        return 0
    sum_gains = np.sum(gains - threshold)
    sum_losses = np.abs(np.sum(losses - threshold))
    omega_ratio = sum_gains / sum_losses
    return omega_ratio

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0):
    downside_returns = returns[returns < risk_free_rate]
    expected_return = returns.mean() - risk_free_rate
    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
    if downside_deviation == 0:
        return 0
    return expected_return / downside_deviation

def fetch_and_rank_perp_assets(info: Info, start_time: int, interval: str):
    global REQUESTS_MADE
    meta = info.meta()
    perp_assets = meta["universe"]
    omega_data, sharpe_data, sortino_data = [], [], []
    historical_data_dict = {}

    for asset in perp_assets:
        asset_name = asset['name']

        # Increase the rate limit throttle to further reduce request frequency
        time.sleep(0.2)  # Sleep 200ms between requests

        historical_data = fetch_with_retries(
            info, 
            asset_name, 
            interval, 
            start_time, 
            int(datetime.now(timezone.utc).timestamp() * 1000)
        )

        if historical_data is None:
            continue

        df_asset = pd.DataFrame(historical_data)

        if 'c' not in df_asset.columns or df_asset.empty:
            continue

        df_asset['c'] = pd.to_numeric(df_asset['c'], errors='coerce')

        if df_asset['c'].isnull().all():
            continue

        df_asset['Returns'] = df_asset['c'].pct_change().dropna()

        if len(df_asset['Returns']) < 1:
            continue

        omega_ratio = calculate_omega_ratio(df_asset['Returns'])
        sharpe_ratio = calculate_sharpe_ratio(df_asset['Returns'])
        sortino_ratio = calculate_sortino_ratio(df_asset['Returns'])

        omega_data.append((asset_name, omega_ratio))
        sharpe_data.append((asset_name, sharpe_ratio))
        sortino_data.append((asset_name, sortino_ratio))
        historical_data_dict[asset_name] = df_asset['c']

    ranked_omega = sorted(omega_data, key=lambda x: x[1], reverse=True)[:10]
    ranked_sharpe = sorted(sharpe_data, key=lambda x: x[1], reverse=True)[:10]
    ranked_sortino = sorted(sortino_data, key=lambda x: x[1], reverse=True)[:10]

    return ranked_omega, ranked_sharpe, ranked_sortino, historical_data_dict

def display_charts(ranked_omega, ranked_sharpe, ranked_sortino):
    # Determine intersection of assets in all three rankings
    assets_omega = {asset: score for asset, score in ranked_omega}
    assets_sharpe = {asset: score for asset, score in ranked_sharpe}
    assets_sortino = {asset: score for asset, score in ranked_sortino}
    
    # Find assets common to all three rankings
    common_assets = set(assets_omega.keys()) & set(assets_sharpe.keys()) & set(assets_sortino.keys())

    # Calculate combined scores for common assets
    combined_scores = []
    for asset in common_assets:
        combined_score = assets_omega[asset] + assets_sharpe[asset] + assets_sortino[asset]
        combined_scores.append((asset, combined_score))
    
    # Sort by combined scores in descending order
    ranked_combined = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    # Create a plot with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Omega ratio chart
    omega_names = [asset for asset, _ in ranked_omega]
    omega_values = [omega_ratio for _, omega_ratio in ranked_omega]
    axs[0, 0].barh(omega_names, omega_values, color='skyblue')
    axs[0, 0].set_title('Top 10 Omega Ratio')
    axs[0, 0].set_xlabel('Omega Ratio')
    axs[0, 0].invert_yaxis()

    # Sharpe ratio chart
    sharpe_names = [asset for asset, _ in ranked_sharpe]
    sharpe_values = [sharpe_ratio for _, sharpe_ratio in ranked_sharpe]
    axs[0, 1].barh(sharpe_names, sharpe_values, color='green')
    axs[0, 1].set_title('Top 10 Sharpe Ratio')
    axs[0, 1].set_xlabel('Sharpe Ratio')
    axs[0, 1].invert_yaxis()

    # Sortino ratio chart
    sortino_names = [asset for asset, _ in ranked_sortino]
    sortino_values = [sortino_ratio for _, sortino_ratio in ranked_sortino]
    axs[1, 0].barh(sortino_names, sortino_values, color='orange')
    axs[1, 0].set_title('Top 10 Sortino Ratio')
    axs[1, 0].set_xlabel('Sortino Ratio')
    axs[1, 0].invert_yaxis()

    # Combined score chart for common assets
    combined_names = [asset for asset, _ in ranked_combined]
    combined_values = [score for _, score in ranked_combined]
    axs[1, 1].barh(combined_names, combined_values, color='purple')
    axs[1, 1].set_title('Assets in All 3 Rankings (Combined Score)')
    axs[1, 1].set_xlabel('Combined Score')
    axs[1, 1].invert_yaxis()

    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Hyperliquid Exchange: Asset Ranking based on Omega, Sharpe, and Sortino Ratios")

    # User input for lookback period
    lookback = st.number_input('Select Lookback Period (in days)', min_value=1, max_value=365, value=14)

    if st.button('Run Analysis'):
        info = Info(base_url=MAINNET_API_URL)
        interval = '1d'
        start_time = int((datetime.now(timezone.utc) - timedelta(days=lookback)).timestamp() * 1000)

        ranked_omega, ranked_sharpe, ranked_sortino, historical_data_dict = fetch_and_rank_perp_assets(info, start_time, interval)
        
        st.subheader("Top 10 performing assets based on Omega ratio:")
        for asset, omega_ratio in ranked_omega:
            st.write(f"{asset}: Omega Ratio = {omega_ratio:.2f}")

        st.subheader("Top 10 performing assets based on Sharpe ratio:")
        for asset, sharpe_ratio in ranked_sharpe:
            st.write(f"{asset}: Sharpe Ratio = {sharpe_ratio:.2f}")

        st.subheader("Top 10 performing assets based on Sortino ratio:")
        for asset, sortino_ratio in ranked_sortino:
            st.write(f"{asset}: Sortino Ratio = {sortino_ratio:.2f}")

        # Display the charts
        display_charts(ranked_omega, ranked_sharpe, ranked_sortino)

if __name__ == "__main__":
    main()
