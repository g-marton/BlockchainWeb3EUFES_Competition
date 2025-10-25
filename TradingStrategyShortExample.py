# Simplified Long-Only Bollinger Bands + RSI Strategy (Streamlit)
# - Single ticker
# - Minimal deps
# - Core signals + ATR-based exits
# - One chart + lightweight backtest (per-trade returns)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BB+RSI Long-Only (Simple)", layout="wide")
st.title("BB+RSI Mean-Reversion • Long-Only (Simplified)")

# -------- Sidebar inputs --------
colA, colB = st.sidebar.columns(2)
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value="MSFT")
lookback_days = st.sidebar.number_input("History (days)", 365, 3650, 1095, 1)
end_date = st.sidebar.date_input("End date", value=dt.date.today())
start_date = end_date - dt.timedelta(days=int(lookback_days))

with colA:
    bb_len = st.number_input("BB length", 5, 100, 20, 1)
    rsi_len = st.number_input("RSI length", 2, 50, 14, 1)
    atr_len = st.number_input("ATR length", 5, 100, 14, 1)
with colB:
    bb_std = st.number_input("BB std", 0.5, 5.0, 2.0, 0.1)
    rsi_low = st.number_input("RSI low (entry)", 5, 50, 30, 1)
    bbw_thr = st.number_input("BB width min", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")

atr_mult = st.sidebar.number_input("Stop-loss ATR x", 0.5, 10.0, 2.0, 0.1)
tp_mult  = st.sidebar.number_input("Take-profit ATR x", 0.5, 10.0, 2.0, 0.1)

# -------- Data --------
@st.cache_data(show_spinner=False)
def get_data(tk, start, end):
    df = yf.download(tk, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df

df = get_data(ticker, start_date, end_date)
if df.empty:
    st.error("No data. Check ticker or date range.")
    st.stop()

# -------- Indicators --------
bb = ta.bbands(df["Close"], length=bb_len, std=bb_std)
rsi = ta.rsi(df["Close"], length=rsi_len).rename("rsi")
atr = ta.atr(df["High"], df["Low"], df["Close"], length=atr_len).rename("atr")

data = pd.concat([df, bb, rsi, atr], axis=1).dropna().copy()
# Standardize column names
data.rename(columns={
    f"BBL_{bb_len}_{float(bb_std):.1f}": "bbl",
    f"BBM_{bb_len}_{float(bb_std):.1f}": "bbm",
    f"BBU_{bb_len}_{float(bb_std):.1f}": "bbh"
}, inplace=True)

data["bb_width"] = (data["bbh"] - data["bbl"]) / data["bbm"]

# -------- Entry rule (long) --------
# Yesterday close < yesterday lower BB
# Yesterday RSI < rsi_low
# Today close > today lower BB
# BB width > threshold
long_signal = (
    (data["Close"].shift(1) < data["bbl"].shift(1)) &
    (data["rsi"].shift(1) < rsi_low) &
    (data["Close"] > data["bbl"]) &
    (data["bb_width"] > bbw_thr)
).astype(int)
data["long_signal"] = long_signal

# -------- Trade simulation: 1 open position at a time; ATR-based SL/TP fixed at entry --------
signal = np.zeros(len(data))
in_pos = False
entry_price = np.nan
entry_atr = np.nan

for i in range(1, len(data)):
    if not in_pos:
        if data["long_signal"].iloc[i] == 1:
            in_pos = True
            entry_price = data["Close"].iloc[i]
            entry_atr = data["atr"].iloc[i]
            signal[i] = 1  # entry
    else:
        stop_price = entry_price - entry_atr * atr_mult
        take_price = entry_price + entry_atr * tp_mult
        px = data["Close"].iloc[i]
        if (px <= stop_price) or (px >= take_price):
            in_pos = False
            signal[i] = -1  # exit
            entry_price = np.nan
            entry_atr = np.nan

data["trade_signal"] = signal
data["entry_px"] = np.where(data["trade_signal"] == 1, data["Close"], np.nan)
data["exit_px"]  = np.where(data["trade_signal"] == -1, data["Close"], np.nan)

# -------- Lightweight per-trade backtest (close-to-close returns) --------
entries = data.index[data["trade_signal"] == 1]
exits   = data.index[data["trade_signal"] == -1]

# Pair entries with the next exit after each entry
trade_returns = []
trade_rows = []

e_ptr, x_ptr = 0, 0
entries = list(entries)
exits   = list(exits)

while e_ptr < len(entries) and x_ptr < len(exits):
    e_idx = entries[e_ptr]
    x_idx = exits[x_ptr]
    if x_idx <= e_idx:
        x_ptr += 1
        continue
    entry_px = float(data.loc[e_idx, "Close"])
    exit_px  = float(data.loc[x_idx, "Close"])
    ret = (exit_px - entry_px) / entry_px
    trade_returns.append(ret)
    trade_rows.append((e_idx, x_idx, entry_px, exit_px, ret))
    e_ptr += 1
    x_ptr += 1

trades_df = pd.DataFrame(trade_rows, columns=["EntryDate","ExitDate","EntryPx","ExitPx","Return"]) if trade_rows else pd.DataFrame(columns=["EntryDate","ExitDate","EntryPx","ExitPx","Return"])

tot_trades = len(trades_df)
wins = (trades_df["Return"] > 0).sum()
win_rate = (wins / tot_trades * 100) if tot_trades else 0.0
tot_return = (np.prod(1 + trades_df["Return"]) - 1) if tot_trades else 0.0
avg_tr = trades_df["Return"].mean() if tot_trades else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Trades", tot_trades)
m2.metric("Win rate", f"{win_rate:.1f}%")
m3.metric("Total return", f"{tot_return*100:.2f}%")
m4.metric("Avg/trade", f"{avg_tr*100:.2f}%")

with st.expander("Show trades"):
    st.dataframe(trades_df.style.format({"EntryPx":"{:.2f}","ExitPx":"{:.2f}","Return":"{:.2%}"}), use_container_width=True)

# -------- Plot --------
plot_df = data.copy()
plot_df["entry_marker"] = np.where(plot_df["trade_signal"] == 1, plot_df["Close"], np.nan)
plot_df["exit_marker"]  = np.where(plot_df["trade_signal"] == -1, plot_df["Close"], np.nan)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
    subplot_titles=(f"{ticker} • Candles + BBands + Entries/Exits", "RSI")
)

# Candles
fig.add_trace(
    go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"],
        name="Price"
    ),
    row=1, col=1
)

# BBands
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bbl"], name="BB Lower", line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bbh"], name="BB Upper", line=dict(width=1)), row=1, col=1)

# Entry/Exit markers
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["entry_marker"], mode="markers",
                         marker=dict(size=9, symbol="triangle-up"), name="Entry"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["exit_marker"], mode="markers",
                         marker=dict(size=9, symbol="x"), name="Exit"), row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi"], name="RSI", line=dict(width=2)), row=2, col=1)
fig.add_hline(y=rsi_low, line_width=1, line_dash="dash", row=2, col=1)

fig.update_layout(xaxis_rangeslider_visible=False, height=800)
st.plotly_chart(fig, use_container_width=True)
