import datetime
from datetime import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="ORB Performance Tracker", page_icon="📈", layout="wide"
)

st.title("🚀 Opening Range Breakout (ORB) Performance Dashboard")
st.markdown(
    "Analyze the probability of price **respecting** or **disrespecting** breakout levels following an Opening Range Breakout."
)

# --- Sidebar Inputs ---
st.sidebar.header("1. Configuration")

# 1. Select Symbol
symbol = st.sidebar.text_input(
    "Ticker Symbol", value="AAPL", help="e.g., AAPL, TSLA, SPY, QQQ"
)

# 2. Select Timeframe
# Note: yfinance intraday data has limits (e.g., 1m data is limited to last 7 days, 5m/15m up to 60 days)
timeframe_options = {"1 Minute": "1m", "5 Minutes": "5m", "15 Minutes": "15m"}
selected_tf_label = st.sidebar.selectbox(
    "Select Timeframe",
    options=list(timeframe_options.keys()),
    index=1,  # Default to 5m
)
timeframe = timeframe_options[selected_tf_label]

# Set data period based on timeframe limitations
if timeframe == "1m":
    period = "7d"
elif timeframe in ["5m", "15m"]:
    period = "59d"
else:
    period = "max"

# 3. Specify Opening Range Time Range
st.sidebar.subheader("2. Opening Range (OR) Settings")
or_start_time = st.sidebar.time_input(
    "OR Start Time", value=time(9, 30)
)  # Default US Market Open
or_duration_minutes = st.sidebar.selectbox(
    "OR Duration (Minutes)", options=[5, 15, 30, 60], index=1
)

# 4. Specify Performance Tracking Time
st.sidebar.subheader("3. Tracking & Evaluation Window")
tracking_end_time = st.sidebar.time_input(
    "Evaluation End Time", value=time(16, 0)
)  # Default US Market Close
breakout_definition = st.sidebar.selectbox(
    "Breakout Trigger",
    options=["Close crosses OR boundary", "High/Low breaches OR boundary"],
    index=0,
)


@st.cache_data(ttl=3600)
def load_data(ticker, period, interval):
  """Fetch data from yfinance"""
  try:
    df = yf.download(
        ticker, period=period, interval=interval, progress=False, auto_adjust=True
    )
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    return df
  except Exception as e:
    st.error(f"Error loading data: {e}")
    return pd.DataFrame()


# Load Data
data_load_state = st.text("Loading market data...")
df = load_data(symbol, period, timeframe)
data_load_state.empty()

if df.empty:
  st.warning(
      "No data found. Try a different symbol or check your internet connection."
  )
else:
  # Ensure index is datetime and localized/converted correctly
  if df.index.tz is None:
    df.index = pd.to_datetime(df.index).tz_localize("UTC")
  else:
    df.index = pd.to_datetime(df.index)

  # Convert to US/Eastern (assuming US equities standard for ORB)
  try:
    df.index = df.index.tz_convert("US/Eastern")
  except Exception:
    pass

  # --- Core ORB Calculation Engine ---
  results = []

  # Group by trading day
  grouped = df.groupby(df.index.date)

  for date, group in grouped:
    # Filter for the opening range period
    market_open = datetime.datetime.combine(date, or_start_time)
    if group.index.tz is not None:
      market_open = pd.Timestamp(market_open).tz_localize(group.index.tz)

    market_end_dt = market_open + datetime.timedelta(
        minutes=or_duration_minutes
    )

    # Opening Range slice
    or_slice = group[(group.index >= market_open) & (group.index < market_end_dt)]

    if len(or_slice) < 2:
      continue  # Not enough data for this day

    or_high = or_slice["High"].max()
    or_low = or_slice["Min"].min() if "Min" in or_slice.columns else or_slice["Low"].min()

    # Tracking window slice (from end of OR to Tracking End Time)
    eval_end_dt = datetime.datetime.combine(date, tracking_end_time)
    if group.index.tz is not None:
      eval_end_dt = pd.Timestamp(eval_end_dt).tz_localize(group.index.tz)

    tracking_slice = group[
        (group.index >= market_end_dt) & (group.index <= eval_end_dt)
    ]

    if tracking_slice.empty:
      continue

    # Check for breakouts and follow-through behavior
    # We check row by row in the tracking window
    breakout_type = None  # 'UP' or 'DOWN'
    breakout_idx = None

    for idx, row in tracking_slice.iterrows():
      if breakout_definition == "Close crosses OR boundary":
        cond_up = row["Close"] > or_high
        cond_down = row["Close"] < or_low
      else:  # High/Low breaches
        cond_up = row["High"] > or_high
        cond_down = row["Low"] < or_low

      if cond_up:
        breakout_type = "UP"
        breakout_idx = idx
        break
      elif cond_down:
        breakout_type = "DOWN"
        breakout_idx = idx
        break

    if breakout_type:
      # Evaluate if price "respected" or "disrespected" the breakout
      # Respected = Moved further in the breakout direction without re-entering the opposite side significantly.
      # Disrespected = FAILED breakout (re-crossed back into or past the opposing boundary).
      sub_tracking = tracking_slice[tracking_slice.index >= breakout_idx]

      respected = True
      max_extension = 0.0

      if breakout_type == "UP":
        max_extension = sub_tracking["High"].max() - or_high
        # Disrespected if price falls back below the OR low or back inside the range significantly
        if (sub_tracking["Low"] < or_low).any():
          respected = False
      elif breakout_type == "DOWN":
        max_extension = or_low - sub_tracking["Low"].min()
        if (sub_tracking["High"] > or_high).any():
          respected = False

      results.append({
          "Date": date,
          "OR_High": or_high,
          "OR_Low": or_low,
          "Breakout": breakout_type,
          "Respected": respected,
          "Max_Extension": max_extension,
      })

  results_df = pd.DataFrame(results)

  # --- Display Dashboard Metrics ---
  if results_df.empty:
    st.info(
        "No breakouts detected with the current parameters. Try adjusting the"
        " timeframe, OR duration, or symbol."
    )
  else:
    total_days = len(grouped)
    total_breakouts = len(results_df)
    up_breakouts = len(results_df[results_df["Breakout"] == "UP"])
    down_breakouts = len(results_df[results_df["Breakout"] == "DOWN"])

    respected_count = len(results_df[results_df["Respected"] == True])
    disrespected_count = len(results_df[results_df["Respected"] == False])

    prob_respect = (
        (respected_count / total_breakouts) * 100 if total_breakouts > 0 else 0
    )
    prob_disrespect = (
        (disrespected_count / total_breakouts) * 100
        if total_breakouts > 0
        else 0
    )

    st.subheader(f"📊 Performance Metrics for {symbol.upper()}")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Days Analyzed", total_days)
    col2.metric("Breakouts Recorded", total_breakouts)
    col3.metric("Respect Rate (Success)", f"{prob_respect:.1f}%")
    col4.metric("Disrespect Rate (Failure)", f"{prob_disrespect:.1f}%")
    col5.metric(
        "Breakout Frequency",
        f"{(total_breakouts / total_days * 100 if total_days > 0 else 0):.1f}%",
    )

    # Breakdown by Direction
    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
      st.markdown("### 🟢 Upward Breakouts")
      up_df = results_df[results_df["Breakout"] == "UP"]
      if not up_df.empty:
        up_respect = (
            len(up_df[up_df["Respected"] == True]) / len(up_df)
        ) * 100
        st.write(f"Total Up Breakouts: **{len(up_df)}**")
        st.write(f"Respect Probability: **{up_respect:.1f}%**")
      else:
        st.write("No upward breakouts recorded.")

    with col_b:
      st.markdown("### 🔴 Downward Breakouts")
      down_df = results_df[results_df["Breakout"] == "DOWN"]
      if not down_df.empty:
        down_respect = (
            len(down_df[down_df["Respected"] == True]) / len(down_df)
        ) * 100
        st.write(f"Total Down Breakouts: **{len(down_df)}**")
        st.write(f"Respect Probability: **{down_respect:.1f}%**")
      else:
        st.write("No downward breakouts recorded.")

    # --- Data Table & Visualization ---
    st.markdown("---")
    st.subheader("📋 Historical Breakdown Log")
    st.dataframe(results_df, use_container_width=True)

    # Optional: Plot sample chart for the most recent day
    st.markdown("---")
    st.subheader("📈 Latest Day Intraday Chart & Opening Range")
    latest_date = df.index[-1].date()
    latest_group = df[df.index.date == latest_date]

    if not latest_group.empty:
      fig = go.Figure()
      fig.add_trace(
          go.Candlestick(
              x=latest_group.index,
              open=latest_group["Open"],
              high=latest_group["High"],
              low=latest_group["Low"],
              close=latest_group["Close"],
              name="Price",
          )
      )

      # Find OR lines for latest day if available
      market_open_latest = datetime.datetime.combine(latest_date, or_start_time)
      if latest_group.index.tz is not None:
        market_open_latest = pd.Timestamp(market_open_latest).tz_localize(
            latest_group.index.tz
        )
      market_end_latest = market_open_latest + datetime.timedelta(
          minutes=or_duration_minutes
      )
      or_slice_latest = latest_group[
          (latest_group.index >= market_open_latest)
          & (latest_group.index < market_end_latest)
      ]

      if not or_slice_latest.empty:
        orh = or_slice_latest["High"].max()
        or_l = or_slice_latest["Low"].min()

        fig.add_hline(
            y=orh,
            line_dash="dash",
            line_color="green",
            annotation_text=f"OR High ({orh})",
        )
        fig.add_hline(
            y=or_l,
            line_dash="dash",
            line_color="red",
            annotation_text=f"OR Low ({or_l})",
        )

      fig.update_layout(
          title=f"{symbol.upper()} Intraday Action on {latest_date}",
          xaxis_title="Time",
          yaxis_title="Price",
          template="plotly_dark",
          height=500,
      )
      st.plotly_chart(fig, use_container_width=True)