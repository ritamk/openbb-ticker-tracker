import yfinance as yf
import pandas as pd
import pandas_ta as ta
import json

def main(symbol: str = "RELIANCE.NS"):
    # NSE; use .BO for BSE
    ticker = yf.Ticker(symbol)
    df = ticker.history(
        start="2024-01-01",
        interval="1d",
        auto_adjust=True, # Adjust for splits/dividends
        prepost=False # Exclude pre/post market data
    )
    # Ensure time order and datetime index regardless of input format
    if "date" in df.columns:
        df = df.sort_values("date").set_index("date")
    else:
        # 'date' is likely already the index
        df = df.sort_index()

    # Make sure index is named 'date' for consistency
    if df.index.name != "date":
        df.index.name = "date"

    # Core indicators (customize as needed)
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = df.join(macd)  # columns: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)
    bb = ta.bbands(df["close"], length=20, std=2.0)
    df = df.join(bb)  # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    df = df.join(stoch)  # STOCHk_14_3_3, STOCHd_14_3_3
    df["adx_14"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["obv"] = ta.obv(df["close"], df["volume"])

    # Prevent lookahead leakage: shift features by 1 bar
    feature_cols = [c for c in df.columns if c not in ["open","high","low","close","volume","vwap","split_ratio","dividend"]]
    df[feature_cols] = df[feature_cols].shift(1)

    # Clean
    df = df.dropna()

    # If feeding to an LLM: keep compact JSON-friendly records
    records = df[["open","high","low","close","volume"] + feature_cols].tail(256).reset_index()
    payload = records.to_dict(orient="records")  # pass this to your LLM
    print(f"Prepared {len(payload)} rows with TA features for {symbol}")

    with open("ta_payload.json", "w") as f:
        json.dump(payload[-64:], f, default=str)  # keep the prompt small
    print("Wrote ta_payload.json")