from __future__ import annotations
import numpy as np
import pandas as pd

def build_features(prices: pd.DataFrame, rates: pd.DataFrame | None = None, horizon: int = 21) -> pd.DataFrame:
    df=prices.copy().sort_values("date"); df["date"]=pd.to_datetime(df["date"])
    if df.duplicated(["ticker","date"]).any(): raise ValueError("duplicate ticker/date observations")
    grouped=df.groupby("ticker",group_keys=False)
    for days in (1,5,21,63): df[f"return_{days}d"]=grouped["close"].pct_change(days)
    df["drawdown_63d"]=df["close"]/grouped["close"].transform(lambda s:s.rolling(63,min_periods=20).max())-1
    daily=grouped["close"].pct_change(); df["volatility_21d"]=daily.groupby(df["ticker"]).transform(lambda s:s.rolling(21,min_periods=15).std()*np.sqrt(252))
    df[f"forward_return_{horizon}d"]=grouped["close"].shift(-horizon)/df["close"]-1
    if rates is not None and not rates.empty:
        rates=rates.copy(); rates["date"]=pd.to_datetime(rates["date"]); df=pd.merge_asof(df.sort_values("date"),rates.sort_values("date"),on="date",direction="backward",tolerance=pd.Timedelta("7d"))
    return df

def assert_point_in_time(frame: pd.DataFrame, feature_columns: list[str], target_col: str):
    if frame[feature_columns].shift(-1).equals(frame[feature_columns]): raise AssertionError("features appear shifted from the future")
    if target_col in feature_columns: raise AssertionError("target cannot be a feature")

