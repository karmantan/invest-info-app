import pandas as pd
from src.features import build_features

def test_missing_trading_dates_do_not_create_calendar_fill():
    p=pd.DataFrame({"date":pd.to_datetime(["2025-01-02","2025-01-03","2025-01-07"]),"ticker":"X","close":[100,101,102]})
    out=build_features(p,horizon=1); assert len(out)==3

def test_benchmark_alignment_is_date_explicit():
    model=pd.DataFrame({"date":pd.to_datetime(["2025-01-02","2025-01-03"]),"r":[.1,.2]}); benchmark=pd.DataFrame({"date":pd.to_datetime(["2025-01-03"]),"b":[.05]})
    aligned=model.merge(benchmark,on="date",how="inner",validate="one_to_one"); assert aligned.date.iloc[0]==pd.Timestamp("2025-01-03")

