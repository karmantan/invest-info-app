import pandas as pd
import numpy as np
from src.events.study import add_event_features, EVENT_FEATURES

def test_event_features_use_only_events_on_or_before_prediction_date():
    dates=pd.bdate_range("2020-01-01",periods=10); panel=pd.DataFrame({"date":dates,"ticker":"SMH"})
    events=pd.DataFrame({"event_type":["major_earnings"],"etf":["SMH"],"effective_market_date":[dates[5]],"event_day_etf_return":[.02],"target_rate_change":[np.nan],"two_year_yield_event_change":[np.nan],"ten_year_yield_event_change":[np.nan],"vix_event_change":[np.nan]})
    out=add_event_features(panel,events)
    assert out.loc[4,"earnings_event_today"]==0
    assert out.loc[4,"days_since_major_earnings"]==365
    assert out.loc[5,"earnings_event_today"]==1
    assert out.loc[5,"days_since_major_earnings"]==0

def test_event_feature_set_is_small_and_prespecified():
    assert len(EVENT_FEATURES)==11
