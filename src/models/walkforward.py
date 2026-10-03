from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    metrics: dict

def yearly_walk_forward(frame, features, target, model_name="ridge", min_train=252):
    data=frame.dropna(subset=features+[target]).sort_values("date").copy(); years=sorted(data.date.dt.year.unique()); rows=[]
    for year in years[1:]:
        train=data[data.date.dt.year<year]; test=data[data.date.dt.year==year]
        if len(train)<min_train or test.empty: continue
        if train.date.max()>=test.date.min(): raise AssertionError("chronological separation failed")
        model=Ridge(alpha=10) if model_name=="ridge" else HistGradientBoostingRegressor(max_depth=3,learning_rate=.05,max_iter=100,random_state=42)
        model.fit(train[features],train[target]); prediction=model.predict(test[features])
        out=test[["date","ticker",target]].copy(); out["prediction"]=prediction; out["training_end"]=train.date.max(); out["model_name"]=model_name; rows.append(out)
    predictions=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    if predictions.empty: return WalkForwardResult(predictions,{"status":"insufficient_history"})
    y=predictions[target]; p=predictions.prediction
    metrics={"rmse":float(mean_squared_error(y,p)**.5),"direction_accuracy":float((np.sign(y)==np.sign(p)).mean()),"mean_return":float(y.mean()),"mean_predicted":float(p.mean()),"observations":len(y)}
    return WalkForwardResult(predictions,metrics)

