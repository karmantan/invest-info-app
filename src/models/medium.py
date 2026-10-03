from __future__ import annotations
import numpy as np
import pandas as pd
from src.models.real_experiment import _model

def medium_walk_forward(panel,features,horizon,min_train=756):
    target=f"forward_excess_{horizon}d"; target_end=f"target_end_{horizon}d"; raw=f"forward_eur_{horizon}d"; bench=f"spy_forward_{horizon}d"; data=panel.dropna(subset=[target,target_end,raw]).sort_values(["date","ticker"]).copy(); rebalance=set(np.array(sorted(data.date.unique()))[::horizon]); rows=[]
    for year in sorted(data.date.dt.year.unique())[1:]:
        test_year=data[(data.date.dt.year==year)&data.date.isin(rebalance)]; train=data[data[target_end]<test_year.date.min()] if not test_year.empty else data.iloc[:0]
        if len(train)<min_train or test_year.empty: continue
        for name in ("ridge","gradient_boosting"):
            model=_model(name); model.fit(train[features],train[target]); pred=model.predict(test_year[features]); residual=train[target]-model.predict(train[features]); rows.append(test_year[["date",target_end,"ticker",target,raw,bench,"drawdown_252d","fundamental_state","fundamental_coverage"]].assign(model=name,prediction=pred,training_rows=len(train),residual_p10=residual.quantile(.1)))
        mean=train[target].mean(); rows.append(test_year[["date",target_end,"ticker",target,raw,bench,"drawdown_252d","fundamental_state","fundamental_coverage"]].assign(model="unconditional",prediction=mean,training_rows=len(train),residual_p10=train[target].quantile(.1)))
        rows.append(test_year[["date",target_end,"ticker",target,raw,bench,"drawdown_252d","fundamental_state","fundamental_coverage"]].assign(model="momentum",prediction=test_year.eur_return_126d.values-test_year.spy_momentum_63d.values,training_rows=len(train),residual_p10=train[target].quantile(.1)))
        rows.append(test_year[["date",target_end,"ticker",target,raw,bench,"drawdown_252d","fundamental_state","fundamental_coverage"]].assign(model="dip",prediction=-test_year.drawdown_252d.values,training_rows=len(train),residual_p10=train[target].quantile(.1)))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
