from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.risk.metrics import maximum_drawdown

def _model(name):
    if name=="ridge": return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=10))
    if name=="gradient_boosting": return make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingRegressor(max_depth=3,learning_rate=.05,max_iter=100,l2_regularization=1,random_state=42))
    raise ValueError(name)

def strict_walk_forward(panel, features, target="forward_excess_21d", horizon=21, min_train=756):
    data=panel.dropna(subset=[target,"target_end_date","eur_return_21d"]).sort_values(["date","ticker"]).copy(); output=[]
    rebalance_dates=set(np.array(sorted(data.date.unique()))[::horizon])
    years=sorted(data.date.dt.year.unique())
    for year in years[1:]:
        test_all=data[data.date.dt.year==year]; train=data[data.target_end_date<test_all.date.min()]
        if len(train)<min_train or test_all.empty: continue
        # One rebalance cohort every 21 benchmark trading dates prevents overlapping test outcomes.
        test=test_all[test_all.date.isin(rebalance_dates)]
        clean=train.dropna(subset=features)
        if len(clean)<min_train: continue
        residual_inputs={}
        for name in ("ridge","gradient_boosting"):
            model=_model(name); model.fit(clean[features],clean[target]); pred=model.predict(test[features])
            insample=train[target]-model.predict(train[features]); residual_inputs[name]=insample
            p10=float(insample.quantile(.1)); prob=float((insample<-.05).mean())
            output.append(test[["date","target_end_date","ticker",target,"forward_eur_21d","benchmark_forward_eur_21d"]].assign(model=name,prediction=pred,training_end=train.target_end_date.max(),training_rows=len(train),residual_p10=p10,residual_prob_loss5=prob))
        unconditional=float(train[target].mean())
        output.append(test[["date","target_end_date","ticker",target,"forward_eur_21d","benchmark_forward_eur_21d"]].assign(model="unconditional",prediction=unconditional,training_end=train.target_end_date.max(),training_rows=len(train),residual_p10=float(train[target].quantile(.1)),residual_prob_loss5=float((train[target]<-.05).mean())))
        output.append(test[["date","target_end_date","ticker",target,"forward_eur_21d","benchmark_forward_eur_21d"]].assign(model="momentum",prediction=test.eur_return_21d.values,training_end=train.target_end_date.max(),training_rows=len(train),residual_p10=float(train[target].quantile(.1)),residual_prob_loss5=float((train[target]<-.05).mean())))
    return pd.concat(output,ignore_index=True) if output else pd.DataFrame()

def statistical_metrics(predictions):
    rows=[]
    for name,x in predictions.groupby("model"):
        err=x.prediction-x.forward_excess_21d
        rows.append({"model":name,"observations":len(x),"mean_prediction_error":float(err.mean()),"mae":float(mean_absolute_error(x.forward_excess_21d,x.prediction)),"rmse":float(mean_squared_error(x.forward_excess_21d,x.prediction)**.5),"correlation":float(x.prediction.corr(x.forward_excess_21d)),"directional_accuracy":float((np.sign(x.prediction)==np.sign(x.forward_excess_21d)).mean())})
    return pd.DataFrame(rows)

def ranked_results(predictions, nominal_eur, buy_fee, sell_fee, spread_bps):
    cost_return=(buy_fee+sell_fee)/nominal_eur+2*spread_bps/10000; rows=[]; selected=[]
    for name,x in predictions.groupby("model"):
        ranks=x.groupby("date").prediction.rank(pct=True,method="first"); top=x[ranks>.66].copy(); top["after_cost_excess"]=top.forward_excess_21d-cost_return
        equity=(1+top.groupby("date").after_cost_excess.mean()).cumprod()
        rows.append({"model":name,"signals":len(top),"mean_asset_return":float(top.forward_eur_21d.mean()),"mean_excess":float(top.forward_excess_21d.mean()),"median_excess":float(top.forward_excess_21d.median()),"probability_outperform":float((top.forward_excess_21d>0).mean()),"p10_excess":float(top.forward_excess_21d.quantile(.1)),"max_drawdown_after_cost":maximum_drawdown(equity),"turnover_round_trips":len(top),"model_gross_profit_eur":float((top.forward_eur_21d*nominal_eur).sum()),"passive_gross_profit_eur":float((top.benchmark_forward_eur_21d*nominal_eur).sum()),"gross_value_added_eur":float((top.forward_excess_21d*nominal_eur).sum()),"costs_eur":float(len(top)*nominal_eur*cost_return),"model_after_cost_profit_eur":float(((top.forward_eur_21d-cost_return)*nominal_eur).sum()),"after_cost_value_added_eur":float((top.after_cost_excess*nominal_eur).sum())})
        selected.append(top.assign(selection="top_bucket"))
    return pd.DataFrame(rows),pd.concat(selected,ignore_index=True)

def bucket_results(predictions):
    rows=[]
    for name,x in predictions.groupby("model"):
        x=x.copy(); x["bucket"]=x.groupby("date").prediction.transform(lambda s:pd.qcut(s.rank(method="first"),3,labels=["lowest","middle","highest"]))
        for bucket,b in x.groupby("bucket",observed=True): rows.append({"model":name,"bucket":str(bucket),"observations":len(b),"average_return":float(b.forward_eur_21d.mean()),"average_excess":float(b.forward_excess_21d.mean()),"median_excess":float(b.forward_excess_21d.median()),"probability_outperform":float((b.forward_excess_21d>0).mean()),"p10_excess":float(b.forward_excess_21d.quantile(.1))})
    return pd.DataFrame(rows)

def leave_one_asset_out(panel, features, target="forward_excess_21d", horizon=21, min_train=756):
    """Frozen pooled specification, excluding the predicted ticker from every training fold."""
    data=panel.dropna(subset=[target,"target_end_date","eur_return_21d"]).sort_values(["date","ticker"]).copy(); output=[]
    rebalance_dates=set(np.array(sorted(data.date.unique()))[::horizon])
    for year in sorted(data.date.dt.year.unique())[1:]:
        test_year=data[(data.date.dt.year==year)&data.date.isin(rebalance_dates)]
        for ticker,test in test_year.groupby("ticker"):
            train=data[(data.target_end_date<test_year.date.min())&(data.ticker!=ticker)].dropna(subset=features)
            if len(train)<min_train: continue
            for name in ("ridge","gradient_boosting"):
                model=_model(name); model.fit(train[features],train[target]); pred=model.predict(test[features])
                output.append(test[["date","target_end_date","ticker",target,"forward_eur_21d","benchmark_forward_eur_21d"]].assign(model=name,prediction=pred,training_end=train.target_end_date.max(),training_rows=len(train),validation="leave_one_asset_out"))
    return pd.concat(output,ignore_index=True) if output else pd.DataFrame()
