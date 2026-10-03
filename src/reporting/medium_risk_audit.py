from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.features.medium import MEDIUM_FEATURES
from src.models.real_experiment import _model
from src.reporting.audit_21d import PERIODS, costs
from src.risk.metrics import maximum_drawdown


POLICIES = ("policy_0_control", "policy_1_two_flags", "policy_2_hard_flags", "policy_3_any_flag", "risk_sizing")
FLAG_COLUMNS = ("high_market_volatility", "high_asset_volatility", "broad_market_downtrend", "extreme_asset_drawdown")


def add_point_in_time_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Pre-specified gates; expanding thresholds use only earlier observations."""
    x = panel.sort_values(["ticker", "date"]).copy()
    x["asset_vol_p80"] = x.groupby("ticker")["volatility_63d"].transform(
        lambda s: s.expanding(min_periods=252).quantile(.8).shift(1)
    )
    vix = x[["date", "vix"]].drop_duplicates("date").sort_values("date")
    vix["vix_p80"] = vix.vix.expanding(min_periods=252).quantile(.8).shift(1)
    x = x.merge(vix[["date", "vix_p80"]], on="date", how="left", validate="many_to_one")
    spy = x[x.ticker == "SPY"][["date", "distance_ma200", "eur_return_63d"]].rename(
        columns={"distance_ma200": "spy_distance_ma200", "eur_return_63d": "spy_return_63d"}
    )
    x = x.merge(spy, on="date", how="left", validate="many_to_one")
    x["high_market_volatility"] = x.vix > x.vix_p80
    x["high_asset_volatility"] = x.volatility_63d > x.asset_vol_p80
    x["broad_market_downtrend"] = (x.spy_distance_ma200 < 0) & (x.spy_return_63d < 0)
    x["extreme_asset_drawdown"] = x.drawdown_252d < -.20
    x["risk_flag_count"] = x[list(FLAG_COLUMNS)].sum(axis=1)
    return x


def policy_weights(x: pd.DataFrame) -> pd.DataFrame:
    hard = x.broad_market_downtrend | x.extreme_asset_drawdown
    out = pd.DataFrame(index=x.index)
    out["policy_0_control"] = 1.0
    out["policy_1_two_flags"] = (x.risk_flag_count < 2).astype(float)
    out["policy_2_hard_flags"] = (~hard).astype(float)
    out["policy_3_any_flag"] = (x.risk_flag_count == 0).astype(float)
    out["risk_sizing"] = np.where(hard | (x.risk_flag_count >= 2), 0.0, np.where(x.risk_flag_count == 1, .5, 1.0))
    return out


def selected_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    context = add_point_in_time_flags(panel)
    pieces = []
    for horizon in (63, 126):
        pred = pd.read_csv(ROOT / f"reports/medium_{horizon}d_predictions.csv", parse_dates=["date", f"target_end_{horizon}d"])
        pred = pred[pred.model.isin(["ridge", "momentum", "gradient_boosting"])].copy()
        pred = pred[pred.groupby(["model", "date"]).prediction.rank(pct=True, method="first") > .66]
        cols = ["date", "ticker", "vix", "volatility_63d", "drawdown_252d", "spy_distance_ma200", "spy_return_63d", "vix_p80", "asset_vol_p80", "risk_flag_count", *FLAG_COLUMNS]
        pred = pred.merge(context[cols], on=["date", "ticker"], how="left", validate="many_to_one")
        urth = panel[panel.ticker == "URTH"][["date", f"forward_eur_{horizon}d"]].rename(columns={f"forward_eur_{horizon}d": "urth_forward"})
        pred = pred.merge(urth, on="date", how="left", validate="many_to_one")
        pred["horizon_days"] = horizon
        pieces.append(pred)
    return pd.concat(pieces, ignore_index=True)


def _metric_row(g: pd.DataFrame, weights: pd.Series, policy: str, group_type="all", group_value="all") -> dict:
    raw = g.raw_return.to_numpy() * weights.to_numpy()
    excess = g.excess_return.to_numpy() * weights.to_numpy()
    p10 = float(np.quantile(raw, .10)); p5 = float(np.quantile(raw, .05))
    by_date = pd.DataFrame({"date": g.date, "return": raw}).groupby("date")["return"].mean().sort_index()
    path = np.r_[1.0, (1 + by_date).cumprod().to_numpy()]
    negative = np.minimum(raw, 0)
    return {
        "horizon_days": int(g.horizon_days.iloc[0]), "model": g.model.iloc[0], "policy": policy,
        "group_type": group_type, "group_value": group_value, "total_signals": len(g),
        "accepted_equivalent": float(weights.sum()), "accepted_nonzero": int((weights > 0).sum()),
        "rejected": int((weights == 0).sum()), "acceptance_rate": float((weights > 0).mean()),
        "mean_raw_return": raw.mean(), "median_raw_return": np.median(raw),
        "mean_excess_return": excess.mean(), "median_excess_return": np.median(excess),
        "probability_positive": np.mean(raw > 0), "probability_outperform": np.mean(excess > 0),
        "probability_loss_gt_5pct": np.mean(raw < -.05), "probability_loss_gt_10pct": np.mean(raw < -.10),
        "probability_loss_gt_15pct": np.mean(raw < -.15), "p25": np.quantile(raw, .25), "p10": p10,
        "p5": p5, "worst": raw.min(), "sequential_max_drawdown": maximum_drawdown(path),
        "downside_deviation": np.sqrt(np.mean(negative ** 2)),
        "expected_shortfall_below_p10": raw[raw <= p10].mean(),
    }


def policy_results(selected: pd.DataFrame):
    summaries=[]; accepted_rejected=[]; by_period=[]; by_asset=[]; recent=[]
    for (horizon, model), g in selected.groupby(["horizon_days", "model"]):
        weights = policy_weights(g)
        control_mean = g.excess_return.mean()
        control_p10 = g.raw_return.quantile(.10)
        for policy in POLICIES:
            w=weights[policy]
            row=_metric_row(g,w,policy)
            row["excess_return_preserved"] = row["mean_excess_return"] / control_mean if control_mean else np.nan
            row["p10_improvement"] = row["p10"] - control_p10
            summaries.append(row)
            for status,mask in (("accepted",w>0),("rejected",w==0)):
                q=g[mask]
                if len(q): accepted_rejected.append(_metric_row(q,pd.Series(1.,index=q.index),policy,"decision",status))
            for lo,hi,label in PERIODS:
                q=g[g.date.dt.year.between(lo,hi)]; qw=w.loc[q.index]
                if len(q): by_period.append(_metric_row(q,qw,policy,"period",label))
            for ticker,q in g.groupby("ticker"):
                by_asset.append(_metric_row(q,w.loc[q.index],policy,"asset",ticker))
            for label,mask in (("pre_2020",g.date.dt.year<2020),("2020_plus",g.date.dt.year>=2020),("full_history",pd.Series(True,index=g.index))):
                q=g[mask]; recent.append(_metric_row(q,w.loc[q.index],policy,"era",label))
    return tuple(pd.DataFrame(v) for v in (summaries,accepted_rejected,by_period,by_asset,recent))


def individual_gates(selected: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (horizon,model),g in selected.groupby(["horizon_days","model"]):
        for flag in FLAG_COLUMNS:
            w=(~g[flag]).astype(float); row=_metric_row(g,w,flag); row["flagged"] = int(g[flag].sum()); rows.append(row)
    return pd.DataFrame(rows)


def economic_results(selected: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    euro=[]; passive=[]; sleeves=[]
    for (horizon,model),g in selected.groupby(["horizon_days","model"]):
        weights=policy_weights(g)
        for policy in POLICIES:
            w=weights[policy]
            effective_raw=g.raw_return*w
            for size in (500,1000,2000,3000):
                active_sizes=size*w
                model_cost=np.array([costs(s,cfg)[0] if s>0 else 0 for s in active_sizes])
                p10=effective_raw.quantile(.1); p5=effective_raw.quantile(.05)
                euro.append({"horizon_days":horizon,"model":model,"policy":policy,"allocation_eur":size,"signals":len(g),"expected_euro_gain":effective_raw.mean()*size,"p10_euro":p10*size,"p5_euro":p5*size,"worst_euro":effective_raw.min()*size,"mean_model_cost_eur":model_cost.mean()})
                for benchmark,col in (("SPY",f"spy_forward_{horizon}d"),("URTH","urth_forward")):
                    valid=g[col].notna(); q=g[valid]; qw=w[valid]; q_sizes=size*qw
                    mc=np.array([costs(s,cfg)[0] if s>0 else 0 for s in q_sizes]); pc=np.array([costs(s,cfg)[1] if s>0 else 0 for s in q_sizes])
                    model_profit=(q.raw_return*qw*size-mc).sum(); benchmark_profit=(q[col]*qw*size-pc).sum()
                    passive.append({"horizon_days":horizon,"model":model,"policy":policy,"benchmark":benchmark,"allocation_eur":size,"signals":len(q),"model_net_profit_eur":model_profit,"passive_net_profit_eur":benchmark_profit,"tactical_net_value_added_eur":model_profit-benchmark_profit,"wealth_value_added_eur":((q.raw_return-q[col])*qw*size-mc).sum()})
            raw=g.raw_return*w
            for sleeve in (.05,.075,.10):
                sleeves.append({"horizon_days":horizon,"model":model,"policy":policy,"tactical_sleeve":sleeve,"portfolio_p10_impact":raw.quantile(.1)*sleeve,"portfolio_p5_impact":raw.quantile(.05)*sleeve,"portfolio_worst_impact":raw.min()*sleeve})
    euro=pd.DataFrame(euro); passive=pd.DataFrame(passive)
    # Attach passive expectation for the economic-size display without implying advice.
    spy=passive[passive.benchmark=="SPY"][["horizon_days","model","policy","allocation_eur","signals","passive_net_profit_eur","tactical_net_value_added_eur"]].copy()
    spy["expected_passive_net_eur"] = spy.passive_net_profit_eur / spy.signals
    spy["expected_value_added_eur"] = spy.tactical_net_value_added_eur / spy.signals
    spy=spy.drop(columns=["signals","passive_net_profit_eur","tactical_net_value_added_eur"])
    euro=euro.merge(spy,on=["horizon_days","model","policy","allocation_eur"],how="left")
    euro["expected_euro_gain_after_cost"] = euro.expected_euro_gain - euro.mean_model_cost_eur
    return euro,passive,pd.DataFrame(sleeves)


def current_scanner(panel: pd.DataFrame) -> pd.DataFrame:
    context=add_point_in_time_flags(panel); latest=context.sort_values("date").groupby("ticker").tail(1).copy(); as_of=latest.date.max()
    rows=[]
    for horizon in (63,126):
        target=f"forward_excess_{horizon}d"; end=f"target_end_{horizon}d"
        train=context[(context[end]<as_of)&context[target].notna()].copy()
        model=_model("ridge"); model.fit(train[MEDIUM_FEATURES],train[target]); candidates=latest.copy(); candidates["prediction"]=model.predict(candidates[MEDIUM_FEATURES]); candidates["rank"]=candidates.prediction.rank(pct=True,method="first")
        history=pd.read_csv(ROOT/f"reports/medium_{horizon}d_predictions.csv",parse_dates=["date"]); history=history[history.model=="ridge"]; history=history[history.groupby("date").prediction.rank(pct=True,method="first")>.66]
        for _,r in candidates.iterrows():
            interesting=r["rank"]>.66; flags=[name for name in FLAG_COLUMNS if bool(r[name])]; hard=r.broad_market_downtrend or r.extreme_asset_drawdown
            status="NO MODEL EDGE" if not interesting else "MODEL INTERESTING — RISK GATE PASSES" if not flags else "MODEL INTERESTING — MARKET DOWNTREND VETO" if r.broad_market_downtrend else "MODEL INTERESTING — EXTREME DRAWDOWN VETO" if r.extreme_asset_drawdown else "MODEL INTERESTING — SOFT VOLATILITY FLAG"
            asset_hist=history[history.ticker==r.ticker][f"forward_eur_{horizon}d"]
            rows.append({"asset":r.ticker,"as_of_date":str(r.date.date()),"horizon_days":horizon,"forecast_excess":r.prediction,"forecast_status":"TOP COHORT" if interesting else "NOT TOP COHORT","active_risk_flags":", ".join(flags) or "none","policy_1_result":"REJECT" if r.risk_flag_count>=2 else "PASS","policy_2_result":"REJECT" if hard else "PASS","policy_3_result":"REJECT" if flags else "PASS","risk_sizing":"0%" if hard or r.risk_flag_count>=2 else "50%" if r.risk_flag_count==1 else "100%","historical_p10":asset_hist.quantile(.1),"historical_p5":asset_hist.quantile(.05),"large_loss_probability":(asset_hist<-.10).mean() if len(asset_hist) else np.nan,"confidence":"LOW" if len(asset_hist)<20 else "MEDIUM","research_status":status})
    return pd.DataFrame(rows)


def run():
    cfg=load_config(); meta=json.loads((ROOT/"reports/medium_research_metadata.json").read_text())
    if meta["specification_hash"] != "c6abcd79817cba9d0912bdc67f67c5f3827990dd684e522393a325b59082be47":
        raise RuntimeError("Frozen medium-term specification hash mismatch")
    panel=pd.read_parquet(ROOT/"data/processed/medium_panel.parquet"); panel.date=pd.to_datetime(panel.date)
    selected=selected_predictions(panel); selected["raw_return"]=selected.apply(lambda r:r[f"forward_eur_{int(r.horizon_days)}d"],axis=1); selected["excess_return"]=selected.apply(lambda r:r[f"forward_excess_{int(r.horizon_days)}d"],axis=1)
    summary,decisions,periods,assets,recent=policy_results(selected); gates=individual_gates(selected); euro,passive,sleeves=economic_results(selected,cfg); scan=current_scanner(panel)
    outputs={"medium_risk_policy_summary.csv":summary,"medium_risk_individual_gates.csv":gates,"medium_risk_accepted_rejected.csv":decisions,"medium_risk_periods.csv":periods,"medium_risk_assets.csv":assets,"medium_risk_recent_dependence.csv":recent,"medium_risk_economic_sizes.csv":euro,"medium_risk_passive_counterfactuals.csv":passive,"medium_risk_sleeve_impact.csv":sleeves,"medium_risk_current_scanner.csv":scan}
    for name,frame in outputs.items(): frame.to_csv(ROOT/"reports"/name,index=False)
    spec={"frozen_medium_specification_hash":meta["specification_hash"],"models":["ridge","momentum","gradient_boosting"],"gates":{"A":"VIX above prior expanding-history P80","B":"asset realized volatility above prior expanding-history P80","C":"SPY below MA200 and 63-day momentum negative","D":"asset drawdown below -20%"},"policies":{"0":"no gate","1":"reject with two or more flags","2":"reject C or D","3":"reject any flag","risk_sizing":"100% no flags; 50% one soft flag; 0% two flags or C/D"}}
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(); audit_meta={"experiment_id":f"medium-risk-{digest[:12]}","mode":"real_medium_risk_research","generated_timestamp_utc":datetime.now(timezone.utc).isoformat(),"frozen_medium_specification_hash":meta["specification_hash"],"risk_specification_hash":digest,"specification":spec,"recommendation_status":"RESEARCH ONLY — LIVE RECOMMENDATIONS DISABLED","artifacts":list(outputs)}
    (ROOT/"reports/medium_risk_metadata.json").write_text(json.dumps(audit_meta,indent=2))
    print(json.dumps(audit_meta,indent=2)); print("\nRIDGE POLICIES\n",summary[summary.model=="ridge"].to_string(index=False)); print("\nCURRENT SCANNER\n",scan.to_string(index=False))


if __name__ == "__main__": run()
