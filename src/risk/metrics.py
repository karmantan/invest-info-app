import numpy as np

def downside_summary(returns):
    r=np.asarray(returns,dtype=float); r=r[np.isfinite(r)]
    if not len(r): return {"sample_size":0}
    return {"sample_size":int(len(r)),"mean":float(np.mean(r)),"median":float(np.median(r)),"p10":float(np.quantile(r,.1)),"p90":float(np.quantile(r,.9)),"worst":float(np.min(r)),"probability_positive":float(np.mean(r>0)),"probability_loss_gt_5pct":float(np.mean(r<-.05))}

def maximum_drawdown(values):
    x=np.asarray(values,dtype=float); peaks=np.maximum.accumulate(x); return float(np.min(x/peaks-1)) if len(x) else 0.0

