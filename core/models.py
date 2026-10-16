"""ML layer: demand forecasting + no-show / cancellation models.

Every model is evaluated on a time-based split (last 20 % of dates held out)
against a simple baseline. If a model does not beat its baseline, the app uses
the baseline — the ranking engine never depends on an unproven model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

from .data import Campus, WEEKDAYS


def _time_split(dates: pd.Series, frac: float = 0.8):
    cut = dates.sort_values().iloc[int(len(dates) * frac)]
    return dates < cut, dates >= cut, cut


# ----------------------------------------------------------------------------- demand forecast
def demand_forecast(c: Campus, horizon: int = 14) -> dict:
    daily = c.requests.groupby("request_date").size().reindex(c.dates, fill_value=0).rename("requests").to_frame()
    daily["wd"] = daily.index.weekday
    daily["t"] = np.arange(len(daily))
    tr, te, cut = _time_split(daily.index.to_series())
    tr, te = tr.values, te.values

    base_pred = np.full(te.sum(), daily.requests[tr].mean())
    wd_mean = daily[tr].groupby("wd").requests.mean()
    seas_pred = daily[te].wd.map(wd_mean).values
    X = pd.get_dummies(daily.wd.astype(str)).assign(t=daily.t)
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0)
    gb.fit(X[tr], daily.requests[tr])
    gb_pred = gb.predict(X[te])
    scores = {"Naive mean": mean_absolute_error(daily.requests[te], base_pred),
              "Weekday profile": mean_absolute_error(daily.requests[te], seas_pred),
              "Gradient boosting": mean_absolute_error(daily.requests[te], gb_pred)}

    future = pd.date_range(c.dates[-1] + pd.Timedelta(days=1), periods=horizon)
    wd_all = daily.groupby("wd").requests.mean()
    fc = pd.DataFrame({"date": future, "forecast": future.weekday.map(wd_all).values})
    resid = (daily.requests[te] - seas_pred).std()
    fc["lo"], fc["hi"] = (fc.forecast - 1.28 * resid).clip(lower=0), fc.forecast + 1.28 * resid

    # expected requests per weekday x hour (requests + recurring classes = load)
    hourly = (c.requests.groupby(["wd", "start_hour"]).size()
              / c.requests.groupby("wd").request_date.nunique()).unstack(fill_value=0)
    hourly.index = [WEEKDAYS[i] for i in hourly.index]
    hourly.columns = [f"{h:02d}:00" for h in hourly.columns]
    return dict(daily=daily, cut=cut, scores=scores, forecast=fc, hourly=hourly,
                test_frame=pd.DataFrame({"date": daily.index[te], "actual": daily.requests[te].values,
                                         "weekday_profile": seas_pred, "gbm": gb_pred}))


# ----------------------------------------------------------------------------- no-show / attendance
def attendance_model(c: Campus) -> dict:
    ev = c.events.copy()
    X = pd.get_dummies(ev[["event_type", "room_type"]]).assign(
        hour=ev.hour, wd=ev.wd, duration=ev.duration_hours, expected=ev.expected_attendance, capacity=ev.capacity)
    y = ev.actual_attendance
    tr, te, cut = _time_split(ev.start_time)
    ratio = (ev.actual_attendance / ev.expected_attendance)[tr].median()
    base = ev.expected_attendance[te] * ratio
    mean_base = np.full(te.sum(), y[tr].mean())
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=0).fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    low = (ev.actual_attendance < 0.5 * ev.expected_attendance)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=0).fit(X[tr], low[tr])
    p = clf.predict_proba(X[te])[:, 1]
    return dict(mae_model=mean_absolute_error(y[te], pred), mae_base=mean_absolute_error(y[te], base),
                mae_mean=mean_absolute_error(y[te], mean_base),
                auc_expected_only=roc_auc_score(low[te], ev.expected_attendance[te]),
                auc_low_turnout=roc_auc_score(low[te], p), low_rate=low.mean(), cut=cut,
                by_type=ev.groupby("event_type").low_turnout.mean().sort_values(),
                importance=pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False).head(8))


def cancellation_model(c: Campus) -> dict:
    q = c.requests.copy()
    X = pd.get_dummies(q[["purpose"]]).assign(
        start=q.start_hour, dur=q.duration_hours, cap=q.required_capacity, proj=q.projector_required,
        comp=q.computers_required, wd=q.wd, month=q.request_date.dt.month)
    y = (q.status == "Cancelled").astype(int)
    tr, te, cut = _time_split(q.request_date)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=0, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    return dict(auc=roc_auc_score(y[te], p), f1=f1_score(y[te], p > 0.5), base_rate=y.mean(), cut=cut,
                by_purpose=q.groupby("purpose").status.apply(lambda s: (s == "Cancelled").mean()))
