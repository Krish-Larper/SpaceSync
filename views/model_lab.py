import plotly.express as px
import streamlit as st

from core.state import ml

m = ml()
st.title("🧪 Model Lab")
st.markdown("Every model is trained on the first 80% of the calendar and scored on the last 20% against a "
            "simple baseline. **A model ships only if it beats its baseline** — otherwise the product uses the baseline.")

st.subheader("1 · Attendance / no-show (events.csv)")
a = m["attendance"]
x, y, z = st.columns(3)
x.metric("Random forest MAE", f"{a['mae_model']:.1f} people")
y.metric("Baseline: mean attendance", f"{a['mae_mean']:.1f} people")
z.metric("Baseline: expected × median turnout", f"{a['mae_base']:.1f} people")
x, y, z = st.columns(3)
x.metric("Low-turnout classifier ROC-AUC", f"{a['auc_low_turnout']:.2f}")
y.metric("Expected attendance alone ROC-AUC", f"{a['auc_expected_only']:.2f}")
z.metric("Events with < 50% turnout", f"{a['low_rate']:.0%}")
st.info("Verdict: the forest only matches the 'predict the mean' baseline, and its low-turnout signal comes almost "
        "entirely from expected attendance (bigger forecasts miss more often). Actual turnout is not predictable from "
        "this data, so SpaceSync uses **empirical rates by event type** and **check-in + auto-release** instead of a "
        "model score.")
l, r = st.columns(2)
fig = px.bar(a["importance"][::-1], orientation="h", color_discrete_sequence=["#0E7C86"],
             labels={"value": "importance", "index": ""})
fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Feature importance")
l.plotly_chart(fig, width="stretch")
fig = px.bar(a["by_type"] * 100, color_discrete_sequence=["#2BA5AE"], labels={"value": "% low turnout", "event_type": ""})
fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, title="Low turnout by event type")
r.plotly_chart(fig, width="stretch")

st.subheader("2 · Request cancellation (room_requests.csv)")
c = m["cancel"]
x, y, z = st.columns(3)
x.metric("ROC-AUC", f"{c['auc']:.2f}", "0.50 = coin flip", delta_color="off")
y.metric("F1", f"{c['f1']:.2f}")
z.metric("Cancellation base rate", f"{c['base_rate']:.0%}")
st.info("Verdict: no signal — cancellations look random with respect to time, size, facilities and purpose. "
        "The Room Finder shows the historical rate for the chosen purpose instead.")

st.subheader("3 · Why the core engine isn't ML")
st.markdown("The brief asks for a solution that respects capacity, facilities, timing and conflicting bookings — "
            "not just predicted availability. Those are hard constraints, so SpaceSync solves them exactly and uses "
            "data only where it has evidence: event history for future-date availability risk, demand profiles for "
            "forecasting, and transparent weights for ranking.")
