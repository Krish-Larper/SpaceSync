import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.state import ml

d = ml()["demand"]
TEAL = ["#F2FBFB", "#9ED9DC", "#2BA5AE", "#0E7C86", "#073B40"]
st.title("📈 Demand Forecast")

st.subheader("Expected booking requests per hour")
fig = px.imshow(d["hourly"].round(2), color_continuous_scale=TEAL, aspect="auto", text_auto=".1f",
                labels=dict(color="requests / day"))
fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")
st.caption("Average new requests starting in each weekday-hour. Requests keep coming until 8 PM, "
           "after the timetable ends — evening study and club space is real demand.")

st.subheader("Next 14 days")
daily, fc = d["daily"], d["forecast"]
fig = go.Figure()
fig.add_scatter(x=daily.index, y=daily.requests, mode="lines", name="history", line=dict(color="#9ED9DC"))
fig.add_scatter(x=fc.date, y=fc.hi, mode="lines", line=dict(width=0), showlegend=False)
fig.add_scatter(x=fc.date, y=fc.lo, mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(14,124,134,0.18)", name="80% band")
fig.add_scatter(x=fc.date, y=fc.forecast, mode="lines+markers", name="forecast", line=dict(color="#0E7C86", width=3))
fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="requests per day")
st.plotly_chart(fig, width="stretch")

st.subheader("Backtest (last 20% of days held out)")
cols = st.columns(len(d["scores"]))
best = min(d["scores"], key=d["scores"].get)
for col, (k, v) in zip(cols, d["scores"].items()):
    col.metric(k, f"MAE {v:.2f}", "chosen" if k == best else None, delta_color="off")
st.caption("Daily request volume is close to flat, so the weekday profile wins narrowly and is what the forecast uses. "
           "The gradient-boosting model adds no accuracy here and is not deployed.")
