import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core import analytics
from core.data import WEEKDAYS, OPEN_HOUR, CLOSE_HOUR
from core.state import campus

c = campus()
u = analytics.room_utilization(c)
TEAL = ["#F2FBFB", "#9ED9DC", "#2BA5AE", "#0E7C86", "#073B40"]

st.title("📊 Admin Dashboard")

f1, f2, f3 = st.columns(3)
blds = f1.multiselect("Building", sorted(c.rooms.building.unique()))
types = f2.multiselect("Room type", sorted(c.rooms.room_type.unique()))
metric = f3.radio("Utilization measure", ["Time", "Seat-weighted"], index=1, horizontal=True,
                  help="Time = booked hours ÷ 72 weekly hours. Seat-weighted also multiplies by how full the room is.")
mask = np.ones(len(u), bool)
if blds: mask &= u.building.isin(blds).values
if types: mask &= u.room_type.isin(types).values
uf = u[mask]
col = "time_util" if metric == "Time" else "effective_util"

a, b, cc, d = st.columns(4)
a.metric("Rooms in view", len(uf))
b.metric(f"Average {metric.lower()} utilization", f"{uf[col].mean():.0%}")
cc.metric("Rooms below 40%", int((uf[col] < 0.4).sum()))
d.metric("Double-booked room-hours", int(uf.clash_hours.sum()))

# ---------------------------------------------------------------- heatmap
st.subheader("When is space busy?")
grid = (c.class_grid[mask][:, :6, OPEN_HOUR:CLOSE_HOUR] > 0).mean(axis=0) * 100
hm = pd.DataFrame(grid, index=WEEKDAYS[:6], columns=[f"{h:02d}:00" for h in range(OPEN_HOUR, CLOSE_HOUR)])
fig = px.imshow(hm, color_continuous_scale=TEAL, aspect="auto", text_auto=".0f",
                labels=dict(color="% rooms in use"))
fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")
st.caption("% of rooms (in the current filter) holding a scheduled class in each weekday-hour. Saturday is the free reserve.")

# ---------------------------------------------------------------- by building / type
l, r = st.columns(2)
with l:
    st.subheader("Utilization by building and type")
    piv = uf.pivot_table(index="building", columns="room_type", values=col, aggfunc="mean") * 100
    fig = px.imshow(piv.round(0), color_continuous_scale=TEAL, text_auto=".0f", aspect="auto",
                    labels=dict(color="%"))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
with r:
    st.subheader("Size vs how full classes are")
    fig = px.scatter(uf, x="capacity", y="seat_fill_median", color="room_type", hover_name="room_id",
                     labels={"seat_fill_median": "median seat fill (attendance ÷ capacity)"})
    fig.add_hline(y=1, line_dash="dash", line_color="#B42318", annotation_text="over capacity")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tables
st.subheader("Rooms to look at")
t1, t2, t3, t4 = st.tabs(["Underused", "Overcrowded", "Double-booked", "Event data quality"])
show = ["room_id", "building", "floor", "room_type", "capacity", "time_util", "seat_fill_median",
        "effective_util", "event_hours_per_week", "over_cap_classes", "clash_hours"]
cfg = {k: st.column_config.ProgressColumn(k, min_value=0, max_value=1, format="percent")
       for k in ["time_util", "effective_util"]}
cfg["seat_fill_median"] = st.column_config.NumberColumn("seat_fill_median", format="%.2f")
cfg["event_hours_per_week"] = st.column_config.NumberColumn("event_hours/wk", format="%.1f")
with t1:
    st.caption("Lowest seat-weighted utilization — candidates for open study space, club bookings or consolidation.")
    st.dataframe(uf.sort_values("effective_util")[show].head(15), column_config=cfg, hide_index=True, width="stretch")
with t2:
    st.caption("Rooms that host the most classes with more students than seats.")
    st.dataframe(uf.sort_values("over_cap_classes", ascending=False)[show].head(15), column_config=cfg,
                 hide_index=True, width="stretch")
with t3:
    st.caption("Room-hours where the timetable puts two or more classes in the same room.")
    clash = c.class_slots.assign(n=c.class_grid[c.class_slots.r, c.class_slots.wd, c.class_slots.hour])
    clash = clash[clash.n > 1].groupby(["room_id", "wd", "hour"]).course_code.agg(", ".join).reset_index()
    clash["slot"] = clash.wd.map(dict(enumerate(WEEKDAYS))) + " " + clash.hour.map(lambda h: f"{h:02d}:00")
    if blds or types:
        clash = clash[clash.room_id.isin(uf.room_id)]
    st.dataframe(clash[["room_id", "slot", "course_code"]].rename(columns={"course_code": "courses sharing the room"}),
                 hide_index=True, width="stretch")
with t4:
    e = c.events
    x, y, z = st.columns(3)
    x.metric("Events starting 8 PM–8 AM", f"{e.out_of_hours.mean():.0%}")
    y.metric("Turnout under 50% of expected", f"{e.low_turnout.mean():.0%}")
    z.metric("Turnout above room capacity", f"{e.overflow.mean():.0%}")
    fig = px.histogram(e, x="hour", nbins=24, color_discrete_sequence=["#0E7C86"],
                       labels={"hour": "event start hour"})
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), bargap=0.1)
    st.plotly_chart(fig, width="stretch")
    st.caption("Event start times are spread evenly over 24 hours — likely timestamp errors. "
               "SpaceSync still treats them as bookings (conservative) but flags them for the facilities office.")
