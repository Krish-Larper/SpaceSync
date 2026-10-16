import plotly.express as px
import streamlit as st

from core.state import campus, class_plan, request_replay

c = campus()
st.title("🛠️ Recommendations")
st.caption("Admin-side actions from the same constraint engine the Room Finder uses.")

t1, t2 = st.tabs(["Timetable repair", "Turned-away requests"])

with t1:
    recs, s = class_plan()
    a, b, cc, d = st.columns(4)
    a.metric("Problem classes", f"{s['problems']:,}", help="Over capacity and/or sharing a room-hour with another class")
    b.metric("Move to a better room", f"{s['moved']:,}")
    cc.metric("Move + shift time", f"{s['shifted']:,}")
    d.metric("Double-booked room-hours", f"{s['clash_after']:,}", f"{s['clash_after'] - s['clash_before']:,} vs today",
             delta_color="inverse")
    st.markdown("**How it works:** classes are processed largest first. Each is lifted out of its room; if it "
                "still clashes or doesn't fit, SpaceSync picks the free room with the best capacity fit that keeps the "
                "original projector/computer setup (same building preferred). If nothing is free at that time, it "
                "tries the nearest free hour the same day. Anything left is flagged for a section split.")
    flt = st.segmented_control("Show", ["All", "Move", "Move + shift", "Unresolved"], default="All")
    view = recs
    if flt == "Move":
        view = recs[recs.action == "Move"]
    elif flt == "Move + shift":
        view = recs[recs.action.str.startswith("Move +")]
    elif flt == "Unresolved":
        view = recs[recs.action == "Unresolved"]
    st.dataframe(view, hide_index=True, width="stretch")
    st.download_button("Download recommendations (CSV)", recs.to_csv(index=False), "spacesync_timetable_repair.csv")

with t2:
    req, s = request_replay()
    a, b, cc = st.columns(3)
    a.metric("Requests rejected or waitlisted", f"{s['turned_away']:,}")
    b.metric("Had a free, fitting room", f"{s['recoverable']:,}", f"{s['recoverable'] / s['turned_away']:.0%}",
             delta_color="off")
    cc.metric("Approved requests placed by SpaceSync", f"{s['approved_placed']}%")
    st.markdown("Every non-cancelled request is replayed in date order — approved ones first (they really got space), "
                "then waitlisted and rejected. A request counts as recoverable only if a room had the seats and "
                "facilities **and** was clear of classes, events and every earlier SpaceSync placement.")
    ta = req[req.status.isin(["Waitlisted", "Rejected"])]
    l, r = st.columns(2)
    fig = px.histogram(ta, x="purpose", color="spacesync_result", barmode="group",
                       color_discrete_map={"Placed": "#0E7C86", "All fitting rooms busy": "#B42318",
                                           "No room has the seats/facilities": "#B26A00"})
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title=None)
    l.plotly_chart(fig, width="stretch")
    fig = px.histogram(ta[ta.spacesync_result == "Placed"], x="start_hour", color_discrete_sequence=["#0E7C86"],
                       labels={"start_hour": "start hour of recovered requests"})
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), bargap=0.1)
    r.plotly_chart(fig, width="stretch")
    st.dataframe(ta[["request_id", "request_date", "start_hour", "duration_hours", "required_capacity",
                     "projector_required", "computers_required", "purpose", "status", "spacesync_room",
                     "spacesync_result"]], hide_index=True, width="stretch")
    st.download_button("Download replay (CSV)", req.to_csv(index=False), "spacesync_request_replay.csv")
