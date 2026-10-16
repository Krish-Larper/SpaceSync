import streamlit as st

from core import analytics
from core.state import campus, request_replay, class_plan

c = campus()
h = analytics.headline_stats(c)
peaks = sorted(analytics.peak_hours(c))
_, replay = request_replay()
_, plan = class_plan()

st.title("The campus has space. Students don't.")
st.markdown("SpaceSync matches every room request to the **best feasible room** — checked against capacity, "
            "facilities, the class timetable and the event calendar — and shows the evidence behind each pick. "
            "Admins get utilization, timetable repairs and demand forecasts from the same engine.")

st.subheader("What the data says")
a, b, cc, d = st.columns(4)
a.metric("Average room utilization", f"{h['avg_time_util']:.0%}",
         help="Share of Mon–Sat 8 AM–8 PM room-hours with a scheduled class")
b.metric("Seat-weighted utilization", f"{h['avg_effective_util']:.0%}",
         help="Time utilization × median seat fill — how much of the space is really used")
cc.metric("Classes in rooms too small", f"{h['over_cap_share']:.0%}", f"{h['over_cap_classes']:,} classes", delta_color="off")
d.metric("Double-booked room-hours", f"{h['clash_room_hours']:,}", f"{h['clash_classes']:,} classes involved", delta_color="off")

a, b, cc, d = st.columns(4)
a.metric("Events with under half the expected crowd", f"{h['low_turnout_events']:.0%}")
b.metric("Requests turned away", f"{h['unmet_requests']:.0%}", "rejected or waitlisted", delta_color="off")
cc.metric("…that had a free, fitting room", f"{replay['recoverable'] / replay['turned_away']:.0%}",
          f"{replay['recoverable']:,} of {replay['turned_away']:,}", delta_color="off")
d.metric("Peak demand hours", f"{peaks[0]:02d}:00–{peaks[-1] + 1:02d}:00")

st.subheader("What SpaceSync does with it")
x, y, z = st.columns(3)
with x.container(border=True):
    st.markdown("**🔎 Room Finder**  \nTime, group size, needs → ranked feasible rooms with a 0–100 score and the "
                "reasons behind it. Book in one tap; the slot is blocked for everyone else.")
with y.container(border=True):
    st.markdown(f"**🛠️ Timetable repair**  \nGreedy constrained re-allocation cuts double-booked room-hours "
                f"from **{plan['clash_before']:,} → {plan['clash_after']:,}** and re-houses over-capacity classes.")
with z.container(border=True):
    st.markdown("**📈 Forecast + Model Lab**  \nDemand by weekday and hour, plus no-show / cancellation models "
                "evaluated honestly against baselines on a time-based split.")

with st.expander("Data cleaning log"):
    for line in c.cleaning_log:
        st.markdown(f"- {line}")
    st.markdown("- Join key: `rooms.room_id` ↔ `class_schedule.room_id` / `events.room_id`. "
                "Requests have no room_id — they are demand before allocation, so SpaceSync assigns one.")
