import datetime as dt

import pandas as pd
import streamlit as st

from core.data import WEEKDAYS
from core.matcher import PURPOSES, WEIGHTS, find_rooms
from core.state import bookings, campus, ml

c = campus()
st.title("🔎 Room Finder")
st.caption("Given a room request, what is the best feasible room — and what evidence supports it?")

# ---------------------------------------------------------------- inputs
mode = st.radio("Start from", ["New search", "A request from the dataset"], horizontal=True)
defaults = dict(date=dt.date(2025, 9, 16), start=14, dur=2, size=40, purpose="Study group",
                proj=False, comp=False, acc=False)
if mode == "A request from the dataset":
    rid = st.selectbox("Request", c.requests.request_id, index=2)
    q = c.requests.set_index("request_id").loc[rid]
    st.info(f"**{rid}** · {q.purpose} · {q.request_date:%a %d %b %Y} {q.start_hour:02d}:00 for {q.duration_hours} h · "
            f"{q.required_capacity} people · projector {'yes' if q.projector_required else 'no'} · "
            f"computers {'yes' if q.computers_required else 'no'} · original status **{q.status}**")
    defaults.update(date=q.request_date.date(), start=int(q.start_hour), dur=int(q.duration_hours),
                    size=int(q.required_capacity), purpose=q.purpose, proj=bool(q.projector_required),
                    comp=bool(q.computers_required))

with st.container(border=True):
    a, b, cc, d = st.columns(4)
    date = a.date_input("Date", defaults["date"], key=f"d{mode}",
                        help="Aug–Nov 2025 checks the real event calendar; later dates use historical event risk")
    start = b.selectbox("Start", list(range(8, 21)), index=defaults["start"] - 8, format_func=lambda h: f"{h:02d}:00", key=f"s{mode}")
    dur = cc.selectbox("Duration (h)", [1, 2, 3, 4], index=defaults["dur"] - 1, key=f"u{mode}")
    size = d.number_input("Group size", 1, 400, defaults["size"], key=f"z{mode}")
    a, b, cc, d, e = st.columns([1.3, 1, 1, 1, 1.3])
    purpose = a.selectbox("Purpose", PURPOSES, index=PURPOSES.index(defaults["purpose"]), key=f"p{mode}")
    proj = b.toggle("Projector", defaults["proj"], key=f"j{mode}")
    comp = cc.toggle("Computers", defaults["comp"], key=f"c{mode}")
    acc = d.toggle("Step-free", defaults["acc"], key=f"a{mode}")
    building = e.selectbox("Preferred building", ["Any"] + sorted(c.rooms.building.unique()), key=f"b{mode}")

res = find_rooms(c, date, start, dur, int(size), purpose, proj, comp, acc,
                 None if building == "Any" else building, bookings())
feas = res[res.feasible]
known = c.date_pos(date) is not None

# ---------------------------------------------------------------- funnel
counts = res.first_block.value_counts()
st.markdown(f"**{len(feas)} of {len(res)} rooms are feasible** for {WEEKDAYS[pd.Timestamp(date).weekday()]} "
            f"{pd.Timestamp(date):%d %b %Y}, {start:02d}:00–{start + dur:02d}:00. "
            + ("Event calendar checked." if known else "Future date — event availability is estimated from history."))
blocked = counts.drop("Feasible", errors="ignore")
if len(blocked):
    st.caption("Ruled out → " + " · ".join(f"{k}: {v}" for k, v in blocked.items()))

if feas.empty:
    st.warning("No room passes every constraint. Try another hour, a shorter slot, or drop a facility requirement.")
    alt = []
    for h in range(8, 21 - dur + 1):
        if h == start:
            continue
        r = find_rooms(c, date, h, dur, int(size), purpose, proj, comp, acc, None, bookings())
        if r.feasible.any():
            alt.append((h, r.iloc[0].room_id, r.iloc[0].score, int(r.feasible.sum())))
    if alt:
        st.markdown("**Nearest slots that work:**")
        st.dataframe(pd.DataFrame(alt, columns=["start", "best room", "score", "feasible rooms"])
                     .assign(start=lambda d: d.start.map(lambda h: f"{h:02d}:00"))
                     .iloc[sorted(range(len(alt)), key=lambda i: abs(alt[i][0] - start))][:5],
                     hide_index=True)
    st.stop()

# ---------------------------------------------------------------- results
cancel_rate = ml()["cancel"]["by_purpose"].get(purpose)
icon = {"ok": "✅", "warn": "⚠️", "bad": "❌"}
for rank, row in enumerate(feas.head(5).itertuples(), 1):
    with st.container(border=True):
        l, r = st.columns([3, 1])
        l.markdown(f"#### #{rank} &nbsp; {row.room_id} · {row.room_type}")
        l.caption(f"{row.building} block, floor {row.floor} · {row.capacity} seats · "
                  f"{'projector · ' if row.projector else ''}{'computers · ' if row.computers else ''}"
                  f"{'step-free · ' if row.accessible else ''}{row.power_outlets} power outlets")
        r.metric("Match score", f"{row.score:.0f}/100")
        r.progress(min(row.score / 100, 1.0))
        ev_html = "".join(f'<div class="ev ev-{k}">{icon[k]} {t}</div>' for k, t in row.evidence)
        l.markdown(ev_html, unsafe_allow_html=True)
        with l.expander("Score breakdown"):
            parts = {"Capacity fit": row.s_cap, "Room-type fit": row.s_type, "Availability confidence": row.s_avail,
                     "Keeps labs free": row.s_cons, "Power outlets": row.s_out, "Preferred building": row.s_bld}
            st.dataframe(pd.DataFrame({"factor": parts.keys(), "weight": WEIGHTS.values(),
                                       "value": [round(v, 2) for v in parts.values()],
                                       "points": [round(100 * w * v, 1) for w, v in zip(WEIGHTS.values(), parts.values())]}),
                         hide_index=True)
        if r.button("Book this room", key=f"book{row.room_id}", type="primary" if rank == 1 else "secondary"):
            bookings().append(dict(room_id=row.room_id, date=pd.Timestamp(date), hours=list(range(start, start + dur)),
                                   purpose=purpose, size=int(size)))
            st.toast(f"Booked {row.room_id} — slot blocked for everyone else.")
            st.rerun()

if cancel_rate is not None:
    st.caption(f"Historical cancellation rate for '{purpose}' requests: {cancel_rate:.0%}. "
               "A check-in reminder is suggested; unchecked bookings release after 15 minutes.")

with st.expander(f"All {len(feas)} feasible rooms"):
    st.dataframe(feas[["room_id", "room_type", "building", "floor", "capacity", "projector", "computers",
                       "accessible", "score"]],
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.0f")},
                 hide_index=True, width="stretch")

if bookings():
    st.subheader("Your bookings this session")
    bk = pd.DataFrame(bookings())
    bk["slot"] = bk.apply(lambda b: f"{b.date:%a %d %b} {b.hours[0]:02d}:00–{b.hours[-1] + 1:02d}:00", axis=1)
    st.dataframe(bk[["room_id", "slot", "purpose", "size"]], hide_index=True)
    if st.button("Clear bookings"):
        st.session_state.bookings = []
        st.rerun()
