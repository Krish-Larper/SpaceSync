"""Utilization metrics and admin recommendations (re-allocation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Campus, WEEKDAYS, OPEN_HOUR, CLOSE_HOUR, WEEKLY_HOURS
from .matcher import TYPE_FIT, capacity_fit


# ----------------------------------------------------------------------------- utilization
def room_utilization(c: Campus) -> pd.DataFrame:
    occupied = (c.class_grid[:, :6, OPEN_HOUR:CLOSE_HOUR] > 0).sum(axis=(1, 2))
    seat = c.schedule.groupby("room_id").seat_fill.median()
    ev_hours = c.events[~c.events.out_of_hours].groupby("room_id").duration_hours.sum()
    weeks = len(c.dates) / 7
    df = c.rooms.copy()
    df["time_util"] = occupied / WEEKLY_HOURS
    df["seat_fill_median"] = df.room_id.map(seat).fillna(0)
    df["effective_util"] = df.time_util * df.seat_fill_median.clip(upper=1)
    df["event_hours_per_week"] = df.room_id.map(ev_hours).fillna(0) / weeks
    df["clash_hours"] = (c.class_grid[:, :6, :] > 1).sum(axis=(1, 2))
    df["over_cap_classes"] = df.room_id.map(c.schedule.groupby("room_id").over_capacity.sum()).fillna(0).astype(int)
    return df


def busy_heatmap(c: Campus) -> pd.DataFrame:
    """% of rooms with a class, per weekday x hour."""
    share = (c.class_grid[:, :6, OPEN_HOUR:CLOSE_HOUR] > 0).mean(axis=0)
    return pd.DataFrame(share * 100, index=WEEKDAYS[:6], columns=[f"{h:02d}:00" for h in range(OPEN_HOUR, CLOSE_HOUR)])


def headline_stats(c: Campus) -> dict:
    u = room_utilization(c)
    ev_in = c.events
    return {
        "avg_time_util": u.time_util.mean(),
        "avg_effective_util": u.effective_util.mean(),
        "rooms_under_50": int((u.time_util < 0.5).sum()),
        "rooms_effective_under_40": int((u.effective_util < 0.4).sum()),
        "over_cap_classes": int(c.schedule.over_capacity.sum()),
        "over_cap_share": c.schedule.over_capacity.mean(),
        "clash_room_hours": int((c.class_grid > 1).sum()),
        "clash_classes": int(c.schedule.in_clash.sum()),
        "low_turnout_events": ev_in.low_turnout.mean(),
        "cancelled_requests": (c.requests.status == "Cancelled").mean(),
        "unmet_requests": c.requests.status.isin(["Rejected", "Waitlisted"]).mean(),
        "events_out_of_hours": ev_in.out_of_hours.mean(),
    }


def peak_hours(c: Campus, top: int = 3) -> list[int]:
    demand = c.requests.groupby("start_hour").size() + c.schedule.groupby("start_hour").size().reindex(range(8, 21), fill_value=0)
    return list(demand.sort_values(ascending=False).head(top).index)


# ----------------------------------------------------------------------------- re-allocation
def reallocate_classes(c: Campus) -> tuple[pd.DataFrame, dict]:
    """Greedy repair of the timetable: move over-capacity / clashing classes into
    free rooms that fit, keeping projector / computer needs of the original room."""
    grid = c.class_grid.copy()
    rooms = c.rooms
    cap = rooms.capacity.values
    proj, comp = rooms.projector.values, rooms.computers.values
    bld = rooms.building.values
    recs = []
    problems = c.schedule[c.schedule.over_capacity | c.schedule.in_clash].sort_values("expected_attendance", ascending=False)

    for row in problems.itertuples():
        r0 = c.room_index[row.room_id]
        hrs = list(range(row.start_hour, row.end_hour))
        grid[r0, row.wd, hrs] -= 1                         # lift the class out
        still_clash = grid[r0, row.wd, hrs].max() > 0
        still_over = row.expected_attendance > cap[r0]
        if not (still_clash or still_over):
            grid[r0, row.wd, hrs] += 1
            continue
        free = grid[:, row.wd, hrs].sum(axis=1) == 0
        ok = free & (cap >= row.expected_attendance) & (proj >= proj[r0]) & (comp >= comp[r0])
        issue = " + ".join(x for x, f in [("Over capacity", still_over), ("Double-booked", still_clash)] if f)
        if not ok.any():
            # second pass: same day, different start time, smallest-fitting free room
            fits = (cap >= row.expected_attendance) & (proj >= proj[r0]) & (comp >= comp[r0])
            best = None
            for alt in sorted(range(OPEN_HOUR, CLOSE_HOUR - row.duration_hours + 1), key=lambda h: abs(h - row.start_hour)):
                ah = list(range(alt, alt + row.duration_hours))
                ok2 = fits & (grid[:, row.wd, ah].sum(axis=1) == 0)
                if ok2.any():
                    r1 = int(np.argmax(np.where(ok2, capacity_fit(row.expected_attendance, cap) + 0.15 * (bld == bld[r0]), -1)))
                    best = (alt, ah, r1)
                    break
            if best is None:
                grid[r0, row.wd, hrs] += 1
                recs.append(dict(schedule_id=row.schedule_id, course=row.course_code, dept=row.department,
                                 slot=f"{row.weekday} {row.start_hour:02d}:00–{row.end_hour:02d}:00",
                                 students=row.expected_attendance, issue=issue, from_room=row.room_id,
                                 from_cap=int(cap[r0]), to_room="—", to_cap=None, action="Unresolved",
                                 evidence="No free fitting room on this day — split the section or move day"))
                continue
            alt, ah, r1 = best
            grid[r1, row.wd, ah] += 1
            recs.append(dict(schedule_id=row.schedule_id, course=row.course_code, dept=row.department,
                             slot=f"{row.weekday} {row.start_hour:02d}:00–{row.end_hour:02d}:00",
                             students=row.expected_attendance, issue=issue, from_room=row.room_id,
                             from_cap=int(cap[r0]), to_room=rooms.room_id[r1], to_cap=int(cap[r1]),
                             action=f"Move + shift to {alt:02d}:00",
                             evidence=f"no fitting room free at {row.start_hour:02d}:00; nearest free slot is {alt:02d}:00 "
                                      f"in a {cap[r1]}-seat room ({row.expected_attendance / cap[r1]:.0%} full)"))
            continue
        score = capacity_fit(row.expected_attendance, cap) + 0.15 * (bld == bld[r0])
        score = np.where(ok, score, -1)
        r1 = int(np.argmax(score))
        grid[r1, row.wd, hrs] += 1
        evid = [f"{cap[r1]} seats for {row.expected_attendance} ({row.expected_attendance / cap[r1]:.0%} full)",
                "free all slot hours"]
        if proj[r0]: evid.append("projector kept")
        if comp[r0]: evid.append("computers kept")
        evid.append("same building" if bld[r1] == bld[r0] else f"moves {bld[r0]} → {bld[r1]}")
        recs.append(dict(schedule_id=row.schedule_id, course=row.course_code, dept=row.department,
                         slot=f"{row.weekday} {row.start_hour:02d}:00–{row.end_hour:02d}:00",
                         students=row.expected_attendance, issue=issue, from_room=row.room_id,
                         from_cap=int(cap[r0]), to_room=rooms.room_id[r1], to_cap=int(cap[r1]),
                         action="Move", evidence=", ".join(evid)))

    out = pd.DataFrame(recs)
    after_clash = int((grid > 1).sum())
    summary = dict(problems=len(problems), moved=int((out.action == "Move").sum()) if len(out) else 0,
                   shifted=int(out.action.str.startswith("Move +").sum()) if len(out) else 0,
                   unresolved=int((out.action == "Unresolved").sum()) if len(out) else 0,
                   clash_before=int((c.class_grid > 1).sum()), clash_after=after_clash)
    return out, summary


def place_requests(c: Campus) -> tuple[pd.DataFrame, dict]:
    """Replay every non-cancelled request through the matcher in date order:
    approved first (real demand that got space), then waitlisted / rejected.
    Answers: how much turned-away demand actually had a feasible room?"""
    rooms = c.rooms
    cap = rooms.capacity.values
    proj, comp = rooms.projector.values, rooms.computers.values
    booked = np.zeros_like(c.event_grid)
    order = {"Approved": 0, "Waitlisted": 1, "Rejected": 2}
    req = c.requests[c.requests.status != "Cancelled"].copy()
    req["prio"] = req.status.map(order)
    req = req.sort_values(["prio", "request_date", "start_hour"])
    assigned, reason = [], []
    for row in req.itertuples():
        hrs = list(range(row.start_hour, min(row.end_hour, 24)))
        d = c.date_pos(row.request_date)
        busy = (c.class_grid[:, row.wd, hrs].sum(axis=1) > 0) | (c.event_grid[d][:, hrs].sum(axis=1) > 0) \
            | (booked[d][:, hrs].sum(axis=1) > 0)
        fits = (cap >= row.required_capacity) & (proj >= row.projector_required) & (comp >= row.computers_required)
        ok = fits & ~busy
        if not ok.any():
            assigned.append(None)
            reason.append("No room has the seats/facilities" if not fits.any() else "All fitting rooms busy")
            continue
        tf = rooms.room_type.map(TYPE_FIT.get(row.purpose, {})).fillna(0.5).values
        score = np.where(ok, capacity_fit(row.required_capacity, cap) + 0.5 * tf
                         - 0.3 * ((comp == 1) & (row.computers_required == 0)), -9)
        r = int(np.argmax(score))
        booked[d][r, hrs] += 1
        assigned.append(rooms.room_id[r])
        reason.append("Placed")
    req["spacesync_room"] = assigned
    req["spacesync_result"] = reason
    turned_away = req[req.status.isin(["Waitlisted", "Rejected"])]
    summary = dict(turned_away=len(turned_away),
                   recoverable=int(turned_away.spacesync_room.notna().sum()),
                   approved_placed=int(req[(req.status == "Approved")].spacesync_room.notna().mean() * 100))
    return req, summary
