"""Constraint-first, explainable room matching.

Step 1 – hard constraints (a room is feasible only if ALL pass):
    capacity >= group size, projector / computers / step-free access if needed,
    no class in that weekday-slot, no event on that date, no SpaceSync booking.
Step 2 – rank feasible rooms with a transparent 0–100 score:
    capacity fit, room-type fit for the purpose, availability confidence,
    keeping specialist rooms (labs) free, power outlets, preferred building.
Every result carries the evidence lines that produced its score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Campus, WEEKDAYS

PURPOSES = ["Class", "Club meeting", "Project work", "Study group", "Workshop", "Seminar", "Guest lecture", "Exam"]

# how well each room type suits each purpose (1 = ideal)
TYPE_FIT = {
    "Class":         {"Classroom": 1.0, "Seminar": 0.7, "Auditorium": 0.6, "Lab": 0.5, "Study room": 0.4},
    "Club meeting":  {"Seminar": 1.0, "Classroom": 0.8, "Study room": 0.8, "Auditorium": 0.5, "Lab": 0.3},
    "Project work":  {"Lab": 1.0, "Study room": 0.9, "Classroom": 0.6, "Seminar": 0.6, "Auditorium": 0.2},
    "Study group":   {"Study room": 1.0, "Classroom": 0.7, "Seminar": 0.6, "Lab": 0.4, "Auditorium": 0.2},
    "Workshop":      {"Lab": 1.0, "Seminar": 0.9, "Classroom": 0.7, "Auditorium": 0.5, "Study room": 0.4},
    "Seminar":       {"Seminar": 1.0, "Auditorium": 0.9, "Classroom": 0.7, "Lab": 0.3, "Study room": 0.3},
    "Guest lecture": {"Auditorium": 1.0, "Seminar": 0.9, "Classroom": 0.6, "Lab": 0.2, "Study room": 0.1},
    "Exam":          {"Classroom": 1.0, "Auditorium": 0.9, "Seminar": 0.6, "Lab": 0.5, "Study room": 0.3},
}
WEIGHTS = {"capacity": 0.38, "type": 0.22, "availability": 0.15, "conserve": 0.10, "outlets": 0.10, "building": 0.05}


def capacity_fit(size: int, capacity: np.ndarray) -> np.ndarray:
    """1.0 when the room is ~85 % full; falls off for wasted seats, 0 if too small."""
    fill = size / capacity
    score = np.clip(1 - np.abs(fill - 0.85) / 0.85, 0, 1)
    return np.where(fill > 1, 0.0, score)


def find_rooms(campus: Campus, date, start_hour: int, duration: int, size: int, purpose: str,
               projector: bool = False, computers: bool = False, accessible: bool = False,
               building: str | None = None, bookings: list | None = None) -> pd.DataFrame:
    rooms = campus.rooms.copy()
    date = pd.Timestamp(date).normalize()
    wd = date.weekday()
    hours = list(range(start_hour, min(start_hour + duration, 24)))
    bookings = bookings or []

    # ---- hard constraints -------------------------------------------------------
    rooms["cap_ok"] = rooms.capacity >= size
    rooms["proj_ok"] = (not projector) | (rooms.projector == 1)
    rooms["comp_ok"] = (not computers) | (rooms.computers == 1)
    rooms["acc_ok"] = (not accessible) | (rooms.accessible == 1)
    rooms["class_busy"] = campus.class_grid[:, wd, hours].sum(axis=1) > 0

    dpos = campus.date_pos(date)
    if dpos is not None:                              # calendar known -> exact check
        rooms["event_busy"] = campus.event_grid[dpos][:, hours].sum(axis=1) > 0
        risk = np.zeros(len(rooms))
    else:                                             # future date -> estimate from history
        rooms["event_busy"] = False
        rate = campus.event_rate[:, wd, hours].clip(0, 1)
        risk = 1 - np.prod(1 - rate, axis=1)
    rooms["event_risk"] = risk

    booked = np.zeros(len(rooms), dtype=bool)
    for b in bookings:
        if pd.Timestamp(b["date"]).normalize() == date and set(b["hours"]) & set(hours):
            booked[campus.room_index[b["room_id"]]] = True
    rooms["spacesync_busy"] = booked

    rooms["feasible"] = (rooms.cap_ok & rooms.proj_ok & rooms.comp_ok & rooms.acc_ok
                         & ~rooms.class_busy & ~rooms.event_busy & ~rooms.spacesync_busy)

    # ---- soft score ---------------------------------------------------------------
    s_cap = capacity_fit(size, rooms.capacity.values)
    s_type = rooms.room_type.map(TYPE_FIT.get(purpose, {})).fillna(0.5).values
    s_avail = 1 - risk
    s_cons = np.where((rooms.computers == 1) & (not computers), 0.4, 1.0)
    s_out = np.clip(rooms.power_outlets.values / max(size / 4, 1), 0, 1)
    s_bld = np.ones(len(rooms)) if not building else (rooms.building == building).astype(float).values
    score = 100 * (WEIGHTS["capacity"] * s_cap + WEIGHTS["type"] * s_type + WEIGHTS["availability"] * s_avail
                   + WEIGHTS["conserve"] * s_cons + WEIGHTS["outlets"] * s_out + WEIGHTS["building"] * s_bld)
    rooms["score"] = np.where(rooms.feasible, score, 0).round(1)
    rooms["s_cap"], rooms["s_type"], rooms["s_avail"] = s_cap, s_type, s_avail
    rooms["s_cons"], rooms["s_out"], rooms["s_bld"] = s_cons, s_out, s_bld

    # ---- evidence -------------------------------------------------------------------
    t = f"{WEEKDAYS[wd]} {date:%d %b} {start_hour:02d}:00–{start_hour + duration:02d}:00"
    busy_courses = (campus.class_slots[(campus.class_slots.wd == wd) & campus.class_slots.hour.isin(hours)]
                    .groupby("room_id").course_code.agg(lambda c: ", ".join(sorted(set(c)))))

    def explain(row) -> list[tuple[str, str]]:
        ev = []
        fill = size / row.capacity
        if row.cap_ok:
            ev.append(("ok", f"Capacity fit: {row.capacity} seats for {size} people ({fill:.0%} full)"))
        else:
            ev.append(("bad", f"Too small: {row.capacity} seats for {size} people"))
        if projector:
            ev.append(("ok", "Projector available") if row.projector else ("bad", "No projector"))
        if computers:
            ev.append(("ok", "Computers available") if row.computers else ("bad", "No computers"))
        if accessible:
            ev.append(("ok", "Step-free access") if row.accessible else ("bad", "Not step-free accessible"))
        if row.class_busy:
            ev.append(("bad", f"Class clash: {busy_courses.get(row.room_id, 'scheduled class')} in this slot"))
        else:
            ev.append(("ok", f"No class clash on {t}"))
        if row.event_busy:
            ev.append(("bad", "Event already booked on this date"))
        elif dpos is None:
            ev.append(("warn" if row.event_risk > 0.1 else "ok",
                       f"Estimated {1 - row.event_risk:.0%} chance free of ad-hoc events (history for this weekday-slot)"))
        else:
            ev.append(("ok", "No event on the calendar for this date"))
        if row.spacesync_busy:
            ev.append(("bad", "Already booked through SpaceSync"))
        ev.append(("ok" if row.s_type >= 0.8 else "warn", f"{row.room_type} suits '{purpose}' ({row.s_type:.0%} type fit)"))
        if row.computers and not computers:
            ev.append(("warn", "Lab with computers — kept lower so lab work can use it"))
        if fill <= 0.4 and row.cap_ok:
            ev.append(("warn", f"{row.capacity - size} seats would sit empty"))
        return ev

    rooms["evidence"] = rooms.apply(explain, axis=1)
    rooms["first_block"] = rooms.apply(_first_block, axis=1)
    return rooms.sort_values(["feasible", "score"], ascending=False).reset_index(drop=True)


def _first_block(row) -> str:
    if row.feasible:
        return "Feasible"
    for col, why in [("cap_ok", "Too small"), ("proj_ok", "No projector"), ("comp_ok", "No computers"),
                     ("acc_ok", "Not accessible")]:
        if not row[col]:
            return why
    if row.class_busy:
        return "Class clash"
    if row.event_busy:
        return "Event clash"
    return "Booked via SpaceSync"
