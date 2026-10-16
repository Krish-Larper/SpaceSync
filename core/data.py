"""Load, clean and index the Datathon campus-space dataset.

Everything downstream (matcher, metrics, allocator, models) works off the
`Campus` object built here, which holds the cleaned tables plus numpy
occupancy grids for fast conflict checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
OPEN_HOUR, CLOSE_HOUR = 8, 20                 # usable teaching hours 8 AM – 8 PM
TEACHING_DAYS = 6                              # Mon–Sat
WEEKLY_HOURS = TEACHING_DAYS * (CLOSE_HOUR - OPEN_HOUR)   # 72 room-hours per week


@dataclass
class Campus:
    rooms: pd.DataFrame
    schedule: pd.DataFrame
    requests: pd.DataFrame
    events: pd.DataFrame
    class_slots: pd.DataFrame                  # one row per (class, hour)
    room_index: dict                           # room_id -> row in grids
    class_grid: np.ndarray                     # [room, weekday, hour] -> number of classes
    dates: pd.DatetimeIndex                    # calendar covered by events/requests
    event_grid: np.ndarray                     # [date, room, hour] -> number of events
    event_rate: np.ndarray                     # [room, weekday, hour] -> P(an event is booked)
    cleaning_log: list = field(default_factory=list)

    def date_pos(self, date) -> int | None:
        date = pd.Timestamp(date).normalize()
        if date < self.dates[0] or date > self.dates[-1]:
            return None
        return int((date - self.dates[0]).days)


def _expand(df: pd.DataFrame, start_col: str, dur_col: str, keep: list[str]) -> pd.DataFrame:
    """Turn bookings with a duration into one row per occupied hour."""
    rep = df.loc[df.index.repeat(df[dur_col])].copy()
    rep["hour"] = rep[start_col] + rep.groupby(level=0).cumcount()
    return rep[keep + ["hour"]].reset_index(drop=True)


def load_campus(data_dir: Path = DATA_DIR) -> Campus:
    log = []
    rooms = pd.read_csv(data_dir / "rooms.csv")
    sched = pd.read_csv(data_dir / "class_schedule.csv")
    req = pd.read_csv(data_dir / "room_requests.csv", parse_dates=["request_date"])
    ev = pd.read_csv(data_dir / "events.csv", parse_dates=["start_time"])

    # ---- de-duplication -------------------------------------------------
    for name, df, key in [("rooms", rooms, "room_id"), ("class_schedule", sched, "schedule_id"),
                          ("room_requests", req, "request_id"), ("events", ev, "event_id")]:
        n = df.duplicated(subset=[c for c in df.columns if c != key]).sum()
        log.append(f"{name}: {len(df):,} rows, {n} exact duplicates removed")
    rooms = rooms.drop_duplicates("room_id").reset_index(drop=True)
    sched = sched.drop_duplicates(subset=[c for c in sched.columns if c != "schedule_id"])
    req = req.drop_duplicates(subset=[c for c in req.columns if c != "request_id"])
    ev = ev.drop_duplicates(subset=[c for c in ev.columns if c != "event_id"])

    room_index = {rid: i for i, rid in enumerate(rooms.room_id)}
    rooms["label"] = rooms.room_id + " · " + rooms.room_type + " · " + rooms.building + " F" + rooms.floor.astype(str)

    # ---- class schedule ---------------------------------------------------
    sched["wd"] = sched.weekday.map({d: i for i, d in enumerate(WEEKDAYS)})
    sched["end_hour"] = sched.start_hour + sched.duration_hours
    sched = sched.merge(rooms[["room_id", "capacity", "room_type", "building", "projector", "computers"]],
                        on="room_id", how="left")
    sched["over_capacity"] = sched.expected_attendance > sched.capacity
    sched["seat_fill"] = sched.expected_attendance / sched.capacity
    slots = _expand(sched, "start_hour", "duration_hours", ["schedule_id", "room_id", "wd", "course_code"])
    slots["r"] = slots.room_id.map(room_index)

    class_grid = np.zeros((len(rooms), 7, 24), dtype=np.int16)
    np.add.at(class_grid, (slots.r.values, slots.wd.values, slots.hour.values), 1)
    clash_cells = class_grid[slots.r.values, slots.wd.values, slots.hour.values] > 1
    clashing_ids = set(slots.loc[clash_cells, "schedule_id"])
    sched["in_clash"] = sched.schedule_id.isin(clashing_ids)
    log.append(f"class_schedule: {int((class_grid > 1).sum()):,} room-hours are double-booked "
               f"({len(clashing_ids):,} classes involved)")
    log.append(f"class_schedule: {int(sched.over_capacity.sum()):,} classes expect more students than room capacity")

    # ---- events ---------------------------------------------------------------
    ev["date"] = ev.start_time.dt.normalize()
    ev["hour"] = ev.start_time.dt.hour
    ev["wd"] = ev.start_time.dt.weekday
    ev["end_hour"] = ev.hour + ev.duration_hours
    ev["crosses_midnight"] = ev.end_hour > 24
    ev["out_of_hours"] = (ev.hour < OPEN_HOUR) | (ev.hour >= CLOSE_HOUR)
    ev = ev.merge(rooms[["room_id", "capacity", "room_type"]], on="room_id", how="left")
    ev["turnout"] = ev.actual_attendance / ev.expected_attendance
    ev["low_turnout"] = ev.turnout < 0.5
    ev["overflow"] = ev.actual_attendance > ev.capacity
    log.append(f"events: {int(ev.out_of_hours.sum()):,} start outside 8 AM–8 PM (kept as bookings, flagged)")
    log.append(f"events: {int(ev.crosses_midnight.sum()):,} cross midnight (spill-over hours clipped)")

    # ---- requests -------------------------------------------------------------
    req["wd"] = req.request_date.dt.weekday
    req["weekday"] = req.wd.map(dict(enumerate(WEEKDAYS)))
    req["end_hour"] = req.start_hour + req.duration_hours
    req["after_close"] = req.end_hour > CLOSE_HOUR + 2

    # ---- dated grids --------------------------------------------------------
    start = min(ev.date.min(), req.request_date.min())
    end = max(ev.date.max(), req.request_date.max())
    dates = pd.date_range(start, end, freq="D")
    event_grid = np.zeros((len(dates), len(rooms), 24), dtype=np.int16)
    ev_slots = _expand(ev, "hour", "duration_hours", ["room_id", "date", "wd"])
    ev_slots = ev_slots[ev_slots.hour < 24]
    d_idx = ((ev_slots.date - dates[0]).dt.days).values
    np.add.at(event_grid, (d_idx, ev_slots.room_id.map(room_index).values, ev_slots.hour.values), 1)

    # empirical probability that an ad-hoc event takes a room in a weekday-hour
    weeks_per_wd = np.array([(dates.weekday == w).sum() for w in range(7)]).clip(min=1)
    ev_counts = np.zeros((len(rooms), 7, 24))
    np.add.at(ev_counts, (ev_slots.room_id.map(room_index).values, ev_slots.wd.values, ev_slots.hour.values), 1)
    event_rate = ev_counts / weeks_per_wd[None, :, None]

    return Campus(rooms=rooms, schedule=sched, requests=req, events=ev, class_slots=slots,
                  room_index=room_index, class_grid=class_grid, dates=dates,
                  event_grid=event_grid, event_rate=event_rate, cleaning_log=log)
