# SpaceSync — Smart Campus Space Finder & Optimizer
Team Larpers · Datathon 2026 · Track 2 (Education)

> Given a room request, what is the best feasible room — and what evidence supports it?

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
Deploy free on Streamlit Community Cloud: push this folder to GitHub → share.streamlit.io → pick `app.py`.

## Pages
| Page | What it does |
|---|---|
| Overview | Headline numbers from the dataset + cleaning log |
| Room Finder | Time, group size, purpose, facilities → feasible rooms ranked 0–100 with evidence lines; one-tap booking blocks the slot; can load any request from `room_requests.csv`; suggests nearest workable slots if nothing fits |
| Admin Dashboard | Weekday × hour busy heatmap, utilization by building/type, underused / overcrowded / double-booked rooms, event data-quality flags |
| Recommendations | Timetable repair (re-house over-capacity & clashing classes) and replay of rejected/waitlisted requests to find recoverable demand; CSV exports |
| Demand Forecast | Requests per weekday-hour, 14-day forecast with 80 % band, backtest vs baselines |
| Model Lab | Attendance/no-show and cancellation models vs baselines on a time-based split, with verdicts |

## How matching works (`core/matcher.py`)
1. **Hard constraints** — capacity ≥ group, projector / computers / step-free if needed, no class in that weekday-slot (`class_schedule`), no event on that date (`events`), no SpaceSync booking. Dates outside the event calendar use the historical probability of an ad-hoc event in that room/weekday/hour.
2. **Transparent score** — capacity fit 38 % (peaks at ~85 % full), room-type fit for the purpose 22 %, availability confidence 15 %, keeping labs free 10 %, power outlets 10 %, preferred building 5 %.
3. **Evidence** — every ✅ / ⚠️ / ❌ line shown to the user comes from those checks.

## Structure
```
app.py              navigation shell
core/data.py        load, clean, flag, build occupancy grids
core/matcher.py     constraint check + scoring + explanations
core/analytics.py   utilization, timetable repair, request replay
core/models.py      demand forecast, attendance & cancellation models
core/state.py       Streamlit caching
views/              one file per page
data/               the four Datathon CSVs + README
```

## Honest notes
- The dataset is synthetic. Turnout and cancellations show no learnable signal (models ≈ baselines), so the product uses empirical rates + check-in/auto-release instead of a model score. The Model Lab page shows the evidence.
- About half the events start between 8 PM and 8 AM — flagged as likely timestamp errors but still treated as bookings.
