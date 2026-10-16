"""Cached singletons shared by every page."""
import streamlit as st

from . import analytics, models
from .data import load_campus


@st.cache_resource(show_spinner="Loading campus data…")
def campus():
    return load_campus()


@st.cache_resource(show_spinner="Repairing the timetable…")
def class_plan():
    return analytics.reallocate_classes(campus())


@st.cache_resource(show_spinner="Replaying 5,000 requests through the matcher…")
def request_replay():
    return analytics.place_requests(campus())


@st.cache_resource(show_spinner="Training models…")
def ml():
    c = campus()
    return dict(demand=models.demand_forecast(c), attendance=models.attendance_model(c),
                cancel=models.cancellation_model(c))


def bookings():
    return st.session_state.setdefault("bookings", [])
