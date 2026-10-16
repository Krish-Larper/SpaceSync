import streamlit as st

st.set_page_config(page_title="SpaceSync · Smart Campus Space", page_icon="🏫", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1250px;}
  h1, h2, h3 {letter-spacing: -0.02em;}
  [data-testid="stMetricValue"] {font-size: 1.9rem;}
  .ev-ok  {color:#0E7C86;}
  .ev-warn{color:#B26A00;}
  .ev-bad {color:#B42318;}
  .ev {margin:0.1rem 0; font-size:0.92rem;}
</style>
""", unsafe_allow_html=True)

pages = [
    st.Page("views/overview.py", title="Overview", icon="🏫", default=True),
    st.Page("views/finder.py", title="Room Finder", icon="🔎"),
    st.Page("views/admin.py", title="Admin Dashboard", icon="📊"),
    st.Page("views/recommend.py", title="Recommendations", icon="🛠️"),
    st.Page("views/forecast.py", title="Demand Forecast", icon="📈"),
    st.Page("views/model_lab.py", title="Model Lab", icon="🧪"),
]
st.sidebar.markdown("### SpaceSync\nTeam Larpers · Datathon 2026 · Track 2 Education")
st.sidebar.caption("Synthetic Datathon dataset — not a live campus booking system.")
st.navigation(pages).run()
