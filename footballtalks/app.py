import os, math, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ---------- Streamlit UI ----------
st.set_page_config(page_title="⚽ FootballTalks", layout="wide")
st.title("⚽ FootballTalks — Live Scoreline & Weather Insights")

# ---------- Environment ----------
load_dotenv()
FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
HEADERS = {"X-Auth-Token": FOOTBALL_KEY}

# ---------- Safe Imports ----------
if not isinstance(pd, type(np)):  # ensure pandas wasn't shadowed
    import importlib
    pd = importlib.import_module("pandas")


# ---------- Helper functions ----------
def get_json(url, headers=None, retry=2):
    for _ in range(retry + 1):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
            time.sleep(0.6)
        except Exception:
            time.sleep(0.6)
    return None


@st.cache_data(show_spinner=False)
def list_competitions():
    j = get_json("https://api.football-data.org/v4/competitions", HEADERS)
    if not j:
        return {}
    comps = [c for c in j["competitions"] if c.get("plan") == "TIER_ONE"]
    return {f"{c['name']} ({c['code']})": c["code"] for c in comps}


@st.cache_data(show_spinner=False)
def list_matches_any_status(comp_code):
    base_url = f"https://api.football-data.org/v4/competitions/{comp_code}/matches"
    for status in ["LIVE", "SCHEDULED", "IN_PLAY", "PAUSED", "TIMED", "FINISHED"]:
        j = get_json(f"{base_url}?status={status}", HEADERS)
        if j and isinstance(j.get("matches"), list) and len(j["matches"]) > 0:
            return j["matches"]
    return []


@st.cache_data(show_spinner=False)
def team_recent_matches(team_id, limit=8):
    j = get_json(f"https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit={limit}", HEADERS)
    if not j or "matches" not in j or not isinstance(j["matches"], list):
        return []
    return j["matches"]


def geocode_coords(team_name):
    if not GOOGLE_KEY:
        st.error("Missing GOOGLE_MAPS_API_KEY in .env")
        st.stop()

    name_map = {
        "CA Mineiro": "Clube Atlético Mineiro Arena MRV Brazil",
        "Fortaleza EC": "Estádio Castelão Fortaleza Brazil",
        "Burnley FC": "Turf Moor Burnley England",
        "Chelsea FC": "Stamford Bridge London",
        "Real Madrid CF": "Santiago Bernabéu Madrid",
        "FC Barcelona": "Camp Nou Barcelona",
        "Liverpool FC": "Anfield Liverpool",
        "Manchester City FC": "Etihad Stadium Manchester",
        "Manchester United FC": "Old Trafford Manchester",
        "Juventus FC": "Allianz Stadium Turin",
        "Paris Saint-Germain FC": "Parc des Princes Paris",
        "Bayern München": "Allianz Arena Munich",
        "Arsenal FC": "Emirates Stadium London",
        "Tottenham Hotspur FC": "Tottenham Hotspur Stadium London",
        "AC Milan": "San Siro Milan",
        "Inter Milano": "San Siro Milan",
        "AS Roma": "Stadio Olimpico Rome"
    }

    query = name_map.get(team_name, f"{team_name} football stadium")
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={query}&key={GOOGLE_KEY}"
    j = get_json(url)
    if j and j.get("status") == "OK":
        loc = j["results"][0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])
    st.error(f"❌ Could not fetch stadium coordinates for {team_name}. Query used: {query}")
    st.stop()


def get_weather(lat, lon):
    if not WEATHER_KEY:
        return ("unknown", 18.0, 60)
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_KEY}&units=metric"
    j = get_json(url)
    if not j or "weather" not in j or "main" not in j:
        return ("unknown", 18.0, 60)
    desc = j["weather"][0]["description"]
    temp = float(j["main"]["temp"])
    hum = int(j["main"]["humidity"])
    return (desc, temp, hum)


def _gf(team_id, match):
    full = match["score"]["fullTime"]
    return full["home"] if team_id == match["homeTeam"]["id"] else full["away"]


def _ga(team_id, match):
    full = match["score"]["fullTime"]
    return full["away"] if team_id == match["homeTeam"]["id"] else full["home"]


def recent_team_stats(team_id, n=8):
    matches = team_recent_matches(team_id, n)
    if not matches:
        return dict(gf=1.2, ga=1.1, pts_per_game=1.5, form_val=0.0)
    gf = ga = pts = 0
    count = 0
    for m in matches:
        ft = m.get("score", {}).get("fullTime", {})
        if ft is None or ft.get("home") is None or ft.get("away") is None:
            continue
        count += 1
        gf += _gf(team_id, m)
        ga += _ga(team_id, m)
        w = m["score"]["winner"]
        if w == "DRAW":
            pts += 1
        elif w == "HOME_TEAM" and team_id == m["homeTeam"]["id"]:
            pts += 3
        elif w == "AWAY_TEAM" and team_id == m["awayTeam"]["id"]:
            pts += 3
    games = max(count, 1)
    ppg = pts / games
    return dict(gf=gf/games, ga=ga/games, pts_per_game=ppg, form_val=ppg - 1.0)


def poisson_prob(k, lam):
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def score_matrix(lh, la, max_goals=6):
    g = np.zeros((max_goals+1, max_goals+1))
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            g[h, a] = poisson_prob(h, lh) * poisson_prob(a, la)
    return g


def outcome_probs(g):
    home = np.tril(g, -1).sum()
    draw = np.trace(g)
    away = np.triu(g, 1).sum()
    s = home + draw + away
    return home/s, draw/s, away/s


# ---------- MAIN ----------
if not FOOTBALL_KEY:
    st.error("Missing FOOTBALL_API_KEY in .env")
    st.stop()

comps = list_competitions()
if not comps:
    st.error("Could not fetch competitions.")
    st.stop()

league_name = st.sidebar.selectbox("🏆 Competition", list(comps.keys()))
league_code = comps[league_name]
matches = list_matches_any_status(league_code)
if not matches:
    st.warning("No matches found for this competition.")
    st.stop()

match_list = [f"{m['homeTeam']['name']} vs {m['awayTeam'].get('name','TBD')} — {m['utcDate'][:10]}" for m in matches]
chosen = st.sidebar.selectbox("⚔️ Match", match_list)
m = matches[match_list.index(chosen)]
home, away = m["homeTeam"], m["awayTeam"]

c1, c2, c3 = st.columns([1, 2, 1])
home_name = home.get("name", "TBD")
away_name = (away or {}).get("name", "TBD")
home_crest = home.get("crest")
away_crest = (away or {}).get("crest")

with c1:
    if home_crest:
        st.image(home_crest, width=90)
with c2:
    st.markdown(f"### 🏟 {home_name} vs {away_name}")
    ko = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")).astimezone(timezone.utc)
    st.caption(f"Kick-off (UTC): {ko.strftime('%Y-%m-%d %H:%M')}")
with c3:
    if away_crest:
        st.image(away_crest, width=90)

lat, lon = geocode_coords(home_name)
desc, temp, hum = get_weather(lat, lon)
wc1, wc2, wc3 = st.columns(3)
with wc1: st.metric("🌤 Weather", desc)
with wc2: st.metric("🌡 Temp (°C)", f"{temp:.1f}")
with wc3: st.metric("💧 Humidity (%)", hum)

map_html = f"""
<iframe width="100%" height="300" frameborder="0" style="border:0"
src="https://www.google.com/maps/embed/v1/view?key={GOOGLE_KEY}&center={lat},{lon}&zoom=13"
allowfullscreen></iframe>
"""
components.html(map_html, height=320)

hs, as_ = recent_team_stats(home["id"]), recent_team_stats((away or {}).get("id", home["id"]))
lh = max(0.2, 0.65*hs["gf"] + 0.35*as_["ga"]) * 1.12
la = max(0.2, 0.65*as_["gf"] + 0.35*hs["ga"])
if desc.split()[0].lower() in ("rain", "snow"):
    lh, la = lh*0.95, la*0.95
lh, la = min(lh, 3.2), min(la, 3.2)
grid = score_matrix(lh, la)
ph, pd, pa = outcome_probs(grid)
hi, ai = np.unravel_index(np.argmax(grid), grid.shape)

st.subheader("🔮 Predicted Result")
st.markdown(f"**Predicted Scoreline:** {home_name} **{hi} – {ai}** {away_name}")

fig = go.Figure(go.Bar(
    x=["Home Win", "Draw", "Away Win"],
    y=[ph*100, pd*100, pa*100],
    text=[f"{ph*100:.1f}%", f"{pd*100:.1f}%", f"{pa*100:.1f}%"],
    textposition="auto",
    marker_color=["#2ecc71", "#f1c40f", "#e74c3c"]
))
fig.update_layout(title="Win / Draw / Away Probabilities", yaxis_title="%", height=350)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 Recent Form (last 5 games)")
hc, ac = st.columns(2)
for team, col in [(home, hc), (away, ac)]:
    try:
        matches = team_recent_matches(team["id"], limit=5)
        if not matches:
            col.markdown(f"**{team.get('name','TBD')}** — no recent games found.")
            continue
        rows = []
        for m in matches:
            ft = m.get("score", {}).get("fullTime", {})
            if not isinstance(ft, dict):
                continue
            rows.append({
                "Opponent": (m["awayTeam"]["name"] if m["homeTeam"]["id"] == team["id"] else m["homeTeam"]["name"]),
                "Result": m["score"]["winner"],
                "Score": f"{ft.get('home','?')} - {ft.get('away','?')}"
            })
        if rows:
            df = pd.DataFrame(rows)
            col.markdown(f"**{team['name']}**")
            col.dataframe(df, use_container_width=True, hide_index=True)
        else:
            col.markdown(f"**{team['name']}** — no valid results yet.")
    except Exception as e:
        col.error(f"Error loading {team.get('name','team')} data: {e}")
