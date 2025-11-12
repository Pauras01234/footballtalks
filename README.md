# footballtalks
ai score predictor based on climate and past records


# ⚽ FootballTalks — Live Scoreline & Weather Insights

An **AI-powered football analytics app** built with **Streamlit**, combining **live match data**, **real-time weather**, and **Poisson-based predictive modeling** to estimate **scorelines and win probabilities**.

![FootballTalks Banner](assets/match_view.png)

---

## 🚀 Features

✅ Live or upcoming matches from top leagues (EPL, LaLiga, Serie A, Bundesliga, UCL, etc.)  
✅ Integrated **OpenWeather** and **Google Maps** APIs for stadium weather and live location  
✅ Dynamic **Poisson model**–based match outcome prediction  
✅ Interactive Streamlit dashboard with real-time refresh  
✅ AI-generated win/draw/away probabilities  

---

## 🧠 How It Works

FootballTalks uses a statistical Poisson model to simulate the number of goals each team scores based on past performance and form metrics:

\[
P(k; λ) = \frac{e^{-λ} λ^k}{k!}
\]

Where:
- **λ (lambda)** = average goals per match adjusted for opponent strength and recent form.  
- **k** = number of goals scored.

The simulation runs for multiple scorelines to estimate **win/draw probabilities**.

---

## 🛠 Tech Stack

- **Python 3.11+**
- **Streamlit** — interactive UI  
- **NumPy, Pandas** — data processing  
- **Plotly** — visualizations  
- **Requests** — API calls  
- **OpenWeather, Google Maps, Football-Data APIs**

---

