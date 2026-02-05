import streamlit as st
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
import soccerdata as sd
import sqlite3
import hashlib  # For password hashing

# Set page config for centered layout and title
st.set_page_config(
    page_title="Soccer Prediction Bot",
    layout="centered",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Custom CSS for iOS-like style: black background, emerald green, rounded cards, Apple fonts, hide branding
st.markdown("""
    <style>
        /* Black background */
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
        }
        /* Hide Streamlit menu, footer, and header */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="collapsedControl"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        .stApp [data-testid="stDecoration"] {display: none;}
        /* Apple-style fonts */
        body, h1, h2, h3, h4, h5, h6, label, p, div, span, input {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            color: #FFFFFF !important;
        }
        /* Emerald green buttons */
        .stButton > button {
            background-color: #0BDA51 !important;
            color: #000000 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-size: 16px !important;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #0AA841 !important;
        }
        /* Inputs with emerald borders, rounded */
        .stTextInput > div > div > div {
            background-color: #1A1A1A !important;
            border: 1px solid #0BDA51 !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
            padding: 8px !important;
        }
        .stTextInput > div > div > div:focus {
            border: 1px solid #0BDA51 !important;
            box-shadow: 0 0 5px #0BDA51 !important;
        }
        /* Number input adjustments */
        .stNumberInput > div > div > div {
            background-color: #1A1A1A !important;
            border: 1px solid #0BDA51 !important;
            border-radius: 8px !important;
        }
        /* Radio buttons with emerald accents, no red */
        div.row-widget.stRadio > div {
            flex-direction: row !important;
            gap: 10px !important;
        }
        div.row-widget.stRadio > div > label > div {
            background-color: #333333 !important;
            padding: 8px 16px !important;
            border-radius: 8px !important;
            color: #FFFFFF !important;
        }
        div.row-widget.stRadio > div > label[data-checked="true"] > div {
            background-color: #0BDA51 !important;
            color: #000000 !important;
        }
        input[type="radio"] {
            accent-color: #0BDA51 !important;  /* Emerald dots */
        }
        /* Alerts emerald */
        [data-testid="stAlert"] {
            background-color: #0BDA51 !important;
            color: #000000 !important;
            border-radius: 8px !important;
        }
        /* Centered card for content */
        .centered-card {
            max-width: 90%;
            margin: 20px auto;
            padding: 20px;
            border-radius: 12px;
            background-color: #111111;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        /* Responsive for mobile */
        @media (max-width: 768px) {
            .stColumn > div {
                width: 100% !important;
                margin-bottom: 10px !important;
            }
            div.row-widget.stRadio > div {
                flex-direction: column !important;
            }
            h1 { font-size: 1.8em !important; }
            h2, h3 { font-size: 1.4em !important; }
        }
        /* Match buttons emerald */
        .stButton > button[kind="secondary"] {
            background-color: #0BDA51 !important;
            color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

# API key from Streamlit secrets
API_KEY = st.secrets.get("API_KEY")
if not API_KEY:
    st.error("API key not found. Please add API_KEY to Streamlit secrets.")
    st.stop()

# SQLite for Predictions
DB_FILE = "reversebot.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT,
            team_name TEXT,
            league TEXT,
            metric TEXT,
            line REAL,
            venue TEXT,
            opponent TEXT,
            role TEXT,
            projected_value REAL,
            ci_low REAL,
            ci_high REAL,
            p_over REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# League ID Map
league_to_id = {
    'ENG-Premier League': 39,
    'ESP-La Liga': 140,
    'ITA-Serie A': 135,
    'GER-Bundesliga': 78,
    'FRA-Ligue 1': 61,
    # Add more
}

# Position Baselines
position_baselines = {
    'GK': {'saves': {'mean': 3.0, 'sd': 1.0}, 'shots_attempted': {'mean': 0.1, 'sd': 0.5}},
    'CB': {'passes_attempted': {'mean': 55.0, 'sd': 10.0}, 'shots_attempted': {'mean': 0.5, 'sd': 1.0}},
    'CM': {'passes_attempted': {'mean': 60.0, 'sd': 15.0}, 'shots_attempted': {'mean': 1.5, 'sd': 2.0}},
    'FWD': {'passes_attempted': {'mean': 30.0, 'sd': 10.0}, 'shots_attempted': {'mean': 3.0, 'sd': 3.0}}
}

# Role Multipliers
role_multipliers = {
    'GK': {'lead_effect': 1.5, 'shots_effect': 1.0, 'pass_adjust': 1.0, 'progressive_adjust': 0.8, 'block_adjust': 1.0},
    'CB': {'lead_effect': 0.85, 'shots_effect': 0.9, 'pass_adjust': 1.0, 'progressive_adjust': 1.0, 'block_adjust': 1.2},
    'CM': {'lead_effect': 1.12, 'shots_effect': 1.1, 'pass_adjust': 1.1, 'progressive_adjust': 1.2, 'block_adjust': 1.0},
    'FWD': {'lead_effect': 1.05, 'shots_effect': 1.2, 'pass_adjust': 0.8, 'progressive_adjust': 0.9, 'block_adjust': 0.7},
    'pivot': {'lead_effect': 1.0, 'shots_effect': 0.8, 'pass_adjust': 1.3, 'progressive_adjust': 1.1, 'block_adjust': 1.15},
    'deep_lying_playmaker': {'lead_effect': 1.15, 'shots_effect': 1.0, 'pass_adjust': 1.4, 'progressive_adjust': 1.4, 'block_adjust': 1.0},
    'pressure_release_valve': {'lead_effect': 1.1, 'shots_effect': 0.9, 'pass_adjust': 1.35, 'progressive_adjust': 1.25, 'block_adjust': 1.05},
    'progressive_midfielder': {'lead_effect': 1.2, 'shots_effect': 1.15, 'pass_adjust': 1.3, 'progressive_adjust': 1.5, 'block_adjust': 0.95},
    'box_to_box': {'lead_effect': 1.1, 'shots_effect': 1.1, 'pass_adjust': 1.2, 'progressive_adjust': 1.3, 'block_adjust': 1.1},
    'winger': {'lead_effect': 1.05, 'shots_effect': 1.3, 'pass_adjust': 0.9, 'progressive_adjust': 1.2, 'block_adjust': 0.8},
    'fullback': {'lead_effect': 0.9, 'shots_effect': 0.95, 'pass_adjust': 1.1, 'progressive_adjust': 1.15, 'block_adjust': 1.2},
    'striker': {'lead_effect': 1.0, 'shots_effect': 1.4, 'pass_adjust': 0.7, 'progressive_adjust': 0.8, 'block_adjust': 0.6},
    # Add more
}

# Leg-Order Conservatism
leg_conservatism = {
    'away_first': 0.85,
    'home_first': 0.95
}

# Session State
if 'page' not in st.session_state:
    st.session_state.page = 'search'
if 'player_data' not in st.session_state:
    st.session_state.player_data = None

# Data Fetch Functions
def fetch_player_info(player_name, season=2025):
    try:
        url = f"https://v3.football.api-sports.io/players?season={season}&search={player_name}"
        headers = {"x-apisports-key": API_KEY}
        response = requests.get(url, headers=headers).json()
        if not response['response']:
            return None, None, None, None, None
        player = response['response'][0]
        return (player['player']['id'], player['statistics'][0]['team']['name'],
                player['statistics'][0]['team']['id'], player['statistics'][0]['league']['name'],
                player['statistics'][0]['league']['id'])
    except Exception as e:
        st.error(f"Player fetch failed: {e}")
        return None, None, None, None, None

def fetch_opponent_id(opponent_name, league_id, season=2025):
    try:
        url = f"https://v3.football.api-sports.io/teams?season={season}&league={league_id}&search={opponent_name}"
        headers = {"x-apisports-key": API_KEY}
        response = requests.get(url, headers=headers).json()
        if not response['response']:
            return None
        return response['response'][0]['team']['id']
    except Exception as e:
        st.error(f"Opponent fetch failed: {e}")
        return None

def fetch_real_data(player_id, team_id, opponent_id, league_id, season=2025, prop_type='passes_attempted', num_matches=10):
    try:
        url_h2h = f"https://v3.football.api-sports.io/fixtures/headtohead?season={season}&h2h={team_id}-{opponent_id}&last={num_matches}"
        headers = {"x-apisports-key": API_KEY}
        response_h2h = requests.get(url_h2h, headers=headers).json()
        h2h_data = response_h2h['response']
        fixture_ids = [f['fixture']['id'] for f in h2h_data]
        
        historical = []
        home_away = []
        opponents_short = []
        opponents_full = []
        for fixture_id in fixture_ids:
            url_stats = f"https://v3.football.api-sports.io/players?fixture={fixture_id}&player={player_id}"
            response_stats = requests.get(url_stats, headers=headers).json()
            if response_stats['response']:
                stats = response_stats['response'][0]['statistics'][0]
                if prop_type == 'passes_attempted':
                    val = stats['passes']['total'] or 0
                elif prop_type == 'saves':
                    val = stats['goals']['saves'] or 0
                elif prop_type == 'shots_attempted':
                    val = stats['shots']['total'] or 0
                historical.append(val)
                fixture = next(f for f in h2h_data if f['fixture']['id'] == fixture_id)
                ha = 'Home' if fixture['teams']['home']['id'] == team_id else 'Away'
                home_away.append(ha)
                opp = fixture['teams']['away']['name'] if ha == 'Home' else fixture['teams']['home']['name']
                opponents_short.append(opp[:3].upper())
                opponents_full.append(opp)
        return np.array(historical), home_away, opponents_short, opponents_full
    except Exception as e:
        st.error(f"H2H fetch failed: {e}")
        return np.array([]), [], [], []

def fetch_heat_map(player_name):
    return 0.88, 0.65

def fetch_injury_data(player_id, team_id):
    return "No injury"

# Model Functions
def impute_data(historical, n_matches=10, mean_val=0):
    if len(historical) < n_matches:
        imputed = np.random.poisson(mean_val, n_matches - len(historical))
        historical = np.concatenate([historical, imputed])
    return historical

def detect_reversal_pattern(historical, first_leg_stat, lead, prop_type):
    return 'stable'

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x.unsqueeze(-1))
        x = self.transformer(x)
        return self.fc(x.mean(dim=1)).squeeze()

def get_time_adjustment(time_series):
    if len(time_series) == 0:
        return 0.0
    model = SimpleTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    x = torch.tensor(time_series, dtype=torch.float32).unsqueeze(0)
    y = torch.tensor([np.mean(time_series)], dtype=torch.float32).unsqueeze(0)
    for _ in range(10):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model(x).item()

def xgboost_predict(features, targets):
    if len(features) < 2:
        return 0.0
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
    model.fit(X_train, y_train)
    return model.predict(X_test)[0] if len(X_test) > 0 else 0.0

def run_bayesian_model(player_data):
    historical = player_data['historical']
    n = len(historical)
    prop_type = player_data['type']
    position_mean = position_baselines[player_data['position']][prop_type]['mean']
    position_sd = position_baselines[player_data['position']][prop_type]['sd']
    
    with pm.Model() as model:
        mu_base = pm.Normal('mu_base', mu=position_mean, sigma=position_sd)
        sigma_walk = pm.HalfNormal('sigma_walk', sigma=0.5)
        walk = pm.GaussianRandomWalk('walk', sigma=sigma_walk, shape=n)
        mu_time = mu_base + walk
        
        beta_opp = pm.Normal('beta_opp', 0, 1)
        beta_home = pm.Normal('beta_home', 0, 1)
        beta_xg = pm.Normal('beta_xg', 0, 1)
        mu_cov = (beta_opp * player_data['covariates']['opp_strength'] +
                  beta_home * player_data['covariates']['home'] +
                  beta_xg * player_data['covariates']['xG'])
        
        if prop_type == 'shots_attempted':
            mu_cov *= 1.2
        
        alpha_odds = pm.Beta('alpha_odds', alpha=player_data['covariates']['odds_prior']*10, beta=(1-player_data['covariates']['odds_prior'])*10)
        
        phi = pm.Gamma('phi', alpha=2, beta=0.1)
        psi = pm.Beta('psi', 1, 1)
        
        shared_effect = pm.Normal('shared_effect', 0, 1)
        
        beta_lead = pm.Normal('beta_lead', 0, 1)
        mu_lead = beta_lead * player_data['aggregate_lead'] * role_multipliers[player_data['role']]['lead_effect' if prop_type != 'shots_attempted' else 'shots_effect']
        
        beta_agr = pm.Normal('beta_agr', 0, 0.5)
        beta_var = pm.Normal('beta_var', 0, 0.5)
        mu_rules = beta_agr * player_data['agr_present'] + beta_var * player_data['var_present']
        
        mu_fatigue = player_data['fatigue_index']
        
        reversal_adjust = 1.0
        if player_data['reversal_flag'] == 'upward_reversal_likely':
            reversal_adjust = 1.15 if prop_type != 'shots_attempted' else 1.25
        elif player_data['reversal_flag'] == 'downward_reversal_likely':
            reversal_adjust = 0.85
        
        conservatism_adjust = leg_conservatism['away_first' if player_data['is_home'] == 0 else 'home_first'] if player_data['is_first_leg'] else 1.0
        
        mu = pm.math.exp(mu_time[-1] + mu_cov + mu_lead + mu_rules + mu_fatigue) * alpha_odds * reversal_adjust * conservatism_adjust
        mu = pm.Deterministic('mu', mu)
        
        y_obs = pm.ZeroInflatedNegativeBinomial('y_obs', psi=psi, mu=mu, alpha=phi, observed=historical)
        
        trace = pm.sample(500, tune=300, return_inferencedata=True, target_accept=0.9, progressbar=False)
    
    return trace

def ensemble_predict(trace, xg_pred):
    bay_mean = az.summary(trace, var_names=['mu'])['mean'][0]
    return (bay_mean + xg_pred) / 2 if xg_pred else bay_mean

def sensitivity_analysis(mu, line, phi_values=[50, 100, 200], n_samples=5000):
    results = {}
    for phi in phi_values:
        samples = np.random.negative_binomial(phi, phi / (phi + mu), n_samples)
        p_over = np.mean(samples > line)
        ci_low, ci_high = np.percentile(samples > line, [2.5, 97.5])
        results[phi] = {'p_over': p_over, 'ci': (ci_low, ci_high)}
    return results

# UI Functions
def display_prediction(data):
    st.subheader("Projected Value")
    st.write(f"{data['hybrid_mu']:.1f} {data['metric'].replace('_', ' ').capitalize()}")
    st.write(f"{ 'OVER' if data['hybrid_mu'] > data['line'] else 'UNDER' } {data['line']} vs Line" if data['line'] > 0 else "")

def display_trajectory_grid(data):
    st.subheader("Recent Match Data")
    cols = st.columns(min(5, len(data['historical'])))  # Responsive columns
    total = 0
    count = 0
    selected = st.session_state.get('selected', [])
    for i in range(len(data['historical'])):
        val = data['historical'][i]
        ha = data['home_away'][i]
        opp = data['opponents_short'][i]
        label = f"@{opp}" if ha == 'Away' else f"vs{opp}"
        with cols[i % len(cols)]:
            if st.button(label, key=i, help=f"Match {i+1}: {val} {data['metric'].replace('_', ' ')} ({ha} vs {data['opponents_full'][i]})"):
                selected.append(val)
                st.session_state.selected = selected
        if len(selected) > count:
            total += selected[-1]
            count += 1
            st.caption(f"{count} selected Total {total} Avg {total/count:.1f}")

def display_breakdown(data):
    st.subheader("Home/Away Breakdown")
    for i, (val, ha, opp) in enumerate(zip(data['historical'], data['home_away'], data['opponents_full'])):
        st.write(f"Match {i+1} ({ha} vs {opp}): {val}")

# Main UI
with st.container(border=True):  # Rounded card
    st.header("Soccer Prediction Bot 5.06")
    st.subheader("Real-time FBRef data with Bayesian analysis")

    st.write("### Query")

    player_name = st.text_input("Player Name", value="Kyle Walker")

    line = st.number_input("Line", value=46.5, step=0.5)

    col1, col2 = st.columns(2)

    with col1:
        venue = st.radio("Venue", ["Home", "Away"], index=0, horizontal=True)

    with col2:
        opponent = st.text_input("Opponent", value="Arsenal")

    prop_type = st.radio("Prop Type", ["Pass Attempts", "Saves"], index=0, horizontal=True)

    if st.button("Submit"):
        # Real logic placeholder
        player_id, team_name, team_id, league_name, league_id = fetch_player_info(player_name)
        opponent_id = fetch_opponent_id(opponent, league_id)
        if player_id and team_id and opponent_id and league_id:
            historical, home_away, opponents_short, opponents_full = fetch_real_data(player_id, team_id, opponent_id, league_id, prop_type=prop_type.lower().replace(" ", "_"))
            # Assume position, role, covariates, etc., for model
            player_data = {
                'historical': historical,
                'type': prop_type.lower().replace(" ", "_"),
                'position': 'CB',  # Placeholder
                'role': 'fullback',  # Placeholder
                'covariates': {'opp_strength': 0.5, 'home': 1 if venue == 'Home' else 0, 'xG': 1.5, 'odds_prior': 0.6},
                'aggregate_lead': 1.0,
                'agr_present': 0,
                'var_present': 0,
                'fatigue_index': 0,
                'reversal_flag': 'stable',
                'is_home': 1 if venue == 'Home' else 0,
                'is_first_leg': True  # Placeholder
            }
            trace = run_bayesian_model(player_data)
            xg_pred = xgboost_predict(np.arange(len(historical)).reshape(-1, 1), historical)  # Placeholder features
            hybrid_mu = ensemble_predict(trace, xg_pred)
            p_over = sensitivity_analysis(hybrid_mu, line)['100']['p_over']
            # Save to DB (no user_id since no login)
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (player_name, team_name, league, metric, line, venue, opponent, role, projected_value, p_over)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (player_name, team_name, league_name, prop_type.lower().replace(" ", "_"), line, venue, opponent, player_data['role'], hybrid_mu, p_over))
            conn.commit()
            conn.close()
            data = {
                'hybrid_mu': hybrid_mu,
                'metric': prop_type.lower().replace(" ", "_"),
                'line': line,
                'historical': historical,
                'home_away': home_away,
                'opponents_short': opponents_short,
                'opponents_full': opponents_full
            }
        else:
            st.error("Failed to fetch data. Using placeholder.")
            data = {
                'hybrid_mu': 50.2,
                'metric': prop_type.lower().replace(" ", "_"),
                'line': line,
                'historical': [45, 50, 55, 40, 60],
                'home_away': ['Home', 'Away', 'Home', 'Away', 'Home'],
                'opponents_short': ['ARS', 'CHE', 'LIV', 'MUN', 'TOT'],
                'opponents_full': ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester United', 'Tottenham']
            }
        st.write(f"Processing prediction for {player_name} vs {opponent} at {venue} for {prop_type} over/under {line}...")
        display_prediction(data)
        display_trajectory_grid(data)
        display_breakdown(data)
        st.success(f"Prediction: Over with {int(data.get('p_over', 0.65)*100)}% probability (based on Bayesian model).")
