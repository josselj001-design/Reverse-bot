import streamlit as st
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Soccer Prediction Bot",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for iOS-like style
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
        }
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="collapsedControl"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        .stApp [data-testid="stDecoration"] {display: none;}
        body, h1, h2, h3, h4, h5, h6, label, p, div, span, input {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            color: #FFFFFF !important;
        }
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
        .stNumberInput > div > div > div {
            background-color: #1A1A1A !important;
            border: 1px solid #0BDA51 !important;
            border-radius: 8px !important;
        }
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
            accent-color: #0BDA51 !important;
            -webkit-accent-color: #0BDA51 !important;
        }
        [data-testid="stAlert"] {
            background-color: #0BDA51 !important;
            color: #000000 !important;
            border-radius: 8px !important;
            border: none !important;
        }
        .centered-card {
            max-width: 90%;
            margin: 20px auto;
            padding: 20px;
            border-radius: 12px;
            background-color: #111111;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
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
        .stButton > button[kind="secondary"] {
            background-color: #0BDA51 !important;
            color: #000000 !important;
            width: auto !important;
        }
    </style>
""", unsafe_allow_html=True)

# API key (placeholder; add to secrets)
API_KEY = st.secrets.get("API_KEY", "your_default_key_for_testing")  # Fallback for local

# Session State
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'data' not in st.session_state:
    st.session_state.data = None

# UI Functions
def render_query_form():
    st.write("### Query")
    player_name = st.text_input("Player Name", value="Kyle Walker")
    line = st.number_input("Line", value=46.5, step=0.5)
    col1, col2 = st.columns(2)
    with col1:
        venue = st.radio("Venue", ["Home", "Away"], index=0, horizontal=True)
    with col2:
        opponent = st.text_input("Opponent", value="Arsenal")
    prop_type = st.radio("Prop Type", ["Pass Attempts", "Saves"], index=0, horizontal=True)
    return player_name, line, venue, opponent, prop_type

def process_submission(player_name, line, venue, opponent, prop_type):
    # Placeholder for fetch/model (implement later)
    st.session_state.submitted = True
    st.session_state.data = {
        'hybrid_mu': 50.2,
        'metric': prop_type.lower().replace(" ", "_"),
        'line': line,
        'historical': np.array([45, 50, 55, 40, 60]),
        'home_away': ['Home', 'Away', 'Home', 'Away', 'Home'],
        'opponents_short': ['ARS', 'CHE', 'LIV', 'MUN', 'TOT'],
        'opponents_full': ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester United', 'Tottenham'],
        'p_over': 0.65,
        'ci_low': 0.6,
        'ci_high': 0.7
    }
    st.rerun()  # Refresh to show results

def display_results(data):
    st.write(f"Prediction for {st.session_state.player_name} vs {st.session_state.opponent} at {st.session_state.venue} for {st.session_state.prop_type} over/under {st.session_state.line}:")
    display_prediction(data)
    display_trajectory_grid(data)
    display_breakdown(data)
    st.success(f"Prediction: Over with {int(data['p_over'] * 100)}% probability (CI: {int(data['ci_low']*100)}% - {int(data['ci_high']*100)}%) based on Bayesian model.")
    # Additional features (placeholders)
    st.info("Injury Status: No injury")
    st.info("Heat Map: Mean 0.88, SD 0.65")
    display_chart(data['historical'])

def display_prediction(data):
    st.subheader("Projected Value")
    st.write(f"{data['hybrid_mu']:.1f} {data['metric'].replace('_', ' ').capitalize()}")
    st.write(f"{ 'OVER' if data['hybrid_mu'] > data['line'] else 'UNDER' } {data['line']} vs Line" if data['line'] > 0 else "")

def display_trajectory_grid(data):
    st.subheader("Recent Match Data")
    cols = st.columns(min(5, len(data['historical'])))
    total = 0
    count = 0
    selected = st.session_state.get('selected', [])
    for i in range(len(data['historical'])):
        val = data['historical'][i]
        ha = data['home_away'][i]
        opp = data['opponents_short'][i]
        label = f"@{opp}" if ha == 'Away' else f"vs{opp}"
        with cols[i % len(cols)]:
            if st.button(label, key=i, help=f"Match {i+1}: {val} {data['metric'].replace('_', ' ')} ({ha} vs {data['opponents_full'][i]})", type="secondary"):
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

def display_chart(historical):
    if len(historical) > 0:
        df = pd.DataFrame({'Match': range(1, len(historical) + 1), 'Value': historical})
        chart = alt.Chart(df).mark_bar(color='#0BDA51').encode(
            x='Match:O',
            y='Value:Q'
        ).properties(width='100%')
        st.altair_chart(chart, use_container_width=True)

# Main UI
st.markdown('<div class="centered-card">', unsafe_allow_html=True)
st.header("Soccer Prediction Bot 5.06")
st.subheader("Real-time FBRef data with Bayesian analysis")

player_name, line, venue, opponent, prop_type = render_query_form()

if st.button("Submit", type="primary"):
    st.session_state.player_name = player_name
    st.session_state.line = line
    st.session_state.venue = venue
    st.session_state.opponent = opponent
    st.session_state.prop_type = prop_type
    process_submission(player_name, line, venue, opponent, prop_type)

if st.session_state.submitted and st.session_state.data:
    display_results(st.session_state.data)

st.markdown('</div>', unsafe_allow_html=True)
