import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import base64
import time
import io
from PIL import Image
import google.generativeai as genai

from engine import (
    DynamicEBShrinker,
    FeatureProcessor,
    NumericSimilarity,
    NLPSimilarity,
    CompositeRanker,
)

genai.configure(api_key=st.secrets["LLM_API_KEY"])

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="The Armchair Scout",
    page_icon="🏏",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* 1. Metric Container / Card Styling */
    [data-testid="stMetric"] {
        background-color: #F5F5F5 !important;
        border: 1px solid #E0E0E0 !important;
        border-top: 4px solid #d0378d !important;
        border-radius: 8px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04) !important;
    }

    /* 2. Metric Label */
    [data-testid="stMetricLabel"] {
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #757575 !important;
        text-transform: capitalize !important;
        letter-spacing: 0.5px !important;
    }

    /* 3. Metric Value */
    [data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 800 !important;
        color: #212121 !important;
    }

    /* 4. Metric Delta */
    [data-testid="stMetricDelta"] {
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* --- SIDEBAR NAVIGATION --- */
    [data-testid="stSidebarNav"] li { margin-bottom: 14px !important; }
    [data-testid="stSidebarNav"] span { font-size: 16px !important; font-weight: 600 !important; letter-spacing: 0.5px !important; }
    [data-testid="stSidebarNav"] a[aria-current="page"] span { color: #F5F5F5 !important; font-weight: 800 !important; }
    [data-testid="stSidebarNav"] a:hover span { color: #F5F5F5 !important; opacity: 0.8; }

    /* --- SELECTBOX / MULTISELECT TEXT COLOR --- */
    .stSelectbox div[data-baseweb="select"] div { color: black !important; }
    .stMultiSelect div[data-baseweb="tag"] span { color: black !important; }
    .stMultiSelect div[data-baseweb="select"] div { color: black !important; }
    div[data-baseweb="popover"] ul li div { color: black !important; }
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.title("🏏 The Armchair Scout: Building a Player Similarity Engine App")
st.markdown("*An enterprise-grade U19 cricket scouting engine powered by Empirical Bayes statistical shrinkage, multidimensional profile matching, and LLM-driven coach insights.*")
st.logo("images/logo.png", size="large")

st.divider()

# --- CACHED DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv("U19_CC_Final.csv")

    pct_cols = ['Dots %', 'Boundary Runs %', 'Six Runs %']
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    num_cols = [
        'Inns', 'Runs', 'Balls', 'Outs', 'Avg', 'Strike rate', 
        'Dots', 'Dots %', 'Boundary Runs %', 'Boundary Rate', 
        'Dots Rate', 'Out Rate', 'Six Runs %', 'Six Rate', 'Non-boundary SR'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

df = load_data()

# --- CONSTANTS ---
ALL_KEYWORDS = [
    'Left-arm', 'Right-arm', 
    'Back of Length', 'Full Toss', 'Half Volley', 'Length Delivery', 'Short Length', 'Yorker',
    'On Stumps', 'Outside Leg', 'Outside Off', 'Wide Off',
    'Back Foot', 'Backs Away', 'Down The Track', 'Front Foot', 'Moves Front',
    'Cut', 'Drive', 'Flick', 'Hook', 'Pull', 'Slog', 'SlogSweep', 'Sweep',
    'Googly', 'In Swing', 'Leg Break', 'Off Break', 'Off Cutter', 'Out Swing', 'Seam Away', 'Slower Ball',
    'Point', 'Square Leg', 'Covers', 'Mid Wicket', 'Fine Leg', 'Long Off', 'Long On', 'Third Man'
]

# --- CACHED IMAGE PROCESSING ---
@st.cache_data
def get_dynamic_avatar_base64(role, img_dir="images"):
    cleaned_role = role.strip().lower().replace(" ", "-")
    filepath = os.path.join(img_dir, f"{cleaned_role}.png")
    try:
        if not os.path.exists(filepath): raise FileNotFoundError
        img = Image.open(filepath).resize((130, 130), Image.Resampling.LANCZOS)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(img_buffer.getvalue()).decode('utf-8')}"
    except FileNotFoundError:
        return "https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_1280.png"


# SIDEBAR CONTROLS & ENGINE CONFIGURATION
selected_roles = st.sidebar.multiselect("Select Player Role(s):", options=df['Player Role'].unique())

if selected_roles:
    players = df[df['Player Role'].isin(selected_roles)]['Player'].unique()
    target_player = st.sidebar.selectbox("Select Target Player:", options=players)
    
    strengths = st.sidebar.segmented_control("Match Strengths (Optional):", selection_mode='multi', options=ALL_KEYWORDS)
    weaknesses = st.sidebar.segmented_control("Match Weaknesses (Optional):", selection_mode='multi', options=ALL_KEYWORDS)
    
    st.sidebar.divider()

    # --- ADVANCED CONFIGURATION ---
    with st.sidebar.expander("Advanced: Feature Weights"):
        st.caption("Adjust metric importance (Multiplier on Distance)")
        user_weights = {
            "Avg":              st.slider("Avg",              0.50, 1.50, 1.30, 0.05),
            "Strike rate":      st.slider("Strike Rate",      0.50, 1.50, 1.30, 0.05),
            "Boundary Runs %":  st.slider("Boundary Runs %",  0.50, 1.50, 1.20, 0.05),
            "Boundary Rate":    st.slider("Boundary Rate",    0.50, 1.50, 1.20, 0.05),
            "Six Runs %":       st.slider("Six Runs %",       0.50, 1.50, 1.20, 0.05),
            "Six Rate":         st.slider("Six Rate",         0.50, 1.50, 1.20, 0.05),
            "Non-boundary SR":  st.slider("Non-boundary SR",  0.50, 1.50, 1.00, 0.05),
            "Out Rate":         st.slider("Out Rate",         0.50, 1.50, 1.00, 0.05),
            "Dots %":           st.slider("Dots %",           0.50, 1.50, 1.00, 0.05),
            "Dots Rate":        st.slider("Dots Rate",        0.50, 1.50, 1.00, 0.05),
        }

    with st.sidebar.expander("Composite Score Weights"):
        st.caption("How much each component drives the final ranking")
        comp_numeric  = st.slider("Statistical Weightage",  0.0, 1.0, 0.70, 0.05)
        comp_strength = st.slider("Strengths Weightage",  0.0, 1.0, 0.20, 0.05)
        comp_weakness = st.slider("Weaknesses Weightage", 0.0, 1.0, 0.10, 0.05)
        composite_weights = {'numeric': comp_numeric, 'strength': comp_strength, 'weakness': comp_weakness}

    st.sidebar.divider()
    top_n = st.sidebar.slider("Top N Results", 5, 20, 10)

    # RUN BUTTON (Must be below all configuration inputs)
    st.sidebar.divider()
    run_engine = st.sidebar.button("Run Similarity Engine", type="primary", use_container_width=True)

    # =========================================================
    # TARGET PLAYER DISPLAY CARD
    # =========================================================
    target_data = df[df['Player'] == target_player].iloc[0]
    dynamic_avatar_src = get_dynamic_avatar_base64(target_data['Player Role'])

    strengths_text = target_data['Player Strengths'] if pd.notna(target_data['Player Strengths']) else "—"
    weaknesses_text = target_data['Player Weaknesses'] if pd.notna(target_data['Player Weaknesses']) else "—"

    st.markdown("""
<style>
.profile-card { background: linear-gradient(135deg, #391b4e 0%, #2a1438 60%, #1a0c24 100%); border-radius: 14px; padding: 22px 26px; display: flex; align-items: center; gap: 24px; box-shadow: 0 8px 24px rgba(57, 27, 78, 0.30); border: 1px solid rgba(208, 55, 141, 0.22); margin-bottom: 12px; position: relative; overflow: hidden; }
.profile-card::before { content: ''; position: absolute; top: -30px; right: -30px; width: 120px; height: 120px; background: radial-gradient(circle, rgba(208,55,141,0.30) 0%, transparent 70%); border-radius: 50%; pointer-events: none; }
.profile-avatar { flex-shrink: 0; width: 135px; height: 135px; border-radius: 50%; object-fit: cover; border: 3px solid #d0378d; box-shadow: 0 0 0 4px rgba(208,55,141,0.20), 0 4px 12px rgba(0,0,0,0.35); transform: translateX(45px); }
.profile-center { flex-shrink: 0; min-width: 160px; }
.profile-name { font-size: 30px !important; font-weight: 1000; color: #FFFFFF !important; margin: 0 0 6px 0 !important; line-height: 1.25; transform: translateX(65px); }
.profile-role-badge { display: inline-block; background: linear-gradient(90deg, #d0378d, #e05aa3); color: #FFFFFF !important; font-size: 11px; font-weight: 700; padding: 3px 12px; border-radius: 20px; letter-spacing: 0.5px; text-transform: uppercase; transform: translateX(65px); }
.profile-sw { flex: 1; min-width: 0; }
.sw-label { font-size: 11px; font-weight: 700; letter-spacing: 0.7px; text-transform: uppercase; margin-bottom: 3px; transform: translateX(95px); }
.sw-label.str { color: #4ade80 !important; }
.sw-label.wk  { color: #f87171 !important; }
.sw-text { font-size: 13px; color: rgba(255,255,255,0.80) !important; line-height: 1.45; margin-bottom: 10px; transform: translateX(95px); }
</style>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="profile-card">
<img class="profile-avatar" src="{dynamic_avatar_src}" alt="avatar"/>
<div class="profile-center">
<p class="profile-name">{target_player}</p>
<div class="profile-role-badge">{target_data['Player Role']} Batter</div>
</div>
<div class="profile-sw">
<div class="sw-label str">● Strengths</div>
<div class="sw-text">{strengths_text}</div>
<div class="sw-label wk">● Weaknesses</div>
<div class="sw-text">{weaknesses_text}</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # TARGET METRICS SNAPSHOT
    peer_df = df[(df['Player Role'].isin(selected_roles)) & (df['Player'] != target_player)]

    def clean_val(v):
        if pd.isna(v): return 0.0
        if isinstance(v, (int, float)): return float(v)
        try: return float(str(v).replace('%', '').strip())
        except ValueError: return 0.0

    metrics_config = [
        ("Inns",             "Inns",            False, False),
        ("Runs",             "Runs",            False, False),
        ("Avg",              "Average",         False, False),
        ("Strike rate",      "Strike Rate",     False, False),
        ("Dots %",           "Dots %",          True,  True), 
        ("Boundary Runs %",  "Boundary Runs %", False, True),
        ("Boundary Rate",    "Boundary Rate",   True,  False),
        ("Out Rate",         "Out Rate",        False, False),
        ("Six Runs %",       "Six Runs %",      False, True),
        ("Six Rate",         "Six Rate",        True,  False)
    ]

    def render_metric_card(col_key: str, label: str, lower_is_better: bool, is_pct: bool):
        val = clean_val(target_data[col_key])
        peer_mean = peer_df[col_key].apply(clean_val).mean() if len(peer_df) > 0 and col_key in peer_df.columns else 0.0
        delta_val = val - peer_mean

        if is_pct:
            display_val, delta_display = f"{val:.2f}%", f"{delta_val:+.2f}%"
        elif col_key in ["Inns", "Runs"]:
            display_val, delta_display = f"{val:.0f}", f"{delta_val:+.1f}"
        else:
            display_val, delta_display = f"{val:.2f}", f"{delta_val:+.2f}"

        st.metric(
            label=label, value=display_val, delta=delta_display,
            delta_color="inverse" if lower_is_better else "normal",
            help=f"vs Peer Avg: {peer_mean:.2f}{'%' if is_pct else ''}"
        )

    cols_row1 = st.columns(5)
    for col_widget, item in zip(cols_row1, metrics_config[:5]):
        key, label, lower_better, is_pct = item
        with col_widget: render_metric_card(key, label, lower_better, is_pct)

    st.write("") 

    cols_row2 = st.columns(5)
    for col_widget, item in zip(cols_row2, metrics_config[5:]):
        key, label, lower_better, is_pct = item
        with col_widget: render_metric_card(key, label, lower_better, is_pct)

    st.divider()


    # ENGINE EXECUTION
    if run_engine:
        # Initialize progress bar
        progress_text = "Initializing Engine..."
        my_bar = st.sidebar.progress(0, text=progress_text)
        
        # ── Filter ──
        role_pool = df[df['Player Role'].isin(selected_roles)].copy()

        # ── Stage 1: EB Shrinkage ──
        shrinker = DynamicEBShrinker()
        shrunk_df = shrinker.fit_transform(role_pool)

        # ── Stage 2: Feature Processing ──
        processor = FeatureProcessor(custom_weights=user_weights)
        processed_df = processor.fit_transform(shrunk_df)

        # ── Stage 3: Euclidean Numeric Similarity ──
        num_engine = NumericSimilarity()
        numeric_scores = num_engine.compute(processed_df, target_player)

        # ── Stage 4: TF-IDF NLP Similarity ──
        nlp_engine = NLPSimilarity()
        nlp_scores = nlp_engine.compute(
            df=role_pool,
            target_player=target_player,
            user_strengths=strengths,
            user_weaknesses=weaknesses
        )

        # ── Stage 5A: Composite Rank ──
        results_raw = numeric_scores.merge(nlp_scores, on='Player', how='inner')
        ranker = CompositeRanker(composite_weights=composite_weights)
        top_results = ranker.rank(results_raw, top_n=top_n)

        # Enrich for Display
        display_cols = ['Player', 'Player Role', 'Avg', 'Strike rate',
                        'Boundary Runs %', 'Six Runs %', 'Dots %', 'Balls',
                        'Player Strengths', 'Player Weaknesses']
        top_results = top_results.merge(
            shrunk_df[[c for c in display_cols if c in shrunk_df.columns]],
            on='Player', how='left'
        )

        # Complete and clear the bar
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)

        time.sleep(0.5)
        my_bar.empty()

        # Stash in Session State
        st.session_state['top_results'] = top_results
        st.session_state['engine_ran']  = True


    # RESULTS DISPLAY
    if st.session_state.get('engine_ran', False):
        top_results = st.session_state['top_results']

        st.markdown("### 🏆 Top Similarity Players")

        # --- ADDED: POLAR AREA CHART ---
        fig = px.bar_polar(
            top_results,
            r="overall_similarity",
            theta="Player",
            color="overall_similarity",
            color_continuous_scale="RdPu", 
            range_r=[0, 100],
            labels={"overall_similarity": "Overall Similarity %"},
            # custom_data is required to pass these columns into the hovertemplate
            custom_data=["Player Role", "numeric_score", "strength_score", "weakness_score"]
        )
        
        # 1. Attractive HTML Hover Template
        fig.update_traces(
            hovertemplate=(
                "<b style='font-size:16px;'>%{theta}</b><br>"
                "<i style='font-size:12px;'>%{customdata[0]}</i><br><br>"
                "<b>Overall Similarity: %{r:.1f}%</b><br>"
                "Stats %: %{customdata[1]:.1f}%<br>"
                "Strengths %: %{customdata[2]:.1f}%<br>"
                "Weaknesses %: %{customdata[3]:.1f}%<br>"
                "<extra></extra>" # This removes the secondary box with the trace name
            )
        )
        
        # 2. Transparent Backgrounds & Styled Hover Box
        fig.update_layout(
            margin=dict(t=30, b=30, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", # Transparent figure background
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot background
            polar=dict(
                bgcolor="rgba(0,0,0,0)",   # Transparent polar grid background
                radialaxis=dict(
                    visible=True, range=[0, 100], 
                    showticklabels=True, 
                    gridcolor="rgba(128, 128, 128, 0.2)" # Subtle grid lines
                ),
                angularaxis=dict(
                    direction="clockwise", 
                    tickfont=dict(size=15), 
                    gridcolor="rgba(128, 128, 128, 0.2)"
                )
            ),
            hoverlabel=dict(
                bgcolor="white",       # Deep purple matching your CSS card
                bordercolor="#d0378d",   # Pink accent border
                font=dict(color="black", size=13, family="sans-serif")
            )
        )
        

        st.plotly_chart(fig, use_container_width=True)
        # ---------------------------------------------------

        # Results table
        show_cols = [
            'Rank', 'Player', 'Player Role', 'overall_similarity',
            'numeric_score', 'strength_score', 'weakness_score',
            'Avg', 'Strike rate', 'Balls'
        ]

        st.divider()
        st.markdown("### 📝 Find Top Players Similarity Scouting Reports")
        st.write('')

        st.sidebar.divider()
        desired_player = st.sidebar.selectbox("Select Desired Player:", options=top_results['Player'].unique())
        desired_data = df[df['Player'] == desired_player].iloc[0]
        
        # Clean roles for display
        d_role_display = str(desired_data['Player Role']).replace("Batter", "").strip() + " Batter"
        t_role_display = str(target_data['Player Role']).replace("Batter", "").strip() + " Batter"

        st.markdown(f"""
        <style>
        .comparison-card {{
            background: linear-gradient(135deg, #391b4e 0%, #2a1438 60%, #1a0c24 100%);
            border-radius: 14px;
            padding: 24px 40px;
            display: grid;
            grid-template-columns: 1fr auto 1fr; /* Locks VS in the dead center */
            align-items: center;
            box-shadow: 0 8px 24px rgba(57, 27, 78, 0.30);
            border: 1px solid rgba(208, 55, 141, 0.22);
            margin-bottom: 24px;
        }}
        .comp-block {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .comp-block.left {{
            justify-self: start;
            align-items: flex-start;
            text-align: left;
            margin-left: 65px;
        }}
        .comp-block.right {{
            justify-self: end;
            align-items: flex-end;
            text-align: right;
            margin-right: 65px;
        }}
        .comp-name {{
            font-size: 26px !important;
            font-weight: 900 !important;
            color: #FFFFFF !important;
            margin: 0 !important;
            line-height: 1.1;
        }}
        .comp-role {{
            display: inline-block;
            background: linear-gradient(90deg, #d0378d, #e05aa3);
            color: #FFFFFF !important;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 14px;
            border-radius: 20px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .comp-vs {{
            font-size: 32px;
            font-weight: 900;
            color: rgba(255, 255, 255, 0.4);
            font-style: italic;
            justify-self: center;
            padding: 0 20px;
        }}
        </style>
        
        <div class="comparison-card"><div class="comp-block left"><p class="comp-name">{desired_player}</p><div class="comp-role">{d_role_display}</div></div><div class="comp-vs">VS</div><div class="comp-block right"><p class="comp-name">{target_player}</p><div class="comp-role">{t_role_display}</div></div></div>
        """, unsafe_allow_html=True)

        st.divider()

        def clean_val(v):
            if pd.isna(v): return 0.0
            if isinstance(v, (int, float)): return float(v)
            try: return float(str(v).replace('%', '').strip())
            except ValueError: return 0.0

        # Configuration: (Column Name, Label, Lower is Better, Is Percentage)
        comparison_metrics = [
            # Row 1
            ("Inns",             "Inns",            False, False),
            ("Runs",             "Runs",            False, False),
            ("Balls",            "Balls",           False, False),
            ("Avg",              "Average",         False, False),
            ("Strike rate",      "Strike Rate",     False, False),
            # Row 2
            ("Dots %",           "Dots %",          True,  True), 
            ("Boundary Runs %",  "Boundary Runs %", False, True),
            ("Boundary Rate",    "Boundary Rate",   True,  False),
            ("Six Runs %",       "Six Runs %",      False, True),
            ("Six Rate",         "Six Rate",        True,  False)
        ]

        def render_comparison_card(col_key, label, lower_is_better, is_pct):
            d_val = clean_val(desired_data[col_key])
            t_val = clean_val(target_data[col_key])
            
            # Delta = Selected Player - Target Player
            delta_val = d_val - t_val

            if is_pct:
                display_val, delta_display = f"{d_val:.2f}%", f"{delta_val:+.2f}%"
            elif col_key in ["Inns", "Runs", "Balls"]:
                display_val, delta_display = f"{d_val:.0f}", f"{delta_val:+.1f}"
            else:
                display_val, delta_display = f"{d_val:.2f}", f"{delta_val:+.2f}"

            st.metric(
                label=label, 
                value=display_val, 
                delta=delta_display,
                # 'inverse' turns negative values green, perfect for "lower is better" stats
                delta_color="inverse" if lower_is_better else "normal",
                help=f"{target_player}: {t_val:.2f}{'%' if is_pct else ''}"
            )

        # Render 2x5 Metric Grid
        c1, c2, c3, c4, c5 = st.columns(5)
        for col_widget, item in zip([c1, c2, c3, c4, c5], comparison_metrics[:5]):
            with col_widget: render_comparison_card(*item)
            
        st.write("") # Vertical spacing

        c6, c7, c8, c9, c10 = st.columns(5)
        for col_widget, item in zip([c6, c7, c8, c9, c10], comparison_metrics[5:]):
            with col_widget: render_comparison_card(*item)

        st.divider()

        # --- LLM REPORT GENERATION ---
        @st.cache_data(show_spinner="Drafting AI-driven Professional Scouting Report via LLM........")
        def generate_llm_report(target_name, desired_name, target_dict, desired_dict, sim_scores):
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""
            Act as an elite T20 Franchise Cricket Scout and Data Analyst. 
            Your task is to generate a highly detailed, visually attractive, and catchy scouting report formatted in pristine Markdown. This output will serve as a printable PDF dossier for franchise owners and head coaches.

            **CONTEXT:**
            We are evaluating {desired_name} as a potential tactical replacement/match for the target profile of {target_name}.
            Dots %, Boundary Rate, Six Rate are lower is better and Boundary Rate, Six Rate are not in percentage.
            The desired player falls in top 10 similarity players engine list.

            **SIMILARITY MATRIX:**
            - Overall Tactical Match: {sim_scores['overall_similarity']}%
            - Statistical Profile Match: {sim_scores['numeric_score']}%
            - Strengths Alignment: {sim_scores['strength_score']}%
            - Weaknesses Alignment: {sim_scores['weakness_score']}%

            **DATA PROFILES:**
            - TARGET ({target_name}): {target_dict}
            - CANDIDATE ({desired_name}): {desired_dict}

            **REPORT STRUCTURE REQUIREMENTS (Use Markdown heavily):**
            1. **Title & Header:** Create a catchy, authoritative dossier title using H2 headers.
            2. **Executive Summary:** A punchy, 3-sentence overview of the stylistic match. Use bold text for impact.
            3. **The Similarity Matrix:** Briefly analyze the component scores provided above. Why did they score {sim_scores['overall_similarity']}% overall?
            4. **Statistical Head-to-Head:** Generate a clean Markdown table comparing their key metrics (Avg, SR, Dots %, Boundary Rates, Six Rates).
            5. **Tactical Blueprint:** Compare their shot-making strengths and vulnerabilities based on the data. Highlight stylistic overlaps.
            6. **The Verdict:** A definitive, sharp conclusion on how {desired_name} will seamlessly fulfill {target_name}'s role.

            Make the tone analytical, sharp, and engaging—similar to a high-stakes 'Moneyball' sports analytics briefing. Use emojis sparingly but effectively to enhance the visual structure (e.g., 📊, 🎯, ⚠️, 🏏). Do not include conversational filler like "Here is the report." Start immediately with the Title.
            """
            response = model.generate_content(prompt)
            return response.text

        # Prepare expanded stats dictionaries for the prompt
        filter_keys = [
            'Player', 'Player Role', 'Inns', 'Runs', 'Balls', 'Avg', 
            'Strike rate', 'Dots %', 'Boundary Rate', 'Player Strengths', 
            'Player Weaknesses', 'Six Rate', 'Six Runs %', 'Boundary Runs %'
        ]
        t_dict = {k: target_data[k] for k in filter_keys if k in target_data}
        d_dict = {k: desired_data[k] for k in filter_keys if k in desired_data}
        
        # Extract all similarity scores from the ranked results
        desired_row = top_results[top_results['Player'] == desired_player].iloc[0]
        scores_dict = {
            'overall_similarity': desired_row['overall_similarity'],
            'numeric_score': desired_row['numeric_score'],
            'strength_score': desired_row['strength_score'],
            'weakness_score': desired_row['weakness_score']
        }

        try:
            report_text = generate_llm_report(target_player, desired_player, t_dict, d_dict, scores_dict)
            
            # Wrap the report in an attractive container
            with st.container(border=True):
                st.markdown(report_text)
            
        except Exception as e:
            st.error(f"Failed to generate AI report. Ensure your API key is valid. Error: {e}")


# --- FOOTER ---
st.sidebar.markdown("<br>" * 10, unsafe_allow_html=True)
st.sidebar.caption("Developed by: [Saiprasad Kagne](https://www.linkedin.com/in/saiprasad-kagne/)")
st.sidebar.caption("Check out: [AthlyticZ Masterclasses](https://athlyticz.com/masterclass?am_id=saiprasad3723)")