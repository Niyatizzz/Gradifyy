"""
Student Performance System — EduSight AI
==========================================
Premium redesigned UI
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib
import os
import io
import hashlib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduSight AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUTPUT_DIR = "outputs"

# ─────────────────────────────────────────────
# GLOBAL CSS — Premium Dark Navy + Amber
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Syne:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --navy:       #080c18;
  --navy-1:     #0d1225;
  --navy-2:     #111827;
  --navy-3:     #1a2235;
  --navy-4:     #242d42;
  --border:     #1e2a40;
  --border-hi:  #2d3f5c;
  --amber:      #f5a623;
  --amber-dim:  #c47d0e;
  --amber-glow: rgba(245,166,35,0.12);
  --emerald:    #10d9a0;
  --rose:       #f43f6e;
  --sky:        #38bdf8;
  --text-1:     #f0f4ff;
  --text-2:     #8899bb;
  --text-3:     #4a5a7a;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
  font-family: 'Syne', sans-serif;
  background: var(--navy) !important;
  color: var(--text-1);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy-1); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }

/* ══ LOGIN ══ */
.login-wrap {
  min-height: 100vh; display: flex; align-items: center;
  justify-content: center;
  background:
    radial-gradient(ellipse 80% 60% at 20% 80%, rgba(245,166,35,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 80% 20%, rgba(56,189,248,0.05) 0%, transparent 60%),
    var(--navy);
  padding: 40px 20px;
}
.login-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--amber-glow); border: 1px solid rgba(245,166,35,0.25);
  border-radius: 999px; padding: 4px 14px;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
  color: var(--amber); text-transform: uppercase; margin-bottom: 24px;
}
.login-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem; font-weight: 900; color: var(--text-1);
  line-height: 1.1; margin-bottom: 8px;
}
.login-title em { color: var(--amber); font-style: normal; }
.login-sub { font-size: 0.82rem; color: var(--text-2); margin-bottom: 36px; line-height: 1.6; }
.demo-box {
  margin-top: 24px; background: rgba(16,217,160,0.04);
  border: 1px solid rgba(16,217,160,0.15); border-radius: 12px;
  padding: 16px 18px; font-size: 0.78rem; color: var(--text-2); line-height: 2;
}
.demo-box strong { color: var(--emerald); }
.demo-box code {
  font-family: 'JetBrains Mono', monospace;
  background: rgba(16,217,160,0.08); padding: 1px 6px;
  border-radius: 4px; color: var(--emerald); font-size: 0.72rem;
}

/* ══ HEADER ══ */
.top-header {
  background: rgba(8,12,24,0.95); backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  padding: 0 40px; height: 64px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 200;
}
.header-logo {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem; font-weight: 900; color: var(--text-1); letter-spacing: -0.02em;
}
.header-logo em { color: var(--amber); font-style: normal; }
.header-right { display: flex; align-items: center; gap: 16px; }
.header-role {
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--amber);
  background: var(--amber-glow); border: 1px solid rgba(245,166,35,0.2);
  padding: 4px 12px; border-radius: 999px;
}
.header-user {
  display: flex; align-items: center; gap: 10px;
  background: var(--navy-3); border: 1px solid var(--border);
  border-radius: 999px; padding: 5px 16px 5px 6px;
  font-size: 0.8rem; color: var(--text-2);
}
.avatar {
  width: 32px; height: 32px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; font-weight: 700; color: #fff; flex-shrink: 0;
}
.av-student { background: linear-gradient(135deg,#6366f1,#8b5cf6); }
.av-teacher { background: linear-gradient(135deg,#f5a623,#f43f6e); }

/* ══ PAGE ══ */
.page-wrap {
  padding: 36px 48px 60px;
  background:
    radial-gradient(ellipse 100% 40% at 50% 0%, rgba(245,166,35,0.04) 0%, transparent 70%),
    var(--navy);
  min-height: calc(100vh - 64px);
}
.page-greeting {
  font-family: 'Playfair Display', serif;
  font-size: 2rem; font-weight: 700; color: var(--text-1);
  line-height: 1.2; margin-bottom: 6px;
}
.page-sub { font-size: 0.85rem; color: var(--text-2); margin-bottom: 28px; }

/* ══ KPI ══ */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 32px; }
.kpi {
  background: var(--navy-2); border: 1px solid var(--border);
  border-radius: 16px; padding: 22px 24px; position: relative; overflow: hidden;
  transition: border-color 0.2s, transform 0.2s;
}
.kpi:hover { border-color: var(--border-hi); transform: translateY(-2px); }
.kpi-accent { position: absolute; top:0; left:0; right:0; height:2px; border-radius:16px 16px 0 0; }
.kpi-icon { font-size: 1.4rem; margin-bottom: 12px; display: block; }
.kpi-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-3); margin-bottom: 6px; }
.kpi-value { font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 700; color: var(--text-1); line-height: 1; margin-bottom: 4px; }
.kpi-delta { font-size: 0.75rem; color: var(--text-2); }
.kpi-delta.up { color: var(--emerald); }
.kpi-delta.down { color: var(--rose); }

/* ══ SECTION ══ */
.sec-head { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
.sec-pip { width: 3px; height: 20px; border-radius: 2px; background: linear-gradient(180deg,var(--amber),transparent); flex-shrink:0; }
.sec-title { font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 700; color: var(--text-1); }
.sec-sub { font-size: 0.78rem; color: var(--text-3); margin-top: 2px; }

/* ══ GLASS ══ */
.glass { background: var(--navy-2); border: 1px solid var(--border); border-radius: 16px; padding: 28px; }

/* ══ PREDICTION ══ */
.pred-result { border-radius: 16px; padding: 32px 28px; text-align: center; position: relative; overflow: hidden; }
.pred-high { border: 1.5px solid var(--emerald); background: rgba(16,217,160,0.05); }
.pred-avg  { border: 1.5px solid var(--amber);   background: rgba(245,166,35,0.05); }
.pred-risk { border: 1.5px solid var(--rose);    background: rgba(244,63,110,0.05); }
.pred-emoji { font-size: 3rem; line-height: 1; margin-bottom: 12px; display: block; }
.pred-class { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; }
.pred-high .pred-class { color: var(--emerald); }
.pred-avg  .pred-class { color: var(--amber); }
.pred-risk .pred-class { color: var(--rose); }
.pred-conf { font-size: 0.78rem; color: var(--text-3); }

/* ══ PROB BARS ══ */
.prob-row { margin-bottom: 14px; }
.prob-meta { display: flex; justify-content: space-between; font-size: 0.76rem; color: var(--text-2); margin-bottom: 5px; }
.prob-track { background: var(--navy-3); border-radius: 999px; height: 6px; overflow: hidden; }
.prob-fill { height: 6px; border-radius: 999px; }

/* ══ REC CARDS ══ */
.rec {
  display: flex; align-items: flex-start; gap: 12px;
  background: var(--navy-2); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;
  font-size: 0.83rem; color: var(--text-2); line-height: 1.5;
}
.rec-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink:0; margin-top:5px; }
.rec.ok   .rec-dot { background: var(--emerald); box-shadow: 0 0 8px var(--emerald); }
.rec.warn .rec-dot { background: var(--amber);   box-shadow: 0 0 8px var(--amber); }
.rec.bad  .rec-dot { background: var(--rose);    box-shadow: 0 0 8px var(--rose); }

/* ══ INTERVENTIONS ══ */
.interv { background: var(--navy-2); border: 1px solid var(--border); border-left: 3px solid var(--amber); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px; font-size: 0.83rem; color: var(--text-2); }
.interv.danger { border-left-color: var(--rose); }
.interv.ok     { border-left-color: var(--emerald); }

/* ══ DRIVERS ══ */
.driver { background: var(--navy-2); border: 1px solid var(--border); border-top: 2px solid var(--amber); border-radius: 14px; padding: 22px 18px; text-align: center; transition: transform 0.2s; }
.driver:hover { transform: translateY(-3px); }
.driver-medal { font-size: 1.8rem; margin-bottom: 10px; }
.driver-name { font-weight: 600; font-size: 0.9rem; color: var(--text-1); margin-bottom: 6px; }
.driver-desc { font-size: 0.74rem; color: var(--text-3); line-height: 1.5; }

/* ══ STREAMLIT OVERRIDES ══ */
.stTextInput > div > div > input {
  background: var(--navy-3) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--text-1) !important;
  font-family: 'Syne', sans-serif !important;
}
.stButton > button {
  background: linear-gradient(135deg, var(--amber), var(--amber-dim)) !important;
  color: #080c18 !important; border: none !important;
  border-radius: 10px !important; font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important; font-size: 0.85rem !important;
  letter-spacing: 0.04em !important; padding: 10px 24px !important;
  box-shadow: 0 4px 20px rgba(245,166,35,0.25) !important; transition: all 0.2s !important;
}
.stButton > button:hover { box-shadow: 0 6px 30px rgba(245,166,35,0.4) !important; transform: translateY(-1px) !important; }
.logout-wrap .stButton > button {
  background: rgba(244,63,110,0.08) !important; color: var(--rose) !important;
  border: 1px solid rgba(244,63,110,0.3) !important; box-shadow: none !important;
  font-size: 0.76rem !important; padding: 6px 14px !important;
}
.logout-wrap .stButton > button:hover { background: rgba(244,63,110,0.18) !important; box-shadow: none !important; transform: none !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: var(--text-3) !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 0.82rem !important; letter-spacing: 0.03em !important; padding: 10px 22px !important; border-bottom: 2px solid transparent !important; }
.stTabs [aria-selected="true"] { color: var(--amber) !important; border-bottom-color: var(--amber) !important; }
.stSuccess { background: rgba(16,217,160,0.08) !important; border: 1px solid rgba(16,217,160,0.2) !important; border-radius: 10px !important; color: var(--emerald) !important; }
.stInfo    { background: rgba(56,189,248,0.08) !important; border: 1px solid rgba(56,189,248,0.2) !important; border-radius: 10px !important; color: var(--sky) !important; }
.stWarning { background: rgba(245,166,35,0.08) !important; border: 1px solid rgba(245,166,35,0.2) !important; border-radius: 10px !important; color: var(--amber) !important; }
.stError   { background: rgba(244,63,110,0.08) !important; border: 1px solid rgba(244,63,110,0.2) !important; border-radius: 10px !important; color: var(--rose) !important; }
hr { border-color: var(--border) !important; margin: 20px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1225",
    "axes.facecolor":    "#0d1225",
    "axes.edgecolor":    "#1e2a40",
    "axes.labelcolor":   "#8899bb",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#4a5a7a",
    "ytick.color":       "#4a5a7a",
    "text.color":        "#f0f4ff",
    "grid.color":        "#1e2a40",
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "sans-serif",
})

C_EMERALD = "#10d9a0"
C_AMBER   = "#f5a623"
C_ROSE    = "#f43f6e"
C_SKY     = "#38bdf8"
PALETTE   = {"At Risk": C_ROSE, "Average": C_AMBER, "High Performer": C_EMERALD}

# ─────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────
USERS = {
    "student1": {"password": hashlib.sha256(b"student123").hexdigest(),
                 "role": "student", "name": "Arjun Mehta",      "id": "S001"},
    "student2": {"password": hashlib.sha256(b"pass456").hexdigest(),
                 "role": "student", "name": "Priya Sharma",     "id": "S002"},
    "teacher1": {"password": hashlib.sha256(b"teacher123").hexdigest(),
                 "role": "teacher", "name": "Dr. Rajesh Kumar", "id": "T001"},
    "teacher2": {"password": hashlib.sha256(b"teach456").hexdigest(),
                 "role": "teacher", "name": "Ms. Anita Patel",  "id": "T002"},
}

def verify(username, password):
    u = USERS.get(username)
    if u and u["password"] == hashlib.sha256(password.encode()).hexdigest():
        return u
    return None

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model    = joblib.load(os.path.join(OUTPUT_DIR, "best_model_xgb.pkl"))
        model_rf = joblib.load(os.path.join(OUTPUT_DIR, "model_rf.pkl"))
        le       = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
        features = joblib.load(os.path.join(OUTPUT_DIR, "feature_cols.pkl"))
        return model, model_rf, le, features, True
    except Exception:
        return None, None, None, None, False

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
ORDINAL_MAPS = {
    "parental_education": {"None":0,"High School":1,"Bachelor":2,"Master":3,"PhD":4},
    "family_income":      {"Low":0,"Medium":1,"High":2},
    "motivation_level":   {"Low":0,"Medium":1,"High":2},
    "teacher_quality":    {"Low":0,"Medium":1,"High":2},
}

def preprocess_input(raw, feature_cols):
    row = raw.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in row and isinstance(row[col], str):
            row[col] = mapping.get(row[col], 0)
    row["internet_access"] = 1 if row.get("internet_access") == "Yes"      else 0
    row["tutoring"]         = 1 if row.get("tutoring")         == "Yes"     else 0
    row["gender"]           = 1 if row.get("gender")           == "Female"  else 0
    row["school_type"]      = 1 if row.get("school_type")      == "Private" else 0
    df = pd.DataFrame([row])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df[feature_cols]

def preprocess_df(df_raw, feature_cols):
    df = df_raw.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    for col, yes_val in [("internet_access","Yes"),("tutoring","Yes"),
                          ("gender","Female"),("school_type","Private")]:
        if col in df.columns:
            df[col] = (df[col] == yes_val).astype(int)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df[feature_cols].fillna(0)

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def section(title, sub=""):
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="sec-head">
      <div class="sec-pip"></div>
      <div><div class="sec-title">{title}</div>{sub_html}</div>
    </div>""", unsafe_allow_html=True)

def kpi_card(icon, label, value, delta="", delta_type=""):
    colors = {"green": C_EMERALD, "amber": C_AMBER, "rose": C_ROSE, "sky": C_SKY}
    color  = colors.get(delta_type, C_AMBER)
    dcls   = "up" if delta_type == "green" else ("down" if delta_type == "rose" else "")
    return f"""
    <div class="kpi">
      <div class="kpi-accent" style="background:linear-gradient(90deg,{color},transparent)"></div>
      <span class="kpi-icon">{icon}</span>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta {dcls}">{delta}</div>
    </div>"""

def prob_bar(label, prob, color):
    return f"""
    <div class="prob-row">
      <div class="prob-meta">
        <span>{label}</span>
        <span style="color:{color};font-weight:600">{prob*100:.1f}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width:{prob*100:.1f}%;background:{color}"></div>
      </div>
    </div>"""


# ═══════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════
def login_page():
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("""
        <div style="padding-top:60px">
          <div class="login-badge">✦ Academic Intelligence Platform</div>
          <div class="login-title">Edu<em>Sight</em><br>AI</div>
          <div class="login-sub">Predict, understand, and improve student performance with explainable machine learning.</div>
        </div>
        """, unsafe_allow_html=True)

        role_choice = st.radio("Login as", ["🎓 Student", "👩‍🏫 Teacher"],
                               horizontal=True, label_visibility="collapsed")
        username = st.text_input("", placeholder="Username  (e.g. student1)")
        password = st.text_input("", placeholder="Password", type="password")

        if st.button("Sign In  →", use_container_width=True):
            user = verify(username.strip(), password.strip())
            if user:
                role_map = {"🎓 Student": "student", "👩‍🏫 Teacher": "teacher"}
                if user["role"] != role_map[role_choice]:
                    st.error("Role mismatch — check your login type.")
                else:
                    st.session_state.user = user
                    st.session_state.logged_in = True
                    st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

        st.markdown("""
        <div class="demo-box">
          <strong>Demo credentials</strong><br>
          Student → <code>student1</code> / <code>student123</code><br>
          Teacher → <code>teacher1</code> / <code>teacher123</code>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER + LOGOUT
# ─────────────────────────────────────────────
def render_header(user):
    av_cls   = "av-student" if user["role"] == "student" else "av-teacher"
    initials = "".join(w[0] for w in user["name"].split()[:2]).upper()
    role_lbl = "Student Portal" if user["role"] == "student" else "Teacher Portal"

    h_left, h_right = st.columns([9, 1])
    with h_left:
        st.markdown(f"""
        <div class="top-header">
          <div class="header-logo">Edu<em>Sight</em> AI</div>
          <div class="header-right">
            <div class="header-role">{role_lbl}</div>
            <div class="header-user">
              <div class="avatar {av_cls}">{initials}</div>
              {user['name']}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with h_right:
        st.markdown('<div class="logout-wrap" style="padding:14px 0 0 4px">', unsafe_allow_html=True)
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# STUDENT DASHBOARD
# ═══════════════════════════════════════════
def student_dashboard(user, model, le, feature_cols):
    render_header(user)
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="page-greeting">Welcome back, {user['name'].split()[0]} 👋</div>
    <div class="page-sub">Enter your details below to get your personalised performance prediction.</div>
    """, unsafe_allow_html=True)

    tab_pred, tab_plan = st.tabs(["🔮  Prediction", "💡  Action Plan"])

    with tab_pred:
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            section("Your Academic Profile", "Adjust to match your current situation")
            c1, c2 = st.columns(2)
            with c1:
                study  = st.slider("📚 Study hrs / day",     0.0, 12.0, 4.0, 0.5)
                attend = st.slider("✅ Attendance %",         30,  100,  75)
                prev   = st.slider("📝 Previous grades %",   20,  100,  65)
                sleep  = st.slider("😴 Sleep hrs / night",   3.0, 12.0, 7.0, 0.5)
            with c2:
                extra  = st.slider("⚽ Extracurricular hrs", 0.0, 8.0,  2.0, 0.5)
                stress = st.slider("😰 Stress level (1–10)", 1,   10,   5)
                dist   = st.slider("📍 Distance to school",  0.5, 50.0, 5.0, 0.5)
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                tutor    = st.selectbox("👨‍🏫 Tutoring",        ["Yes", "No"])
                internet = st.selectbox("🌐 Internet access",  ["Yes", "No"])
                motivate = st.selectbox("💪 Motivation level", ["Low", "Medium", "High"])
                gender   = st.selectbox("👤 Gender",           ["Male", "Female"])
            with c4:
                par_edu  = st.selectbox("🎓 Parent education", ["None","High School","Bachelor","Master","PhD"])
                income   = st.selectbox("💰 Family income",    ["Low","Medium","High"])
                school   = st.selectbox("🏫 School type",      ["Public","Private"])
                t_qual   = st.selectbox("⭐ Teacher quality",  ["Low","Medium","High"])
            st.markdown('</div>', unsafe_allow_html=True)

        raw = dict(study_hours_per_day=study, attendance_rate=attend,
                   previous_grades=prev, sleep_hours=sleep,
                   extracurricular_hours=extra, stress_level=stress,
                   distance_to_school_km=dist, tutoring=tutor,
                   internet_access=internet, motivation_level=motivate,
                   gender=gender, parental_education=par_edu,
                   family_income=income, school_type=school, teacher_quality=t_qual)
        X     = preprocess_input(raw, feature_cols)
        proba = model.predict_proba(X)[0]
        pred  = le.classes_[np.argmax(proba)]
        conf  = np.max(proba) * 100

        with right:
            box_cls = {"High Performer":"pred-high","Average":"pred-avg","At Risk":"pred-risk"}[pred]
            emoji   = {"High Performer":"🏆","Average":"📊","At Risk":"⚠️"}[pred]
            st.markdown(f"""
            <div class="pred-result {box_cls}">
              <span class="pred-emoji">{emoji}</span>
              <div class="pred-class">{pred}</div>
              <div class="pred-conf">Confidence: {conf:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            section("Class Probabilities")
            c_map = {"At Risk": C_ROSE, "Average": C_AMBER, "High Performer": C_EMERALD}
            st.markdown("".join(prob_bar(cls, p, c_map[cls]) for cls, p in zip(le.classes_, proba)),
                        unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            section("Why This Prediction?", "SHAP feature impact")
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)
                pred_idx  = np.argmax(proba)

                if isinstance(shap_vals, list):
                    sv_arr = shap_vals[pred_idx][0]
                elif shap_vals.ndim == 3:
                    sv_arr = shap_vals[0, :, pred_idx]
                else:
                    sv_arr = shap_vals[0]

                df_sv = pd.DataFrame({"Feature": feature_cols, "SHAP": sv_arr})
                df_sv = df_sv.reindex(df_sv["SHAP"].abs().sort_values(ascending=False).index).head(8)

                fig, ax = plt.subplots(figsize=(6, 4))
                colors_s = [C_EMERALD if v > 0 else C_ROSE for v in df_sv["SHAP"][::-1]]
                ax.barh(df_sv["Feature"][::-1], df_sv["SHAP"][::-1],
                        color=colors_s, height=0.5, edgecolor="none")
                ax.axvline(0, color=C_SKY, linewidth=0.8, alpha=0.5)
                ax.set_xlabel("SHAP Impact", fontsize=8)
                ax.set_title(f"Impact on '{pred}' prediction", fontsize=9, color="#8899bb", pad=10)
                ax.tick_params(labelsize=8)
                ax.grid(axis="x", alpha=0.25)
                plt.tight_layout(pad=1.5)
                st.pyplot(fig); plt.close()
                st.markdown('<div style="font-size:0.72rem;color:#4a5a7a;margin-top:-8px">🟢 Positive impact &nbsp;|&nbsp; 🔴 Negative impact</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

    with tab_plan:
        col_recs, col_radar = st.columns([1, 1], gap="large")

        with col_recs:
            section("Personalised Action Plan", "Based on your inputs")
            recs = []
            if study    < 3:              recs.append(("bad",  "📚 Increase study time to at least 4–5 hours/day."))
            if attend   < 75:             recs.append(("bad",  "✅ Attendance below 75%. Aim for 80%+ to stay on track."))
            if prev     < 50:             recs.append(("warn", "📝 Previous grades are below average. Revisit core topics."))
            if not (6 <= sleep <= 9):     recs.append(("warn", "😴 Target 7–9 hrs sleep. Both extremes hurt performance."))
            if stress   > 7:              recs.append(("bad",  "😰 High stress detected. Consider mindfulness or counselling."))
            if tutor    == "No":          recs.append(("warn", "👨‍🏫 A tutor can significantly boost your results."))
            if motivate == "Low":         recs.append(("bad",  "💪 Low motivation is a key risk. Set small daily goals."))
            if extra    > 5:              recs.append(("warn", "⚽ Extracurriculars very high — balance with study time."))
            if internet == "No":          recs.append(("warn", "🌐 No internet — explore school library resources."))
            if not recs:
                st.markdown('<div class="rec ok"><div class="rec-dot"></div>🌟 Your profile looks excellent! Keep up these habits.</div>', unsafe_allow_html=True)
            for sev, txt in recs:
                st.markdown(f'<div class="rec {sev}"><div class="rec-dot"></div>{txt}</div>', unsafe_allow_html=True)

        with col_radar:
            section("Performance Radar", "Across 6 key dimensions")
            cats = ["Study","Attendance","Grades","Sleep","Low Stress","Motivation"]
            vals = [
                min(study / 8, 1),
                attend / 100,
                prev / 100,
                1 - abs(sleep - 7.5) / 4.5,
                1 - (stress - 1) / 9,
                {"Low":0.2,"Medium":0.6,"High":1.0}[motivate],
            ]
            vals  += vals[:1]
            angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
            angles += angles[:1]
            fig2, ax2 = plt.subplots(figsize=(5, 4.8), subplot_kw={"polar": True})
            ax2.set_facecolor("#0d1225"); fig2.patch.set_facecolor("#0d1225")
            ax2.plot(angles, vals, "o-", linewidth=2.5, color=C_AMBER, zorder=3)
            ax2.fill(angles, vals, alpha=0.15, color=C_AMBER, zorder=2)
            for r in [0.25, 0.5, 0.75, 1.0]:
                ax2.plot(angles, [r]*len(angles), color="#1e2a40", linewidth=0.8, zorder=1)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(cats, fontsize=8.5, color="#8899bb")
            ax2.set_yticks([]); ax2.set_ylim(0, 1)
            ax2.spines["polar"].set_color("#1e2a40")
            ax2.grid(color="#1e2a40", linewidth=0.6)
            ax2.set_title("Your Performance Radar", fontsize=10, color="#f0f4ff", pad=18, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# TEACHER DASHBOARD
# ═══════════════════════════════════════════
def teacher_dashboard(user, model, le, feature_cols):
    render_header(user)
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="page-greeting">Class Overview &mdash; {user['name']} 📋</div>
    <div class="page-sub">Upload a CSV to analyse and predict student outcomes at scale.</div>
    """, unsafe_allow_html=True)

    tab_up, tab_class, tab_risk, tab_exp = st.tabs([
        "📤  Upload Data","📊  Class Analytics","⚠️  At-Risk Students","🧠  Explainability"
    ])

    with tab_up:
        section("Upload Class Dataset", "CSV with student feature columns")
        st.markdown("""
        <div style="font-size:0.8rem;color:#4a5a7a;margin-bottom:16px;line-height:1.8">
        Required: <code style="background:#1a2235;padding:1px 6px;border-radius:4px;color:#f5a623">
        study_hours_per_day, attendance_rate, previous_grades, sleep_hours,
        extracurricular_hours, parental_education, internet_access, tutoring,
        family_income, motivation_level, gender, school_type, distance_to_school_km,
        teacher_quality, stress_level</code>
        &nbsp;·&nbsp; Optional: <code style="background:#1a2235;padding:1px 6px;border-radius:4px;color:#38bdf8">student_name, student_id</code>
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop CSV here", type=["csv"])
        use_demo = st.checkbox("Use demo dataset (outputs/student_data.csv)", value=not bool(uploaded))

        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(df_raw)} rows.")
        elif use_demo and os.path.exists(os.path.join(OUTPUT_DIR, "student_data.csv")):
            df_raw = pd.read_csv(os.path.join(OUTPUT_DIR, "student_data.csv"))
            st.info(f"ℹ️ Using demo dataset — {len(df_raw)} students.")
        else:
            st.warning("Upload a CSV file or enable the demo dataset.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        X_all   = preprocess_df(df_raw, feature_cols)
        preds   = le.inverse_transform(model.predict(X_all))
        probas  = model.predict_proba(X_all)
        df_pred = df_raw.copy()
        df_pred["Predicted_Performance"] = preds
        df_pred["Confidence_%"] = (np.max(probas, axis=1) * 100).round(1)
        st.session_state["teacher_df"]     = df_pred
        st.session_state["teacher_probas"] = probas

        st.markdown("<br>", unsafe_allow_html=True)
        section("Predictions Preview")
        show_cols = [c for c in ["student_name","student_id","study_hours_per_day",
                                  "attendance_rate","previous_grades",
                                  "Predicted_Performance","Confidence_%"]
                     if c in df_pred.columns]
        def color_perf(val):
            m = {"High Performer":"background-color:#0a1f0f;color:#10d9a0;font-weight:600",
                 "Average":       "background-color:#1a1400;color:#f5a623;font-weight:600",
                 "At Risk":       "background-color:#1f0a0a;color:#f43f6e;font-weight:600"}
            return m.get(val, "")
        styled = df_pred[show_cols].style.applymap(color_perf, subset=["Predicted_Performance"])
        st.dataframe(styled, use_container_width=True, height=340)

        buf = io.BytesIO()
        df_pred.to_csv(buf, index=False)
        st.download_button("⬇️ Download Predictions CSV", buf.getvalue(), "predictions.csv", "text/csv")

    with tab_class:
        if "teacher_df" not in st.session_state:
            st.info("Upload data in the 'Upload Data' tab first.")
        else:
            df  = st.session_state["teacher_df"]
            n   = len(df)
            nh  = (df["Predicted_Performance"] == "High Performer").sum()
            na  = (df["Predicted_Performance"] == "Average").sum()
            nr  = (df["Predicted_Performance"] == "At Risk").sum()
            avg_att = df["attendance_rate"].mean() if "attendance_rate" in df.columns else 0

            st.markdown(f"""
            <div class="kpi-grid">
              {kpi_card("🏆","High Performers", nh, f"{nh/n*100:.0f}% of class","green")}
              {kpi_card("📊","Average",          na, f"{na/n*100:.0f}% of class","amber")}
              {kpi_card("⚠️","At Risk",          nr, f"{nr/n*100:.0f}% of class","rose")}
              {kpi_card("📅","Avg Attendance",  f"{avg_att:.0f}%","Class mean","sky")}
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                section("Performance Distribution")
                fig, ax = plt.subplots(figsize=(5, 3.8))
                labels = list(PALETTE.keys())
                sizes  = [(df["Predicted_Performance"] == l).sum() for l in labels]
                colors = [PALETTE[l] for l in labels]
                _, _, pcts = ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                    startangle=90, textprops={"color":"#f0f4ff","fontsize":8.5},
                    wedgeprops={"linewidth":2.5,"edgecolor":"#0d1225"}, pctdistance=0.78)
                for p in pcts: p.set_fontsize(10); p.set_fontweight("bold")
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with c2:
                if "attendance_rate" in df.columns and "study_hours_per_day" in df.columns:
                    section("Attendance vs Study Hours")
                    fig2, ax2 = plt.subplots(figsize=(5, 3.8))
                    for cls, color in PALETTE.items():
                        sub = df[df["Predicted_Performance"]==cls]
                        ax2.scatter(sub["attendance_rate"], sub["study_hours_per_day"],
                                    alpha=0.55, s=20, c=color, label=cls, edgecolors="none")
                    ax2.set_xlabel("Attendance Rate (%)", fontsize=8)
                    ax2.set_ylabel("Study Hours / Day",   fontsize=8)
                    ax2.legend(fontsize=7.5, framealpha=0.15)
                    ax2.grid(True, alpha=0.2)
                    plt.tight_layout(); st.pyplot(fig2); plt.close()

            numeric_feats = ["study_hours_per_day","attendance_rate","previous_grades",
                             "sleep_hours","stress_level","extracurricular_hours"]
            existing = [c for c in numeric_feats if c in df.columns]
            if existing:
                st.markdown("<br>", unsafe_allow_html=True)
                section("Average Features by Group")
                grp = df.groupby("Predicted_Performance")[existing].mean().round(2)
                st.dataframe(grp.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)

    with tab_risk:
        if "teacher_df" not in st.session_state:
            st.info("Upload data in the 'Upload Data' tab first.")
        else:
            df      = st.session_state["teacher_df"]
            df_risk = df[df["Predicted_Performance"] == "At Risk"].copy()
            section(f"⚠️ {len(df_risk)} Students Need Immediate Attention",
                    "Flagged by the model — review and act")

            if df_risk.empty:
                st.success("🎉 No at-risk students detected!")
            else:
                show_cols = [c for c in ["student_name","student_id","attendance_rate",
                             "study_hours_per_day","previous_grades","stress_level","Confidence_%"]
                             if c in df_risk.columns]
                st.dataframe(df_risk[show_cols].reset_index(drop=True), use_container_width=True, height=280)

                risk_numeric = ["attendance_rate","study_hours_per_day","previous_grades","stress_level"]
                exist_risk   = [c for c in risk_numeric if c in df_risk.columns]
                if exist_risk:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section("At-Risk Group vs Class Average")
                    fig3, axes = plt.subplots(1, len(exist_risk), figsize=(4.2*len(exist_risk), 3.5))
                    if len(exist_risk)==1: axes=[axes]
                    for ax3, col in zip(axes, exist_risk):
                        ax3.hist(df_risk[col].dropna(), bins=15, color=C_ROSE, alpha=0.75,
                                 edgecolor="#0d1225", label="At Risk")
                        if col in df.columns:
                            ax3.axvline(df[col].mean(), color=C_EMERALD, linewidth=2,
                                        linestyle="--", label="Class avg")
                        ax3.set_title(col.replace("_"," ").title(), fontsize=9, fontweight="bold")
                        ax3.legend(fontsize=7); ax3.grid(axis="y", alpha=0.2)
                    plt.suptitle("At-Risk Group Distributions", fontsize=11, fontweight="bold",
                                 color="#f0f4ff", y=1.02)
                    plt.tight_layout(); st.pyplot(fig3); plt.close()

                st.markdown("<br>", unsafe_allow_html=True)
                section("Recommended Interventions")
                if "attendance_rate" in df_risk.columns and df_risk["attendance_rate"].mean() < 70:
                    st.markdown('<div class="interv danger">📅 Low attendance — send parent/guardian alerts immediately.</div>', unsafe_allow_html=True)
                if "study_hours_per_day" in df_risk.columns and df_risk["study_hours_per_day"].mean() < 3:
                    st.markdown('<div class="interv danger">📚 Study hours critically low — introduce after-school study programmes.</div>', unsafe_allow_html=True)
                if "stress_level" in df_risk.columns and df_risk["stress_level"].mean() > 7:
                    st.markdown('<div class="interv">😰 High average stress — schedule counsellor sessions.</div>', unsafe_allow_html=True)
                st.markdown('<div class="interv ok">👥 Schedule individual 1-on-1 check-ins for all flagged students this week.</div>', unsafe_allow_html=True)

    with tab_exp:
        if "teacher_df" not in st.session_state:
            st.info("Upload data in the 'Upload Data' tab first.")
        else:
            section("Global SHAP Explainability", "Computed over your uploaded class dataset")
            df    = st.session_state["teacher_df"]
            X_exp = preprocess_df(df, feature_cols)
            try:
                with st.spinner("Computing SHAP values…"):
                    exp2 = shap.TreeExplainer(model)
                    sv2  = exp2.shap_values(X_exp)

                hp_idx = list(le.classes_).index("High Performer") \
                         if "High Performer" in le.classes_ else -1
                if isinstance(sv2, list):
                    sv_hp = sv2[hp_idx]
                elif sv2.ndim == 3:
                    sv_hp = sv2[:, :, hp_idx]
                else:
                    sv_hp = sv2

                mean_sv = np.abs(sv_hp).mean(axis=0)
                idx_s   = np.argsort(mean_sv)[::-1][:12]

                c_l, c_r = st.columns(2, gap="large")
                with c_l:
                    section("Feature Importance", "Mean |SHAP| value")
                    fig4, ax4 = plt.subplots(figsize=(6, 5.2))
                    bar_vals   = mean_sv[idx_s[::-1]]
                    bar_lbls   = [feature_cols[i] for i in idx_s[::-1]]
                    bar_colors = plt.cm.YlGn(np.linspace(0.3, 0.9, len(bar_vals)))
                    ax4.barh(bar_lbls, bar_vals, color=bar_colors, height=0.55, edgecolor="none")
                    ax4.set_xlabel("Mean |SHAP value|", fontsize=8)
                    ax4.grid(axis="x", alpha=0.25)
                    plt.tight_layout(pad=1.5); st.pyplot(fig4); plt.close()

                with c_r:
                    section("SHAP Beeswarm", "Feature value vs prediction impact")
                    fig5, ax5 = plt.subplots(figsize=(6, 5.2))
                    plt.sca(ax5)
                    shap.summary_plot(sv_hp, X_exp, show=False, max_display=12,
                                      plot_type="dot", color_bar=True)
                    plt.title("Beeswarm — High Performer Class",
                              fontsize=9, color="#8899bb", pad=10)
                    plt.tight_layout(pad=1.5); st.pyplot(fig5); plt.close()

                st.markdown("<br>", unsafe_allow_html=True)
                section("Top 3 Performance Drivers")
                top3  = [feature_cols[i] for i in np.argsort(mean_sv)[::-1][:3]]
                descs = {
                    "study_hours_per_day": "Daily study time is the single strongest predictor of academic success.",
                    "attendance_rate":     "Consistent class attendance strongly correlates with better outcomes.",
                    "previous_grades":     "Past performance is the most reliable indicator of future results.",
                    "sleep_hours":         "Optimal sleep (7–9 hrs) directly boosts cognitive performance.",
                    "motivation_level":    "Highly motivated students vastly outperform their peers.",
                    "stress_level":        "Elevated stress is a top suppressor of academic performance.",
                }
                d_cols = st.columns(3, gap="large")
                for i, (dc, feat) in enumerate(zip(d_cols, top3)):
                    with dc:
                        st.markdown(f"""
                        <div class="driver">
                          <div class="driver-medal">{"🥇🥈🥉"[i]}</div>
                          <div class="driver-name">{feat.replace("_"," ").title()}</div>
                          <div class="driver-desc">{descs.get(feat,"Key predictor of student performance.")}</div>
                        </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"SHAP computation error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
        return

    user = st.session_state.user
    model, model_rf, le, feature_cols, ok = load_artifacts()

    if not ok:
        st.error("⚠️ No trained model found. Run `python student_performance_ml.py` first.")
        return

    if user["role"] == "student":
        student_dashboard(user, model, le, feature_cols)
    else:
        teacher_dashboard(user, model_rf, le, feature_cols)


if __name__ == "__main__":
    main()