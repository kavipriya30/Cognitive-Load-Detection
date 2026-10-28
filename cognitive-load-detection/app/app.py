"""
Streamlit Dashboard Application for Cognitive Load Detection using Typing Patterns.

Pages / Sections:
1. 🏠 Home: Project overview, privacy notice banner, system workflow, 15 features table, ethical disclaimer.
2. ⌨️ Typing Test: Interactive browser-based keystroke collector, live counters, timer & explainable prediction.
3. 📊 Current Session Analytics: Biometric breakdown & scale profile bar charts for active session.
4. 📜 Local History Log: Persistent local CSV history without requiring user authentication.
5. 🤖 Model Performance Dashboard: Actual calculated metrics, Random Forest feature importances, and confusion matrix.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

# Ensure root src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engineering import compute_typing_features, FEATURE_NAMES
from src.prediction import predict_cognitive_load, MODEL_PATH_DEFAULT

SESSION_HISTORY_FILE = os.path.join("data", "session_history.csv")
MODEL_METRICS_FILE = os.path.join("models", "model_metrics.json")

# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIG & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cognitive Load AI | Typing Patterns",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Neutral Theme
st.markdown("""
<style>
    /* Main App Background & Typography */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    
    /* Status Badges */
    .badge-low {
        background-color: #064e3b;
        color: #34d399;
        font-weight: 700;
        padding: 8px 20px;
        border-radius: 9999px;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #059669;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.2);
    }
    
    .badge-medium {
        background-color: #78350f;
        color: #fbbf24;
        font-weight: 700;
        padding: 8px 20px;
        border-radius: 9999px;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #d97706;
        box-shadow: 0 0 12px rgba(251, 191, 36, 0.2);
    }
    
    .badge-high {
        background-color: #7f1d1d;
        color: #f87171;
        font-weight: 700;
        padding: 8px 20px;
        border-radius: 9999px;
        font-size: 1.25rem;
        display: inline-block;
        border: 1px solid #dc2626;
        box-shadow: 0 0 12px rgba(248, 113, 113, 0.2);
    }

    /* Privacy Banner */
    .privacy-banner {
        background-color: rgba(6, 78, 59, 0.6);
        border-left: 4px solid #10b981;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #a7f3d0;
        font-size: 0.95rem;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: rgba(30, 27, 75, 0.7);
        border-left: 4px solid #6366f1;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 24px;
        color: #c7d2fe;
        line-height: 1.6;
    }

    /* Target Prompt Box */
    .prompt-box {
        background-color: #0f2942;
        border: 1px solid #2563eb;
        border-radius: 10px;
        padding: 18px 22px;
        font-size: 1.15rem;
        color: #93c5fd;
        margin-bottom: 20px;
        line-height: 1.6;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }

    /* Insight Bullet Item */
    .insight-item {
        background: #1e293b;
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


def save_session_to_history(session_data: dict):
    """Save session results to local CSV history file."""
    os.makedirs("data", exist_ok=True)
    df_new = pd.DataFrame([session_data])
    if os.path.exists(SESSION_HISTORY_FILE):
        df_existing = pd.read_csv(SESSION_HISTORY_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(SESSION_HISTORY_FILE, index=False)


# -----------------------------------------------------------------------------
# 2. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric-folders/100/brain.png", width=65)
st.sidebar.title("Cognitive Load AI")
st.sidebar.caption("Data Science Keystroke Biometrics")

st.sidebar.markdown("""
<div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); padding: 10px; border-radius: 10px; margin-bottom: 15px; text-align: center;">
    <div style="font-size: 11px; font-weight: 600; color: #a5b4fc; margin-bottom: 6px;">🌟 Standalone Real-Website UI</div>
    <a href="http://localhost:8000" target="_blank" style="display: block; background: linear-gradient(90deg, #6366f1, #06b6d4); color: white; padding: 7px 12px; border-radius: 6px; font-size: 11px; font-weight: bold; text-decoration: none; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">Open Website in New Tab ↗</a>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🌐 Real Website Suite (Live Arena)",
        "🏠 Home",
        "⌨️ Typing Test",
        "📊 Current Session Analytics",
        "📜 Local History Log",
        "🤖 Model Performance Dashboard"
    ],
    index=0
)

st.sidebar.divider()
st.sidebar.info(
    "**Project Highlights**\n"
    "- **ML Models**: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting\n"
    "- **Leak-Free Pipeline**: StandardScaler + Estimator\n"
    "- **Features**: 15 Behavioral Biometrics\n"
    "- **Privacy**: Anonymized Timing Only"
)


# -----------------------------------------------------------------------------
# PAGE 0: REAL WEBSITE SUITE
# -----------------------------------------------------------------------------
if page == "🌐 Real Website Suite (Live Arena)":
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%); padding: 12px 20px; border-radius: 12px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);">
        <div style="color: white; font-weight: 600; font-size: 14px;">
            ✨ <strong>Real Website UI is Active!</strong> Live letter-by-letter typing arena, mechanical keyboard audio, What-If simulator, and Radar biometrics.
        </div>
        <a href="http://localhost:8000" target="_blank" style="background: white; color: #4338ca; padding: 7px 16px; border-radius: 8px; font-weight: 700; text-decoration: none; font-size: 12px;">
            Open in Full Tab ↗
        </a>
    </div>
    """, unsafe_allow_html=True)
    components.iframe("http://localhost:8000", height=950, scrolling=True)

# -----------------------------------------------------------------------------
# PAGE 1: HOME
# -----------------------------------------------------------------------------
elif page == "🏠 Home":
    st.title("🧠 Cognitive Load Detection using Typing Patterns")
    st.subheader("Predicting Mental Effort via Keystroke Biometrics & Machine Learning")

    st.markdown("""
    <div class="privacy-banner">
        🔒 <strong>Privacy & Security Guaranteed:</strong> This system processes timing biometrics only (keydown/keyup timestamps in milliseconds, delays, hesitation pauses, backspace counts). <strong>Your typed text content is never recorded or stored.</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Academic & Ethical Disclaimer:</strong><br>
        This Data Science system estimates cognitive load levels (LOW, MEDIUM, HIGH) strictly from statistical typing mechanics. 
        <strong>It is NOT a medical diagnostic system</strong> and does NOT diagnose stress, anxiety, ADHD, depression, or any neurological condition.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        ### 📌 Project Objective & Motivation
        Cognitive load refers to the amount of working memory resources demanded during cognitive tasks. 
        This project uses non-intrusive software biometrics to detect subtle changes in motor fluency, hesitation pauses, and error revisions during typing.

        #### ⚡ 15 Keystroke Biometrics Captured:
        - **Speed Metrics**: Words Per Minute (WPM), Characters Per Minute (CPM), Keystrokes Per Second
        - **Motor Fluency**: Key hold duration (ms) & inter-key flight intervals (ms)
        - **Planning & Hesitation**: Pause count (>1.5s), average & maximum pause durations
        - **Revision Activity**: Backspace count & correction rate
        - **Cadence & Precision**: Typing rhythm (std dev of flight times) & Levenshtein error rate
        """)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#60a5fa; margin-top:0;">⚙️ System Workflow Pipeline</h4>
            <ol style="line-height:1.8; color:#cbd5e1;">
                <li><strong>Browser Event Logger</strong>: High-resolution millisecond keydown/keyup tracking.</li>
                <li><strong>Feature Extractor</strong>: Calculation of 15 numerical biometric metrics.</li>
                <li><strong>Leak-Free ML Pipeline</strong>: Scaler + Winner Classifier (Random Forest / Logistic Regression).</li>
                <li><strong>Explainable Inference</strong>: Prediction class, confidence %, and top behavioral drivers.</li>
                <li><strong>Local History Log</strong>: Persistent session history saved to <code>data/session_history.csv</code>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📊 15 Keystroke Biometric Features Table")
    feat_df = pd.DataFrame([
        {"Feature": "typing_speed_wpm", "Category": "Speed", "Description": "Standard WPM formula ((chars/5) / minute)"},
        {"Feature": "characters_per_minute", "Category": "Speed", "Description": "Total characters typed per minute"},
        {"Feature": "average_key_hold_time", "Category": "Motor", "Description": "Mean key press-to-release duration (ms)"},
        {"Feature": "average_inter_key_interval", "Category": "Motor", "Description": "Mean flight time between consecutive key presses (ms)"},
        {"Feature": "median_inter_key_interval", "Category": "Motor", "Description": "Median flight time (ms)"},
        {"Feature": "typing_time", "Category": "Duration", "Description": "Total active typing duration (seconds)"},
        {"Feature": "pause_count", "Category": "Planning", "Description": "Count of inter-key delays > 1.5 seconds"},
        {"Feature": "average_pause_duration", "Category": "Planning", "Description": "Mean duration of hesitation pauses (ms)"},
        {"Feature": "maximum_pause_duration", "Category": "Planning", "Description": "Maximum hesitation pause duration (ms)"},
        {"Feature": "backspace_count", "Category": "Revision", "Description": "Total Backspace key presses"},
        {"Feature": "correction_rate", "Category": "Revision", "Description": "Ratio of backspaces to total keystrokes"},
        {"Feature": "error_rate", "Category": "Accuracy", "Description": "Edit distance mismatch ratio against prompt text"},
        {"Feature": "typing_rhythm", "Category": "Cadence", "Description": "Standard deviation of inter-key flight intervals"},
        {"Feature": "keystrokes_per_second", "Category": "Speed", "Description": "Keystroke frequency per second"},
        {"Feature": "total_keystrokes", "Category": "Volume", "Description": "Total keydown events recorded"}
    ])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 2: TYPING TEST & PREDICTION
# -----------------------------------------------------------------------------
elif page == "⌨️ Typing Test":
    st.title("⌨️ Interactive Typing Test & Real-Time Prediction")
    st.write("Select a text passage complexity below, click **'Start Test'**, type the passage into the box, and click **'Submit Typing Sample'**.")

    prompt_option = st.selectbox(
        "Select Test Passage Complexity:",
        [
            "Easy Passage (Simple Narrative)",
            "Medium Passage (Technical Concepts)",
            "Complex Passage (Dense Academic Text)"
        ]
    )

    PROMPTS = {
        "Easy Passage (Simple Narrative)": "The sun shines brightly over the calm blue ocean. Birds fly across the clear sky while waves gently touch the sandy shore.",
        "Medium Passage (Technical Concepts)": "Algorithm efficiency requires balancing spatial memory consumption with temporal complexity, optimizing iterative data processing loops.",
        "Complex Passage (Dense Academic Text)": "Heterogeneous multi-agent cognitive architectures synthesize non-stationary environmental telemetry via asynchronous backpropagation mechanisms."
    }

    target_prompt = PROMPTS[prompt_option]

    st.markdown(f"""
    <div class="prompt-box">
        <strong style="color: #60a5fa;">Target Passage Prompt:</strong><br>
        <em>"{target_prompt}"</em>
    </div>
    """, unsafe_allow_html=True)

    # JavaScript Keystroke Logging Component with Timer & Live Counters
    js_component_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: 'Segoe UI', sans-serif; margin: 0; background: transparent; color: #fff; }
            .toolbar { display: flex; gap: 12px; margin-bottom: 10px; font-size: 14px; color: #94a3b8; align-items: center; flex-wrap: wrap; }
            .badge { background: #1e293b; padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; font-size: 13px; }
            .badge strong { color: #60a5fa; }
            textarea {
                width: 100%;
                height: 125px;
                background-color: #1e293b;
                border: 2px solid #3b82f6;
                border-radius: 8px;
                color: #f8fafc;
                font-size: 16px;
                padding: 14px;
                box-sizing: border-box;
                outline: none;
                resize: vertical;
                font-family: inherit;
            }
            textarea:focus { border-color: #60a5fa; box-shadow: 0 0 10px rgba(96,165,250,0.4); }
            textarea:disabled { background-color: #0f172a; border-color: #334155; opacity: 0.65; }
            .btn-container { margin-top: 12px; display: flex; gap: 12px; }
            button {
                background-color: #2563eb;
                color: white;
                border: none;
                padding: 10px 22px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            button:hover { background-color: #1d4ed8; transform: translateY(-1px); }
            button:disabled { background-color: #475569; cursor: not-allowed; transform: none; opacity: 0.6; }
            #startBtn { background-color: #10b981; }
            #startBtn:hover { background-color: #059669; }
        </style>
    </head>
    <body>
        <div class="toolbar">
            <button id="startBtn">▶ Start Test</button>
            <span class="badge">⏱ Elapsed: <strong id="timerText">0.0s</strong></span>
            <span class="badge">🔤 Characters: <strong id="charCount">0</strong></span>
            <span class="badge">⌫ Backspaces: <strong id="bsCount">0</strong></span>
        </div>

        <textarea id="typingBox" placeholder="Click 'Start Test' above to begin..." disabled></textarea>

        <div class="btn-container">
            <button id="submitBtn" disabled>Submit Typing Sample</button>
        </div>

        <script>
            const typingBox = document.getElementById('typingBox');
            const startBtn = document.getElementById('startBtn');
            const submitBtn = document.getElementById('submitBtn');
            const timerText = document.getElementById('timerText');
            const charCount = document.getElementById('charCount');
            const bsCount = document.getElementById('bsCount');

            let keydowns = [];
            let keyups = [];
            let backspaces = 0;
            let startTime = null;
            let timerInterval = null;

            startBtn.addEventListener('click', () => {
                keydowns = [];
                keyups = [];
                backspaces = 0;
                typingBox.value = "";
                typingBox.disabled = false;
                submitBtn.disabled = false;
                startBtn.disabled = true;
                typingBox.focus();

                startTime = performance.now();
                timerInterval = setInterval(() => {
                    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
                    timerText.innerText = elapsed + "s";
                }, 100);
            });

            typingBox.addEventListener('keydown', (e) => {
                if (typingBox.disabled) return;
                const now = performance.now();
                keydowns.push({ key: e.key, timestamp: now });

                if (e.key === 'Backspace') {
                    backspaces++;
                    bsCount.innerText = backspaces;
                }
                charCount.innerText = typingBox.value.length + (e.key.length === 1 ? 1 : 0);
            });

            typingBox.addEventListener('keyup', (e) => {
                if (typingBox.disabled) return;
                const now = performance.now();
                keyups.push({ key: e.key, timestamp: now });
            });

            submitBtn.addEventListener('click', () => {
                clearInterval(timerInterval);
                typingBox.disabled = true;
                submitBtn.disabled = true;
                startBtn.disabled = false;

                const payload = {
                    keydowns: keydowns,
                    keyups: keyups,
                    typed_text: typingBox.value
                };

                if (window.parent && window.parent.postMessage) {
                    window.parent.postMessage({
                        isStreamlitMessage: true,
                        type: "streamlit:setComponentValue",
                        value: JSON.stringify(payload)
                    }, "*");
                }
            });
        </script>
    </body>
    </html>
    """

    component_data = components.html(js_component_code, height=270)

    # Optional Simulation Inputs for Quick Testing
    with st.expander("🛠️ Manual Input / Simulation Tester"):
        manual_text = st.text_area("Manual Typed Text", target_prompt)
        col_m1, col_m2, col_m3 = st.columns(3)
        sim_wpm = col_m1.number_input("Typing Speed (WPM)", 5.0, 120.0, 45.0)
        sim_hold = col_m2.number_input("Avg Hold Time (ms)", 50.0, 400.0, 140.0)
        sim_inter = col_m3.number_input("Avg Inter-key Flight (ms)", 80.0, 600.0, 220.0)
        col_m4, col_m5 = st.columns(2)
        sim_backspaces = col_m4.number_input("Backspace Count", 0, 50, 4)
        sim_pauses = col_m5.number_input("Pause Count (>1.5s)", 0, 20, 2)
        run_manual = st.button("Run Manual Simulation")

    # Process Payload
    features = None
    if component_data:
        try:
            payload = json.loads(component_data)
            keydowns = payload.get("keydowns", [])
            keyups = payload.get("keyups", [])
            typed_text = payload.get("typed_text", "")

            if len(keydowns) < 5:
                st.warning("Please start test and type at least a few words before submitting.")
            else:
                features = compute_typing_features(keydowns, keyups, typed_text, target_prompt)
        except Exception as err:
            st.error(f"Error processing key events: {err}")

    elif run_manual:
        cpm = sim_wpm * 5.0
        kps = cpm / 60.0
        typing_time = len(manual_text) / max(0.1, kps)
        features = {
            "typing_speed_wpm": sim_wpm,
            "characters_per_minute": cpm,
            "average_key_hold_time": sim_hold,
            "average_inter_key_interval": sim_inter,
            "median_inter_key_interval": sim_inter * 0.95,
            "typing_time": round(typing_time, 2),
            "pause_count": sim_pauses,
            "average_pause_duration": 2000.0 if sim_pauses > 0 else 0.0,
            "maximum_pause_duration": 2500.0 if sim_pauses > 0 else 0.0,
            "backspace_count": sim_backspaces,
            "correction_rate": round(sim_backspaces / max(1, len(manual_text)), 4),
            "error_rate": 0.04,
            "typing_rhythm": 110.0,
            "keystrokes_per_second": round(kps, 2),
            "total_keystrokes": len(manual_text) + sim_backspaces
        }

    if features:
        st.session_state["latest_features"] = features
        res = predict_cognitive_load(features, MODEL_PATH_DEFAULT)
        st.session_state["latest_result"] = res

        # Log to Session History
        sess_id = f"SESS-{int(time.time())}"
        session_record = {
            "session_id": sess_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "typing_speed_wpm": features["typing_speed_wpm"],
            "typing_time": features["typing_time"],
            "pause_count": features["pause_count"],
            "backspace_count": features["backspace_count"],
            "average_inter_key_interval": features["average_inter_key_interval"],
            "predicted_cognitive_load": res["cognitive_load"],
            "confidence_pct": res["confidence_pct"]
        }
        save_session_to_history(session_record)

    if "latest_result" in st.session_state:
        res = st.session_state["latest_result"]
        feats = st.session_state["latest_features"]
        load = res["cognitive_load"]
        conf = res["confidence_pct"]
        badge_class = f"badge-{load.lower()}"

        st.divider()
        st.subheader("🎯 Prediction & Explainable Behavioral Result")

        col_res1, col_res2 = st.columns([1, 1.2])

        with col_res1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3 style="margin-top:0; color:#94a3b8; font-size:1.1rem;">Predicted Cognitive Load</h3>
                <span class="{badge_class}">{load}</span>
                <h4 style="margin-top: 18px; color: #60a5fa;">Confidence Score: {conf}%</h4>
                <p style="font-size: 0.85rem; color: #64748b; margin-bottom:0;">Winning Architecture: {res['model_used']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("**Class Probability Distribution:**")
            for cls_name, prob_val in res["class_probabilities"].items():
                st.write(f"**{cls_name}**: {prob_val}%")
                st.progress(min(1.0, prob_val / 100.0))

        with col_res2:
            st.markdown("#### 🔍 Observed Behavioral Patterns (Model Drivers):")
            for pattern in res["observed_patterns"]:
                st.markdown(f"""<div class="insight-item">📌 {pattern}</div>""", unsafe_allow_html=True)

            st.markdown(f"<p style='color:#94a3b8; font-size:0.85rem; margin-top:12px;'><em>(Note: {res['disclaimer']})</em></p>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 3: CURRENT SESSION ANALYTICS
# -----------------------------------------------------------------------------
elif page == "📊 Current Session Analytics":
    st.title("📊 Active Session Biometric Analytics")

    if "latest_features" not in st.session_state:
        st.info("No active typing test session found. Complete a test on the '⌨️ Typing Test' page first.")
    else:
        feats = st.session_state["latest_features"]
        res = st.session_state.get("latest_result", {})

        st.subheader("Active Session Metric Grid")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Typing Speed", f"{feats['typing_speed_wpm']} WPM")
        m_col2.metric("Characters / Min", f"{feats['characters_per_minute']}")
        m_col3.metric("Key Hold Time", f"{feats['average_key_hold_time']} ms")
        m_col4.metric("Flight Interval", f"{feats['average_inter_key_interval']} ms")

        m_col5, m_col6, m_col7, m_col8 = st.columns(4)
        m_col5.metric("Pause Count (>1.5s)", f"{feats['pause_count']}")
        m_col6.metric("Backspace Count", f"{feats['backspace_count']}")
        m_col7.metric("Correction Rate", f"{feats['correction_rate'] * 100:.1f}%")
        m_col8.metric("Typing Rhythm", f"{feats['typing_rhythm']} ms")

        st.divider()
        st.subheader("Session Feature Profile Chart")
        
        norm_data = [
            {"Metric": "WPM", "Value": feats['typing_speed_wpm']},
            {"Metric": "Hold Time (ms)", "Value": feats['average_key_hold_time']},
            {"Metric": "Flight Time (ms)", "Value": feats['average_inter_key_interval']},
            {"Metric": "Rhythm Std (ms)", "Value": feats['typing_rhythm']},
            {"Metric": "Backspaces (x10)", "Value": feats['backspace_count'] * 10},
            {"Metric": "Pauses (x20)", "Value": feats['pause_count'] * 20}
        ]
        df_norm = pd.DataFrame(norm_data)
        fig_bar = px.bar(
            df_norm,
            x="Metric",
            y="Value",
            color="Metric",
            title="Session Biometric Scale Profile",
            template="plotly_dark"
        )
        fig_bar.update_layout(height=420)
        st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 4: LOCAL SESSION HISTORY LOG
# -----------------------------------------------------------------------------
elif page == "📜 Local History Log":
    st.title("📜 Local User Session History Log")
    st.write("View past typing test session results stored locally in `data/session_history.csv`.")

    if not os.path.exists(SESSION_HISTORY_FILE):
        st.info("No session history logged yet. Complete a typing test to generate your first entry.")
    else:
        df_hist = pd.read_csv(SESSION_HISTORY_FILE)
        st.write(f"Total Logged Sessions: **{len(df_hist)}**")
        
        st.dataframe(df_hist.sort_values(by="timestamp", ascending=False), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Historical Speed vs Pause Trend")
        fig_trend = px.line(
            df_hist,
            x="timestamp",
            y=["typing_speed_wpm", "average_inter_key_interval"],
            markers=True,
            title="Historical Speed (WPM) & Flight Delay Trends",
            template="plotly_dark"
        )
        fig_trend.update_layout(height=420)
        st.plotly_chart(fig_trend, use_container_width=True)

        if st.button("🗑️ Clear Session History"):
            os.remove(SESSION_HISTORY_FILE)
            st.success("Session history cleared successfully!")
            st.rerun()


# -----------------------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE DASHBOARD
# -----------------------------------------------------------------------------
elif page == "🤖 Model Performance Dashboard":
    st.title("🤖 Machine Learning Model Performance Dashboard")
    st.write("Actual calculated evaluation metrics loaded dynamically from `models/model_metrics.json`.")

    if not os.path.exists(MODEL_METRICS_FILE):
        st.warning("Model metrics file not found. Run `python src/train_model.py` to calculate model metrics.")
    else:
        with open(MODEL_METRICS_FILE, "r") as f:
            metrics_data = json.load(f)

        win_model = metrics_data.get("winning_model", "Logistic Regression")
        st.success(f"🏆 Winning Architecture Selected: **{win_model}**")

        models_dict = metrics_data.get("models", {})
        summary_rows = []
        for name, perf in models_dict.items():
            summary_rows.append({
                "Model": name,
                "Accuracy": perf["accuracy"],
                "Precision": perf["precision"],
                "Recall": perf["recall"],
                "F1-Score": perf["f1_score"],
                "5-Fold CV F1": perf["cv_f1_mean"]
            })

        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.subheader("Model Benchmark Comparison")
            fig_comp = px.bar(
                df_summary,
                x="Model",
                y=["Accuracy", "F1-Score"],
                barmode="group",
                title="Accuracy & F1-Score by Model",
                template="plotly_dark"
            )
            fig_comp.update_layout(height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_p2:
            st.subheader("Feature Importance (Random Forest)")
            rf_perf = models_dict.get("Random Forest", {})
            rf_fi = rf_perf.get("feature_importance", {})
            if rf_fi:
                df_fi = pd.DataFrame(list(rf_fi.items()), columns=["Feature", "Importance"]).head(10)
                fig_fi = px.bar(
                    df_fi,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Biometric Feature Importances",
                    template="plotly_dark"
                )
                fig_fi.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_fi, use_container_width=True)

        st.divider()
        st.subheader("Winning Model Confusion Matrix")
        viz_cm = os.path.join("visualizations", "confusion_matrix.png")
        if os.path.exists(viz_cm):
            st.image(viz_cm, caption="Confusion Matrix Visualization")
