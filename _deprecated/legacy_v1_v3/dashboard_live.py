import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import torch
import plotly.graph_objects as go
import plotly.express as px
import time

st.set_page_config(page_title="MATHIR - Live Brain", page_icon="🧠", layout="wide")

# --- PREMIUM CSS (GLASSMORPHISM & NEON) ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(20, 20, 30) 0%, rgb(0, 0, 0) 90%);
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Title with Gradient */
    .main-title { 
        font-size: 3em; 
        background: -webkit-linear-gradient(45deg, #00ff9d, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center; 
        font-weight: 900; 
        letter-spacing: 2px;
        text-shadow: 0px 0px 30px rgba(0, 255, 157, 0.3);
        margin-bottom: 20px;
    }

    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 212, 255, 0.2);
        border: 1px solid rgba(0, 212, 255, 0.3);
    }

    .metric-label {
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 2.2em;
        font-weight: 700;
        color: #fff;
    }

    /* Color Accents */
    .color-mathir { color: #00ff9d; text-shadow: 0 0 15px rgba(0, 255, 157, 0.4); }
    .color-lstm { color: #ff9900; text-shadow: 0 0 15px rgba(255, 153, 0, 0.4); }
    .color-vram { color: #00d4ff; }
    .color-ram-ok { color: #00ff9d; }
    .color-ram-warn { color: #ff9900; }
    .color-ram-crit { color: #ff0000; text-shadow: 0 0 15px rgba(255, 0, 0, 0.6); } 
    .diff-pos { color: #00ff9d; font-weight: bold; background: rgba(0, 255, 157, 0.1); padding: 5px 10px; border-radius: 8px; }
    .diff-neg { color: #ff4b4b; font-weight: bold; background: rgba(255, 75, 75, 0.1); padding: 5px 10px; border-radius: 8px; }

    /* Separator */
    hr { border-color: rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)

# Functions
def get_file_mtime(filepath):
    try:
        return os.path.getmtime(filepath)
    except:
        return 0

@st.cache_data(ttl=1)
def load_live_data_cached(mtime):
    """Load data (cache invalidated if mtime changes)"""
    for _ in range(3):
        try:
            if os.path.exists('training_log.json'):
                with open('training_log.json', 'r') as f:
                    content = f.read().strip()
                    if content:
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            lines = content.split('\n')
                            for line in reversed(lines):
                                try:
                                    if line.strip():
                                        return json.loads(line)
                                except:
                                    continue
            time.sleep(0.1)
        except:
            time.sleep(0.1)
    return None

def load_live_data():
    current_mtime = get_file_mtime('training_log.json')
    return load_live_data_cached(current_mtime)

def load_live_weights():
    try:
        ckpt_path = "checkpoints/mathir_live.pth"
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            stats = { "mean": [], "std": [] }
            layers_analysed = 0
            for key, tensor in state_dict.items():
                if 'weight' in key and len(tensor.shape) >= 2:
                    stats["mean"].append(tensor.mean().item())
                    stats["std"].append(tensor.std().item())
                    layers_analysed += 1
                    if layers_analysed > 5: break
            return stats
    except:
        pass
    return None

# --- HEADER ---
st.markdown('<div class="main-title">MATHIR RESEARCH LAB</div>', unsafe_allow_html=True)

# --- SIDEBAR CONTEXT ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/brain.png", width=50)
    st.markdown("### 🔬 The Cognitive Labyrinth")
    st.info("""
    **This is not a simple road.**
    It is a mental torture test designed to break classic LSTM models.
    """)
    
    with st.expander("🌪️ Distractions & Chaos", expanded=True):
        st.markdown("""
        - **⚡ Unpredictable Weather** : Rain, Fog, Night (Changes friction).
        - **🎭 Visual Distractions** : Misleading signs, animals, ads (Must be ignored).
        - **🕸️ Latency** : Random delay between decision and action.
        """)
        
    with st.expander("🧠 Memory Challenge", expanded=True):
        st.markdown("""
        - **🚦 Deferred Orders** : "Turn left... after the 3rd red light".
        - **📦 Overload** : Up to 5 active rules at the same time.
        - **⏳ Forgetting** : Info must be kept in memory for ~200 steps.
        """)
        
    st.markdown("---")
    st.markdown("v2.4.0 - Evolution Core")

# --- DATA ---
data = load_live_data()

if not data:
    st.warning("⏳ SYNC IN PROGRESS... Waiting for live data.")
else:
    # --- METRICS GRID (6 COLUMNS) ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    step = data.get('step', 0)
    m_score = data.get('mathir_avg_reward', 0)
    l_score = data.get('lstm_avg_reward', 0)
    vram = data.get('vram_gb', 0)
    ram_gb = data.get('ram_gb', 0)
    ram_pct = data.get('ram_percent', 0)
    diff = (m_score - l_score) * 100
    
    # Tooltip definitions
    tooltips = {
        "step": "Total number of training iterations performed.",
        "mathir": "Average accuracy of MATHIR model (Max score = 1.0).",
        "lstm": "Average accuracy of classic LSTM model (Competitor).",
        "adv": "Performance difference: Positive = MATHIR wins, Negative = LSTM wins.",
        "vram": "Video Memory (GPU) Reserved.",
        "ram": "System Memory (RAM) Used."
    }

    with c1:
        st.markdown(f'<div class="metric-card" title="{tooltips["step"]}"><div class="metric-label">Training Step ℹ️</div><div class="metric-value">{step:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card" title="{tooltips["mathir"]}"><div class="metric-label">MATHIR Accuracy ℹ️</div><div class="metric-value color-mathir">{m_score:.3f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card" title="{tooltips["lstm"]}"><div class="metric-label">LSTM Accuracy ℹ️</div><div class="metric-value color-lstm">{l_score:.3f}</div></div>', unsafe_allow_html=True)
    with c4:
        diff_html = f'<span class="diff-pos">+{diff:.1f}%</span>' if diff >= 0 else f'<span class="diff-neg">{diff:.1f}%</span>'
        st.markdown(f'<div class="metric-card" title="{tooltips["adv"]}"><div class="metric-label">Advantage ℹ️</div><div class="metric-value">{diff_html}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card" title="{tooltips["vram"]}"><div class="metric-label">VRAM (Reserved) ℹ️</div><div class="metric-value color-vram">{vram:.2f} GB</div></div>', unsafe_allow_html=True)
    with c6:
        ram_color = "color-ram-ok"
        if ram_pct > 90: ram_color = "color-ram-crit"
        elif ram_pct > 80: ram_color = "color-ram-warn"
        st.markdown(f'<div class="metric-card" title="{tooltips["ram"]}"><div class="metric-label">System RAM ⚠️</div><div class="metric-value {ram_color}">{ram_gb:.1f} GB <span style="font-size:0.5em">({ram_pct:.0f}%)</span></div></div>', unsafe_allow_html=True)

    # --- BATTLE ARENA (GAMIFICATION) ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Emoji Logic
    battle_emoji = "😐"
    battle_status = "Perfect Draw"
    battle_color = "#888"
    
    if diff > 20:
        battle_emoji = "😂🔥"
        battle_status = "MATHIR HUMILIATES LSTM"
        battle_color = "#00ff9d"
    elif diff > 5:
        battle_emoji = "😎💪"
        battle_status = "MATHIR DOMINATES"
        battle_color = "#00cc7a"
    elif diff > 0:
        battle_emoji = "🙂🤏"
        battle_status = "MATHIR WINS BARELY"
        battle_color = "#aaffd3"
    elif diff > -5:
        battle_emoji = "👀⚔️"
        battle_status = "TIGHT FIGHT..."
        battle_color = "#ffaa00"
    elif diff > -20:
        battle_emoji = "😬🩹"
        battle_status = "MATHIR STRUGGLING"
        battle_color = "#ff6600"
    else:
        battle_emoji = "😭🚑"
        battle_status = "MATHIR CRUSHED"
        battle_color = "#ff0000"

    st.markdown(f"""
    <div style="text-align:center; padding: 20px; border-radius: 20px; background: rgba(0,0,0,0.4); border: 2px solid {battle_color};">
        <div style="font-size: 4em; margin-bottom: 10px;">{battle_emoji}</div>
        <div style="font-size: 2em; font-weight: bold; color: {battle_color}; text-transform: uppercase; letter-spacing: 3px;">
            {battle_status}
        </div>
        <div style="color: #aaa; margin-top: 5px;">LSTM attempts to survive the Labyrinth...</div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- LAST TORTURE RESULT ---
    if 'benchmarks' in data and data['benchmarks']:
        last_bench = data['benchmarks'][-1]
        lm_score = last_bench['mathir_score']
        ll_score = last_bench['lstm_score']
        lwinner = last_bench['winner']
        
        lcolor = "#00ff9d" if lwinner == "MATHIR" else "#ff4b4b"
        
        st.markdown(f"""
        <div style="text-align:center; margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 10px;">
            <span style="font-size: 1.2em; color: #ccc;">LAST TORTURE TEST : </span>
            <span style="font-size: 1.5em; font-weight: bold; color: #00ff9d;">MATHIR {lm_score:.1f}</span> 
            <span style="color: #666;"> vs </span>
            <span style="font-size: 1.5em; font-weight: bold; color: #ff9900;">LSTM {ll_score:.1f}</span>
            <span style="margin-left: 10px; font-weight: bold; color: {lcolor}; border: 1px solid {lcolor}; padding: 2px 8px; border-radius: 5px;">{lwinner} WINS</span>
        </div>
        <br>
        """, unsafe_allow_html=True)

    # --- CHARTS ---
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 🧬 Performance Comparison (Live)")
        if 'history' in data:
            steps = data['history'].get('steps', [])[-500:] # Zoom on last 500
            mathir_hist = data['history'].get('mathir', [])[-500:]
            lstm_hist = data['history'].get('lstm', [])[-500:]
            
            if steps:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=steps, y=mathir_hist, name='MATHIR', line=dict(color='#00ff9d', width=3), mode='lines'))
                fig.add_trace(go.Scatter(x=steps, y=lstm_hist, name='LSTM', line=dict(color='#ff9900', width=2, dash='dot'), mode='lines'))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ccc'),
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                    legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)')
                )
                st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### 🧠 Neuro-Plasticity")
        stats = load_live_weights()
        if stats:
            mean_val = np.mean(stats["mean"]) if stats["mean"] else 0
            std_val = np.mean(stats["std"]) if stats["std"] else 0
            
            # Custom Progress Bars with Explanations
            st.write(f"Synaptic Density: **{mean_val:.4f}**")
            st.markdown(f"""<div style="background:#333;border-radius:10px;height:8px;width:100%;margin-bottom:5px;"><div style="background:#00d4ff;width:{min(abs(mean_val)*1000, 100)}%;height:100%;border-radius:10px;box-shadow:0 0 10px #00d4ff;"></div></div>""", unsafe_allow_html=True)
            
            st.write(f"Adaptation Rate (Variance): **{std_val:.4f}**")
            st.markdown(f"""<div style="background:#333;border-radius:10px;height:8px;width:100%;margin-bottom:15px;"><div style="background:#ff00ff;width:{min(std_val*500, 100)}%;height:100%;border-radius:10px;box-shadow:0 0 10px #ff00ff;"></div></div>""", unsafe_allow_html=True)
            
            with st.expander("ℹ️ Understanding these gauges"):
                st.markdown("""
                - **Synaptic Density** : The average strength of connections. If it increases, the brain strengthens its memories.
                - **Adaptation Rate** : Indicates if the brain is modifying its structure significantly. High value = Intense learning in progress.
                """)
            
            # Params & AutoML Logic
            if 'current_hyperparams' in data:
                st.write("---")
                st.subheader("🧬 Genetics & AutoML")
                
                params = data['current_hyperparams']
                decay = params.get('retention_decay', [])
                lstm_lr = params.get('lstm_lr', 0.0001)
                mathir_lr = params.get('mathir_lr', 0.0001)
                scenario = params.get('scenario', "PHASE 1: EVOLUTION")

                # Scenario Badge
                st.info(f"**ACTIVE SCENARIO:** {scenario}")

                # Summary Metrics
                c_p1, c_p2, c_p3 = st.columns(3)
                with c_p1:
                    mean_decay = float(np.mean(decay)) if decay else 0
                    st.metric("MATHIR Retention", f"{mean_decay:.3f}", help="Average retention rates.")
                with c_p2:
                    st.metric("MATHIR LR", f"{mathir_lr:.5f}", help="Learning speed of our champion.")
                with c_p3:
                    is_capped = lstm_lr >= 0.001
                    lr_display = f"{lstm_lr:.5f}"
                    if is_capped:
                        lr_display += " (MAX)"
                        st.metric("LSTM LR", lr_display, delta="SATURATED", delta_color="inverse", help="LSTM is maxed out!")
                    else:
                        st.metric("LSTM LR", lr_display, help="Learning speed of the competitor.")

                # Detailed Clickable Explanation
                with st.expander("🔍 Detailed Analysis: How it works?", expanded=False):
                    st.markdown(f"""
                    ### 🎯 4-Phase Scientific Protocol
                    We test robustness over 30k step cycles:
                    
                    1.  **PHASE 1 (Evolution)** : LSTM is dynamically doped to challenge MATHIR.
                    2.  **PHASE 2 (Standard)** : "Fair Fight". No doping, standard LRs (3e-4).
                    3.  **PHASE 3 (Unleashed)** : MATHIR unleashed (3e-4) vs Doped LSTM (1e-3).
                    4.  **PHASE 4 (Chaos)** : Both models at max (1e-3).
                    
                    **Current State** : `{scenario}`
                    """)

# --- BENCHMARK / HALL OF FAME ---
    if 'benchmarks' in data and data['benchmarks']:
        st.markdown("---")
        st.subheader("🏆 Hall of Fame (Torture Test Results)")
        
        bench_df = pd.DataFrame(data['benchmarks'])
        
        if not bench_df.empty:
            # Stability Score
            wins = bench_df['winner'].value_counts()
            mathir_wins = wins.get('MATHIR', 0)
            total_benches = len(bench_df)
            stability = (mathir_wins / total_benches) * 100
            
            c_b1, c_b2 = st.columns([1, 3])
            
            with c_b1:
                st.metric("Win Rate (5 last)", f"{stability:.0f}%", help="Percentage of benchmarks won by MATHIR.")
                last_winner = bench_df.iloc[-1]['winner']
                color = "green" if last_winner == "MATHIR" else "red"
                st.markdown(f"Last Winner : <b style='color:{color};font-size:1.2em;'>{last_winner}</b>", unsafe_allow_html=True)
                
            with c_b2:
                # Chart
                fig_bench = px.line(bench_df, x='step', y=['mathir_score', 'lstm_score'], 
                                    markers=True, title="Pure Performance Evolution (Same Seed)",
                                    color_discrete_map={"mathir_score": "#00ff9d", "lstm_score": "#ff9900"})
                fig_bench.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#ccc'))
                st.plotly_chart(fig_bench, use_container_width=True)

# --- REPORTING & EXPORT ---
    st.markdown("---")
    st.subheader("📑 Report & Export")
    
    col_rep1, col_rep2 = st.columns([1, 1])
    
    with col_rep1:
        st.markdown("### 1. AI Analysis (Ollama)")
        
        # Load Best Params (Safe Load)
        best_params = None
        if os.path.exists("mathir_best_params.json"):
            try:
                with open("mathir_best_params.json", "r") as f:
                    best_params = json.load(f)
            except:
                pass

        if st.button("🧠 Generate Strategist Analysis (Expert)"):
            with st.spinner("Consulting Llama 3.2 Oracle for optimal parameter analysis..."):
                
                # --- DATA GATHERING ---
                h_mathir = data.get('history', {}).get('mathir', [])
                h_lstm = data.get('history', {}).get('lstm', [])
                
                if not h_mathir:
                    st.error("Not enough data for analysis.")
                else:
                    # Metrics
                    current_score = h_mathir[-1] if h_mathir else 0
                    avg_m = np.mean(h_mathir[-50:]) if len(h_mathir) > 50 else np.mean(h_mathir)
                    avg_l = np.mean(h_lstm[-50:]) if len(h_lstm) > 50 else np.mean(h_lstm)
                    
                    # Best Params Context
                    bp_text = "No 'Best Run' recorded yet."
                    bp_json_str = "{}"
                    if best_params:
                        bp_score = best_params.get('score', 0)
                        bp_step = best_params.get('step', 0)
                        bp_vals = best_params.get('params', {}).get('retention_decay', [])
                        bp_text = f"Record at Step {bp_step} (Score: {bp_score:.3f}) with Decay={bp_vals}"
                        bp_json_str = json.dumps(best_params.get('params', {}))
                    
                    # Scenario Context for Ollama
                    current_scenario = data.get('current_hyperparams', {}).get('scenario', "Unknown")
                    mathir_lr_val = data.get('current_hyperparams', {}).get('mathir_lr', 0.0001)

                    # --- OPTION A: REAL OLLAMA CALL ---
                    ollama_response = None
                    try:
                        import subprocess
                        
                        prompt = f"""
                        Your mission is to write a punchy FINAL verdict (Style: Futuristic AI Expert & Enthusiastic).
                        
                        FIGHT CONTEXT:
                        - CURRENT SCENARIO: {current_scenario}
                        - MATHIR (New Gen) Score: {avg_m:.3f} (LR: {mathir_lr_val})
                        - LSTM (Old Gen) Score: {avg_l:.3f}
                        - Winning Parameters: {bp_json_str}
                        
                        Structure your response like this:
                        
                        1. 🏆 THE UNDISPUTED WINNER
                           - Declare the winner with emphasis. Use emojis (🚀, 🧠, 🥇). If MATHIR wins, it's a revolution. If LSTM resists, it's a tenacious veteran.
                           
                        2. 💥 T.K.O. ANALYSIS
                           - Explain THE REASONS for the victory. 
                           - If Decay > 0.9: "Elephant Memory" for long delays.
                           - If Decay Mixed: "Hybrid Brain" (Reflexes + Strategy).
                           - Talk about "Synaptic Plasticity" vs "LSTM Entropy".
                           
                        3. 🔮 FUTURISTIC VISION
                           - How does this architecture change the game for tomorrow's autonomous vehicles?
                           - Imagine cars that "understand" the road instead of suffering it.
                        
                        4. 📝 DEPLOYMENT RECOMMENDATION
                           - Go / No Go for production. Be decisive!
                           
                        Be concise, direct, and 'WOW'. No useless fluff. Language: English.
                        """
                        
                        # Call Ollama
                        result = subprocess.run(
                            ["ollama", "run", "llama3.2:3b", prompt],
                            capture_output=True, text=True, encoding='utf-8', errors='ignore',
                            timeout=60 # Give it time to think
                        )
                        
                        if result.returncode == 0 and result.stdout.strip():
                            ollama_response = result.stdout.strip()
                    except Exception as e:
                        print(f"Ollama Error: {e}")
                        ollama_response = None

                    # --- OPTION B: FALLBACK EXPERT SYSTEM (If Ollama is dead/missing) ---
                    if not ollama_response:
                        # Logic to simulate the explanation
                        decay = best_params.get('params', {}).get('retention_decay', [0.5, 0.5, 0.5]) if best_params else [0.5, 0.5, 0.5]
                        mean_d = np.mean(decay)
                        
                        if mean_d > 0.8:
                            reason = "These high values (>0.8) indicate the model favors **Long-Term Episodic Memory**. This is crucial for surviving the '100 step Delays' of the Torture Test, where the visual cue must be retained for a long time."
                        elif mean_d < 0.4:
                            reason = "These low values (<0.4) suggest a **Reflex (Ultra-Short Term)** strategy. The model ignores memory traps and focuses on immediate physics (Ice/Friction) to avoid skidding."
                        else:
                            reason = "This **Hybrid (Multi-Scale)** configuration is ideal. High rates (0.9) handle deferred navigation, while low rates (0.3-0.5) handle immediate friction and noise filtering."

                        ollama_response = f"""
### 🤖 Backup Analysis (Expert Rule-Based)

**Optimal Parameters Detected**: `{decay}`

**Why are they the best?**
{reason}

*Note: Ollama did not respond, this is a heuristic approximation.*
"""

                    # --- FINAL DISPLAY ---
                    st.session_state['ollama_analysis'] = f"""
### 🧠 Strategist Report (Ollama)

**Champion Configuration Analysis**
{ollama_response}

---
**Raw Data**: {bp_text}
"""
                    st.success("Analysis generated successfully!")
        
        if 'ollama_analysis' in st.session_state:
            st.markdown(st.session_state['ollama_analysis'])

    with col_rep2:
        st.markdown("### 2. Export Web Report (Interactive)")
        
        if st.button("🌐 Generate Web Report (Premium)"):
            with st.spinner("Creating high-definition interactive report..."):
                try:
                    import plotly.io as pio
                    
                    # --- 1. STATISTICS CALCULATION ---
                    # Global Stats (From Benchmarks)
                    if not bench_df.empty:
                        total_torture_tests = len(bench_df)
                        mathir_wins = len(bench_df[bench_df['winner'] == 'MATHIR'])
                        lstm_wins = len(bench_df[bench_df['winner'] == 'LSTM'])
                        win_rate = (mathir_wins / total_torture_tests * 100) if total_torture_tests > 0 else 0
                        
                        avg_mathir_long = bench_df['mathir_score'].mean()
                        avg_lstm_long = bench_df['lstm_score'].mean()
                        global_diff = ((avg_mathir_long - avg_lstm_long) / avg_lstm_long) * 100 if avg_lstm_long > 0 else 0
                    else:
                        total_torture_tests = 0
                        mathir_wins = 0
                        lstm_wins = 0
                        win_rate = 0
                        avg_mathir_long = 0
                        avg_lstm_long = 0
                        global_diff = 0

                    # --- 2. CHARTS GENERATION ---
                    # Chart 1: Long Term Evolution (Benchmarks)
                    if not bench_df.empty:
                        fig_bench = px.line(bench_df, x="step", y=["mathir_score", "lstm_score"],
                                          title="Benchmark War: Evolution since beginning (Step 5000+)",
                                          labels={'value': 'Score (Performance)', 'step': 'Training Steps', 'variable': 'Model'},
                                          color_discrete_map={"mathir_score": "#00ff00", "lstm_score": "#ff0000"})
                        fig_bench.update_traces(mode="lines+markers", marker=dict(size=6))
                        fig_bench.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                              xaxis=dict(showgrid=True, gridcolor='#333'), yaxis=dict(showgrid=True, gridcolor='#333'))
                        html_chart_bench = pio.to_html(fig_bench, full_html=False, include_plotlyjs='cdn')
                    else:
                        html_chart_bench = "<p style='color:red;'>No historical benchmark data available yet.</p>"

                    # Chart 2: Short Term Dynamics (Training Logs)
                    h_steps = data.get('history', {}).get('steps', [])
                    h_mathir = data.get('history', {}).get('mathir', [])
                    h_lstm = data.get('history', {}).get('lstm', [])
                    min_len = min(len(h_steps), len(h_mathir), len(h_lstm))
                    
                    if min_len > 0:
                        df_hist = pd.DataFrame({
                            'Step': h_steps[:min_len],
                            'MATHIR': h_mathir[:min_len],
                            'LSTM': h_lstm[:min_len]
                        })
                        fig_hist = px.line(df_hist, x='Step', y=['MATHIR', 'LSTM'], 
                                         title="Zoom: Learning Dynamics (Last 1000 Steps)",
                                         color_discrete_map={"MATHIR": "#00aa00", "LSTM": "#aa0000"})
                        fig_hist.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        html_chart_hist = pio.to_html(fig_hist, full_html=False, include_plotlyjs=False)
                    else:
                        html_chart_hist = "<p>No recent data.</p>"

                    # --- 3. FORMAT TEXT ---
                    analysis_text = st.session_state.get('ollama_analysis', "No generated analysis.")
                    import re
                    html_analysis = analysis_text
                    html_analysis = re.sub(r'### (.*)', r'<h3 style="color: #4da6ff; border-bottom: 1px solid #333; padding-bottom: 5px;">\1</h3>', html_analysis)
                    html_analysis = re.sub(r'## (.*)', r'<h2 style="color: #66b3ff;">\1</h2>', html_analysis)
                    html_analysis = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #e6f7ff;">\1</strong>', html_analysis)
                    html_analysis = html_analysis.replace('\n', '<br>')
                    
                    # --- 4. BUILD HTML ---
                    timestamp = time.strftime("%d/%m/%Y %H:%M")
                    full_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>MATHIR Report - {timestamp}</title>
                        <style>
                            body {{ font-family: 'Segoe UI', Roboto, sans-serif; background-color: #0d1117; color: #c9d1d9; max-width: 1400px; margin: 0 auto; padding: 20px; }}
                            .container {{ background-color: #161b22; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
                            h1 {{ color: #58a6ff; text-align: center; border-bottom: 2px solid #30363d; padding-bottom: 20px; font-size: 2.5em; }}
                            h2 {{ color: #58a6ff; margin-top: 40px; border-left: 5px solid #238636; padding-left: 15px; }}
                            
                            /* Metrics Grid */
                            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                            .metric-box {{ background-color: #21262d; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #30363d; }}
                            .metric-box.highlight {{ background-color: #238636; color: white; border: none; }}
                            .metric-title {{ font-size: 0.9em; color: #8b949e; margin-bottom: 5px; }}
                            .metric-box.highlight .metric-title {{ color: #e6f7ff; }}
                            .metric-val {{ font-size: 1.8em; font-weight: bold; color: #ffffff; }}
                            
                            .chart-container {{ background-color: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin: 20px 0; }}
                            .analysis-box {{ background-color: #1f2937; padding: 25px; border-radius: 8px; border-left: 5px solid #1f6feb; line-height: 1.6; font-size: 1.1em; }}
                            
                            .details-section {{ background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 8px; margin-top: 20px; }}
                            .details-title {{ color: #d29922; font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
                            .details-text {{ color: #8b949e; font-size: 0.95em; line-height: 1.5; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>📑 Strategic Expertise Report : MATHIR vs LSTM</h1>
                            <p style="text-align: center; color: #8b949e;">Report Date : {timestamp} | Current Step : {step}</p>
                            
                            <!-- GLOBAL STATS -->
                            <h2>🏆 Global Statistics (Since Start)</h2>
                            <div class="metrics-grid">
                                <div class="metric-box highlight">
                                    <div class="metric-title">MATHIR Wins</div>
                                    <div class="metric-val">{mathir_wins} <span style="font-size:0.5em;">({win_rate:.1f}%)</span></div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-title">LSTM Wins</div>
                                    <div class="metric-val">{lstm_wins}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-title">Total Torture Tests</div>
                                    <div class="metric-val">{total_torture_tests}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-title">Global Average Score</div>
                                    <div class="metric-val">{avg_mathir_long:.1f}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-title">Global Advantage</div>
                                    <div class="metric-val" style="color: {'#238636' if global_diff > 0 else '#da3633'}">{'+' if global_diff > 0 else ''}{global_diff:.1f}%</div>
                                </div>
                            </div>

                            <!-- ANALYSIS -->
                            <h2>🧠 AI Analysis & Verdict</h2>
                            <div class="analysis-box">
                                {html_analysis}
                            </div>

                            <!-- VISUALS -->
                            <h2>📊 Visual Proofs & Evolution</h2>
                            
                            <div class="chart-container">
                                {html_chart_bench}
                                <p style="text-align:center; color:#8b949e; font-size:0.9em;">*This chart shows the complete history of benchmarks (Torture Tests) performed since step 5000.*</p>
                            </div>
                            
                            <div class="chart-container">
                                {html_chart_hist}
                            </div>

                            <!-- METHODOLOGY DETAILS -->
                            <h2>📖 Methodological Details & Protocols</h2>
                            
                            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                                <div class="details-section" style="flex: 1; min-width: 300px;">
                                    <div class="details-title">⚔️ The Fighters: MATHIR vs LSTM</div>
                                    <div class="details-text">
                                        <ul>
                                            <li><strong>MATHIR (Memory-Augmented Tensor Hybrid with Intelligent Routing)</strong> : A new generation architecture using Manifold Constrained Hyper-Connections (mHC). It projects prediction error via Sinkhorn to stabilize gradients over the long term. It has explicit separated memory (Working, Episodic, Semantic).</li>
                                            <li><strong>LSTM (Long Short-Term Memory)</strong> : The industrial standard for recurrent networks. Efficient for short sequences, but suffering from "entropic degradation" over the long term (catastrophic forgetting). In this benchmark, LSTM often benefits from a "doped" Learning Rate (LR) to make it more competitive.</li>
                                        </ul>
                                    </div>
                                </div>

                                <div class="details-section" style="flex: 1; min-width: 300px;">
                                    <div class="details-title">🔥 "Torture Test" Protocol</div>
                                    <div class="details-text">
                                        <p>To validate robustness, we do not perform calm learning. We subject agents to intense stress every 5000 steps:</p>
                                        <ul>
                                            <li><strong>Noise Injections</strong> : Random sensor perturbation simulating failures.</li>
                                            <li><strong>Parameter Shift</strong> : Brutal change of friction coefficients or simulated physical gravity.</li>
                                            <li><strong>Forced Relearning</strong> : We observe recovery speed (Plasticity) after a partial amnesiac "shock".</li>
                                        </ul>
                                        <p>Only a model with stable structural memory (MATHIR) can survive this regime in the long run.</p>
                                    </div>
                                </div>
                            </div>
                            
                            <p style="text-align: center; margin-top: 50px; color: #484f58;">
                                Official Report - Re-Information Research Laboratory
                            </p>
                        </div>
                    </body>
                    </html>
                    """
                    
                    st.session_state['html_report'] = full_html
                    st.session_state['html_filename'] = f"MATHIR_Full_Report_{step}.html"
                    st.success("✅ Full Web Report (Stats + Methodology + Charts) Generated!")
                    
                except Exception as e:
                    st.error(f"HTML Error: {e}")

        if 'html_report' in st.session_state:
            st.download_button(
                label="📄 Download Web Report (.html)",
                data=st.session_state['html_report'],
                file_name=st.session_state['html_filename'],
                mime="text/html",
                key='dl_html_btn'
            )


# AUTO-REFRESH
time.sleep(1) # Faster refresh (1s)
try:
    st.rerun()
except AttributeError:
    st.experimental_rerun()
