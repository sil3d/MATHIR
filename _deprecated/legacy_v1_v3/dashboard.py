import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import time
import os

st.set_page_config(
    page_title="MATHIR Evolution Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #30364d;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #00d2ff;
    }
    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 MATHIR Evolution Suite (Powered by DeepSeek-mHC)")

LOG_FILE = "training_log.json"

def load_data():
    if not os.path.exists(LOG_FILE):
        return None
    try:
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

# Placeholder pour refresh automatique
placeholder = st.empty()

while True:
    data = load_data()
    
    with placeholder.container():
        if data is None:
            st.warning("⏳ Waiting for training to start... (Run 'run_experiment.bat')")
            time.sleep(2)
            continue
            
        # --- HEADER ---
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Step</div>
                <div class="metric-value">{data['step']:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            m_score = data['mathir_avg_reward']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">MATHIR Accuracy</div>
                <div class="metric-value" style="color: #00ffaa">{m_score:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            l_score = data['lstm_avg_reward']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">LSTM Accuracy</div>
                <div class="metric-value" style="color: #ffaa00">{l_score:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            delta = (m_score - l_score) / (l_score + 1e-6) * 100 # Avoid div by zero
            color = "#00ffaa" if delta > 0 else "#ff4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Relative Gain</div>
                <div class="metric-value" style="color: {color}">{delta:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            vram = data.get('vram_gb', 0.0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">VRAM Usage</div>
                <div class="metric-value" style="color: #d142f5">{vram:.2f} GB</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # --- CHARTS ---
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            # Chart 1: Training Accuracy
            st.subheader("🏎️ Real-Time Learning Curve")
            
            history = data.get('history', {})
            steps = history.get('steps', [])
            m_hist = history.get('mathir', [])
            l_hist = history.get('lstm', [])
            
            if len(steps) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=steps, y=m_hist,
                    mode='lines',
                    name='MATHIR (Train)',
                    line=dict(color='#00ffaa', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=steps, y=l_hist,
                    mode='lines',
                    name='LSTM (Train)',
                    line=dict(color='#ffaa00', width=1, dash='dot')
                ))
                
                fig.update_layout(
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#1e2130',
                    font=dict(color='white'),
                    xaxis_title="Steps",
                    yaxis_title="Accuracy (1 - MSE)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                unique_key = f"chart_{data['step']}_{time.time()}"
                try:
                    st.plotly_chart(fig, use_container_width=True, key=unique_key)
                except:
                    st.plotly_chart(fig, width="stretch", key=unique_key)

            # Chart 2: Memory Retention Benchmark
            st.subheader("🧠 Long-Term Memory Capacity (Periodic Test)")
            
            CAP_FILE = "capacity_log.json"
            if os.path.exists(CAP_FILE):
                try:
                    with open(CAP_FILE, 'r') as f:
                        cap_data = json.load(f)
                        
                    if len(cap_data) > 0:
                        cap_steps = [d['step'] for d in cap_data]
                        cap_m = [d['mathir_retention'] for d in cap_data]
                        cap_l = [d['lstm_retention'] for d in cap_data]
                        
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(
                            x=cap_steps, y=cap_m,
                            name='MATHIR Retention',
                            marker_color='#00ffaa'
                        ))
                        fig2.add_trace(go.Bar(
                            x=cap_steps, y=cap_l,
                            name='LSTM Retention',
                            marker_color='#ffaa00'
                        ))
                        
                        fig2.update_layout(
                            paper_bgcolor='#0e1117',
                            plot_bgcolor='#1e2130',
                            font=dict(color='white'),
                            xaxis_title="Checkpoint Step",
                            yaxis_title="Retention Score (0-1)",
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=300,
                            barmode='group'
                        )
                        st.plotly_chart(fig2, use_container_width=True, key=f"cap_{time.time()}")
                    else:
                        st.info("Waiting for first benchmark (every 10k steps)...")
                except:
                    st.error("Error reading capacity log.")
            else:
                st.info("Waiting for first benchmark (every 10k steps)...")
        
        with col_side:
            st.subheader("🧬 MATHIR DNA (AI Optimized)")
            
            if 'current_hyperparams' in data:
                decay = data.get('current_hyperparams', {}).get('retention_decay', [0,0,0])
                lstm_lr = data.get('current_hyperparams', {}).get('lstm_lr', 0.0)
                
                st.write("**Retention Rates (Llama 3.2 Tuned):**")
                for i, val in enumerate(decay):
                     st.write(f"Level {i+1}: **{val:.3f}**")
                     st.progress(float(val))
                     
                st.write("---")
                st.write("**LSTM Learning Rate:**")
                st.code(f"{lstm_lr:.6f}")
            else:
                st.info("Waiting for first optimization...")
            
    # Refresh rate
    time.sleep(1)
