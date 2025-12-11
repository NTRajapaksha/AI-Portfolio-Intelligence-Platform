"""
Streamlit UI for AI Portfolio Intelligence Platform
"""
import streamlit as st
import os
import time
from PIL import Image
import glob
import pandas as pd
import ast

# Import local modules
from config import config
from agents import agent
from tools import reset_state, DATA_STATE

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Portfolio Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS & Styling ---
st.markdown("""
    <style>
    /* Main Title Styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Subtitle Styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Metric Cards */
    .metric-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Logs/Steps Styling */
    .log-step {
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        padding: 5px 0;
        border-bottom: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Helper Functions ---

def clean_ai_response(response):
    """
    Cleans the raw output from Google Gemini which sometimes returns 
    a list of dictionaries instead of a plain string.
    """
    if isinstance(response, str):
        # Check if it looks like a list string "[{'type': ...}]"
        if response.strip().startswith("[") and "type" in response:
            try:
                # Safely evaluate the string as a python object
                parsed = ast.literal_eval(response)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed[0].get("text", str(parsed))
            except:
                return response
        return response
    
    elif isinstance(response, list) and len(response) > 0:
        if isinstance(response[0], dict):
            return response[0].get("text", str(response))
        return str(response[0])
        
    return str(response)

# --- 4. Sidebar Interface ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910312.png", width=50)
    st.header("Configuration")
    
    # API Status
    with st.expander("🔌 API Status", expanded=True):
        if config.GOOGLE_API_KEY:
            st.success("Gemini AI: Connected", icon="✅")
        else:
            st.error("Gemini AI: Disconnected", icon="❌")
            
        if config.ENABLE_SENTIMENT:
            if config.NEWS_API_KEY:
                st.success("News API: Connected", icon="✅")
            else:
                st.warning("News API: Missing Key", icon="⚠️")
    
    st.divider()
    
    # User Guide
    st.subheader("📘 User Guide")
    with st.expander("How to run an analysis", expanded=False):
        st.markdown("""
        **1. Input Tickers:** Enter symbols like `AAPL, MSFT`.
        
        **2. Select Mode:**
        * **Hybrid (Best):** Reliable math + AI summary.
        * **Autonomous (Experimental):** AI decides the workflow.
        * **Manual:** Raw data only.
        
        **3. Analyze:**
        Click the launch button and wait 30-60s.
        """)
    
    st.divider()
    
    # Controls
    st.subheader("⚙️ Analysis Settings")
    mode = st.radio(
        "Execution Mode:",
        ["Hybrid (Recommended)", "Manual", "Autonomous"],
        index=0
    )
    
    forecast_days = st.slider("Forecast Horizon (Days)", 30, 180, 60, 30)
    include_sentiment = st.checkbox("Analyze News Sentiment", value=config.ENABLE_SENTIMENT)

# --- 5. Main Content Area ---

st.markdown('<h1 class="main-header">AI Portfolio Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Multi-Agent Orchestration • Meta Prophet • Gemini 1.5</div>', unsafe_allow_html=True)
st.divider()

# Tabs
tab_dash, tab_docs, tab_debug = st.tabs(["🚀 Dashboard", "📚 Documentation", "🔧 System State"])

with tab_dash:
    # Input Section
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        tickers_input = st.text_input(
            "Target Assets", 
            value="AAPL, MSFT, GOOGL", 
            placeholder="e.g. NVDA, TSLA, AMD",
            help="Enter comma-separated stock tickers"
        )
    with col_btn:
        st.write("##") # Spacer
        analyze_btn = st.button("Run Analysis", type="primary")

    # Analysis Logic
    if analyze_btn and tickers_input:
        reset_state()
        
        # Clear old assets
        if os.path.exists(config.ASSETS_DIR):
            for f in glob.glob(f"{config.ASSETS_DIR}/*.png"):
                try: os.remove(f)
                except: pass
        else:
            os.makedirs(config.ASSETS_DIR)
            
        # Progress Container
        status_container = st.status("Initializing AI Agents...", expanded=True)
        
        try:
            results = {}
            
            # --- EXECUTION ---
            if mode == "Hybrid (Recommended)":
                status_container.write("Step 1: Fetching Market Data...")
                status_container.write("Step 2: Calculating Risk Metrics...")
                status_container.write("Step 3: Running Prophet Forecasts...")
                results = agent.run_hybrid(tickers_input, forecast_days, include_sentiment)
                status_container.write("Step 4: Synthesizing AI Report...")
                
            elif mode == "Manual":
                status_container.write("Running Quantitative Tools...")
                results = agent.run_manual(tickers_input, forecast_days, include_sentiment)
                
            elif mode == "Autonomous":
                status_container.write("⚠️ Waking up Autonomous Agent...")
                status_container.write("Note: This mode is slower to prevent rate limits.")
                query = f"Analyze {tickers_input}. Fetch data, calculate risk, forecast for {forecast_days} days, and compare."
                results = agent.run_autonomous(query)

            status_container.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # --- RESULTS DISPLAY ---
            
            # 1. Executive Summary
            st.subheader("📝 Executive Summary")
            
            # Clean the text response
            final_text = clean_ai_response(results.get("final_response", "No response generated."))
            
            st.markdown(final_text)
            
            # Download Button
            st.download_button(
                label="📥 Download Report",
                data=final_text,
                file_name=f"portfolio_report_{int(time.time())}.md",
                mime="text/markdown"
            )
            
            st.divider()
            
            # 2. Forecast Visualizations
            st.subheader("📈 Technical Forecasts")
            plot_files = glob.glob(f"{config.ASSETS_DIR}/*.png")
            
            if plot_files:
                # Dynamic grid layout
                cols = st.columns(2)
                for idx, plot_path in enumerate(plot_files):
                    with cols[idx % 2]:
                        img = Image.open(plot_path)
                        st.image(img, caption=os.path.basename(plot_path).replace("_forecast.png", "").upper(), use_container_width=True)
            else:
                st.info("No charts generated. This usually means the data fetch failed or mode is set to text-only.")

            # 3. Detailed Logs (Accordion)
            with st.expander("🔍 View Agent Thought Process"):
                for step in results.get("steps", []):
                    st.markdown(f"**🔹 {step.get('step', 'Action')}**")
                    # Clean the result text for logs too
                    clean_result = clean_ai_response(step.get('result', ''))
                    st.code(clean_result[:500] + ("..." if len(clean_result) > 500 else ""), language="text")

        except Exception as e:
            status_container.update(label="Error Occurred", state="error")
            st.error(f"Critical System Error: {str(e)}")
            if "429" in str(e):
                st.warning("ℹ️ You hit the Google Gemini Free Tier rate limit. Please wait 1 minute and try again.")

with tab_docs:
    st.markdown("## System Architecture")
    


    st.markdown("""
    ### How it Works
    1. **Data Layer**: Connects to Yahoo Finance via `yfinance` to fetch OHLCV data.
    2. **Math Layer**: Uses `numpy` for Sharpe/Beta calculations and `prophet` for time-series forecasting.
    3. **Agent Layer**: `LangGraph` orchestrates the flow. In **Autonomous Mode**, the LLM decides the order of operations.
    4. **Presentation**: Streamlit renders the UI and visualization assets.
    """)

with tab_debug:
    st.warning("⚠️ For Developer Use Only")
    if st.button("Clear System Memory"):
        reset_state()
        st.success("Memory Wiped")
        st.rerun()
        
    st.json(DATA_STATE)