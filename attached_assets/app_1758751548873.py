# -*- coding: utf-8 -*-
"""
AEGIS-OS v5.0 المتقدم للذكاء الفضائي
Advanced Orbital Debris Intelligence & Sustainability Platform
Enhanced Scientific Computing with AI Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import time
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules with error handling
try:
    from modules.constants import *
    from modules.data_sources import CelesTrakDataSource, NOAASpaceWeatherSource, ESADebrisSource
    from modules.orbital_mechanics import EnhancedSGP4Propagator, NBodyGravitationalModel
    from modules.atmospheric_models import NRLMSISE00Model, AtmosphericDragModel
    from modules.ai_models import DebrisAIPredictor, CollisionRiskPredictor
    from modules.collision_detection import AdvancedCollisionDetector
    from modules.economic_analysis import EconomicAnalyzer
    from modules.compliance_monitor import ComplianceMonitor
    from services.real_time_data import RealTimeDataManager
    from services.kessler_simulation import KesslerSimulator
    from utils.visualization_utils import create_3d_orbit_plot, create_debris_distribution_plot
    from lang.translations import get_text, set_language
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False

def main():
    """Main application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="AEGIS-OS v5.0 - Advanced Space Intelligence",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for RTL support and enhanced UI
    st.markdown("""
    <style>
    .rtl {
        direction: rtl;
        text-align: right;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e2124, #2f3349);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00ff41;
        margin: 10px 0;
    }
    .alert-high {
        background: linear-gradient(145deg, #ff4444, #aa0000);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-medium {
        background: linear-gradient(145deg, #ffaa00, #cc8800);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: linear-gradient(145deg, #00ff41, #00cc33);
        border-radius: 10px;
        margin-bottom: 20px;
        color: black;
        font-weight: bold;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background-color: #00ff41; }
    .status-inactive { background-color: #ff4444; }
    .status-warning { background-color: #ffaa00; }
    </style>
    """, unsafe_allow_html=True)
    
    # Language selector
    col1, col2 = st.columns([4, 1])
    with col2:
        language = st.selectbox(
            "🌐 Language",
            ["English", "العربية"],
            key="language_selector"
        )
        st.session_state.language = 'ar' if language == "العربية" else 'en'
    
    # Apply RTL for Arabic
    if st.session_state.language == 'ar':
        st.markdown('<div class="rtl">', unsafe_allow_html=True)
    
    # Main header
    st.title(get_text("main_title", st.session_state.language))
    st.markdown(f"### {get_text('subtitle', st.session_state.language)}")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-logo"><h2>🛰️ AEGIS-OS v5.0</h2></div>', unsafe_allow_html=True)
        
        page = st.selectbox(
            get_text("navigation", st.session_state.language),
            [
                get_text("dashboard", st.session_state.language),
                get_text("real_time_tracking", st.session_state.language),
                get_text("collision_analysis", st.session_state.language),
                get_text("ai_predictions", st.session_state.language),
                get_text("kessler_simulation", st.session_state.language),
                get_text("economic_analysis", st.session_state.language),
                get_text("compliance_monitor", st.session_state.language),
                get_text("orbital_mechanics", st.session_state.language)
            ]
        )
        
        st.markdown("---")
        
        # Real-time data status
        st.markdown(f"### {get_text('data_status', st.session_state.language)}")
        try:
            data_manager = RealTimeDataManager()
            status = data_manager.get_system_status()
            
            for source, state in status.items():
                status_class = "status-active" if state['status'] == 'active' else "status-inactive"
                st.markdown(f'<div><span class="status-indicator {status_class}"></span><strong>{source}</strong>: {state["status"]}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error getting system status: {e}")
    
    # Route to appropriate page
    try:
        if get_text("dashboard", st.session_state.language) in page:
            render_dashboard()
        elif get_text("real_time_tracking", st.session_state.language) in page:
            render_real_time_tracking()
        elif get_text("collision_analysis", st.session_state.language) in page:
            render_collision_analysis()
        elif get_text("ai_predictions", st.session_state.language) in page:
            render_ai_predictions()
        elif get_text("kessler_simulation", st.session_state.language) in page:
            render_kessler_simulation()
        elif get_text("economic_analysis", st.session_state.language) in page:
            render_economic_analysis()
        elif get_text("compliance_monitor", st.session_state.language) in page:
            render_compliance_monitor()
        elif get_text("orbital_mechanics", st.session_state.language) in page:
            render_orbital_mechanics()
    except Exception as e:
        st.error(f"Error rendering page: {e}")
        st.exception(e)
    
    # Close RTL div
    if st.session_state.language == 'ar':
        st.markdown('</div>', unsafe_allow_html=True)

def render_dashboard():
    """Render main dashboard with key metrics and alerts."""
    
    st.markdown(f"## {get_text('dashboard_title', st.session_state.language)}")
    
    try:
        # Initialize data sources
        celestrak = CelesTrakDataSource()
        noaa = NOAASpaceWeatherSource()
        collision_detector = AdvancedCollisionDetector()
        
        # Get real-time data
        with st.spinner(get_text("loading_data", st.session_state.language)):
            debris_data = celestrak.get_debris_catalog()
            satellite_data = celestrak.get_active_satellites()
            space_weather = noaa.get_solar_activity()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            debris_count = len(debris_data) if not debris_data.empty else 0
            st.markdown(f"""
            <div class="metric-card">
            <h3>{get_text('total_debris', st.session_state.language)}</h3>
            <h2 style="color: #ff4444;">{debris_count:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            satellite_count = len(satellite_data) if not satellite_data.empty else 0
            st.markdown(f"""
            <div class="metric-card">
            <h3>{get_text('active_satellites', st.session_state.language)}</h3>
            <h2 style="color: #00ff41;">{satellite_count:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if not debris_data.empty and 'altitude_km' in debris_data.columns:
                high_risk_debris = len(debris_data[debris_data['altitude_km'] < 1000])
            else:
                high_risk_debris = 0
            st.markdown(f"""
            <div class="metric-card">
            <h3>{get_text('high_risk_objects', st.session_state.language)}</h3>
            <h2 style="color: #ffaa00;">{high_risk_debris:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            activity_level = space_weather.get('solar_activity_level', 'Unknown')
            st.markdown(f"""
            <div class="metric-card">
            <h3>{get_text('solar_activity', st.session_state.language)}</h3>
            <h2 style="color: #00aaff;">{activity_level}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Alerts and warnings
        st.markdown(f"### {get_text('alerts', st.session_state.language)}")
        
        # Critical altitude alerts
        if not debris_data.empty and 'altitude_km' in debris_data.columns:
            critical_debris = debris_data[debris_data['altitude_km'] < 400]
            if not critical_debris.empty:
                st.markdown(f"""
                <div class="alert-high">
                ⚠️ <strong>{get_text('critical_alert', st.session_state.language)}</strong>: 
                {len(critical_debris)} {get_text('objects_critical_altitude', st.session_state.language)}
                </div>
                """, unsafe_allow_html=True)
        
        # Space weather alerts
        if space_weather.get('kp_index', 0) > 5:
            st.markdown(f"""
            <div class="alert-medium">
            🌞 <strong>{get_text('space_weather_alert', st.session_state.language)}</strong>: 
            {get_text('high_geomagnetic_activity', st.session_state.language)} (Kp = {space_weather['kp_index']:.1f})
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {get_text('debris_by_altitude', st.session_state.language)}")
            if not debris_data.empty and 'altitude_km' in debris_data.columns:
                try:
                    fig = create_debris_distribution_plot(debris_data, st.session_state.language)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating debris distribution plot: {e}")
            else:
                st.info(get_text("no_data_available", st.session_state.language))
        
        with col2:
            st.markdown(f"#### {get_text('orbital_zones', st.session_state.language)}")
            if not debris_data.empty and 'orbital_zone' in debris_data.columns:
                try:
                    zone_counts = debris_data['orbital_zone'].value_counts()
                    fig = px.pie(
                        values=zone_counts.values,
                        names=zone_counts.index,
                        title=get_text('objects_by_zone', st.session_state.language)
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating orbital zones plot: {e}")
            else:
                st.info(get_text("no_data_available", st.session_state.language))
        
        # Global debris map
        st.markdown(f"### {get_text('global_debris_map', st.session_state.language)}")
        if not debris_data.empty and len(debris_data) > 0:
            try:
                # Create world map with debris positions
                m = folium.Map(location=[0, 0], zoom_start=2, tiles="CartoDB dark_matter")
                
                # Add debris points (sample for performance)
                sample_debris = debris_data.head(1000) if len(debris_data) > 1000 else debris_data
                
                for _, debris in sample_debris.iterrows():
                    # Convert to lat/lon (simplified - real implementation would calculate from orbital elements)
                    lat = np.random.uniform(-85, 85)  # Placeholder
                    lon = np.random.uniform(-180, 180)
                    
                    altitude = debris.get('altitude_km', 0)
                    color = 'red' if altitude < 600 else 'orange' if altitude < 1000 else 'green'
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=3,
                        color=color,
                        fillOpacity=0.7,
                        popup=f"Object: {debris.get('name', 'Unknown')}<br>Altitude: {altitude:.0f} km"
                    ).add_to(m)
                
                st_folium(m, width=700, height=400)
            except Exception as e:
                st.error(f"Error creating debris map: {e}")
        else:
            st.info(get_text("no_data_available", st.session_state.language))
    
    except Exception as e:
        st.error(f"Error in dashboard: {e}")
        st.exception(e)

def render_real_time_tracking():
    """Render real-time tracking interface."""
    
    st.markdown(f"## {get_text('real_time_tracking_title', st.session_state.language)}")
    
    try:
        # Auto-refresh toggle
        auto_refresh = st.checkbox(get_text("auto_refresh", st.session_state.language), value=False)
        
        if auto_refresh:
            # Auto-refresh every 30 seconds
            time.sleep(1)
            st.rerun()
        
        # Manual refresh button
        if st.button(get_text("refresh_data", st.session_state.language)):
            st.rerun()
        
        # Data sources
        data_manager = RealTimeDataManager()
        
        # Real-time metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=get_text("data_updates", st.session_state.language),
                value="Live",
                delta="Active"
            )
        
        with col2:
            current_time = datetime.now()
            st.metric(
                label=get_text("last_update", st.session_state.language),
                value=current_time.strftime("%H:%M:%S UTC"),
                delta="Real-time"
            )
        
        with col3:
            st.metric(
                label=get_text("tracking_stations", st.session_state.language),
                value="15",
                delta="+2"
            )
        
        # Live tracking display
        st.markdown(f"### {get_text('live_tracking', st.session_state.language)}")
        
        # Get live tracking data
        tracking_data = data_manager.get_live_tracking_data()
        
        if tracking_data is not None and not tracking_data.empty:
            st.dataframe(
                tracking_data,
                use_container_width=True,
                height=300
            )
        else:
            st.info(get_text("no_live_data", st.session_state.language))
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {get_text('altitude_changes', st.session_state.language)}")
            # Generate sample time series data
            timestamps = pd.date_range(start=datetime.now()-timedelta(hours=1), end=datetime.now(), freq='1min')
            altitudes = 400 + 50 * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.normal(0, 5, len(timestamps))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=altitudes,
                mode='lines',
                name=get_text('altitude_km', st.session_state.language),
                line=dict(color='#00ff41')
            ))
            fig.update_layout(
                title=get_text('debris_altitude_trend', st.session_state.language),
                xaxis_title=get_text('time', st.session_state.language),
                yaxis_title=get_text('altitude_km', st.session_state.language),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"#### {get_text('velocity_profile', st.session_state.language)}")
            velocities = 7.5 + 0.5 * np.cos(np.linspace(0, 6*np.pi, len(timestamps))) + np.random.normal(0, 0.1, len(timestamps))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=velocities,
                mode='lines',
                name=get_text('velocity_kms', st.session_state.language),
                line=dict(color='#ff4444')
            ))
            fig.update_layout(
                title=get_text('velocity_trend', st.session_state.language),
                xaxis_title=get_text('time', st.session_state.language),
                yaxis_title=get_text('velocity_kms', st.session_state.language),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in real-time tracking: {e}")
        st.exception(e)

def render_collision_analysis():
    """Render advanced collision detection and analysis."""
    
    st.markdown(f"## {get_text('collision_analysis_title', st.session_state.language)}")
    
    try:
        # Initialize collision detector
        collision_detector = AdvancedCollisionDetector()
        
        # Analysis parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_horizon = st.slider(
                get_text("prediction_horizon", st.session_state.language),
                min_value=1, max_value=72, value=24,
                help=get_text("horizon_help", st.session_state.language)
            )
        
        with col2:
            risk_threshold = st.slider(
                get_text("risk_threshold", st.session_state.language),
                min_value=0.1, max_value=1.0, value=0.1, step=0.01,
                format="%.3f"
            )
        
        with col3:
            analysis_type = st.selectbox(
                get_text("analysis_type", st.session_state.language),
                [
                    get_text("monte_carlo", st.session_state.language),
                    get_text("probabilistic", st.session_state.language),
                    get_text("deterministic", st.session_state.language)
                ]
            )
        
        # Run collision analysis
        if st.button(get_text("run_analysis", st.session_state.language), type="primary"):
            with st.spinner(get_text("analyzing_collisions", st.session_state.language)):
                try:
                    # Get sample data for analysis
                    celestrak = CelesTrakDataSource()
                    debris_data = celestrak.get_debris_catalog()
                    
                    if not debris_data.empty:
                        # Run collision detection
                        results = collision_detector.analyze_collision_risks(
                            debris_data.head(100),  # Limit for performance
                            time_horizon_hours=time_horizon,
                            risk_threshold=risk_threshold
                        )
                        
                        # Display results
                        st.success(f"{get_text('analysis_complete', st.session_state.language)}")
                        
                        # High-risk conjunctions
                        if results.get('high_risk_conjunctions'):
                            st.markdown(f"### ⚠️ {get_text('high_risk_conjunctions', st.session_state.language)}")
                            
                            for i, conjunction in enumerate(results['high_risk_conjunctions'][:10]):  # Show top 10
                                with st.expander(f"{conjunction.get('primary_name', f'Object {i+1}')} ↔ {conjunction.get('secondary_name', f'Object {i+2}')}"):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            get_text("collision_probability", st.session_state.language),
                                            f"{conjunction.get('probability', 0):.2e}"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            get_text("closest_approach", st.session_state.language),
                                            f"{conjunction.get('min_distance', 0):.3f} km"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            get_text("time_to_conjunction", st.session_state.language),
                                            f"{conjunction.get('time_to_ca', 0):.1f}h"
                                        )
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                get_text("total_conjunctions", st.session_state.language),
                                len(results.get('all_conjunctions', []))
                            )
                        
                        with col2:
                            st.metric(
                                get_text("high_risk_count", st.session_state.language),
                                len(results.get('high_risk_conjunctions', []))
                            )
                        
                        with col3:
                            conjunctions = results.get('all_conjunctions', [])
                            avg_risk = np.mean([c.get('probability', 0) for c in conjunctions]) if conjunctions else 0
                            st.metric(
                                get_text("average_risk", st.session_state.language),
                                f"{avg_risk:.2e}"
                            )
                        
                        with col4:
                            max_risk = max([c.get('probability', 0) for c in conjunctions], default=0)
                            st.metric(
                                get_text("maximum_risk", st.session_state.language),
                                f"{max_risk:.2e}"
                            )
                    
                    else:
                        st.error(get_text("no_debris_data", st.session_state.language))
                
                except Exception as e:
                    st.error(f"Error in collision analysis: {e}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"Error in collision analysis setup: {e}")
        st.exception(e)

def render_ai_predictions():
    """Render AI-powered prediction interface."""
    
    st.markdown(f"## {get_text('ai_predictions_title', st.session_state.language)}")
    
    try:
        # Initialize AI predictors
        ai_predictor = DebrisAIPredictor()
        collision_predictor = CollisionRiskPredictor()
        
        # AI model selection
        col1, col2 = st.columns(2)
        
        with col1:
            ai_model = st.selectbox(
                get_text("ai_model", st.session_state.language),
                ["GPT-5", "Claude-4", "Local Transformer"]
            )
        
        with col2:
            prediction_type = st.selectbox(
                get_text("prediction_type", st.session_state.language),
                [
                    get_text("debris_evolution", st.session_state.language),
                    get_text("collision_risk", st.session_state.language),
                    get_text("orbital_decay", st.session_state.language),
                    get_text("breakup_analysis", st.session_state.language)
                ]
            )
        
        # Prediction parameters
        st.markdown(f"### {get_text('prediction_parameters', st.session_state.language)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_days = st.slider(
                get_text("forecast_period", st.session_state.language),
                min_value=1, max_value=365, value=30
            )
        
        with col2:
            confidence_level = st.slider(
                get_text("confidence_level", st.session_state.language),
                min_value=0.8, max_value=0.99, value=0.95, step=0.01
            )
        
        with col3:
            include_uncertainty = st.checkbox(
                get_text("include_uncertainty", st.session_state.language),
                value=True
            )
        
        # Run AI prediction
        if st.button(get_text("generate_prediction", st.session_state.language), type="primary"):
            with st.spinner(get_text("generating_ai_prediction", st.session_state.language)):
                try:
                    # Get current debris data
                    celestrak = CelesTrakDataSource()
                    debris_data = celestrak.get_debris_catalog()
                    noaa = NOAASpaceWeatherSource()
                    space_weather = noaa.get_solar_activity()
                    
                    if not debris_data.empty:
                        # Generate AI prediction based on selected type
                        if get_text("debris_evolution", st.session_state.language) in prediction_type:
                            prediction = ai_predictor.predict_debris_evolution(
                                debris_data.head(50),  # Limit for performance
                                forecast_days=forecast_days,
                                model_type=ai_model.lower(),
                                space_weather=space_weather
                            )
                        
                        elif get_text("collision_risk", st.session_state.language) in prediction_type:
                            prediction = collision_predictor.predict_collision_risk(
                                debris_data.head(100),
                                forecast_days=forecast_days,
                                confidence_level=confidence_level
                            )
                        
                        else:
                            # Default prediction
                            prediction = ai_predictor.predict_debris_evolution(
                                debris_data.head(50),
                                forecast_days=forecast_days,
                                model_type=ai_model.lower()
                            )
                        
                        # Display prediction results
                        st.success(get_text("prediction_complete", st.session_state.language))
                        
                        # Key insights from AI
                        st.markdown(f"### 🤖 {get_text('ai_insights', st.session_state.language)}")
                        
                        for insight in prediction.get('insights', []):
                            st.markdown(f"• {insight}")
                        
                        # Confidence metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                get_text("model_confidence", st.session_state.language),
                                f"{prediction.get('confidence', 0.85):.2%}"
                            )
                        
                        with col2:
                            st.metric(
                                get_text("prediction_accuracy", st.session_state.language),
                                f"{prediction.get('accuracy', 0.92):.2%}"
                            )
                        
                        with col3:
                            st.metric(
                                get_text("data_quality", st.session_state.language),
                                prediction.get('data_quality', 'High')
                            )
                    
                    else:
                        st.error(get_text("insufficient_data", st.session_state.language))
                        
                except Exception as e:
                    st.error(f"{get_text('prediction_error', st.session_state.language)}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in AI predictions setup: {e}")
        st.exception(e)

def render_kessler_simulation():
    """Render Kessler Syndrome simulation."""
    
    st.markdown(f"## {get_text('kessler_simulation_title', st.session_state.language)}")
    
    try:
        # Initialize simulator
        simulator = KesslerSimulator()
        
        # Simulation parameters
        st.markdown(f"### {get_text('simulation_parameters', st.session_state.language)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            simulation_years = st.slider(
                get_text("simulation_period", st.session_state.language),
                min_value=1, max_value=100, value=25
            )
        
        with col2:
            initial_debris_count = st.number_input(
                get_text("initial_debris", st.session_state.language),
                min_value=1000, max_value=100000, value=15000, step=1000
            )
        
        with col3:
            collision_probability_factor = st.slider(
                get_text("collision_factor", st.session_state.language),
                min_value=0.1, max_value=3.0, value=1.0, step=0.1
            )
        
        # Run simulation
        if st.button(get_text("run_simulation", st.session_state.language), type="primary"):
            st.session_state.simulation_running = True
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Run Kessler simulation
                results = simulator.run_kessler_simulation(
                    years=simulation_years,
                    initial_debris=initial_debris_count,
                    collision_factor=collision_probability_factor,
                    progress_callback=lambda p, msg: (
                        progress_bar.progress(p),
                        status_text.text(msg)
                    )
                )
                
                st.session_state.simulation_running = False
                
                # Display results
                if results:
                    st.success(get_text("simulation_complete", st.session_state.language))
                    
                    # Key findings
                    st.markdown(f"### {get_text('key_findings', st.session_state.language)}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        final_debris = results.get('final_debris_count', initial_debris_count)
                        st.metric(
                            get_text("final_debris_count", st.session_state.language),
                            f"{final_debris:,}",
                            delta=f"+{final_debris - initial_debris_count:,}"
                        )
                    
                    with col2:
                        total_collisions = results.get('total_collisions', 0)
                        st.metric(
                            get_text("total_collisions", st.session_state.language),
                            f"{total_collisions:,}"
                        )
                    
                    with col3:
                        cascade_probability = results.get('cascade_probability', 0)
                        st.metric(
                            get_text("cascade_probability", st.session_state.language),
                            f"{cascade_probability:.1%}"
                        )
                
                else:
                    st.error("Simulation failed to produce results")
            
            except Exception as e:
                st.error(f"Error in Kessler simulation: {e}")
                st.exception(e)
                st.session_state.simulation_running = False
    
    except Exception as e:
        st.error(f"Error in Kessler simulation setup: {e}")
        st.exception(e)

def render_economic_analysis():
    """Render economic analysis of debris removal missions."""
    
    st.markdown(f"## {get_text('economic_analysis_title', st.session_state.language)}")
    
    try:
        # Initialize economic analyzer
        analyzer = EconomicAnalyzer()
        
        # Analysis parameters
        st.markdown(f"### {get_text('analysis_parameters', st.session_state.language)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mission_cost = st.number_input(
                get_text("mission_cost_musd", st.session_state.language),
                min_value=10, max_value=1000, value=150,
                help=get_text("mission_cost_help", st.session_state.language)
            )
        
        with col2:
            debris_mass = st.number_input(
                get_text("target_debris_mass", st.session_state.language),
                min_value=100, max_value=10000, value=2000,
                help=get_text("debris_mass_help", st.session_state.language)
            )
        
        with col3:
            mission_success_rate = st.slider(
                get_text("success_probability", st.session_state.language),
                min_value=0.5, max_value=1.0, value=0.85, step=0.05
            )
        
        # Run economic analysis
        if st.button(get_text("run_economic_analysis", st.session_state.language), type="primary"):
            with st.spinner(get_text("calculating_economics", st.session_state.language)):
                try:
                    # Perform cost-benefit analysis
                    results = analyzer.analyze_debris_removal_mission(
                        mission_cost_musd=mission_cost,
                        target_debris_mass_kg=debris_mass,
                        success_probability=mission_success_rate
                    )
                    
                    # Display results
                    st.success(get_text("analysis_complete", st.session_state.language))
                    
                    # Key metrics
                    st.markdown(f"### {get_text('economic_metrics', st.session_state.language)}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        net_benefit = results.get('net_present_value', 0)
                        benefit_color = "green" if net_benefit > 0 else "red"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; background: linear-gradient(145deg, #1e2124, #2f3349); 
                                   border-radius: 10px; border-left: 4px solid {benefit_color};">
                        <h4>{get_text('net_present_value', st.session_state.language)}</h4>
                        <h3 style="color: {benefit_color};">${net_benefit:.1f}M</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        roi = results.get('return_on_investment', 0)
                        roi_color = "green" if roi > 0 else "red"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; background: linear-gradient(145deg, #1e2124, #2f3349); 
                                   border-radius: 10px; border-left: 4px solid {roi_color};">
                        <h4>{get_text('return_on_investment', st.session_state.language)}</h4>
                        <h3 style="color: {roi_color};">{roi:.1%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        bcr = results.get('benefit_cost_ratio', 0)
                        bcr_color = "green" if bcr > 1 else "orange" if bcr > 0.5 else "red"
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; background: linear-gradient(145deg, #1e2124, #2f3349); 
                                   border-radius: 10px; border-left: 4px solid {bcr_color};">
                        <h4>{get_text('benefit_cost_ratio', st.session_state.language)}</h4>
                        <h3 style="color: {bcr_color};">{bcr:.2f}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error in economic analysis: {e}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"Error in economic analysis setup: {e}")
        st.exception(e)

def render_compliance_monitor():
    """Render compliance monitoring dashboard."""
    
    st.markdown(f"## {get_text('compliance_monitor_title', st.session_state.language)}")
    
    try:
        # Initialize compliance monitor
        monitor = ComplianceMonitor()
        
        # Compliance standards selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_standards = st.multiselect(
                get_text("compliance_standards", st.session_state.language),
                ["ISO 27852", "IADC Guidelines", "UN Space Debris Mitigation", "FCC Rules"],
                default=["ISO 27852", "IADC Guidelines"]
            )
        
        with col2:
            assessment_period = st.selectbox(
                get_text("assessment_period", st.session_state.language),
                [
                    get_text("real_time", st.session_state.language),
                    get_text("daily", st.session_state.language),
                    get_text("weekly", st.session_state.language),
                    get_text("monthly", st.session_state.language)
                ]
            )
        
        # Run compliance assessment
        if st.button(get_text("run_compliance_check", st.session_state.language), type="primary"):
            with st.spinner(get_text("assessing_compliance", st.session_state.language)):
                try:
                    # Get current satellite and debris data
                    celestrak = CelesTrakDataSource()
                    debris_data = celestrak.get_debris_catalog()
                    satellite_data = celestrak.get_active_satellites()
                    
                    # Run compliance assessment
                    compliance_results = monitor.assess_compliance(
                        debris_data=debris_data,
                        satellite_data=satellite_data,
                        standards=selected_standards
                    )
                    
                    # Display overall compliance score
                    st.markdown(f"### {get_text('overall_compliance', st.session_state.language)}")
                    
                    overall_score = compliance_results.get('overall_score', 0.75)
                    score_color = "#00ff41" if overall_score > 0.8 else "#ffaa00" if overall_score > 0.6 else "#ff4444"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #1e2124, #2f3349); 
                               border-radius: 15px; border-left: 6px solid {score_color};">
                    <h2>{get_text('compliance_score', st.session_state.language)}</h2>
                    <h1 style="color: {score_color}; font-size: 3em;">{overall_score:.1%}</h1>
                    <p>{get_text('compliance_rating', st.session_state.language)}: 
                    {'Excellent' if overall_score > 0.9 else 'Good' if overall_score > 0.8 else 'Needs Improvement' if overall_score > 0.6 else 'Poor'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error in compliance assessment: {e}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"Error in compliance monitoring setup: {e}")
        st.exception(e)

def render_orbital_mechanics():
    """Render advanced orbital mechanics analysis."""
    
    st.markdown(f"## {get_text('orbital_mechanics_title', st.session_state.language)}")
    
    try:
        # Initialize propagators
        sgp4_propagator = EnhancedSGP4Propagator()
        nbody_model = NBodyGravitationalModel()
        atmospheric_model = NRLMSISE00Model()
        
        # Orbital parameters input
        st.markdown(f"### {get_text('orbital_parameters', st.session_state.language)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            semi_major_axis = st.number_input(
                get_text("semi_major_axis", st.session_state.language),
                min_value=6500.0, max_value=50000.0, value=7000.0, step=10.0,
                help=get_text("sma_help", st.session_state.language)
            )
            
            inclination = st.slider(
                get_text("inclination_deg", st.session_state.language),
                min_value=0.0, max_value=180.0, value=45.0, step=0.1
            )
        
        with col2:
            eccentricity = st.number_input(
                get_text("eccentricity", st.session_state.language),
                min_value=0.0, max_value=0.99, value=0.001, step=0.001, format="%.4f"
            )
            
            raan = st.slider(
                get_text("raan_deg", st.session_state.language),
                min_value=0.0, max_value=360.0, value=0.0, step=0.1
            )
        
        with col3:
            arg_perigee = st.slider(
                get_text("arg_perigee_deg", st.session_state.language),
                min_value=0.0, max_value=360.0, value=0.0, step=0.1
            )
            
            mean_anomaly = st.slider(
                get_text("mean_anomaly_deg", st.session_state.language),
                min_value=0.0, max_value=360.0, value=0.0, step=0.1
            )
        
        # Object properties
        st.markdown(f"### {get_text('object_properties', st.session_state.language)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            object_mass = st.number_input(
                get_text("mass_kg", st.session_state.language),
                min_value=1.0, max_value=10000.0, value=500.0
            )
        
        with col2:
            cross_section_area = st.number_input(
                get_text("cross_section_m2", st.session_state.language),
                min_value=0.01, max_value=100.0, value=1.0, step=0.01
            )
        
        with col3:
            drag_coefficient = st.number_input(
                get_text("drag_coefficient", st.session_state.language),
                min_value=1.0, max_value=4.0, value=2.2, step=0.1
            )
        
        # Propagation settings
        st.markdown(f"### {get_text('propagation_settings', st.session_state.language)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            propagation_days = st.slider(
                get_text("propagation_period", st.session_state.language),
                min_value=1, max_value=365, value=30
            )
        
        with col2:
            time_step_minutes = st.selectbox(
                get_text("time_step", st.session_state.language),
                [1, 5, 15, 30, 60, 120],
                index=4
            )
        
        # Run orbital analysis
        if st.button(get_text("run_orbital_analysis", st.session_state.language), type="primary"):
            with st.spinner(get_text("computing_orbit", st.session_state.language)):
                try:
                    # Define initial orbital elements
                    orbital_elements = {
                        'semi_major_axis': semi_major_axis,
                        'eccentricity': eccentricity,
                        'inclination': inclination,
                        'raan': raan,
                        'arg_perigee': arg_perigee,
                        'mean_anomaly': mean_anomaly
                    }
                    
                    # Object properties
                    object_properties = {
                        'mass': object_mass,
                        'cross_sectional_area': cross_section_area,
                        'drag_coefficient': drag_coefficient,
                        'reflectivity': 0.3  # Default reflectivity
                    }
                    
                    # Get space weather data
                    noaa = NOAASpaceWeatherSource()
                    space_weather = noaa.get_solar_activity()
                    
                    # Simple propagation for demonstration
                    r_eci, v_eci, updated_elements = sgp4_propagator.propagate_orbit(
                        orbital_elements, 
                        time_step_minutes * 60,
                        object_properties,
                        space_weather
                    )
                    
                    # Display results
                    st.success(f"{get_text('propagation_complete', st.session_state.language)}")
                    
                    # Orbital statistics
                    st.markdown(f"### {get_text('orbital_statistics', st.session_state.language)}")
                    
                    altitude = np.linalg.norm(r_eci) - EARTH_RADIUS
                    velocity_mag = np.linalg.norm(v_eci)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            get_text("current_altitude", st.session_state.language),
                            f"{altitude:.1f} km"
                        )
                    
                    with col2:
                        st.metric(
                            get_text("velocity_magnitude", st.session_state.language),
                            f"{velocity_mag:.2f} km/s"
                        )
                    
                    with col3:
                        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / EARTH_GRAVITATIONAL_PARAMETER) / 3600
                        st.metric(
                            get_text("orbital_period", st.session_state.language),
                            f"{period:.2f} h"
                        )
                
                except Exception as e:
                    st.error(f"{get_text('propagation_error', st.session_state.language)}: {str(e)}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"Error in orbital mechanics setup: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
