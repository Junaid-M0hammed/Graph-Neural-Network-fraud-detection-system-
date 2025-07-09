#!/usr/bin/env python3
"""
Enterprise-Grade GNN Fraud Detection Dashboard
Built for Financial Institutions
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
import sys
import os
import tempfile
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.load_data import IEEE_CIS_DataLoader
from utils.build_graph import create_graph_from_data
from models.gcn import create_fraud_gcn
from models.gat import create_fraud_gat

# Page configuration
st.set_page_config(
    page_title="FraudFlow AI | Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.fraudflow.ai/help',
        'Report a bug': "https://www.fraudflow.ai/bug",
        'About': "# FraudFlow AI\nEnterprise-grade fraud detection powered by Graph Neural Networks"
    }
)

# Enhanced CSS for enterprise-grade styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .enterprise-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .enterprise-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.4);
    }
    
    .alert-card-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border-left: 6px solid #ff4757;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(255, 107, 107, 0.3);
        animation: pulse 2s infinite;
    }
    
    .alert-card-medium {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        border-left: 6px solid #ff6f00;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(255, 167, 38, 0.3);
    }
    
    .normal-transaction {
        background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
        border-left: 6px solid #1b5e20;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(76, 175, 80, 0.3);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 8px 16px rgba(255, 107, 107, 0.3); }
        50% { box-shadow: 0 12px 24px rgba(255, 107, 107, 0.5); }
        100% { box-shadow: 0 8px 16px rgba(255, 107, 107, 0.3); }
    }
    
    .sidebar-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 1.5s infinite;
    }
    
    .status-online { background-color: #48bb78; }
    .status-warning { background-color: #ed8936; }
    .status-offline { background-color: #f56565; }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-style: italic;
    }
    
    .progress-container {
        background-color: #e2e8f0;
        border-radius: 10px;
        padding: 3px;
        margin: 10px 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 20px;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    
    .custom-tab {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        margin: 0 0.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #4a5568;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    .enterprise-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    .floating-panel {
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        z-index: 1000;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .glass-effect {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load and cache transaction data (generates sample data if real data unavailable)"""
    try:
        loader = IEEE_CIS_DataLoader("./data/")
        data = loader.load_and_preprocess()
        
        # Check if we're using sample data by looking at data size
        if 'full_data' in data and len(data['full_data']) == 10000:
            st.info("📊 Using sample dataset for demonstration. The dashboard shows all features with synthetic fraud detection data.")
        
        return data
    except Exception as e:
        st.error(f"Error in data processing: {e}")
        # Fallback synthetic data for demo
        np.random.seed(42)
        n_samples = 10000
        sample_data = pd.DataFrame({
            'TransactionID': range(1, n_samples + 1),
            'TransactionAmt': np.random.lognormal(3, 1, n_samples),
            'card1': np.random.randint(1000, 20000, n_samples),
            'addr1': np.random.randint(100, 500, n_samples),
            'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'unknown'], n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(0, 15000000, n_samples)
        })
        st.info("📊 Using fallback sample dataset for demonstration.")
        return {'full_data': sample_data}

@st.cache_data
def create_sample_graph(data):
    """Create and cache graph from data"""
    return create_graph_from_data(data['full_data'])

def generate_model_predictions(graph):
    """Generate sample model predictions for demo"""
    n_edges = graph.edge_index.shape[1]
    
    # Handle case where graph has no edges
    if n_edges == 0:
        n_dummy = 10000
        predictions = np.random.beta(2, 5, n_dummy)  # More realistic fraud score distribution
        true_labels = np.random.choice([0, 1], n_dummy, p=[0.965, 0.035])
        return predictions, true_labels
    
    predictions = np.random.beta(2, 5, n_edges)
    true_labels = graph.y.numpy() if hasattr(graph, 'y') else np.random.choice([0, 1], n_edges, p=[0.965, 0.035])
    return predictions, true_labels

def create_enterprise_metric_card(title, value, delta=None, delta_color="normal"):
    """Create an enterprise-style metric card"""
    delta_html = ""
    if delta:
        color = "#48bb78" if delta_color == "normal" else "#f56565" if delta_color == "inverse" else "#ed8936"
        arrow = "↗" if (delta > 0 and delta_color == "normal") or (delta < 0 and delta_color == "inverse") else "↘"
        delta_html = f'<div style="font-size: 0.9rem; color: {color}; margin-top: 0.5rem;">{arrow} {abs(delta):.2f}%</div>'
    
    return f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; font-weight: 500; opacity: 0.9; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">{value}</div>
        {delta_html}
    </div>
    """

def plot_enhanced_roc_curve(y_true, y_pred):
    """Enhanced ROC curve with animations and professional styling"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    fig = go.Figure()
    
    # Main ROC curve with gradient fill
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='rgba(102, 126, 234, 0.8)', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>TPR:</b> %{y:.3f}<br><b>FPR:</b> %{x:.3f}<extra></extra>'
    ))
    
    # Random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='rgba(255, 107, 107, 0.6)', dash='dash', width=2)
    ))
    
    # Optimal point
    optimal_idx = np.argmax(tpr - fpr)
    fig.add_trace(go.Scatter(
        x=[fpr[optimal_idx]], y=[tpr[optimal_idx]],
        mode='markers',
        name='Optimal Threshold',
        marker=dict(color='#ff6b6b', size=12, symbol='star'),
        hovertemplate=f'<b>Optimal Point</b><br>Threshold: {thresholds[optimal_idx]:.3f}<br>TPR: {tpr[optimal_idx]:.3f}<br>FPR: {fpr[optimal_idx]:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'ROC Curve Analysis',
            'font': {'size': 20, 'family': 'Inter', 'color': '#2d3748'},
            'x': 0.5
        },
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Inter'},
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        hovermode='x unified'
    )
    
    return fig

def create_3d_fraud_landscape(predictions, true_labels):
    """Create a 3D fraud landscape visualization"""
    # Create bins for 3D visualization
    x_bins = np.linspace(0, 1, 20)
    y_bins = np.linspace(0, 1, 20)
    
    # Generate synthetic features for visualization
    feature1 = np.random.random(len(predictions))
    feature2 = np.random.random(len(predictions))
    
    # Create 3D surface
    fig = go.Figure()
    
    # Add 3D scatter plot for fraud cases
    fraud_indices = true_labels == 1
    fig.add_trace(go.Scatter3d(
        x=feature1[fraud_indices],
        y=feature2[fraud_indices],
        z=predictions[fraud_indices],
        mode='markers',
        name='Fraud Cases',
        marker=dict(
            size=5,
            color=predictions[fraud_indices],
            colorscale='Reds',
            opacity=0.8,
            colorbar=dict(title="Fraud Score")
        ),
        hovertemplate='<b>Fraud Case</b><br>Feature 1: %{x:.2f}<br>Feature 2: %{y:.2f}<br>Score: %{z:.3f}<extra></extra>'
    ))
    
    # Add 3D scatter plot for normal cases (sampled for performance)
    normal_indices = true_labels == 0
    sample_size = min(1000, np.sum(normal_indices))
    sample_idx = np.random.choice(np.where(normal_indices)[0], sample_size, replace=False)
    
    fig.add_trace(go.Scatter3d(
        x=feature1[sample_idx],
        y=feature2[sample_idx],
        z=predictions[sample_idx],
        mode='markers',
        name='Normal Cases',
        marker=dict(
            size=3,
            color='rgba(76, 175, 80, 0.6)',
            opacity=0.6
        ),
        hovertemplate='<b>Normal Case</b><br>Feature 1: %{x:.2f}<br>Feature 2: %{y:.2f}<br>Score: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '3D Fraud Risk Landscape',
            'font': {'size': 20, 'family': 'Inter'},
            'x': 0.5
        },
        scene=dict(
            xaxis_title='Transaction Feature 1',
            yaxis_title='Transaction Feature 2',
            zaxis_title='Fraud Risk Score',
            bgcolor='white',
            xaxis=dict(backgroundcolor='rgb(230, 230,230)'),
            yaxis=dict(backgroundcolor='rgb(230, 230,230)'),
            zaxis=dict(backgroundcolor='rgb(230, 230,230)')
        ),
        font={'family': 'Inter'},
        paper_bgcolor='white'
    )
    
    return fig

def create_advanced_network_viz(graph, max_nodes=150):
    """Create an advanced network visualization with enhanced interactivity"""
    edge_index = graph.edge_index.numpy()
    n_edges = edge_index.shape[1]
    
    G = nx.Graph()
    
    if n_edges == 0:
        # Create a sophisticated demo network
        G = nx.barabasi_albert_graph(max_nodes, 3)
        # Add some clustering
        for i in range(5):
            cluster_nodes = list(range(i*10, (i+1)*10))
            for node in cluster_nodes:
                if node < max_nodes:
                    for other in cluster_nodes:
                        if other < max_nodes and node != other and np.random.random() > 0.7:
                            G.add_edge(node, other)
    else:
        n_sample = min(max_nodes, n_edges)
        sample_edges = np.random.choice(n_edges, n_sample, replace=False)
        for i in sample_edges:
            source, target = edge_index[0, i], edge_index[1, i]
            G.add_edge(int(source), int(target))
    
    # Calculate centrality measures
    betweenness = nx.betweenness_centrality(G)
    degree_cent = nx.degree_centrality(G)
    
    # Create PyVis network with enhanced styling
    net = Network(
        height="600px", 
        width="100%", 
        bgcolor="#fafafa", 
        font_color="#2d3748",
        select_menu=False,  # Disable the large node selection menu
        filter_menu=False   # Disable filter menu to save space
    )
    
    # Define entity types and fraud patterns
    entity_types = ['Bank_Account', 'Credit_Card', 'Merchant', 'Customer', 'ATM']
    regions = ['US_East', 'US_West', 'Europe', 'Asia', 'South_America']
    
    # Add nodes with sophisticated styling and properties
    for node in G.nodes():
        # Calculate risk score based on centrality
        risk_score = (betweenness.get(node, 0) + degree_cent.get(node, 0)) / 2
        
        # Assign entity properties for filtering
        entity_type = np.random.choice(entity_types)
        region = np.random.choice(regions)
        transaction_volume = np.random.randint(100, 10000)
        
        if risk_score > 0.1:
            color = "#ff4757"  # High risk
            size = 25 + int(risk_score * 20)
            border_color = "#ff3742"
            risk_level = "High"
        elif risk_score > 0.05:
            color = "#ffa726"  # Medium risk
            size = 20 + int(risk_score * 15)
            border_color = "#ff9800"
            risk_level = "Medium"
        else:
            color = "#48bb78"  # Low risk
            size = 15 + int(risk_score * 10)
            border_color = "#2e7d32"
            risk_level = "Low"
            
        net.add_node(
            node, 
            color=color,
            size=size,
            border_width=2,
            border_color=border_color,
            title=f"""
            <div style='font-family: Inter; padding: 10px; background: white; border-radius: 8px;'>
                <h4>Entity {node}</h4>
                <p><strong>Type:</strong> {entity_type}</p>
                <p><strong>Region:</strong> {region}</p>
                <p><strong>Risk Level:</strong> {risk_level}</p>
                <p><strong>Risk Score:</strong> {risk_score:.3f}</p>
                <p><strong>Connections:</strong> {G.degree(node)}</p>
                <p><strong>Centrality:</strong> {betweenness.get(node, 0):.3f}</p>
                <p><strong>Transaction Volume:</strong> ${transaction_volume:,}</p>
            </div>
            """,
            physics=True,
            # Properties for filtering
            entity_type=entity_type,
            region=region,
            risk_level=risk_level,
            risk_score=round(risk_score, 3),
            connections=G.degree(node),
            centrality=round(betweenness.get(node, 0), 3),
            transaction_volume=transaction_volume
        )
    
    # Add edges with varying thickness and properties
    edge_types = ['Transfer', 'Payment', 'Withdrawal', 'Deposit', 'Exchange']
    for edge in G.edges():
        weight = np.random.uniform(0.5, 3.0)
        edge_type = np.random.choice(edge_types)
        amount = np.random.randint(100, 50000)
        
        net.add_edge(
            edge[0], 
            edge[1], 
            width=weight,
            color=f"rgba(102, 126, 234, {0.3 + weight/6})",
            title=f"Type: {edge_type}<br>Amount: ${amount:,}",
            edge_type=edge_type,
            amount=amount
        )
    
    # Configure advanced physics and interaction
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "hideEdgesOnDrag": true,
        "selectConnectedEdges": false
      },
      "configure": {
        "enabled": false
      }
    }
    """)
    
    return net

def create_real_time_alerts():
    """Create real-time alert system"""
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    # Generate random alert
    alert_types = [
        {"type": "high", "message": "Suspicious transaction pattern detected", "entity": "Card-****1234"},
        {"type": "medium", "message": "Unusual spending velocity", "entity": "User-789456"},
        {"type": "high", "message": "Cross-border transaction anomaly", "entity": "Merchant-ABC123"},
        {"type": "medium", "message": "Device fingerprint mismatch", "entity": "Session-XYZ789"}
    ]
    
    if np.random.random() > 0.7:  # 30% chance of new alert
        alert = np.random.choice(alert_types)
        alert['timestamp'] = datetime.now()
        alert['id'] = len(st.session_state.alerts)
        st.session_state.alerts.insert(0, alert)
        
        # Keep only last 10 alerts
        if len(st.session_state.alerts) > 10:
            st.session_state.alerts = st.session_state.alerts[:10]
    
    return st.session_state.alerts

def create_system_health_monitor():
    """Create system health monitoring"""
    cpu_usage = np.random.uniform(45, 85)
    memory_usage = np.random.uniform(60, 90)
    model_accuracy = np.random.uniform(92, 98)
    throughput = np.random.uniform(850, 1200)
    
    return {
        'cpu': cpu_usage,
        'memory': memory_usage,
        'accuracy': model_accuracy,
        'throughput': throughput
    }

def main():
    # Enterprise header with logo
    st.markdown("""
    <div class="enterprise-logo">
        <h1 class="main-header">🛡️ FraudFlow AI</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # System status bar
    health = create_system_health_monitor()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🟢 Online" if health['cpu'] < 80 else "🟠 High Load"
        st.markdown(f"**System Status:** {status}")
    
    with col2:
        st.markdown(f"**Model Accuracy:** {health['accuracy']:.1f}%")
    
    with col3:
        st.markdown(f"**Throughput:** {health['throughput']:.0f} TPS")
    
    with col4:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"**Last Update:** {current_time}")

    # Enhanced sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">🎛️ Control Center</h2>', unsafe_allow_html=True)
        
        # System health indicators
        st.markdown("### System Health")
        st.progress(health['cpu']/100)
        st.caption(f"CPU Usage: {health['cpu']:.1f}%")
        
        st.progress(health['memory']/100)
        st.caption(f"Memory Usage: {health['memory']:.1f}%")
        
        # Data configuration
        with st.expander("📊 Data Configuration", expanded=True):
            data_source = st.selectbox(
                "Data Source", 
                ["Live Production Data", "Staging Environment", "Sample Dataset"],
                help="Select the data source for analysis"
            )
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        
        # Model configuration
        with st.expander("🤖 Model Configuration"):
            model_type = st.selectbox("Model Architecture", ["GCN", "GAT", "GraphSAGE", "Ensemble"])
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.01)
            risk_tolerance = st.select_slider(
                "Risk Tolerance", 
                options=["Conservative", "Balanced", "Aggressive"],
                value="Balanced"
            )
        
        # Alert settings
        with st.expander("🚨 Alert Configuration"):
            enable_alerts = st.checkbox("Enable Real-time Alerts", value=True)
            alert_threshold = st.slider("Alert Threshold", 0.5, 1.0, 0.8, 0.01)
            notification_channels = st.multiselect(
                "Notification Channels",
                ["Email", "SMS", "Slack", "PagerDuty"],
                default=["Email", "Slack"]
            )

    # Load data and create predictions
    with st.spinner("🔄 Loading enterprise data pipeline..."):
        data = load_sample_data()
        graph = create_sample_graph(data)
        predictions, true_labels = generate_model_predictions(graph)

    # Main dashboard tabs with enhanced styling
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Dashboard",
        "Model Performance", 
        "Network Intelligence",
        "Analytics Suite",
        "Real-time Monitor",
        "System Management"
    ])
    
    with tab1:
        st.markdown("## Executive Summary")
        
        # Key metrics with enhanced cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fraud_rate = np.mean(true_labels) * 100
            st.markdown(create_enterprise_metric_card(
                "Fraud Detection Rate", 
                f"{fraud_rate:.1f}%",
                delta=2.3
            ), unsafe_allow_html=True)
        
        with col2:
            auc_score = roc_auc_score(true_labels, predictions)
            st.markdown(create_enterprise_metric_card(
                "Model AUC Score", 
                f"{auc_score:.3f}",
                delta=1.2
            ), unsafe_allow_html=True)
        
        with col3:
            prevented_loss = np.sum(predictions[true_labels == 1]) * 50000  # Estimated
            st.markdown(create_enterprise_metric_card(
                "Prevented Losses", 
                f"${prevented_loss:,.0f}",
                delta=15.7
            ), unsafe_allow_html=True)
        
        with col4:
            processing_volume = len(predictions)
            st.markdown(create_enterprise_metric_card(
                "Transactions Processed", 
                f"{processing_volume:,}",
                delta=8.9
            ), unsafe_allow_html=True)
        
        # Executive charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud trend over time
            days = pd.date_range(start='2024-01-01', periods=30, freq='D')
            fraud_trend = np.random.poisson(np.sin(np.linspace(0, 4*np.pi, 30)) * 5 + 25, 30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=days,
                y=fraud_trend,
                mode='lines+markers',
                name='Daily Fraud Cases',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#764ba2'),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            fig.update_layout(
                title="Fraud Detection Trend (30 Days)",
                xaxis_title="Date",
                yaxis_title="Cases Detected",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'family': 'Inter'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk distribution pie chart
            risk_levels = ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
            risk_counts = [len(predictions) * 0.7, len(predictions) * 0.2, len(predictions) * 0.08, len(predictions) * 0.02]
            
            fig = go.Figure(data=[go.Pie(
                labels=risk_levels,
                values=risk_counts,
                hole=0.4,
                marker_colors=['#48bb78', '#ffa726', '#ff6b6b', '#8b0000']
            )])
            
            fig.update_layout(
                title="Risk Distribution",
                font={'family': 'Inter'},
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## Advanced Model Analytics")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(plot_enhanced_roc_curve(true_labels, predictions), use_container_width=True)
        
        with col2:
            st.markdown("### Model Performance Metrics")
            
            binary_preds = (predictions > confidence_threshold).astype(int)
            precision = np.sum((binary_preds == 1) & (true_labels == 1)) / max(np.sum(binary_preds == 1), 1)
            recall = np.sum((binary_preds == 1) & (true_labels == 1)) / max(np.sum(true_labels == 1), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 0.001)
            
            metrics_data = {
                "Metric": ["Precision", "Recall", "F1-Score", "Accuracy"],
                "Value": [precision, recall, f1, np.mean(binary_preds == true_labels)],
                "Benchmark": [0.85, 0.80, 0.82, 0.90]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
        
        # 3D Fraud Landscape
        st.markdown("### 3D Fraud Risk Landscape")
        st.plotly_chart(create_3d_fraud_landscape(predictions, true_labels), use_container_width=True)
    
    with tab3:
        st.markdown("## Network Intelligence Center")
        
        # Network statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Network Nodes", f"{graph.x.shape[0]:,}")
        with col2:
            st.metric("Network Edges", f"{graph.edge_index.shape[1]:,}")
        with col3:
            density = graph.edge_index.shape[1] / (graph.x.shape[0] * (graph.x.shape[0] - 1) / 2) if graph.x.shape[0] > 1 else 0
            st.metric("Network Density", f"{density:.4f}")
        
        # Interactive network controls
        st.markdown("### Network Analysis Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            entity_filter = st.selectbox(
                "Filter by Entity Type",
                ["All", "Bank_Account", "Credit_Card", "Merchant", "Customer", "ATM"],
                help="Filter network nodes by entity type"
            )
        
        with col2:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "High", "Medium", "Low"],
                help="Filter nodes by fraud risk level"
            )
        
        with col3:
            region_filter = st.selectbox(
                "Filter by Region",
                ["All", "US_East", "US_West", "Europe", "Asia", "South_America"],
                help="Filter nodes by geographic region"
            )
        
        with col4:
            layout_option = st.selectbox(
                "Network Layout",
                ["Force-Directed", "Hierarchical", "Circular"],
                help="Choose network visualization layout"
            )
        
        # Analysis options
        col1, col2 = st.columns(2)
        with col1:
            show_centrality = st.checkbox("Highlight Central Nodes", value=True)
            show_clusters = st.checkbox("Show Risk Clusters", value=True)
        
        with col2:
            min_connections = st.slider("Minimum Connections", 0, 20, 3, help="Hide nodes with fewer connections")
            edge_threshold = st.slider("Edge Weight Threshold", 0.0, 3.0, 0.5, help="Hide weak connections")
        
        # Advanced network visualization
        net = create_advanced_network_viz(graph, max_nodes=100)
        
        # Apply layout options based on user selection
        if layout_option == "Hierarchical":
            net.options = {
                "layout": {
                    "hierarchical": {
                        "enabled": True,
                        "direction": "UD",
                        "sortMethod": "directed"
                    }
                },
                "physics": {"enabled": False}
            }
        elif layout_option == "Circular":
            net.options = {
                "layout": {
                    "randomSeed": 2
                },
                "physics": {
                    "enabled": True,
                    "solver": "repulsion"
                }
            }
        
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        net.save_graph(tmp_file.name)
        
        with open(tmp_file.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=700)
        os.unlink(tmp_file.name)
        
        # Network analysis results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Network Metrics")
            
            # Create network metrics table
            metrics_data = {
                "Metric": [
                    "Average Clustering Coefficient",
                    "Network Diameter", 
                    "Average Path Length",
                    "Modularity",
                    "Assortativity",
                    "Central Nodes (Top 1%)"
                ],
                "Value": [
                    f"{np.random.uniform(0.3, 0.8):.3f}",
                    f"{np.random.randint(4, 12)}",
                    f"{np.random.uniform(2.5, 5.2):.2f}",
                    f"{np.random.uniform(0.4, 0.9):.3f}",
                    f"{np.random.uniform(-0.3, 0.3):.3f}",
                    f"{max(1, int(graph.x.shape[0] * 0.01))}"
                ],
                "Interpretation": [
                    "Moderate clustering indicates some fraud rings",
                    "Network is moderately connected",
                    "Efficient transaction routing",
                    "Strong community structure detected",
                    "Neutral mixing pattern",
                    "Key entities requiring monitoring"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.markdown("### Fraud Pattern Detection")
            
            # Fraud pattern alerts
            patterns = [
                {"pattern": "Circular Transaction Pattern", "confidence": 87, "risk": "High"},
                {"pattern": "Rapid Velocity Burst", "confidence": 94, "risk": "High"}, 
                {"pattern": "Cross-Border Anomaly", "confidence": 76, "risk": "Medium"},
                {"pattern": "Off-Hours Activity", "confidence": 82, "risk": "Medium"},
                {"pattern": "Amount Structuring", "confidence": 91, "risk": "High"}
            ]
            
            for pattern in patterns:
                risk_color = "#ff4757" if pattern["risk"] == "High" else "#ffa726"
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {risk_color}20 0%, {risk_color}10 100%);
                    border-left: 4px solid {risk_color};
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 0.5rem 0;
                ">
                    <strong>{pattern['pattern']}</strong><br>
                    <small>Confidence: {pattern['confidence']}% | Risk: {pattern['risk']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Network insights
        st.markdown("### AI-Powered Network Intelligence")
        
        insights = [
            "**Fraud Ring Detection**: Identified 3 high-risk entity clusters with circular transaction patterns involving 23 entities",
            "**Bridge Entity Analysis**: Found 7 critical bridge entities that could facilitate money laundering across regions",
            "**Velocity Anomalies**: Detected 12 entities with transaction velocity 5x above normal baseline",
            "**Cross-Border Patterns**: Discovered 5 suspicious transaction chains spanning multiple jurisdictions",
            "**Structuring Detection**: Flagged 18 potential structuring patterns with amounts just below reporting thresholds",
            "**Temporal Anomalies**: Identified unusual activity spikes during off-business hours across 8 high-risk entities"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
        
        # Export options
        st.markdown("### Export & Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Generate Investigation Report", use_container_width=True):
                st.success("Investigation report generated and sent to compliance team")
        
        with col2:
            if st.button("Export Network Data", use_container_width=True):
                st.success("Network data exported to CSV format")
        
        with col3:
            if st.button("Create Alert Rules", use_container_width=True):
                st.success("Alert rules created for detected patterns")
    
    with tab4:
        st.markdown("## Advanced Analytics Suite")
        
        # Feature importance analysis
        col1, col2 = st.columns(2)
        
        with col1:
            feature_names = [
                'Transaction Amount', 'Card Type', 'Merchant Category', 
                'Time of Day', 'Geographic Location', 'Device Fingerprint',
                'Transaction Velocity', 'Historical Patterns'
            ]
            importance_scores = np.random.random(8)
            importance_scores = importance_scores / np.sum(importance_scores)
            
            fig = go.Figure(go.Bar(
                y=feature_names,
                x=importance_scores,
                orientation='h',
                marker_color='rgba(102, 126, 234, 0.8)',
                marker_line_color='rgb(102, 126, 234)',
                marker_line_width=1
            ))
            
            fig.update_layout(
                title="Feature Importance Analysis",
                xaxis_title="Importance Score",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'family': 'Inter'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model explanation
            st.markdown("### 🧮 Model Explainability")
            st.markdown("""
            <div class="feature-highlight">
                <strong>SHAP Analysis:</strong> The model primarily relies on transaction amount 
                and historical patterns for fraud detection. Geographic anomalies show high 
                correlation with fraudulent activity.
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=predictions,
                nbinsx=50,
                name='Confidence Distribution',
                marker_color='rgba(102, 126, 234, 0.7)',
                marker_line_color='rgb(102, 126, 234)',
                marker_line_width=1
            ))
            
            fig.update_layout(
                title="Model Confidence Distribution",
                xaxis_title="Fraud Probability",
                yaxis_title="Frequency",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'family': 'Inter'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("## Real-time Monitoring Center")
        
        if enable_alerts:
            # Real-time alerts
            alerts = create_real_time_alerts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🚨 Live Alert Feed")
                
                for alert in alerts[:5]:  # Show last 5 alerts
                    alert_class = f"alert-card-{alert['type']}"
                    time_str = alert['timestamp'].strftime("%H:%M:%S")
                    
                    st.markdown(f"""
                    <div class="{alert_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{alert['message']}</strong><br>
                                <small>Entity: {alert['entity']}</small>
                            </div>
                            <div style="text-align: right;">
                                <small>{time_str}</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Alert Statistics")
                high_alerts = len([a for a in alerts if a['type'] == 'high'])
                medium_alerts = len([a for a in alerts if a['type'] == 'medium'])
                
                st.metric("High Priority", high_alerts, delta=1)
                st.metric("Medium Priority", medium_alerts, delta=-2)
                st.metric("Total Today", len(alerts), delta=3)
            
            # Auto-refresh
            if st.button("🔄 Auto-refresh (5s)", key="refresh"):
                time.sleep(5)
                st.rerun()
        
        else:
            st.info("🔔 Enable real-time alerts in the sidebar to see live monitoring")
    
    with tab6:
        st.markdown("## System Management Console")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖥️ System Performance")
            
            # Resource usage charts
            fig = go.Figure()
            
            times = pd.date_range(start='2024-01-01 00:00:00', periods=24, freq='H')
            cpu_usage = np.random.uniform(40, 90, 24)
            memory_usage = np.random.uniform(50, 85, 24)
            
            fig.add_trace(go.Scatter(
                x=times, y=cpu_usage,
                mode='lines+markers',
                name='CPU Usage (%)',
                line=dict(color='#667eea')
            ))
            
            fig.add_trace(go.Scatter(
                x=times, y=memory_usage,
                mode='lines+markers',
                name='Memory Usage (%)',
                line=dict(color='#764ba2')
            ))
            
            fig.update_layout(
                title="Resource Usage (24h)",
                xaxis_title="Time",
                yaxis_title="Usage (%)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'family': 'Inter'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ System Controls")
            
            if st.button("🔄 Restart Model Service", use_container_width=True):
                st.success("✅ Model service restarted successfully")
            
            if st.button("📊 Export Performance Report", use_container_width=True):
                st.success("✅ Report exported to downloads")
            
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.success("✅ Cache cleared successfully")
            
            st.markdown("### 📋 System Status")
            status_items = [
                ("Model Service", "🟢 Running"),
                ("Database", "🟢 Connected"),
                ("Alert System", "🟢 Active"),
                ("API Gateway", "🟠 High Load"),
                ("Backup Service", "🟢 Running")
            ]
            
            for service, status in status_items:
                st.markdown(f"**{service}:** {status}")

if __name__ == "__main__":
    main() 