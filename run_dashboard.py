#!/usr/bin/env python3
"""
Streamlit Dashboard Launcher for GNN Fraud Detection System
"""
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    # Map package names to their import names
    required_packages = {
        'streamlit': 'streamlit',
        'plotly': 'plotly', 
        'umap-learn': 'umap',
        'pyvis': 'pyvis',
        'torch': 'torch',
        'torch_geometric': 'torch_geometric',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn'
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    return True

def main():
    print("GNN Fraud Detection Dashboard Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('app/streamlit_dashboard.py'):
        print("Error: Please run this script from the project root directory")
        print("Expected structure: gnn-fraud-detection-fintech/")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Launch Streamlit dashboard
    dashboard_path = Path("app/streamlit_dashboard.py")
    
    print(f"Starting Streamlit dashboard...")
    print(f"Dashboard will be available at: http://localhost:8501")
    print("\nTo stop the dashboard, press Ctrl+C in this terminal")
    print("-" * 40)
    
    try:
        # Run streamlit with custom configuration
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.serverAddress", "localhost",
            "--server.headless", "false"
        ]
        
        subprocess.run(cmd, cwd=os.getcwd())
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 