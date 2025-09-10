#!/usr/bin/env python3
"""
Energy Advisor MVP - Streamlit App Launcher
Simple launcher script to run the Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Streamlit app
    app_path = os.path.join(script_dir, 'src', 'streamlit_app.py')
    
    # Check if the app exists
    if not os.path.exists(app_path):
        print(f"❌ Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching Energy Advisor MVP...")
    print(f"📁 App location: {app_path}")
    print("🌐 Opening in your default browser...")
    print("⏹️  Press Ctrl+C to stop the app")
    print()
    
    try:
        # Run the Streamlit app (let Streamlit handle port/address automatically)
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 