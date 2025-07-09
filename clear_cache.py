#!/usr/bin/env python3
"""
Streamlit Cache Clearing Utility
"""
import os
import shutil
from pathlib import Path

def clear_streamlit_cache():
    """Clear all Streamlit cache directories"""
    
    # Common cache locations
    cache_locations = [
        Path.home() / ".streamlit" / "cache",
        Path.cwd() / ".streamlit" / "cache", 
        Path("/tmp") / "streamlit",
        Path.home() / ".cache" / "streamlit"
    ]
    
    cleared_count = 0
    
    for cache_path in cache_locations:
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"✅ Cleared cache: {cache_path}")
                cleared_count += 1
            except Exception as e:
                print(f"❌ Failed to clear {cache_path}: {e}")
    
    if cleared_count == 0:
        print("ℹ️  No cache directories found to clear")
    else:
        print(f"🎉 Successfully cleared {cleared_count} cache location(s)")

if __name__ == "__main__":
    print("🧹 Clearing Streamlit Cache...")
    clear_streamlit_cache()
    print("✨ Cache clearing complete!") 