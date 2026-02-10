#!/usr/bin/env python3
"""
Launcher script for TrueSight API
Sets up proper Python path and starts the server
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Change to the project directory
os.chdir(Path(__file__).parent)

# Import and run the main application
from api.main import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting TrueSight API Server...")
    print("📍 Access at: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )