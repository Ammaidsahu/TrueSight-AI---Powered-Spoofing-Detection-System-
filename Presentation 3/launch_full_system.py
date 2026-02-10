"""
TrueSight Complete System Launcher
Starts both backend API and frontend servers
"""

import subprocess
import threading
import time
import sys
import os

def start_backend():
    """Start the backend API server"""
    print("🚀 Starting TrueSight Backend API Server...")
    os.chdir('src')
    subprocess.run([sys.executable, 'api/main.py'])

def start_frontend():
    """Start the frontend server"""
    print("🎨 Starting TrueSight Frontend Server...")
    time.sleep(3)  # Wait for backend to start
    subprocess.run([sys.executable, '../serve_frontend.py'])

def main():
    print("=" * 60)
    print("TRUE SIGHT COMPLETE SYSTEM LAUNCHER")
    print("=" * 60)
    print()
    print("Starting both backend API and frontend servers...")
    print()
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Give backend time to initialize
    time.sleep(5)
    
    # Start frontend
    start_frontend()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 System shutdown initiated...")
        print("👋 Thank you for using TrueSight!")