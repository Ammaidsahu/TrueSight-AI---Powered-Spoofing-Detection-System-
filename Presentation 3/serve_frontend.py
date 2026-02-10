"""
Simple HTTP Server for TrueSight Frontend
"""

import http.server
import socketserver
import os
import threading
import webbrowser
from pathlib import Path

class FrontendServer:
    def __init__(self, port=3000):
        self.port = port
        self.directory = Path(__file__).parent / "frontend"
        
    def start_server(self):
        """Start the frontend server"""
        os.chdir(self.directory)
        
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", self.port), handler) as httpd:
            print(f"🚀 TrueSight Frontend Server running at http://localhost:{self.port}")
            print("📁 Serving files from:", self.directory)
            print("🔄 Press Ctrl+C to stop the server")
            print("-" * 50)
            
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n🛑 Frontend server stopped")
                httpd.shutdown()

def main():
    """Main function to start the frontend server"""
    server = FrontendServer(port=3000)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=server.start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Give server time to start
    import time
    time.sleep(2)
    
    # Open browser
    print("🌐 Opening browser...")
    webbrowser.open(f"http://localhost:3000")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()