# TrueSight - Complete System

## 🎯 System Overview
TrueSight is a comprehensive AI-powered multi-modal deepfake detection and forensic attribution system with a complete web interface.

## 🚀 Quick Start

### Option 1: Start Everything Together
```bash
python launch_full_system.py
```

### Option 2: Start Components Separately

**Backend API Server:**
```bash
cd src
python api/main.py
```

**Frontend Server:**
```bash
python serve_frontend.py
```

## 🌐 Access Points

- **Frontend Interface**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📋 Features Available

### 🔍 Deepfake Detection
- Upload video/audio files for analysis
- Real-time detection results
- Confidence scoring and artifact detection
- Multi-modal analysis (video + audio)

### 📊 Dashboard
- Real-time system statistics
- Processing metrics and performance data
- Activity visualization
- Detection history overview

### 🔬 Digital Forensics
- Device attribution tools
- PRNU analysis capabilities
- Compression artifact detection
- Source identification

### ⚙️ System Management
- User authentication interface
- System settings and configuration
- API integration controls
- Monitoring and logging

## 🛠️ Technology Stack

### Frontend
- Pure HTML5/CSS3/JavaScript (No build tools required)
- Responsive design with modern UI components
- Interactive data visualization
- Real-time notifications

### Backend
- FastAPI (Python) REST API
- Asynchronous processing
- Prometheus monitoring
- JWT authentication

## 📁 Project Structure
```
true-sight/
├── src/                 # Backend API source code
├── frontend/            # Web interface files
├── infrastructure/      # Docker and deployment configs
├── models/              # AI/ML models
├── launch_full_system.py # Complete system launcher
├── serve_frontend.py    # Frontend server
└── README.md           # This file
```

## 🔧 Development Notes

The frontend simulates API calls for demonstration purposes. In production:
- Connect frontend to actual backend endpoints
- Implement proper authentication flows
- Add real-time WebSocket connections
- Integrate with actual ML models

## 🎉 Getting Started

1. Click the "TrueSight Frontend" preview button to access the web interface
2. Explore the dashboard and detection features
3. Upload sample media files for analysis
4. View real-time system metrics and results

The system is ready for immediate demonstration and testing!