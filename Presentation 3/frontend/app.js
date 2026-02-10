// TrueSight Frontend Application Logic

class TrueSightApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentView = 'dashboard';
        this.uploadedFiles = [];
        this.init();
    }

    init() {
        // Show loading screen initially
        setTimeout(() => {
            this.hideLoadingScreen();
            this.setupEventListeners();
            this.loadDashboardData();
            this.showToast('Welcome to TrueSight!', 'success');
        }, 2000);
    }

    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        const app = document.getElementById('app');
        
        loadingScreen.classList.add('hidden');
        setTimeout(() => {
            loadingScreen.style.display = 'none';
            app.classList.remove('hidden');
        }, 500);
    }

    setupEventListeners() {
        // Navigation
        document.getElementById('dashboard-btn').addEventListener('click', () => this.switchView('dashboard'));
        document.getElementById('detection-btn').addEventListener('click', () => this.switchView('detection'));
        document.getElementById('forensics-btn').addEventListener('click', () => this.switchView('forensics'));
        document.getElementById('settings-btn').addEventListener('click', () => this.switchView('settings'));

        // Logout
        document.getElementById('logout-btn').addEventListener('click', () => this.logout());

        // File Upload
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#2563eb';
            uploadArea.style.backgroundColor = 'rgba(37, 99, 235, 0.1)';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '';
            uploadArea.style.backgroundColor = '';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '';
            uploadArea.style.backgroundColor = '';
            const files = Array.from(e.dataTransfer.files);
            this.handleFiles(files);
        });

        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            this.handleFiles(files);
        });

        // Detection
        document.getElementById('analyze-btn').addEventListener('click', () => this.analyzeMedia());

        // Demo data refresh
        setInterval(() => this.updateDashboardStats(), 5000);
    }

    switchView(viewName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
        document.getElementById(`${viewName}-btn`).classList.add('active');

        // Update views
        document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
        document.getElementById(`${viewName}-view`).classList.add('active');

        this.currentView = viewName;

        // Load view-specific data
        if (viewName === 'dashboard') {
            this.loadDashboardData();
        }
    }

    handleFiles(files) {
        this.uploadedFiles = files;
        const uploadArea = document.getElementById('upload-area');
        
        if (files.length > 0) {
            uploadArea.innerHTML = `
                <i class="fas fa-check-circle" style="color: #10b981; font-size: 48px; margin-bottom: 15px;"></i>
                <h3>${files.length} file(s) selected</h3>
                <p>${files.map(f => f.name).join(', ')}</p>
                <button id="clear-files" class="btn-secondary" style="margin-top: 15px;">
                    <i class="fas fa-times"></i> Clear Files
                </button>
            `;
            
            document.getElementById('clear-files').addEventListener('click', (e) => {
                e.stopPropagation();
                this.clearFiles();
            });
            
            this.showToast(`${files.length} file(s) ready for analysis`, 'success');
        }
    }

    clearFiles() {
        this.uploadedFiles = [];
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        uploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt upload-icon"></i>
            <h3>Drop files here or click to upload</h3>
            <p>Supports MP4, AVI, MOV videos and MP3, WAV audio files</p>
        `;
        
        fileInput.value = '';
    }

    async analyzeMedia() {
        if (this.uploadedFiles.length === 0) {
            this.showToast('Please select files to analyze', 'warning');
            return;
        }

        const analyzeBtn = document.getElementById('analyze-btn');
        const resultsSection = document.getElementById('results-section');
        const resultsContainer = document.getElementById('results-container');

        // Disable button and show loading
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

        try {
            // Simulate API call to backend
            const results = await this.simulateDetection(this.uploadedFiles);
            
            // Display results
            this.displayResults(results);
            resultsSection.classList.remove('hidden');
            
            this.showToast('Analysis completed successfully!', 'success');
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showToast('Analysis failed. Please try again.', 'error');
        } finally {
            // Re-enable button
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Media';
        }
    }

    async simulateDetection(files) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 2000));

        return files.map((file, index) => ({
            id: `result_${Date.now()}_${index}`,
            filename: file.name,
            fileType: file.type.startsWith('video/') ? 'video' : 'audio',
            isDeepfake: Math.random() > 0.7, // 30% chance of being a deepfake
            confidence: Math.floor(Math.random() * 40) + 60, // 60-99%
            processingTime: Math.floor(Math.random() * 100) + 50, // 50-150ms
            artifactsDetected: [
                'frame_consistency',
                'lighting_inconsistency',
                'edge_artifacts'
            ].filter(() => Math.random() > 0.5),
            timestamp: new Date().toISOString()
        }));
    }

    displayResults(results) {
        const container = document.getElementById('results-container');
        
        container.innerHTML = results.map(result => {
            const confidenceClass = result.confidence >= 90 ? 'high' : 
                                  result.confidence >= 75 ? 'medium' : 'low';
            
            return `
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">
                            <i class="fas ${result.fileType === 'video' ? 'fa-video' : 'fa-volume-up'}"></i>
                            ${result.filename}
                        </div>
                        <span class="confidence-badge confidence-${confidenceClass}">
                            ${result.isDeepfake ? '⚠️ Deepfake' : '✓ Authentic'} 
                            (${result.confidence}%)
                        </span>
                    </div>
                    <div class="result-details">
                        <p><strong>Processing Time:</strong> ${result.processingTime}ms</p>
                        <p><strong>Artifacts Detected:</strong> ${result.artifactsDetected.length > 0 ? 
                            result.artifactsDetected.join(', ') : 'None detected'}</p>
                        <p><strong>Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}</p>
                        ${result.isDeepfake ? 
                            '<p class="warning-text" style="color: #f59e0b; margin-top: 10px;">⚠️ This media has been identified as potentially manipulated</p>' : 
                            '<p class="success-text" style="color: #10b981; margin-top: 10px;">✓ This media appears authentic</p>'
                        }
                    </div>
                </div>
            `;
        }).join('');
    }

    async loadDashboardData() {
        try {
            // Simulate API calls
            const stats = await this.simulateStats();
            this.updateDashboardStats(stats);
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }

    async simulateStats() {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));
        
        return {
            totalDetections: Math.floor(Math.random() * 1000) + 500,
            deepfakesFound: Math.floor(Math.random() * 100) + 25,
            accuracyRate: Math.floor(Math.random() * 5) + 93,
            avgProcessing: Math.floor(Math.random() * 20) + 40
        };
    }

    updateDashboardStats(stats = null) {
        if (!stats) {
            // Generate random updates for demo
            stats = {
                totalDetections: parseInt(document.getElementById('total-detections').textContent) + Math.floor(Math.random() * 3),
                deepfakesFound: parseInt(document.getElementById('deepfakes-found').textContent) + (Math.random() > 0.7 ? 1 : 0),
                accuracyRate: 93 + Math.floor(Math.random() * 5),
                avgProcessing: 40 + Math.floor(Math.random() * 20)
            };
        }

        document.getElementById('total-detections').textContent = stats.totalDetections;
        document.getElementById('deepfakes-found').textContent = stats.deepfakesFound;
        document.getElementById('accuracy-rate').textContent = `${stats.accuracyRate}%`;
        document.getElementById('avg-processing').textContent = `${stats.avgProcessing}ms`;

        // Animate chart bars
        const bars = document.querySelectorAll('.chart-bar');
        bars.forEach(bar => {
            const newHeight = Math.floor(Math.random() * 80) + 20;
            bar.style.height = `${newHeight}%`;
        });
    }

    logout() {
        if (confirm('Are you sure you want to logout?')) {
            this.showToast('Logged out successfully', 'success');
            // In a real app, this would redirect to login page
            setTimeout(() => {
                location.reload();
            }, 1000);
        }
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas ${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                toastContainer.removeChild(toast);
            }, 300);
        }, 3000);
    }

    getToastIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        return icons[type] || icons.info;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.trueSightApp = new TrueSightApp();
});

// Add slideOut animation to CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);