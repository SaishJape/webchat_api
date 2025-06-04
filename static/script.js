// Constants
const API_BASE_URL = 'http://localhost:8000/api';
const TOAST_DURATION = 3000;

// State management
const state = {
    currentUser: null,
    currentTaskId: null,
    isCompleted: false,
    isProcessing: false
};

// Toast notification system
const Toast = {
    show(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type} animate-fade-in`;
        toast.innerHTML = `
            <span class="toast-icon">${type === 'success' ? '✓' : '✕'}</span>
            <span class="toast-message">${message}</span>
        `;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, TOAST_DURATION);
    }
};

// UI Components
const UI = {
    showLoading() {
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        document.body.appendChild(spinner);
    },
    
    hideLoading() {
        const spinner = document.querySelector('.spinner');
        if (spinner) spinner.remove();
    },
    
    updateProgress(status) {
        const steps = {
            'crawling': 1,
            'processing': 2,
            'generating_embeddings': 3,
            'storing': 4,
            'completed': 5
        };

        const messages = {
            'crawling': 'Crawling website...',
            'processing': 'Processing content...',
            'generating_embeddings': 'Generating embeddings...',
            'storing': 'Storing data...',
            'completed': 'Processing completed successfully!'
        };

        // Update steps
        for (let i = 1; i <= 5; i++) {
            const step = document.getElementById(`step${i}`);
            if (i < steps[status]) {
                step.className = 'step step-completed';
            } else if (i === steps[status]) {
                step.className = 'step step-active';
            } else {
                step.className = 'step step-pending';
            }
        }

        // Update message
        const messageEl = document.getElementById('progressMessage');
        messageEl.textContent = messages[status] || '';
        messageEl.className = 'animate-fade-in';
    },
    
    showChatWidgetCode(collectionName) {
        const chatWidgetCode = document.getElementById('chatWidgetCode');
        const iframeCode = document.getElementById('iframeCode');
        
        // Format the iframe code with syntax highlighting
        const code = `<span class="tag">&lt;iframe</span>
  <span class="attr">src</span>=<span class="string">"http://localhost:8080?collection_name=${collectionName}"</span>
  <span class="attr">width</span>=<span class="string">"500"</span>
  <span class="attr">height</span>=<span class="string">"800"</span>
  <span class="attr">style</span>=<span class="string">"background: transparent; border: none; position: fixed; bottom: 20px; right: 20px; z-index: 9999;"</span>
  <span class="attr">allowtransparency</span>=<span class="string">"true"</span>
  <span class="attr">title</span>=<span class="string">"My Chatbot"</span>&gt;
<span class="tag">&lt;/iframe&gt;</span>`;
        
        iframeCode.innerHTML = code;
        chatWidgetCode.classList.remove('hidden');
        chatWidgetCode.classList.add('animate-fade-in');
    }
};

// Authentication handlers
const Auth = {
    async login(email, password) {
        try {
            const response = await fetch(`${API_BASE_URL}/login/json`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            if (!response.ok) throw new Error('Login failed');

            const data = await response.json();
            localStorage.setItem('token', data.access_token);
            state.currentUser = { email };
            return true;
        } catch (error) {
            Toast.show(error.message, 'error');
            return false;
        }
    },

    async signup(email, username, password) {
        try {
            const response = await fetch(`${API_BASE_URL}/signup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, username, password })
            });

            if (!response.ok) throw new Error('Signup failed');

            Toast.show('Signup successful! Please login.');
            return true;
        } catch (error) {
            Toast.show(error.message, 'error');
            return false;
        }
    },

    logout() {
        // Clear user data
        localStorage.removeItem('token');
        state.currentUser = null;
        state.currentTaskId = null;
        state.isCompleted = false;
        state.isProcessing = false;

        // Show auth container and hide main content
        document.getElementById('authContainer').classList.remove('hidden');
        document.getElementById('mainContent').classList.add('hidden');

        // Reset forms
        document.getElementById('loginForm').reset();
        document.getElementById('signupForm').reset();
        document.getElementById('scrapingForm').reset();

        // Show success message
        Toast.show('Logged out successfully!');
    }
};

// Processing handlers
const Processing = {
    async handleScraping(event) {
        event.preventDefault();
        const url = document.getElementById('url').value;
        const pdfFile = document.getElementById('pdfFile').files[0];
        
        if (state.isProcessing) return;
        state.isProcessing = true;
        UI.showLoading();

        try {
            if (pdfFile) {
                await this.handlePdfUpload(pdfFile);
            } else if (url) {
                await this.handleUrlScraping(url);
            } else {
                throw new Error('Please provide either a URL or upload a PDF file');
            }
        } catch (error) {
            Toast.show(error.message, 'error');
        } finally {
            state.isProcessing = false;
            UI.hideLoading();
        }
    },

    async handlePdfUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/upload-and-process`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` },
            body: formData
        });

        if (!response.ok) throw new Error('Failed to process PDF');

        const data = await response.json();
        UI.showChatWidgetCode(data.collection_name);
        Toast.show('PDF processed successfully!');
    },

    async handleUrlScraping(url) {
        const response = await fetch(`${API_BASE_URL}/scrape-and-ingest`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({ url })
        });

        if (!response.ok) throw new Error('Failed to start scraping');

        const data = await response.json();
        state.currentTaskId = data.task_id;
        this.startProgressCheck();
    },

    async checkProgress() {
        if (!state.currentTaskId || state.isCompleted) return;

        try {
            const response = await fetch(`${API_BASE_URL}/scraping-progress/${state.currentTaskId}`, {
                headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
            });

            if (!response.ok) throw new Error('Failed to check progress');

            const data = await response.json();
            UI.updateProgress(data.status);

            if (data.status === 'completed') {
                state.isCompleted = true;
                UI.showChatWidgetCode(data.result.collection_name);
                Toast.show('Processing completed successfully!');
            } else if (data.error) {
                throw new Error(data.error);
            } else {
                setTimeout(() => this.checkProgress(), 2000);
            }
        } catch (error) {
            Toast.show(error.message, 'error');
            state.isCompleted = true;
        }
    },

    startProgressCheck() {
        this.checkProgress();
    }
};

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Auth form handlers
    document.getElementById('loginForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('loginEmail').value;
        const password = document.getElementById('loginPassword').value;
        
        if (await Auth.login(email, password)) {
            document.getElementById('authContainer').classList.add('hidden');
            document.getElementById('mainContent').classList.remove('hidden');
        }
    });

    document.getElementById('signupForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('signupEmail').value;
        const username = document.getElementById('signupUsername').value;
        const password = document.getElementById('signupPassword').value;
        
        if (await Auth.signup(email, username, password)) {
            toggleAuthForms();
        }
    });

    // Logout button handler
    document.getElementById('logoutButton')?.addEventListener('click', () => {
        Auth.logout();
    });

    // Scraping form handler
    document.getElementById('scrapingForm')?.addEventListener('submit', (e) => {
        Processing.handleScraping(e);
    });

    // Copy code button handler
    document.getElementById('copyButton')?.addEventListener('click', () => {
        const code = document.getElementById('iframeCode').textContent;
        navigator.clipboard.writeText(code).then(() => {
            Toast.show('Code copied to clipboard!');
        }).catch(() => {
            Toast.show('Failed to copy code', 'error');
        });
    });

    // Check if user is logged in
    const token = localStorage.getItem('token');
    if (token) {
        document.getElementById('authContainer').classList.add('hidden');
        document.getElementById('mainContent').classList.remove('hidden');
    }
});

// Utility functions
function toggleAuthForms() {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');
    loginForm.classList.toggle('hidden');
    signupForm.classList.toggle('hidden');
} 