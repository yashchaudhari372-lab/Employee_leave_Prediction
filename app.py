import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model loading logic
MODEL_PATH = "model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# Embedded Single-Page Dashboard Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction & Analytics Dashboard</title>
    
    <!-- Google Fonts & FontAwesome -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --panel-bg: rgba(30, 41, 59, 0.7);
            --panel-border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
            --accent-glow: rgba(99, 102, 241, 0.35);
            --card-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            --badge-bg: rgba(99, 102, 241, 0.2);
            --chart-grid: rgba(255, 255, 255, 0.05);
        }

        [data-theme="light"] {
            --bg-gradient: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 50%, #f3e8ff 100%);
            --panel-bg: rgba(255, 255, 255, 0.85);
            --panel-border: rgba(0, 0, 0, 0.08);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --input-bg: rgba(255, 255, 255, 0.9);
            --input-border: rgba(0, 0, 0, 0.12);
            --accent-glow: rgba(99, 102, 241, 0.15);
            --card-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            --badge-bg: rgba(99, 102, 241, 0.1);
            --chart-grid: rgba(0, 0, 0, 0.05);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
            overflow-x: hidden;
        }

        /* Glassmorphism Containers */
        .glass-panel {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            box-shadow: var(--card-shadow);
        }

        /* Animated Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            animation: slideDown 0.8s ease-out;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .brand-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: white;
            box-shadow: 0 0 20px var(--accent-glow);
            animation: pulseGlow 3s infinite alternate;
        }

        .brand-text h1 {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text p {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Theme Toggle */
        .theme-toggle {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            color: var(--text-main);
            padding: 0.6rem 1.2rem;
            border-radius: 50px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 0.9rem;
        }

        .theme-toggle:hover {
            transform: translateY(-2px);
        }

        /* Main Grid Layout */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
            font-size: 1.2rem;
            font-weight: 700;
        }

        /* Colorful Parameter Grid */
        .param-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .input-group {
            position: relative;
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid var(--panel-border);
            background: rgba(255, 255, 255, 0.02);
            transition: all 0.3s ease;
        }

        .input-group:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.2);
        }

        /* Color accents for parameters */
        .input-group:nth-child(1) { border-left: 4px solid #3b82f6; }
        .input-group:nth-child(2) { border-left: 4px solid #10b981; }
        .input-group:nth-child(3) { border-left: 4px solid #f59e0b; }
        .input-group:nth-child(4) { border-left: 4px solid #ec4899; }
        .input-group:nth-child(5) { border-left: 4px solid #8b5cf6; }
        .input-group:nth-child(6) { border-left: 4px solid #06b6d4; }
        .input-group:nth-child(7) { border-left: 4px solid #84cc16; }
        .input-group:nth-child(8) { border-left: 4px solid #f97316; }

        .input-group label {
            display: block;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: #818cf8;
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        /* Submit Button */
        .btn-submit {
            grid-column: span 2;
            margin-top: 1rem;
            padding: 1rem;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            color: white;
            font-weight: 700;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .btn-submit:hover {
            opacity: 0.95;
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.6);
        }

        /* Analytics Panel */
        .analytics-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .prediction-badge {
            text-align: center;
            padding: 1.5rem;
            border-radius: 16px;
            background: var(--badge-bg);
            border: 1px dashed var(--panel-border);
            animation: fadeIn 0.5s ease-in;
        }

        .prediction-badge h2 {
            font-size: 2.2rem;
            font-weight: 800;
            margin-top: 0.5rem;
        }

        .chart-card {
            position: relative;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(0, 0, 0, 0.1);
            height: 320px;
        }

        /* Animations */
        @keyframes slideDown {
            from { transform: translateY(-30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        @keyframes pulseGlow {
            0% { box-shadow: 0 0 15px rgba(99, 102, 241, 0.4); }
            100% { box-shadow: 0 0 30px rgba(236, 72, 153, 0.7); }
        }
    </style>
</head>
<body>

    <header class="glass-panel">
        <div class="brand">
            <div class="brand-icon">
                <i class="fa-solid fa-brain"></i>
            </div>
            <div class="brand-text">
                <h1>AI Prediction Suite</h1>
                <p>Logistic Regression Real-time Analytics</p>
            </div>
        </div>
        <button class="theme-toggle" id="themeBtn" onclick="toggleTheme()">
            <i class="fa-solid fa-moon"></i> Dark Mode
        </button>
    </header>

    <main class="dashboard-grid">
        <!-- Input Form Panel -->
        <section class="glass-panel" style="padding: 2rem;">
            <div class="section-header">
                <i class="fa-solid fa-sliders" style="color: #818cf8;"></i>
                <span>Model Parameters</span>
            </div>

            <form id="predictionForm" onsubmit="submitForm(event)">
                <div class="param-grid">
                    <div class="input-group">
                        <label>Education</label>
                        <select id="Education" required>
                            <option value="Bachelors">Bachelors</option>
                            <option value="Masters">Masters</option>
                            <option value="PHD">PHD</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Joining Year</label>
                        <input type="number" id="JoiningYear" value="2017" min="2000" max="2030" required>
                    </div>

                    <div class="input-group">
                        <label>City</label>
                        <select id="City" required>
                            <option value="Bangalore">Bangalore</option>
                            <option value="Pune">Pune</option>
                            <option value="New Delhi">New Delhi</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Payment Tier</label>
                        <select id="PaymentTier" required>
                            <option value="1">Tier 1</option>
                            <option value="2">Tier 2</option>
                            <option value="3" selected>Tier 3</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Age</label>
                        <input type="number" id="Age" value="28" min="18" max="100" required>
                    </div>

                    <div class="input-group">
                        <label>Gender</label>
                        <select id="Gender" required>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Ever Benched</label>
                        <select id="EverBenched" required>
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>

                    <div class="input-group">
                        <label>Domain Exp (Years)</label>
                        <input type="number" id="ExperienceInCurrentDomain" value="3" min="0" max="30" required>
                    </div>

                    <button type="submit" class="btn-submit">
                        <i class="fa-solid fa-bolt"></i> Run Prediction
                    </button>
                </div>
            </form>
        </section>

        <!-- Output & Visualizations Panel -->
        <section class="glass-panel" style="padding: 2rem;">
            <div class="section-header">
                <i class="fa-solid fa-chart-pie" style="color: #ec4899;"></i>
                <span>Analytics Dashboard</span>
            </div>

            <div class="analytics-container">
                <div class="prediction-badge">
                    <span style="font-weight: 600; color: var(--text-muted); font-size: 0.9rem;">CURRENT PREDICTION</span>
                    <h2 id="predictionResult" style="color: #818cf8;">--</h2>
                </div>

                <div class="chart-card">
                    <canvas id="probabilityChart"></canvas>
                </div>
            </div>
        </section>
    </main>

    <script>
        // Dark/Light Theme Switching Logic
        function toggleTheme() {
            const html = document.documentElement;
            const themeBtn = document.getElementById('themeBtn');
            const isDark = html.getAttribute('data-theme') === 'dark';
            
            if (isDark) {
                html.setAttribute('data-theme', 'light');
                themeBtn.innerHTML = '<i class="fa-solid fa-sun"></i> Light Mode';
            } else {
                html.setAttribute('data-theme', 'dark');
                themeBtn.innerHTML = '<i class="fa-solid fa-moon"></i> Dark Mode';
            }
            updateChartTheme();
        }

        // Initialize Dynamic Chart.js Visualization
        let chart;
        function initChart() {
            const ctx = document.getElementById('probabilityChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Class 0 (Negative)', 'Class 1 (Positive)'],
                    datasets: [{
                        label: 'Probability Confidence',
                        data: [0.5, 0.5],
                        backgroundColor: [
                            'rgba(236, 72, 153, 0.7)',
                            'rgba(99, 102, 241, 0.7)'
                        ],
                        borderColor: [
                            '#ec4899',
                            '#6366f1'
                        ],
                        borderWidth: 2,
                        borderRadius: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: 'Model Confidence Output Spectrum',
                            color: '#94a3b8'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' }
                        },
                        x: {
                            grid: { display: false }
                        }
                    }
                }
            });
        }

        function updateChartTheme() {
            if(!chart) return;
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            const color = isDark ? '#94a3b8' : '#64748b';
            chart.options.plugins.title.color = color;
            chart.update();
        }

        // Form Submit Handler
        async function submitForm(e) {
            e.preventDefault();
            
            const payload = {
                Education: document.getElementById('Education').value,
                JoiningYear: parseInt(document.getElementById('JoiningYear').value),
                City: document.getElementById('City').value,
                PaymentTier: parseInt(document.getElementById('PaymentTier').value),
                Age: parseInt(document.getElementById('Age').value),
                Gender: document.getElementById('Gender').value,
                EverBenched: document.getElementById('EverBenched').value,
                ExperienceInCurrentDomain: parseInt(document.getElementById('ExperienceInCurrentDomain').value)
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await res.json();
                
                if(data.status === 'success') {
                    // Update UI elements
                    document.getElementById('predictionResult').innerText = "Class " + data.prediction;
                    
                    // Update Chart Data
                    chart.data.datasets[0].data = data.probabilities;
                    chart.update();
                } else {
                    alert('Prediction Error: ' + data.error);
                }
            } catch(err) {
                console.error(err);
                alert('Connection to Flask backend failed!');
            }
        }

        window.onload = () => {
            initChart();
        };
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Convert incoming JSON payload into DataFrame with proper schema
        df = pd.DataFrame([data])
        
        if model is None:
            return jsonify({
                "status": "error",
                "error": "Model file 'model.pkl' was not found on the server."
            }), 500

        # Execute Prediction
        prediction = model.predict(df)[0]
        
        # Calculate class probabilities (if supported by LogisticRegression)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[0].tolist()
        else:
            probabilities = [0.5, 0.5]

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
