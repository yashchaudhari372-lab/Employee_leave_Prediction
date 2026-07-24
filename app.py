import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained Scikit-Learn Logistic Regression model
MODEL_PATH = "lr_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML/CSS/JS Template string with Attrition Output Section
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Retention & Attrition Predictor</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        brand: {
                            50: '#eef2ff',
                            500: '#6366f1',
                            600: '#4f46e5',
                            700: '#4338ca',
                        }
                    },
                    animation: {
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        'float': 'float 6s ease-in-out infinite',
                    },
                    keyframes: {
                        float: {
                            '0%, 100%': { transform: 'translateY(0px)' },
                            '50%': { transform: 'translateY(-10px)' },
                        }
                    }
                }
            }
        }
    </script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .glass {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        .dark .glass {
            background: rgba(17, 24, 39, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .glow-effect {
            box-shadow: 0 0 25px -5px rgba(99, 102, 241, 0.4);
        }
        .gradient-text {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: rgba(156, 163, 175, 0.4);
            border-radius: 10px;
        }
    </style>
</head>
<body class="bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition-colors duration-300 min-h-screen custom-scrollbar font-sans">

    <!-- Top Navigation Header -->
    <header class="sticky top-0 z-50 glass border-b border-slate-200 dark:border-slate-800 px-6 py-4">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center text-white shadow-lg animate-float">
                    <i class="fa-solid fa-users-gear text-xl"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold gradient-text tracking-wide">Employee Attrition Intelligence</h1>
                    <p class="text-xs text-slate-500 dark:text-slate-400">Logistic Regression Retention Analytics</p>
                </div>
            </div>

            <div class="flex items-center gap-4">
                <!-- Dark Mode Toggle Button -->
                <button id="themeToggle" class="p-2.5 rounded-xl bg-slate-200/80 dark:bg-slate-800/80 hover:bg-slate-300 dark:hover:bg-slate-700 transition-all duration-200 text-slate-700 dark:text-amber-400 shadow-sm">
                    <i id="themeIcon" class="fa-solid fa-moon text-lg"></i>
                </button>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <!-- Left Column: Interactive Feature Form -->
        <section class="lg:col-span-5 flex flex-col gap-6">
            <div class="glass p-6 rounded-2xl shadow-xl hover:shadow-2xl transition-all border border-slate-200/50 dark:border-slate-800">
                <div class="flex items-center gap-3 mb-6 pb-3 border-b border-slate-200 dark:border-slate-800">
                    <i class="fa-solid fa-user-pen text-indigo-500 text-lg"></i>
                    <h2 class="text-lg font-semibold">Employee Details</h2>
                </div>

                <form id="predictionForm" class="space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Education</label>
                            <select id="Education" name="Education" class="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                                <option value="0">Bachelors (0)</option>
                                <option value="1">Masters (1)</option>
                                <option value="2">PHD (2)</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Joining Year</label>
                            <input type="number" id="JoiningYear" name="JoiningYear" value="2018" min="2010" max="2026" class="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">City</label>
                            <select id="City" name="City" class="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                                <option value="0">Bangalore (0)</option>
                                <option value="1">Pune (1)</option>
                                <option value="2">New Delhi (2)</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Payment Tier</label>
                            <select id="PaymentTier" name="PaymentTier" class="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                                <option value="1">Tier 1</option>
                                <option value="2">Tier 2</option>
                                <option value="3" selected>Tier 3</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Age</label>
                            <input type="number" id="Age" name="Age" value="28" min="18" max="70" class="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Gender</label>
                            <select id="Gender" name="Gender" class="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                                <option value="0">Male (0)</option>
                                <option value="1">Female (1)</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Ever Benched</label>
                            <select id="EverBenched" name="EverBenched" class="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                                <option value="0">No (0)</option>
                                <option value="1">Yes (1)</option>
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Domain Experience (Yrs)</label>
                            <input type="number" id="ExperienceInCurrentDomain" name="ExperienceInCurrentDomain" value="3" min="0" max="25" class="w-full bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-xl px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all">
                        </div>
                    </div>

                    <button type="submit" class="w-full mt-4 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:opacity-90 text-white font-semibold py-3 rounded-xl shadow-lg transition-all duration-300 flex items-center justify-center gap-2 transform active:scale-95 glow-effect">
                        <i class="fa-solid fa-wand-magic-sparkles"></i>
                        Predict Retention Status
                    </button>
                </form>
            </div>
        </section>

        <!-- Right Column: Visual Dashboard & Attrition Output Section -->
        <section class="lg:col-span-7 flex flex-col gap-6">
            
            <!-- Dynamic Attrition Prediction Result Card -->
            <div id="resultCard" class="glass p-6 rounded-2xl border border-slate-200/50 dark:border-slate-800 transition-all duration-300">
                <div class="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3 mb-4">
                    <span class="text-xs uppercase tracking-wider font-semibold text-slate-500 dark:text-slate-400">Prediction Analysis</span>
                    <span id="riskBadge" class="px-3 py-1 rounded-full text-xs font-semibold bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                        Awaiting Inputs
                    </span>
                </div>

                <div class="flex flex-col md:flex-row items-center justify-between gap-6">
                    <div class="flex items-center gap-4">
                        <div id="statusIconBg" class="w-16 h-16 rounded-2xl bg-slate-200/50 dark:bg-slate-800/50 text-slate-500 flex items-center justify-center text-3xl shadow-inner transition-all">
                            <i id="statusIcon" class="fa-solid fa-user-clock"></i>
                        </div>
                        <div>
                            <h2 id="predictionText" class="text-2xl font-bold">Will Employee Stay or Leave?</h2>
                            <p id="predictionSubtext" class="text-xs text-slate-500 dark:text-slate-400 mt-1">Submit employee details to compute retention likelihood.</p>
                        </div>
                    </div>

                    <div class="flex flex-col items-center md:items-end border-t md:border-t-0 md:border-l border-slate-200 dark:border-slate-800 pt-4 md:pt-0 md:pl-6 w-full md:w-auto">
                        <span class="text-xs text-slate-500 dark:text-slate-400 font-medium">Confidence Score</span>
                        <span id="confidenceValue" class="text-3xl font-extrabold gradient-text">--%</span>
                    </div>
                </div>
            </div>

            <!-- Charts Container Grid -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Retention vs Attrition Probability Chart -->
                <div class="glass p-5 rounded-2xl border border-slate-200/50 dark:border-slate-800 flex flex-col justify-between">
                    <h3 class="text-sm font-semibold text-slate-600 dark:text-slate-300 mb-2 flex items-center gap-2">
                        <i class="fa-solid fa-chart-pie text-pink-500"></i> Probability Breakdown
                    </h3>
                    <div class="relative h-48 w-full flex items-center justify-center">
                        <canvas id="probabilityChart"></canvas>
                    </div>
                </div>

                <!-- Input Parameters Overview Bar Chart -->
                <div class="glass p-5 rounded-2xl border border-slate-200/50 dark:border-slate-800 flex flex-col justify-between">
                    <h3 class="text-sm font-semibold text-slate-600 dark:text-slate-300 mb-2 flex items-center gap-2">
                        <i class="fa-solid fa-chart-simple text-indigo-500"></i> Parameter Values
                    </h3>
                    <div class="relative h-48 w-full">
                        <canvas id="featureChart"></canvas>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <!-- Client Scripting -->
    <script>
        // Dark Mode Logic
        const themeToggleBtn = document.getElementById('themeToggle');
        const themeIcon = document.getElementById('themeIcon');
        
        themeToggleBtn.addEventListener('click', () => {
            const isDark = document.documentElement.classList.toggle('dark');
            themeIcon.className = isDark ? 'fa-solid fa-moon text-lg' : 'fa-solid fa-sun text-lg text-amber-500';
            updateChartThemes(isDark);
        });

        // Chart.js Instances
        let probChart, featChart;

        function initCharts() {
            const ctxProb = document.getElementById('probabilityChart').getContext('2d');
            probChart = new Chart(ctxProb, {
                type: 'doughnut',
                data: {
                    labels: ['Will Stay', 'Will Leave'],
                    datasets: [{
                        data: [0.5, 0.5],
                        backgroundColor: ['#10b981', '#f43f5e'],
                        borderWidth: 0,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: '#94a3b8' } }
                    },
                    cutout: '70%'
                }
            });

            const ctxFeat = document.getElementById('featureChart').getContext('2d');
            featChart = new Chart(ctxFeat, {
                type: 'bar',
                data: {
                    labels: ['Edu', 'Year', 'City', 'Tier', 'Age', 'Gender', 'Bench', 'Exp'],
                    datasets: [{
                        label: 'Value',
                        data: [0, 2018, 0, 3, 28, 0, 0, 3],
                        backgroundColor: '#8b5cf6',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        function updateChartThemes(isDark) {
            const textColor = isDark ? '#94a3b8' : '#475569';
            if(probChart) {
                probChart.options.plugins.legend.labels.color = textColor;
                probChart.update();
            }
            if(featChart) {
                featChart.options.scales.x.ticks.color = textColor;
                featChart.options.scales.y.ticks.color = textColor;
                featChart.update();
            }
        }

        // Form Submit Handler
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {};
            formData.forEach((val, key) => data[key] = parseFloat(val));

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const res = await response.json();

                if (res.success) {
                    const statusIconBg = document.getElementById('statusIconBg');
                    const statusIcon = document.getElementById('statusIcon');
                    const predictionText = document.getElementById('predictionText');
                    const predictionSubtext = document.getElementById('predictionSubtext');
                    const riskBadge = document.getElementById('riskBadge');
                    const confidenceValue = document.getElementById('confidenceValue');

                    const stayProb = (res.probabilities[0] * 100).toFixed(1);
                    const leaveProb = (res.probabilities[1] * 100).toFixed(1);

                    if (res.prediction === 0) {
                        // Employee WILL STAY
                        predictionText.innerText = "Employee Will STAY";
                        predictionText.className = "text-2xl font-bold text-emerald-500 dark:text-emerald-400";
                        predictionSubtext.innerText = "High retention likelihood based on parameters.";
                        
                        statusIconBg.className = "w-16 h-16 rounded-2xl bg-emerald-500/10 text-emerald-500 flex items-center justify-center text-3xl shadow-inner";
                        statusIcon.className = "fa-solid fa-user-check";

                        riskBadge.innerText = "Low Risk / Retained";
                        riskBadge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-500 border border-emerald-500/20";
                        
                        confidenceValue.innerText = `${stayProb}%`;
                    } else {
                        // Employee WILL LEAVE
                        predictionText.innerText = "Employee Will LEAVE";
                        predictionText.className = "text-2xl font-bold text-rose-500 dark:text-rose-400";
                        predictionSubtext.innerText = "High attrition risk detected. Consider engagement actions.";
                        
                        statusIconBg.className = "w-16 h-16 rounded-2xl bg-rose-500/10 text-rose-500 flex items-center justify-center text-3xl shadow-inner";
                        statusIcon.className = "fa-solid fa-user-xmark";

                        riskBadge.innerText = "High Attrition Risk";
                        riskBadge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/10 text-rose-500 border border-rose-500/20";

                        confidenceValue.innerText = `${leaveProb}%`;
                    }
                    
                    // Update Charts
                    probChart.data.datasets[0].data = res.probabilities;
                    probChart.update();

                    featChart.data.datasets[0].data = Object.values(data);
                    featChart.update();
                } else {
                    alert('Prediction Error: ' + res.error);
                }
            } catch (err) {
                console.error(err);
            }
        });

        // Initialize on DOM Ready
        window.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model file lr_model.pkl not found on server.'})

    try:
        data = request.get_json(force=True)
        # Sequence of variables strictly mapped according to model feature names
        feature_order = [
            'Education', 'JoiningYear', 'City', 'PaymentTier',
            'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
        ]
        
        input_features = [float(data.get(feat, 0)) for feat in feature_order]
        features_array = np.array([input_features])

        prediction = int(model.predict(features_array)[0])
        probabilities = model.predict_proba(features_array)[0].tolist()

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
