# Employee Leave Prediction | Employee Attrition Intelligence

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Logistic%20Regression-orange?logo=scikitlearn) ![Vercel](https://img.shields.io/badge/Deployment-Vercel-black?logo=vercel)

**An interactive employee retention and attrition prediction application powered by Logistic Regression.**

**[🚀 Live Demo](https://employee-leave-prediction-ebon.vercel.app/)** · **[💻 Source Code](https://github.com/yashchaudhari372-lab/Employee_leave_Prediction)**

## 📌 Project Overview

Employee attrition can disrupt workforce planning, increase recruitment costs, and affect team productivity. This project demonstrates how a machine learning classification model can support exploratory HR decision-making by estimating whether an employee is likely to leave or stay.

The application combines a saved **Scikit-learn Logistic Regression** model with a **Flask** backend and a responsive dashboard featuring **Tailwind CSS** and **Chart.js**. The interface is titled **Employee Attrition Intelligence**.

> **Disclaimer:** Predictions are estimates for educational and analytical purposes. They should not be used as the sole basis for employment, promotion, or termination decisions.

## ✨ Key Features

- **Employee attrition prediction:** Submit employee details and receive a model-generated classification when the saved model is available.
- **Retention analytics interface:** Explore results through a modern, dashboard-style application.
- **Interactive visualizations:** Chart.js supports visual presentation of prediction information.
- **Responsive design:** Tailwind CSS-based UI with dark-mode styling.
- **Saved ML model:** The Flask application loads the trained classifier from `lr_model.pkl`.
- **Cloud deployment:** Publicly accessible Vercel application.

## 🧰 Technology Stack

| Layer | Technologies |
|---|---|
| Programming | Python |
| Machine learning | Logistic Regression, Scikit-learn |
| Numerical processing | NumPy |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript, Tailwind CSS |
| Visualization | Chart.js |
| Model persistence | Python Pickle (`.pkl`) |
| Deployment | Vercel |

## 🧠 Machine Learning Approach

The system uses **Logistic Regression**, a supervised learning algorithm for binary classification. It learns a relationship between employee-related input features and an attrition outcome. For a new employee profile, the classifier generates a prediction based on the model learned during training.

### Prediction Workflow

```text
Employee information
        ↓
Web form / Flask request
        ↓
Input validation and feature preparation
        ↓
Saved Logistic Regression model (lr_model.pkl)
        ↓
Employee attrition prediction
        ↓
Dashboard result / visualization
```

**Important:** Input features must follow the same encoding, order, and preprocessing used during model training. See `app.py` for the application's current input mapping.

## 📊 Model Evaluation

The repository includes a serialized trained model, but publicly verifiable test-set metrics are not provided in this README. Report results only after evaluating the model on held-out data.

| Evaluation Metric | Result |
|---|---|
| Accuracy | 67.0% |
| Precision | 60.0% |
| Recall | 36.0% |
| F1-score | 45.0% |
| ROC-AUC | 61.0% |

**Suggested evaluation:** Use a confusion matrix and classification report, and pay particular attention to recall for employees who actually leave. Check class balance and compare against a majority-class baseline.

## 📂 Repository Structure

```text
Employee_leave_Prediction/
├── app.py              # Flask app, UI, and prediction handling
├── lr_model.pkl        # Serialized Logistic Regression model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## ⚙️ Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashchaudhari372-lab/Employee_leave_Prediction.git
cd Employee_leave_Prediction
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask application

```bash
python app.py
```

Open the local address printed in your terminal (commonly `http://127.0.0.1:5000`). Ensure that `lr_model.pkl` is present in the project directory.

## 🌐 Live Application

**[Launch Employee Leave Prediction](https://employee-leave-prediction-ebon.vercel.app/)**

> The repository also lists `https://elp-xi.vercel.app/` in its About section. The demo link above is the one documented in the repository README; check both URLs and retain your preferred working deployment.

## 💼 Business Use Cases

- Explore employee attrition patterns and workforce retention signals.
- Support HR analytics and workforce planning discussions.
- Demonstrate end-to-end deployment of a classification model.
- Help prioritize further investigation into employee experience and retention.

Model predictions should be reviewed for fairness, privacy, and possible bias before any real-world HR use.

## 🚀 Future Enhancements

- Publish a reproducible training notebook and dataset documentation.
- Add verified accuracy, precision, recall, F1-score, and ROC-AUC results.
- Display a confusion matrix and feature-level explanations.
- Add stronger input validation, error handling, and automated tests.
- Assess model calibration, drift, fairness, and data privacy.
- Include application screenshots and a short demonstration video.

## 👨‍💻 Author

**Yash Chaudhari**  
Aspiring Data Scientist & Data Analyst

- **GitHub:** https://github.com/yashchaudhari372-lab
- **LinkedIn:** https://www.linkedin.com/in/yashchaudhari08/

---

⭐ If you find this project useful, consider starring the repository.
