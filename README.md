# Cyber-Threat-Analysis

An advanced AI-powered web application that analyzes, predicts, and visualizes cyber threats in real time using Machine Learning, Deep Learning, and Anomaly Detection.

🚀 Features

📊 Interactive Dashboard: Real-time analytics of web traffic and threat metrics

🔍 Threat Analysis: Detect suspicious IPs, ports, and anomaly patterns

🤖 Machine Learning Models: Gradient Boosting, Random Forest, and Neural Network models for classification and scoring

🛑 Auto Blocklist: Automatically detects and adds malicious IPs to blocklist

🌍 Geographical Visualization: Global cyber threat heatmap using Plotly

🔐 Security Insights: Displays cybersecurity best practices and risk assessment

🧩 Tech Stack

Frontend: Streamlit

Backend / ML: Python, Scikit-learn, TensorFlow, Keras

Visualization: Seaborn, Matplotlib, Plotly

Model Persistence: Joblib

Data Handling: Pandas, NumPy

🛠️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/Sanketborhade33/Cyber-Threat-Analysis.git
cd Cyber-Threat-Analysis

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate     # on Windows
# or
source venv/bin/activate  # on Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py


Then open the app in your browser:
👉 http://localhost:8501

📦 Project Structure
Cyber-Threat-Analysis/
│
├── app.py                   # Main Streamlit app
├── data/
│   └── sample_data.csv       # Sample dataset
├── models/
│   ├── scaler.pkl
│   ├── attack_classifier.pkl
│   └── threat_score_regressor.pkl
├── blocklist.txt             # Auto-generated suspicious IPs
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation

⚙️ Deployment (Render / Streamlit Cloud)
Render

Service Type: Web Service

Build Command:

pip install -r requirements.txt