

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, classification_report


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"




import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


st.set_page_config(page_title="Cyber Threat Analyzer", layout="wide")


DATA_PATH = "data/sample_data.csv"
SCALER_PATH = "models/scaler.pkl"
REGRESSOR_PATH = "models/threat_score_regressor.pkl"
CLASSIFIER_PATH = "models/attack_classifier.pkl"


os.makedirs("models", exist_ok=True)


if os.path.exists(DATA_PATH):
    data = pd.read_csv(DATA_PATH)
else:
    st.error("⚠️ Dataset not found! Please upload a CSV file.")
    st.stop()







from sklearn.ensemble import IsolationForest

# Check if 'anomaly_score' exists in the dataset
if 'anomaly_score' not in data.columns:
    # st.warning("🚨 `anomaly_score` not found! Running anomaly detection model...")

    # Select numerical features for anomaly detection
    features = ['bytes_in', 'bytes_out', 'duration_seconds']
    
    # Ensure the required features exist in the dataset
    if all(col in data.columns for col in features):
        # Apply Isolation Forest for anomaly detection
        anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        data['anomaly_score'] = anomaly_detector.fit_predict(data[features])

        # Show success message
        st.success("✅ Anomaly detection completed!")
    else:
        st.error("❌ Required columns missing for anomaly detection: 'bytes_in', 'bytes_out', 'duration_seconds'")




if 'duration_seconds' not in data.columns:
    data['creation_time'] = pd.to_datetime(data['creation_time'], errors='coerce')
    data['end_time'] = pd.to_datetime(data['end_time'], errors='coerce')
    data['duration_seconds'] = (data['end_time'] - data['creation_time']).dt.total_seconds().fillna(0)
    data.to_csv(DATA_PATH, index=False)

if 'threat_score' not in data.columns:
    def calculate_threat_score(row):
        score = 0
        if row['bytes_in'] > 50000:
            score += 5
        elif row['bytes_in'] > 30000:
            score += 3

        if row['bytes_out'] > 25000:
            score += 4
        elif row['bytes_out'] > 20000:
            score += 2

        if row['duration_seconds'] > 1200:
            score += 5
        elif row['duration_seconds'] > 600:
            score += 3

        if row.get('src_ip_country_code', '') in ['RU', 'CN', 'KP']:
            score += 3

        return min(score, 10)

    data['threat_score'] = data.apply(calculate_threat_score, axis=1)
    data.to_csv(DATA_PATH, index=False)


scaler = StandardScaler()
X = data[['bytes_in', 'bytes_out', 'duration_seconds']]
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['threat_score'], test_size=0.3, random_state=42)


regressor = GradientBoostingRegressor(n_estimators=700, max_depth=10, learning_rate=0.1, random_state=42)
regressor.fit(X_train, y_train)
joblib.dump(regressor, REGRESSOR_PATH)


y_class = (data['threat_score'] >= 5).astype(int)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.3, random_state=42)
classifier = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
classifier.fit(X_train_class, y_train_class)
joblib.dump(classifier, CLASSIFIER_PATH)

st.sidebar.success("✅ Models trained and saved successfully!")



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


X_nn = data[['bytes_in', 'bytes_out', 'duration_seconds']]  # Ensure these columns exist
y_nn = (data['threat_score'] > 5).astype(int)  # Convert threat scores to binary labels


X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y_nn, test_size=0.3, random_state=42)


scaler_nn = StandardScaler()
X_train_nn = scaler_nn.fit_transform(X_train_nn)
X_test_nn = scaler_nn.transform(X_test_nn)


model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_nn.shape[1],)),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification (0: Low Risk, 1: High Risk)
])


model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(X_train_nn, y_train_nn, epochs=10, batch_size=32, validation_split=0.2, verbose=1)


loss, accuracy = model.evaluate(X_test_nn, y_test_nn)
# st.sidebar.text(f"🔍 Neural Network Test Accuracy: {accuracy:.4f}")



def predict_threat_score(bytes_in, bytes_out, duration_seconds):
    """Predicts threat score using the trained Gradient Boosting model."""
    scaler = joblib.load(SCALER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    input_scaled = scaler.transform([[bytes_in, bytes_out, duration_seconds]])
    prediction = regressor.predict(input_scaled)
    return round(prediction[0], 2)

def classify_attack(bytes_in, bytes_out, duration_seconds):
    """Predicts attack type using the trained classification model."""
    scaler = joblib.load(SCALER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    input_scaled = scaler.transform([[bytes_in, bytes_out, duration_seconds]])
    prediction_proba = classifier.predict_proba(input_scaled)
    return "High Risk" if prediction_proba[0][1] > 0.5 else "Low Risk"


y_pred_threat = regressor.predict(X_test)
y_pred_class = classifier.predict(X_test_class)

mse = mean_squared_error(y_test, y_pred_threat)
r2 = r2_score(y_test, y_pred_threat)
accuracy = accuracy_score(y_test_class, y_pred_class)

# st.sidebar.markdown("### 🔥 Model Accuracy")
# st.sidebar.text(f"📊 Threat Score R² Score: {r2:.4f}")
# st.sidebar.text(f"✅ Classification Accuracy: {accuracy:.4f}")




if 'anomaly_score' not in data.columns:
    st.warning("🚨 `anomaly_score` not found! Running anomaly detection model...")
    features = ['bytes_in', 'bytes_out', 'duration_seconds']
    if all(col in data.columns for col in features):
        anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        data['anomaly_score'] = anomaly_detector.fit_predict(data[features])
        st.success("✅ Anomaly detection completed!")
    else:
        st.error("❌ Required columns missing for anomaly detection: 'bytes_in', 'bytes_out', 'duration_seconds'")


tabs = ["📊 Dashboard", "🔍 Threat Analysis", "🤖 ML Predictions", "🛑 Blocklist", "🔐 Security Insights"]
selected_tab = st.sidebar.radio("Navigate", tabs)

if selected_tab == "📊 Dashboard":
    st.title("📊 Cyber Threat Analysis Dashboard")
    st.write("An interactive overview of network traffic and detected cyber threats.")

    # ✅ **Key Performance Indicators (KPIs)**
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="🚨 Total Threats", value=len(data))
        
    with col2:
        st.metric(label="🌍 Unique IPs", value=data['src_ip'].nunique())

    with col3:
        high_risk_count = (data['threat_score'] > 7).sum()
        st.metric(label="⚠️ High-Risk Alerts", value=high_risk_count)
    
    with col4:
        anomaly_count = (data['anomaly_score'] == -1).sum() if 'anomaly_score' in data.columns else 0
        st.metric(label="🔴 Anomalies Detected", value=anomaly_count)

    st.subheader("📈 Cyber Threat Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔥 Feature Correlation Heatmap")
        numeric_data = data.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.write("### 🌍 Top Attacking IPs")
        top_ips = data['src_ip'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(y=top_ips.index, x=top_ips.values, palette="Reds_r", ax=ax)
        ax.set_xlabel("Number of Attacks")
        ax.set_ylabel("Source IP")
        st.pyplot(fig)

 
    import plotly.express as px
    import pycountry  

    st.write("### 🌍 Global Cyber Threat Heatmap")

 
    def convert_to_iso3(country_code):
        try:
            return pycountry.countries.get(alpha_2=country_code.upper()).alpha_3
        except AttributeError:
            return None  # Returns None for unrecognized country codes

  
    data['iso3'] = data['src_ip_country_code'].apply(convert_to_iso3)

  
    country_data = data['iso3'].dropna().value_counts().reset_index()
    country_data.columns = ['Country', 'Threats']

    fig = px.choropleth(
        country_data,
        locations="Country",
        locationmode="ISO-3",
        color="Threats",
        color_continuous_scale=px.colors.sequential.YlOrRd[::-1],  # Yellow-Red gradient
        range_color=[0, country_data['Threats'].quantile(0.95)],  
        hover_name="Country",
        hover_data={'Threats': ':,d'},
        projection="natural earth",
        title="<b>Global Cyber Threat Distribution</b>",
        height=600
    )

    
    fig.update_geos(
        showcountries=True,
        countrycolor="rgba(255,255,255,0.5)",
        coastlinecolor="lightblue",
        showocean=True,
        oceancolor="rgba(0, 102, 255, 0.2)",
        showland=True,
        landcolor="rgba(200, 200, 200, 0.3)",
        showframe=False,
        bgcolor='#0e1117'
    )

    
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

   
    st.subheader("📌 Threat Classification")

    if 'threat_score' in data.columns:
        threat_labels = ["Low Risk (0-4)", "Medium Risk (5-7)", "High Risk (8-10)"]
        threat_counts = [
            (data['threat_score'] <= 4).sum(),
            ((data['threat_score'] > 4) & (data['threat_score'] <= 7)).sum(),
            (data['threat_score'] > 7).sum()
        ]

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#f5f5f5')

        colors = ["#3498db", "#f39c12", "#e74c3c"]

        wedges, texts, autotexts = ax.pie(
            threat_counts, 
            labels=threat_labels, 
            autopct=lambda p: f'{p:.1f}%\n({int(p*sum(threat_counts)/100)})',
            colors=colors,
            startangle=140, 
            explode=(0.05, 0.05, 0.1), 
            wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
            textprops={'fontsize': 10, 'fontweight':'bold', 'color':'#333'},
            pctdistance=0.85,
            shadow=True
        )

        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)

        plt.suptitle("Threat Score Distribution Analysis", 
                     y=1.05, 
                     fontsize=16, 
                     fontweight='bold', 
                     color='#2c3e50')
        plt.title("Classification of Network Security Threats by Risk Level", 
                  fontsize=10, 
                  pad=20, 
                  color='#7f8c8d')

        st.pyplot(fig)

    # 🎚️ **Interactive Threat Filter**
    st.subheader("🔍 Threat Intelligence Explorer")
    st.markdown("---")

    # ✅ Create filtered dataset based on slider input
    threat_range = st.slider(
        "**Threat Severity Range** 🔐 → 💥",
        min_value=0,
        max_value=10,
        value=(3, 8),
        help="Adjust the range to focus on specific threat levels"
    )

    filtered_data = data[(data["threat_score"] >= threat_range[0]) & 
                         (data["threat_score"] <= threat_range[1])]

    # ✅ Create key threat statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Filtered Records", f"{len(filtered_data):,}")
    with col2:
        avg_score = filtered_data["threat_score"].mean() if not filtered_data.empty else 0
        st.metric("⚖️ Average Threat", f"{avg_score:.1f}/10")
    with col3:
        top_country = filtered_data["src_ip_country_code"].mode()[0] if not filtered_data.empty else "N/A"
        st.metric("🌍 Top Country", top_country)

    # ✅ Format and display dataframe
    st.markdown("### 🕵️‍♂️ Threat Event Log")
    if not filtered_data.empty:
        styled_df = filtered_data.head(20).style.applymap(lambda val: 
                    'background-color: red; color: white' if val >= 8 else 
                    'background-color: orange; color: black' if val >= 5 else 
                    'background-color: green; color: white', subset=["threat_score"])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.warning("🚨 No threats found in selected range - try widening your filters!")

    # ✅ CSV Export Feature
    if not filtered_data.empty:
        csv = filtered_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Filtered Data (CSV)",
            data=csv,
            file_name="threat_intel_report.csv",
            mime="text/csv"
        )

    st.markdown("---")



# Threat Analysis Section
if selected_tab == "🔍 Threat Analysis":
    st.title("🔍 Threat Analysis & Visualizations")

    # Cyberattack Heatmap Visualization
    if st.checkbox("📊 Show Cyberattack Heatmap"):
        plt.figure(figsize=(12, 6))
        sns.heatmap(pd.crosstab(data['src_ip_country_code'], data['threat_score']), cmap="Reds", linewidths=0.5)
        st.pyplot()

    # Network Interaction Graph Visualization
    if st.checkbox("📌 Show Network Interaction Graph"):
        st.write("### 🔌 Network Interaction Graph")
        G = nx.Graph()
        for idx, row in data.iterrows():
            G.add_edge(row['src_ip'], row['dst_ip'], weight=row['threat_score'])
        plt.figure(figsize=(12, 8))
        nx.draw_networkx(G, with_labels=True, node_size=500, node_color="skyblue", font_size=10)
        st.pyplot()

    # Suspicious Activities Based on Destination Port Visualization
    if st.checkbox("📌 Show Suspicious Activities Based on Destination Port"):  # Checkbox for user interaction
        st.write("### 🚨 Suspicious Activities Based on Destination Port")

        # Check if the 'dst_port' column exists before plotting
        if 'dst_port' in data.columns:
            plt.figure(figsize=(14, 8))

            # Get the top 10 destination ports
            top_ports = data['dst_port'].value_counts().index[:10]

            sns.countplot(x='dst_port', data=data, palette='coolwarm', order=top_ports)

            plt.title("Top 10 Destination Ports", fontsize=16, fontweight='bold', pad=20)
            plt.xlabel("Destination Port", fontsize=12, labelpad=15)
            plt.ylabel("Number of Events", fontsize=12, labelpad=15)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            sns.despine(left=True)
            plt.tight_layout()
            st.pyplot()
        else:
            st.error("❌ Column 'dst_port' is missing from the dataset.")




    # Web Traffic Analysis Over Time Visualization
    if st.checkbox("📌 Show Web Traffic Analysis Over Time"):
        st.write("### 📶 Web Traffic Analysis Over Time")
        data.set_index('creation_time', inplace=True)
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['bytes_in'], label='Bytes In', marker='o')
        plt.plot(data.index, data['bytes_out'], label='Bytes Out', marker='o')
        plt.title('Web Traffic Analysis Over Time')
        plt.xlabel('Time')
        plt.ylabel('Bytes')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    # Anomaly Detection Scatter Plot Visualization
    if st.checkbox("📌 Show Anomaly Detection Scatter Plot"):
        if 'anomaly_score' in data.columns:
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=data, x='bytes_in', y='bytes_out', hue='anomaly_score', palette={1: 'blue', -1: 'red'})
            plt.title("Anomaly Detection: Bytes In vs Bytes Out")
            plt.xlabel("Bytes In")
            plt.ylabel("Bytes Out")
            st.pyplot()
        else:
            st.error("❌ `anomaly_score` column is missing. Run anomaly detection first.")


    # Top Destination Ports with WAF Rule Triggers Visualization
    if st.checkbox("📌 Show Top Destination Ports with WAF Rule Triggers"):
        plt.figure(figsize=(14, 8))
        ax = sns.countplot(
            data=data[data['detection_types'] == 'waf_rule'], 
            x='dst_port', 
            order=data[data['detection_types'] == 'waf_rule']['dst_port'].value_counts().index[:10],  
            palette='viridis',
            saturation=0.9
        )
        # Add annotations
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=10,
                        color='black')
        plt.title("Top 10 Destination Ports with WAF Rule Triggers", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("Destination Port", fontsize=12, labelpad=15)
        plt.ylabel("Number of Suspicious Events", fontsize=12, labelpad=15)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        sns.despine(left=True)
        plt.gca().set_facecolor('#f5f5f5')
        plt.figtext(0.5, 0.01, 
                   "Analysis of Web Application Firewall (WAF) rule triggers by destination port",
                   ha="center", fontsize=10, color='#666666')
        plt.tight_layout()
        st.pyplot()

 
   
    if st.checkbox("📌 Show Model Training History"):
        try:
            # Check if 'history' is defined
            if "history" in globals() or "history" in locals():
                plt.figure(figsize=(12, 6))

                # Accuracy Plot
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label="Training Accuracy")
                plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
                plt.title("Model Accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()

                # Loss Plot
                plt.subplot(1, 2, 2)
                plt.plot(history.history["loss"], label="Training Loss")
                plt.plot(history.history["val_loss"], label="Validation Loss")
                plt.title("Model Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()

                st.pyplot()
            else:
                st.info("ℹ️ Model training history is not available yet. Train the model first.")

        except NameError:
            st.warning("⚠️ The model has not been trained yet. Train the model first.")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")






elif selected_tab == "🤖 ML Predictions":
    st.title("🤖 Predict Cyber Threats using Machine Learning")

    bytes_in = st.number_input("📥 Bytes In", min_value=0, step=1)
    bytes_out = st.number_input("📤 Bytes Out", min_value=0, step=1)
    duration_seconds = st.number_input("⏳ Duration (seconds)", min_value=0.0, step=0.1)
    src_ip = st.text_input("🌍 Source IP Address", placeholder="Enter IP to check threat level")

    if st.button("🔍 Predict Threat Level"):
        threat_score = predict_threat_score(bytes_in, bytes_out, duration_seconds)
        attack_class = classify_attack(bytes_in, bytes_out, duration_seconds)

        st.success(f"🚨 Predicted Threat Score: {threat_score}")
        st.warning(f"⚠️ Attack Classification: {attack_class}")

        # ✅ **Automatically Add "High Risk" IPs to Blocklist**
        if attack_class == "High Risk" and src_ip:
            blocklist.add(src_ip)  # Add to blocklist
            save_blocklist(blocklist)  # Persist to file
            st.error(f"🛑 IP `{src_ip}` added to Blocklist!")





BLOCKLIST_PATH = "blocklist.txt"

def load_blocklist():
    """Loads existing blocklisted IPs from file (if it exists)."""
    if os.path.exists(BLOCKLIST_PATH):
        with open(BLOCKLIST_PATH, "r") as f:
            return set(f.read().splitlines())  # Store as set for fast lookups
    return set()

def save_blocklist(blocklist):
    """Saves the updated blocklist back to the file."""
    with open(BLOCKLIST_PATH, "w") as f:
        for ip in blocklist:
            f.write(ip + "\n")


if selected_tab == "🛑 Blocklist":  
    st.title("🛑 Auto Blocklist for Suspicious IPs")
  
    blocklist = load_blocklist()

   
    suspicious_ips = set(data[data['threat_score'] > 7]['src_ip'].unique())
    blocklist.update(suspicious_ips)  # Merge dataset IPs into blocklist
    save_blocklist(blocklist)  # Save updated blocklist

  
    st.write(f"### 🚨 {len(blocklist)} Suspicious IPs Blocked")
    
    if blocklist:
        st.table(pd.DataFrame(list(blocklist), columns=["Blocked IPs"]))  # ✅ Convert set to list before displaying
    
    
    if st.button("📥 Download Blocklist"):
        save_blocklist(blocklist)  # Ensure latest IPs are saved
        st.success("✅ Blocklist saved! Check 'blocklist.txt'")






elif selected_tab == "🔐 Security Insights":
    st.title("🔐 Security Best Practices & Threat Mitigation")

    st.markdown("""
    Cybersecurity threats are constantly evolving. Implementing strong security measures is essential to protect sensitive data and networks. Below are key best practices:
    """)

   
    st.subheader("🔑 Strong Authentication & Access Control")
    st.markdown("""
    - Use **multi-factor authentication (MFA)** to protect user accounts.
    - Enforce **strong password policies** (length, complexity, expiration).
    - Limit access to sensitive systems using **role-based access control (RBAC)**.
    """)

    # ✅ **2️⃣ Network Security**
    st.subheader("🌐 Network Security & Monitoring")
    st.markdown("""
    - **Enable firewalls** and intrusion detection/prevention systems (**IDS/IPS**).
    - **Encrypt sensitive data** during transmission (**TLS/SSL**).
    - Monitor logs and network traffic for **suspicious activities**.
    """)

    # ✅ **3️⃣ Malware & Attack Prevention**
    st.subheader("🚨 Preventing Cyber Attacks & Malware")
    st.markdown("""
    - **Regularly update software** to patch vulnerabilities.
    - Use **endpoint security solutions** (antivirus, anti-malware, EDR).
    - Educate employees about **phishing attacks** and **social engineering** tactics.
    """)

    # ✅ **4️⃣ Incident Response & Recovery**
    st.subheader("📌 Incident Response & Data Backup")
    st.markdown("""
    - Implement an **incident response plan** for handling cyber threats.
    - Maintain **secure, offsite backups** to recover from ransomware attacks.
    - Conduct regular **penetration testing** to identify security weaknesses.
    """)

    # ✅ **5️⃣ Emerging Threats & AI-based Security**
    st.subheader("🧠 AI & Machine Learning in Cybersecurity")
    st.markdown("""
    - Use **AI-driven threat detection models** to identify anomalies.
    - Deploy **automated security operations (SOAR)** to respond to threats in real-time.
    - Monitor **zero-day exploits** and emerging cyber threats.
    """)

    # ✅ Cybersecurity Score / Risk Assessment
    st.subheader("🔍 Cybersecurity Risk Assessment")
    risk_score = np.random.randint(40, 100)  # Simulated risk score
    if risk_score > 75:
        st.success(f"✅ Your estimated Cybersecurity Score: **{risk_score}/100** (Good Security)")
    elif risk_score > 50:
        st.warning(f"⚠️ Your estimated Cybersecurity Score: **{risk_score}/100** (Moderate Risk)")
    else:
        st.error(f"🚨 Your estimated Cybersecurity Score: **{risk_score}/100** (High Risk) - Take Action!")

    # ✅ Security Resources & Threat Intelligence
    st.subheader("📚 Useful Cybersecurity Resources")
    st.markdown("""
    - 🔗 [OWASP Security Guidelines](https://owasp.org/)
    - 🔗 [MITRE ATT&CK Framework](https://attack.mitre.org/)
    - 🔗 [National Cyber Security Centre (NCSC)](https://www.ncsc.gov.uk/)
    - 🔗 [SANS Cybersecurity Training](https://www.sans.org/)
    """)

    st.success("🔒 Stay proactive and safeguard your systems from cyber threats!")


st.sidebar.markdown("---")
st.sidebar.markdown("📍 Developed by Sanket")


















