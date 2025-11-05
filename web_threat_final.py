# 1️⃣ Module Importing
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import streamlit as st


# Data Handling & Processing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score

# Deep Learning (TensorFlow & Keras)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Visualization
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# 2️⃣ Load the Data
# data = pd.read_csv("CloudWatch_Traffic_Web_Attack (1).csv")
data = pd.read_csv(r"C:\Users\ASUS\Downloads\Cyber-Threat-Analysis-main\Cyber-Threat-Analysis-main\data\sample_data.csv")
data.drop_duplicates(inplace=True)

# Convert time-related columns to datetime format
data['creation_time'] = pd.to_datetime(data['creation_time'])
data['end_time'] = pd.to_datetime(data['end_time'])
data['time'] = pd.to_datetime(data['time'])

# Standardize text data
data['src_ip_country_code'] = data['src_ip_country_code'].astype(str).str.upper()

# Feature Engineering: Duration Calculation
data['duration_seconds'] = (data['end_time'] - data['creation_time']).dt.total_seconds()

# 3️⃣ Identify Repeated Attack IPs
attack_ip_list = data[data['detection_types'] == 'waf_rule']['src_ip'].value_counts()
attack_ip_list = attack_ip_list[attack_ip_list > 1].index.tolist()

# 4️⃣ Threat Scoring System Function
def calculate_threat_score(row):
    score = 0
    if row['bytes_in'] > 30000:
        score += 3
    if row['bytes_out'] > 20000:
        score += 2
    if row['src_ip_country_code'] in ['RU', 'CN', 'KP']:
        score += 3
    if row['src_ip'] in attack_ip_list:
        score += 3
    return min(score, 10)

# Apply the function
data['threat_score'] = data.apply(calculate_threat_score, axis=1)

# ✅ Print top 10 threats
print(data[['src_ip', 'threat_score']].sort_values(by='threat_score', ascending=False).head(10))

# 5️⃣ **Auto Blocklist for Suspicious IPs**
suspicious_ips = data[data['threat_score'] > 7]['src_ip'].unique()
with open("blocklist.txt", "w") as f:
    for ip in suspicious_ips:
        f.write(ip + "\n")
print("✅ Blocklist Created: blocklist.txt")

# 6️⃣ Data Preprocessing
scaler = StandardScaler()
data[['scaled_bytes_in', 'scaled_bytes_out', 'scaled_duration_seconds']] = scaler.fit_transform(
    data[['bytes_in', 'bytes_out', 'duration_seconds']]
)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_features = encoder.fit_transform(data[['src_ip_country_code']])
encoded_columns = encoder.get_feature_names_out(['src_ip_country_code'])
encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=data.index)
data = pd.concat([data, encoded_df], axis=1)

data['is_suspicious'] = (data['detection_types'] == 'waf_rule').astype(int)

# 7️⃣ Data Visualization

## Heatmap for Correlation Matrix
plt.figure(figsize=(12, 10))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
heatmap = sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap=cmap, center=0, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()

## Stacked Bar Chart for Detection Types by Country
detection_types_by_country = pd.crosstab(data['src_ip_country_code'], data['detection_types'])
detection_types_by_country.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Detection Types by Country Code')
plt.xlabel('Country Code')
plt.ylabel('Frequency of Detection Types')
plt.xticks(rotation=45)
plt.legend(title='Detection Type')
plt.show()

## **Real-Time Attack Monitoring: Top 10 Attackers**
top_attackers = data['src_ip'].value_counts().head(10)
plt.figure(figsize=(10, 5))
top_attackers.plot(kind='bar', color='red')
plt.title("Top 10 Attacking IPs")
plt.xlabel("IP Address")
plt.ylabel("Attack Count")
plt.xticks(rotation=45)
plt.show()

## **Cyberattack Heatmap**
plt.figure(figsize=(12, 6))
sns.heatmap(pd.crosstab(data['src_ip_country_code'], data['threat_score']), cmap="Reds", linewidths=0.5)
plt.title("Cyberattack Heatmap by Country")
plt.xlabel("Threat Score")
plt.ylabel("Country Code")
plt.xticks(rotation=45)
plt.show()

# ✅ **NEW FEATURE 1: Anomaly Detection & Visualization**
anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
data['anomaly_score'] = anomaly_detector.fit_predict(data[['bytes_in', 'bytes_out', 'scaled_duration_seconds']])
print("✅ Anomaly Detection Completed")

# ✅ **Anomaly Scatter Plot**
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='bytes_in', y='bytes_out', hue='anomaly_score', palette={1: 'blue', -1: 'red'})
plt.title("Anomaly Detection: Bytes In vs Bytes Out")
plt.xlabel("Bytes In")
plt.ylabel("Bytes Out")
plt.show()

# ✅ **NEW FEATURE 2: Suspicious Activities Based on Destination Port**
plt.figure(figsize=(14, 8))
ax = sns.countplot(
    data=data[data['detection_types'] == 'waf_rule'], 
    x='dst_port', 
    order=data[data['detection_types'] == 'waf_rule']['dst_port'].value_counts().index[:10],  # Get top 10 from filtered data
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

# Style enhancements
plt.title("Top 10 Destination Ports with WAF Rule Triggers", 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Destination Port", fontsize=12, labelpad=15)
plt.ylabel("Number of Suspicious Events", fontsize=12, labelpad=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Add grid and background
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine(left=True)
plt.gca().set_facecolor('#f5f5f5')

# Add footer text
plt.figtext(0.5, 0.01, 
           "Analysis of Web Application Firewall (WAF) rule triggers by destination port",
           ha="center", fontsize=10, color='#666666')

plt.tight_layout()
plt.show()

# ✅ **NEW FEATURE 3: Explicit Neural Network Accuracy Print**
X = data[['bytes_in', 'bytes_out', 'scaled_duration_seconds']]
y = data['is_suspicious']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ **Define the Neural Network Model**
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification output
])

# ✅ **Compile the Model**
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# ✅ **Train the Model**
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# ✅ **Evaluate the Model**
print(f"Neural Network Test Accuracy: {model.evaluate(X_test_scaled, y_test)[1] * 100:.2f}%")

# 8️⃣ Machine Learning Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and Accuracy
y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))






# ✅ Web Traffic Analysis Over Time
data.set_index('creation_time', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['bytes_in'], label='Bytes In', marker='o')
plt.plot(data.index, data['bytes_out'], label='Bytes Out', marker='o')
plt.title('Web Traffic Analysis Over Time')
plt.xlabel('Time')
plt.ylabel('Bytes')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# ✅ Network Interaction between Source and Destination IPs
G = nx.Graph()
for _, row in data.iterrows():
    G.add_edge(row['src_ip'], row['dst_ip'])

plt.figure(figsize=(14, 10))
nx.draw_networkx(G, with_labels=True, node_size=20, font_size=8, node_color='skyblue', font_color='darkblue')
plt.title('Network Interaction between Source and Destination IPs')
plt.axis('off')  # Hide axis
plt.show()



# ✅ Training History Visualization
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()



