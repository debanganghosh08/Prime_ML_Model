import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="ML Model Dashboard", layout="wide")

# Load model results BEFORE using it
results_df = pd.read_csv("tuned_model_results_with_auc.csv")

# Add custom CSS
st.markdown("""
    <style>
    ::-webkit-scrollbar { display: none; }
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    h1, h2, h3 {
        color: #4ADEDE;
        text-align: center;
    }
    div.stButton > button {
        background-color: #4ADEDE;
        color: black;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #38B6FF;
        color: white;
        transform: scale(1.05);
    }
    .stFileUploader label {
        color: #4ADEDE;
    }
    </style>
""", unsafe_allow_html=True)

# Main content starts here
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.title("Model Performance Dashboard 📊")
st.dataframe(results_df)

st.markdown('</div>', unsafe_allow_html=True)

st.title("ML Model Comparison Dashboard")
st.markdown("Compare accuracy, AUC scores, and feature importance of tuned ML models.")

st.subheader("Model Evaluation Summary📝")
st.dataframe(results_df)

# Plot Accuracy vs AUC
#st.subheader("Accuracy vs AUC Score📊")
#
#fig, ax = plt.subplots(figsize=(20, 5))
#sns.barplot(data=results_df, x="Model", y="Tuned Mean CV Accuracy", color="skyblue", label="Accuracy")
#sns.barplot(data=results_df, x="Model", y="AUC Score", color="salmon", label="AUC")
#
#plt.xticks(rotation=45)
#plt.ylabel("Score")
#plt.legend()
#st.pyplot(fig)

st.title("Accuracy vs AUC Score 📊")

# Create the figure
fig = go.Figure()

# Bar for Accuracy
fig.add_trace(go.Bar(
    x=results_df["Model"],
    y=results_df["Tuned Mean CV Accuracy"],
    name="Accuracy",
    marker_color='skyblue'
))

# Bar for AUC Score
fig.add_trace(go.Bar(
    x=results_df["Model"],
    y=results_df["AUC Score"],
    name="AUC Score",
    marker_color='salmon'
))

# Update layout for better appearance
fig.update_layout(
    barmode='group',
    xaxis_title="Model",
    yaxis_title="Score",
    title="Model Accuracy vs AUC Comparison",
    template="plotly_dark",
    font=dict(family="Fira Code", size=14),
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("🔢 Interactive Model Comparison: Accuracy vs AUC vs F1 vs Precision")

# --- 1️⃣ Model Multiselect Filter ---
all_models = results_df["Model"].tolist()
selected_models = st.multiselect("📌 Select Models to Compare", all_models, default=all_models)

# Filter the dataframe
filtered_df = results_df[results_df["Model"].isin(selected_models)]

# --- 2️⃣ Theme Selector ---
theme = st.radio("🎨 Select Chart Theme", ["Dark", "Light"], index=0)
plot_theme = "plotly_dark" if theme == "Dark" else "plotly_white"

# --- 3️⃣ Optional Metric Selector ---
st.markdown("📊 Select Additional Metrics to Display:")
show_f1 = st.checkbox("F1 Score", value=True)
show_precision = st.checkbox("Precision", value=False)

# --- Plotly Interactive Clustered Bar Chart ---
fig = go.Figure()

# Always include Accuracy & AUC
fig.add_trace(go.Bar(
    x=filtered_df["Model"],
    y=filtered_df["Tuned Mean CV Accuracy"],
    name="Accuracy",
    marker_color='skyblue'
))
fig.add_trace(go.Bar(
    x=filtered_df["Model"],
    y=filtered_df["AUC Score"],
    name="AUC Score",
    marker_color='salmon'
))

# Optional: Add F1
if show_f1 and "F1 Score" in filtered_df.columns:
    fig.add_trace(go.Bar(
        x=filtered_df["Model"],
        y=filtered_df["F1 Score"],
        name="F1 Score",
        marker_color='lightgreen'
    ))

# Optional: Add Precision
if show_precision and "Precision" in filtered_df.columns:
    fig.add_trace(go.Bar(
        x=filtered_df["Model"],
        y=filtered_df["Precision"],
        name="Precision",
        marker_color='violet'
    ))

# Layout updates
fig.update_layout(
    barmode='group',
    xaxis_title="Model",
    yaxis_title="Score",
    title="📊 Model Performance Comparison",
    template=plot_theme,
    font=dict(family="Fira Code", size=14),
    height=550,
    legend=dict(title="Metrics")
)

st.plotly_chart(fig, use_container_width=True)