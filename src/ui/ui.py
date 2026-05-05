import streamlit as st
import requests
import base64

st.set_page_config(page_title="Cancer Prediction Visualizer", page_icon="🔬")

st.title("🔬 Cancer Prediction Visualizer")
st.markdown("Train a PyTorch model and visualize outcomes using Matplotlib.")

API_URL = "http://api-service:8000" # K8s internal service name

# 1. Fetch initial data & PCA plot
try:
    info_res = requests.get(f"{API_URL}/info", timeout=5)
    data_info = info_res.json()
except requests.exceptions.RequestException:
    data_info = None
    st.error("🚨 Could not connect to the ML API. Is the K8s port-forward running?")

if data_info:
    st.success(f"✅ Dataset loaded: {data_info['data']['train_size']} training samples")
    
    try:
        pca_res = requests.get(f"{API_URL}/plot/pca", timeout=5)
        st.image(base64.b64decode(pca_res.json()["image_base64"]), caption="PCA Feature Space")
    except:
        st.warning("Could not load initial plot.")

st.divider()

# 2. Training Configuration
st.subheader("⚙️ Model Training")
col1, col2, col3 = st.columns(3)
with col1: epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50)
with col2: lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, format="%4f")
with col3: hidden = st.number_input("Hidden Dim", min_value=8, max_value=128, value=32)

if st.button("Train PyTorch Model", type="primary", use_container_width=True):
    if not data_info:
        st.warning("API not connected.")
    else:
        with st.spinner("Training model..."):
            try:
                response = requests.post(
                    f"{API_URL}/train",
                    json={"epochs": epochs, "lr": lr, "hidden_dim": hidden},
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                st.subheader("Training Results")
                m1, m2 = st.columns(2)
                m1.metric(label="Test Accuracy", value=f"{data['accuracy']:.2%}")
                m2.metric(label="Confusion Matrix", value=str(data['confusion_matrix']))
                
                st.image(base64.b64decode(data["training_plot_base64"]), caption="Loss & Accuracy Curves")
                st.image(base64.b64decode(data["confusion_plot_base64"]), caption="Confusion Matrix Heatmap")
                
            except requests.exceptions.RequestException as e:
                st.error("🚨 Training failed or API crashed.")

st.divider()

# 3. Prediction Section
st.subheader("🔮 Make a Prediction")
if data_info:
    features = data_info["data"]["features"]
    inputs = []
    cols = st.columns(3)
    for i, feat in enumerate(features):
        with cols[i % 3]:
            val = st.number_input(feat, key=f"feat_{i}", format="%4f")
            inputs.append(val)

    if st.button("Predict Outcome", type="secondary", use_container_width=True):
        with st.spinner("Predicting..."):
            try:
                res = requests.post(f"{API_URL}/predict", json={"features": inputs}, timeout=10)
                res.raise_for_status()
                pred_data = res.json()
                
                if pred_data["prediction"] == "Benign":
                    st.success(f"✅ Result: **Benign** (Confidence: {pred_data['probability']:.2%})")
                else:
                    st.error(f"⚠️ Result: **Malignant** (Confidence: {1 - pred_data['probability']:.2%})")
                    
            except requests.exceptions.RequestException as e:
                st.error("🚨 Prediction failed. Make sure you trained the model first!")