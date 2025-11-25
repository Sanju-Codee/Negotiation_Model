import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Negotiation ML Predictor", page_icon="⚖️")

MODEL_PATH = "Negotiation_MLproject.joblib"     # your model
ENCODER_PRODUCT = "product_encoder.joblib"      # Product_Name label encoder
ENCODER_SENTIMENT = "sentiment_encoder.joblib"  # Sentiment label encoder

st.title("⚖️ Negotiation & Bargaining Prediction App")
st.write("Predict negotiation outcomes using your Machine Learning model.")

# -----------------------------------------------------
# Load Model & Encoders
# -----------------------------------------------------
def load_all():
    try:
        st.session_state.model = joblib.load(MODEL_PATH)
        st.session_state.prod_enc = joblib.load(ENCODER_PRODUCT)
        st.session_state.sent_enc = joblib.load(ENCODER_SENTIMENT)
        st.success("✅ Model & Encoders Loaded Successfully!")
    except Exception as e:
        import traceback
        st.error(f"❌ Could not load model or encoders:\n{traceback.format_exc()}")

if st.button("Load Model"):
    load_all()

# -----------------------------------------------------
# Input Section
# -----------------------------------------------------
st.subheader("📥 Enter Input Data")

product_name = st.text_input("Product Name")

listed_price = st.number_input("Listed Price", min_value=0.0, step=1.0)
offered_price = st.number_input("Offered Price", min_value=0.0, step=1.0)
final_price = st.number_input("Final Price", min_value=0.0, step=1.0)
discount = st.number_input("Discount %", min_value=0.0, step=0.1)

sentiment = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"])

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
def predict():
    if "model" not in st.session_state:
        st.warning("⚠️ Please load the model first!")
        return

    model = st.session_state.model
    prod_enc = st.session_state.prod_enc
    sent_enc = st.session_state.sent_enc

    try:
        product_encoded = prod_enc.transform([product_name])[0]
        sentiment_encoded = sent_enc.transform([sentiment])[0]

        features = np.array([[product_encoded,
                              listed_price,
                              offered_price,
                              final_price,
                              discount,
                              sentiment_encoded]])

        pred = model.predict(features)[0]

        if pred == 0:
            st.success("🟢 Predicted Outcome: ACCEPTED")
        elif pred == 1:
            st.warning("🟡 Predicted Outcome: PENDING")
        else:
            st.error("🔴 Predicted Outcome: REJECTED")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

if st.button("Predict"):
    predict()

st.markdown("---")
