import joblib
import streamlit as st

from text_utils import clean_text   # same cleaning code used during training

# -------------------------------
# 1. Page configuration
# -------------------------------
st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="centered")

# -------------------------------
# 2. Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl")

artifact = load_model()
model = artifact["model"]
default_threshold = artifact.get("threshold", 0.7)

# -------------------------------
# 3. Interface
# -------------------------------
st.title("📧 Email Spam Detection")
st.write("Enter an email message below to check whether it is Spam or a legitimate email.")
st.divider()

email_text = st.text_area("Enter Email Text", placeholder="Paste your email message here...", height=200)

with st.expander("Advanced settings"):
    threshold = st.slider(
        "Spam threshold",
        min_value=0.30, max_value=0.95, value=float(default_threshold), step=0.05,
        help="An email is marked SPAM when its spam probability is at or above this value. "
             "Higher = fewer good emails wrongly marked as spam, but more spam missed.",
    )

# -------------------------------
# 4. Prediction
# -------------------------------
if st.button("Check Email", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter an email message first.")
    else:
        processed = clean_text(email_text)
        spam_prob = float(model.predict_proba([processed])[0][1])

        st.divider()
        if spam_prob >= threshold:
            st.error("🚨 This email is SPAM!")
        else:
            st.success("✅ This email is HAM (Legitimate).")

        col1, col2 = st.columns(2)
        col1.metric("Spam probability", f"{spam_prob * 100:.1f}%")
        col2.metric("Ham probability", f"{(1 - spam_prob) * 100:.1f}%")
        st.progress(min(max(spam_prob, 0.0), 1.0))

        if abs(spam_prob - threshold) < 0.15:
            st.info("The model is not very sure about this email. Check the sender and links manually.")

st.divider()
st.caption("Built with Python, NLP, Scikit-learn and Streamlit. Predictions are not a guarantee that an email is safe.")
