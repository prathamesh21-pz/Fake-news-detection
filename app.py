import streamlit as st
import pickle
import re
import string

# -----------------------------
# Load Saved Model & Vectorizer
# -----------------------------

MODEL_DIR = Path("models")

model = pickle.load(open(MODEL_DIR / "fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open(MODEL_DIR / "vectorizer.pkl", "rb"))
# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.title("📰 Fake News Detection System")
st.markdown(
    """
    This application uses a Machine Learning model to classify news as:

    - ✅ **Real News**
    - ❌ **Fake News**
    """
)

# -----------------------------
# User Input
# -----------------------------
news_text = st.text_area(
    "Enter News Article/Text",
    height=250,
    placeholder="Paste news content here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:

        cleaned_text = clean_text(news_text)

        transformed_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(transformed_text)

        probability = model.predict_proba(transformed_text)

        confidence = max(probability[0]) * 100

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("✅ This appears to be REAL News")
        else:
            st.error("❌ This appears to be FAKE News")

        st.info(f"Confidence Score: {confidence:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "MSc Data Science Project | Fake News Detection Using Machine Learning and Streamlit"
)
