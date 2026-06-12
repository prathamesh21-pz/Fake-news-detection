import streamlit as st
import pickle
import re
import string

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1E3A8A;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#555;
}

.result-box {
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.real-news {
    background-color:#d4edda;
    color:#155724;
}

.fake-news {
    background-color:#f8d7da;
    color:#721c24;
}

.footer {
    text-align:center;
    color:gray;
    padding-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = pickle.load(open("models/fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\\S+', '', text)
    text = re.sub(r'\\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2965/2965879.png",
        width=120
    )

    st.title("Project Info")

    st.write("""
    **MSc Data Science Project**

    Fake News Detection using:

    - Machine Learning
    - TF-IDF Vectorization
    - Logistic Regression
    - Streamlit Deployment
    """)

    st.markdown("---")

    st.write("### Model Features")
    st.success("Real News Detection")
    st.error("Fake News Detection")
    st.info("Confidence Score")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<p class="title">📰 Fake News Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Powered News Verification Platform</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# INPUT AREA
# -----------------------------
st.subheader("Enter News Content")

news = st.text_area(
    "",
    height=300,
    placeholder="Paste the news article here..."
)

# -----------------------------
# BUTTON
# -----------------------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict_btn = st.button(
        "🔍 Analyze News",
        use_container_width=True
    )

# -----------------------------
# PREDICTION
# -----------------------------
if predict_btn:

    if news.strip() == "":
        st.warning("Please enter news text.")
    else:

        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)

        prob = model.predict_proba(vector)

        confidence = round(max(prob[0]) * 100, 2)

        st.markdown("## Analysis Result")

        if prediction[0] == 1:

            st.markdown(
                f'''
                <div class="result-box real-news">
                ✅ REAL NEWS
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'''
                <div class="result-box fake-news">
                ❌ FAKE NEWS
                </div>
                ''',
                unsafe_allow_html=True
            )

        st.markdown("### Confidence Score")

        st.progress(confidence/100)

        st.metric(
            label="Model Confidence",
            value=f"{confidence}%"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.markdown(
    """
    <div class="footer">
    MSc Data Science Final Year Project<br>
    Fake News Detection using Machine Learning & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
