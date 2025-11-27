# app.py
import re
import joblib
import nltk
import pandas as pd
import streamlit as st
import speech_recognition as sr
from nltk.corpus import stopwords
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

# --------------------------------------------------
# Streamlit basic config
# --------------------------------------------------
st.set_page_config(page_title="CineSentiment", layout="wide")

# --------------------------------------------------
# NLTK stopwords (same as training)
# --------------------------------------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
exclude_words = {'not', 'no', 'nor', "don't", "won't", "didn't", "isn't"}
stop_words = stop_words - exclude_words


def clean_text(text: str) -> str:
    """Clean text in the same way as training."""
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    text = text.lower().split()
    text = [w for w in text if w not in stop_words]
    return " ".join(text)


@st.cache_resource
def load_model_files():
    """Load the trained model and vectorizer."""
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model_files()

# --------------------------------------------------
# Basic Netflix-style CSS (ASCII only)
# --------------------------------------------------
st.markdown(
    """
<style>
.main, .block-container {
    background: radial-gradient(circle at top, #1d1d1d 0%, #000000 60%);
    color: #f5f5f5;
}
.section-card {
    background: #121212;
    padding: 1.5rem 1.8rem;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.7);
    border: 1px solid #292929;
}
.stButton > button {
    background: linear-gradient(135deg, #e50914, #b81d24);
    color: white;
    border-radius: 100px;
    padding: 10px 24px;
    font-weight: 600;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff2027, #e50914);
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
<div style="padding:2.3rem 1rem; text-align:center;
            background:linear-gradient(135deg,#000000,#111111,#b81d24);
            border-radius:0 0 24px 24px; box-shadow:0 15px 40px rgba(0,0,0,0.7);">
    <h1 style="font-size:2.8rem; font-weight:850; letter-spacing:2px;">
        CineSentiment
    </h1>
    <div style="font-size:1.1rem; margin-top:8px;">
        Movie Review Sentiment Analyzer (Voice, Text, YouTube Comments)
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Prediction helper
# --------------------------------------------------
def predict_sentiment(text: str, neutral_threshold: float = 0.60):
    """
    Returns (label, confidence_percent, cleaned_text)
    label in {"positive", "negative", "neutral"}
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "neutral", 0.0, cleaned

    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    labels = model.classes_
    idx = proba.argmax()
    label = labels[idx]
    confidence = float(proba[idx]) * 100.0

    if confidence < neutral_threshold * 100.0:
        return "neutral", confidence, cleaned
    return label, confidence, cleaned


# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_voice, tab_text, tab_youtube = st.tabs(
    ["Voice Review", "Text Review", "YouTube Comments / CSV"]
)

# ===================== TAB 1: VOICE =====================
with tab_voice:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Voice Review")

    if st.button("Start Recording (5 seconds)"):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                st.info("Listening... please speak now.")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

            try:
                text = recognizer.recognize_google(audio)
                st.write("You said:")
                st.info(text)

                label, conf, cleaned = predict_sentiment(text)
                st.write("Cleaned text:", cleaned)

                if label == "positive":
                    st.success("Positive (confidence: {:.1f}%)".format(conf))
                elif label == "negative":
                    st.error("Negative (confidence: {:.1f}%)".format(conf))
                else:
                    st.info("Neutral (confidence: {:.1f}%)".format(conf))

            except sr.UnknownValueError:
                st.error("Could not understand the speech.")
            except sr.RequestError:
                st.error("Speech recognition service error. Check your internet.")

        except Exception as e:
            st.error("Microphone error: {}".format(e))

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== TAB 2: TEXT =====================
with tab_text:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Text Review")

    text_input = st.text_area("Enter your movie review:", height=140)

    if st.button("Analyze Text"):
        if not text_input.strip():
            st.warning("Please type something first.")
        else:
            label, conf, cleaned = predict_sentiment(text_input)
            st.write("Cleaned text:", cleaned)
            if label == "positive":
                st.success("Positive (confidence: {:.1f}%)".format(conf))
            elif label == "negative":
                st.error("Negative (confidence: {:.1f}%)".format(conf))
            else:
                st.info("Neutral (confidence: {:.1f}%)".format(conf))

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== TAB 3: YOUTUBE + CSV =====================
with tab_youtube:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("YouTube Comments or CSV Analysis")

    sub1, sub2 = st.tabs(["YouTube URL", "Upload CSV File"])

    # -------- Subtab 1: YouTube URL --------
    with sub1:
        url = st.text_input("Paste YouTube video URL:")
        limit = st.slider("Number of comments to analyze:", 20, 500, 100, step=20)

        if st.button("Fetch and Analyze YouTube Comments"):
            if not url.strip():
                st.warning("Please paste a YouTube link.")
            else:
                try:
                    st.info("Fetching comments from YouTube...")
                    downloader = YoutubeCommentDownloader()
                    comments = []
                    for c in downloader.get_comments_from_url(
                        url, sort_by=SORT_BY_POPULAR
                    ):
                        comments.append(c["text"])
                        if len(comments) >= limit:
                            break

                    if not comments:
                        st.error("No comments found for this video.")
                    else:
                        st.success("Fetched {} comments. Analyzing...".format(len(comments)))
                        pos = neg = neu = 0
                        progress = st.progress(0.0)
                        total = len(comments)

                        for i, cm in enumerate(comments):
                            label, _, _ = predict_sentiment(cm)
                            if label == "positive":
                                pos += 1
                            elif label == "negative":
                                neg += 1
                            else:
                                neu += 1
                            progress.progress((i + 1) / float(total))

                        st.write("Positive:", pos)
                        st.write("Negative:", neg)
                        st.write("Neutral:", neu)

                        df_chart = pd.DataFrame(
                            {"Sentiment": ["positive", "negative", "neutral"],
                             "Count": [pos, neg, neu]}
                        ).set_index("Sentiment")
                        st.bar_chart(df_chart)

                except Exception as e:
                    st.error("Error while fetching comments: {}".format(e))

    # -------- Subtab 2: CSV upload --------
    with sub2:
        file = st.file_uploader("Upload CSV file (first column should have comments):",
                                type=["csv"])
        if st.button("Analyze CSV Comments"):
            if file is None:
                st.warning("Please upload a CSV file first.")
            else:
                try:
                    df = pd.read_csv(file)
                    column = df.columns[0]
                    comments = df[column].dropna().astype(str).tolist()

                    if not comments:
                        st.warning("No comments found in CSV.")
                    else:
                        st.success("Loaded {} comments. Analyzing...".format(len(comments)))
                        pos = neg = neu = 0
                        progress = st.progress(0.0)
                        total = len(comments)

                        for i, cm in enumerate(comments):
                            label, _, _ = predict_sentiment(cm)
                            if label == "positive":
                                pos += 1
                            elif label == "negative":
                                neg += 1
                            else:
                                neu += 1
                            progress.progress((i + 1) / float(total))

                        st.write("Positive:", pos)
                        st.write("Negative:", neg)
                        st.write("Neutral:", neu)

                        df_chart = pd.DataFrame(
                            {"Sentiment": ["positive", "negative", "neutral"],
                             "Count": [pos, neg, neu]}
                        ).set_index("Sentiment")
                        st.bar_chart(df_chart)

                except Exception as e:
                    st.error("Error while reading CSV: {}".format(e))

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("CineSentiment - Movie Review Opinion Analyzer (ML + NLP + Streamlit)")
