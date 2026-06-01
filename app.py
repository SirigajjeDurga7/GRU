import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* Text Area */
textarea {
    color: black !important;
    background-color: white !important;
    font-size: 16px !important;
}

/* Placeholder */
textarea::placeholder {
    color: gray !important;
}

/* Positive Box */
.sentiment-positive{
    background-color:#d4edda;
    color:black !important;
    padding:20px;
    border-radius:12px;
    border:2px solid #28a745;
    margin-top:10px;
}

/* Negative Box */
.sentiment-negative{
    background-color:#f8d7da;
    color:black !important;
    padding:20px;
    border-radius:12px;
    border:2px solid #dc3545;
    margin-top:10px;
}

/* Force all text black */
.sentiment-positive h1,
.sentiment-positive h2,
.sentiment-positive h3,
.sentiment-positive p,
.sentiment-negative h1,
.sentiment-negative h2,
.sentiment-negative h3,
.sentiment-negative p{
    color:black !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

simple_rnn_model = tf.keras.models.load_model(
    "simple_rnn_model.h5"
)

lstm_model = tf.keras.models.load_model(
    "lstm_model.h5"
)

gru_model = tf.keras.models.load_model(
    "gru_model.h5"
)

# --------------------------------------------------
# IMDB WORD INDEX
# --------------------------------------------------

word_index = imdb.get_word_index()

VOCAB_SIZE = 10000
MAX_LENGTH = 200

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def encode_review(text):

    words = text.lower().split()

    encoded = []

    for word in words:

        if word in word_index:

            idx = word_index[word] + 3

            if idx < VOCAB_SIZE:
                encoded.append(idx)

    return encoded


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_sentiment(review, model):

    sequence = encode_review(review)

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0][0]

    positive_prob = float(prediction)
    negative_prob = 1 - positive_prob

    sentiment = (
        "Positive"
        if positive_prob >= 0.5
        else "Negative"
    )

    confidence = max(
        positive_prob,
        negative_prob
    )

    return (
        sentiment,
        confidence,
        positive_prob,
        negative_prob
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(
    "🎬 Movie Review Sentiment Analysis System"
)

st.subheader(
    "Deep Learning Based Sentiment Classification"
)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

selected_model = st.radio(
    "Select Model",
    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# --------------------------------------------------
# REVIEW INPUT
# --------------------------------------------------

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        model_map = {

            "SimpleRNN": simple_rnn_model,
            "LSTM": lstm_model,
            "GRU": gru_model
        }

        model = model_map[selected_model]

        sentiment, confidence, pos, neg = \
            predict_sentiment(
                review,
                model
            )

        # ------------------------------------------
        # SENTIMENT CARD
        # ------------------------------------------

        if sentiment == "Positive":

            st.markdown(
                f"""
                <div class="sentiment-positive">
                    <h2>😊 Sentiment: {sentiment}</h2>
                    <h3>Confidence: {confidence*100:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="sentiment-negative">
                    <h2>😞 Sentiment: {sentiment}</h2>
                    <h3>Confidence: {confidence*100:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ------------------------------------------
        # PROBABILITY CHART
        # ------------------------------------------

        st.subheader(
            "Probability Distribution"
        )

        prob_df = pd.DataFrame({

            "Class":
            ["Positive", "Negative"],

            "Probability":
            [pos, neg]
        })

        st.bar_chart(
            prob_df.set_index("Class")
        )

        # ------------------------------------------
        # MODEL COMPARISON
        # ------------------------------------------

        st.subheader(
            "Compare Predictions from All Models"
        )

        results = []

        for name, mdl in {

            "SimpleRNN": simple_rnn_model,
            "LSTM": lstm_model,
            "GRU": gru_model

        }.items():

            s, c, p, n = predict_sentiment(
                review,
                mdl
            )

            results.append([
                name,
                s,
                round(c*100, 2)
            ])

        comparison_df = pd.DataFrame(

            results,

            columns=[
                "Model",
                "Sentiment",
                "Confidence (%)"
            ]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        # ------------------------------------------
        # CONFIDENCE CHART
        # ------------------------------------------

        st.subheader(
            "Confidence Comparison"
        )

        st.bar_chart(
            comparison_df.set_index(
                "Model"
            )["Confidence (%)"]
        )