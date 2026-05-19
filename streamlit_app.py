import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

st.set_page_config(page_title="AI Sentiment Dashboard", layout="centered")

# ------------------------------
# Load Model
# ------------------------------

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# ------------------------------
# Header
# ------------------------------

st.title("🧠 AI-Powered Ecommerce Sentiment Analysis")
st.write("Real-time customer sentiment analysis using DistilBERT")

# ------------------------------
# Single Review Analysis
# ------------------------------

st.header("✍️ Single Review Analysis")

user_input = st.text_area(
    "Enter Customer Review"
)

if st.button("Analyze Sentiment"):

    if user_input.strip() != "":

        result = classifier(user_input)[0]

        sentiment = result["label"]
        confidence = round(result["score"] * 100, 2)

        if sentiment == "POSITIVE":
            st.success(f"Positive Sentiment")

        else:
            st.error(f"Negative Sentiment")

        st.info(f"Confidence Score: {confidence}%")

# ------------------------------
# CSV Dataset Analysis
# ------------------------------

st.header("📂 Dataset-Level Analysis")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data)

    if "Review" in data.columns:

        predictions = []

        for review in data["Review"]:

            result = classifier(str(review))[0]

            if result["label"] == "POSITIVE":
                predictions.append("Positive")
            else:
                predictions.append("Negative")

        data["Predicted_Sentiment"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(data)

        # ------------------------------
        # FIXED SENTIMENT DISTRIBUTION
        # ------------------------------

        positive_count = (data["Predicted_Sentiment"] == "Positive").sum()
        negative_count = (data["Predicted_Sentiment"] == "Negative").sum()

        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [positive_count, negative_count]
        })

        st.subheader("📊 Sentiment Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.bar(
            sentiment_df["Sentiment"],
            sentiment_df["Count"]
        )

        ax.set_title("Sentiment Distribution")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Reviews")

        # Show values above bars
        for i, value in enumerate(sentiment_df["Count"]):
            ax.text(i, value + 0.1, str(value), ha='center')

        st.pyplot(fig)

        # Statistics
        st.subheader("📈 Statistics")

        st.write(f"✅ Positive Reviews: {positive_count}")
        st.write(f"❌ Negative Reviews: {negative_count}")

    else:
        st.error("CSV must contain a 'Review' column")

# ------------------------------
# Footer
# ------------------------------

st.markdown("---")
st.caption("Built using Streamlit + DistilBERT + Hugging Face Transformers")
