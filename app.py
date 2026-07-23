import streamlit as st
import joblib
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf.pkl")

stop_words = ENGLISH_STOP_WORDS

# Text Cleaning Function
def clean_text(text):
    text = text.lower()

    words = text.split()

    cleaned = []

    for word in words:
        word = word.strip(string.punctuation)

        if word not in stop_words and word != "":
            cleaned.append(word)

    return " ".join(cleaned)


# Page Configuration
st.set_page_config(
    page_title="Spam Detection",
    page_icon="📩",
    layout="centered"
)

# Title
st.title("📩 Spam Message Detection")
st.write("Predict whether a message is **Spam** or **Ham** using Machine Learning.")

st.divider()

# Input Text
message = st.text_area(
    "Enter your message:",
    height=180,
    placeholder="Type your SMS here..."
)

# Prediction Button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        clean_msg = clean_text(message)

        vector = tfidf.transform([clean_msg])

        prediction = model.predict(vector)

        probability = model.predict_proba(vector)

        spam_prob = probability[0][1] * 100
        ham_prob = probability[0][0] * 100

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("🚨 This is a SPAM message.")
        else:
            st.success("✅ This is a HAM message.")

        st.write("---")

        st.write(f"**Spam Probability :** {spam_prob:.2f}%")
        st.progress(float(spam_prob)/100)

        st.write(f"**Ham Probability :** {ham_prob:.2f}%")
        st.progress(float(ham_prob)/100)

st.divider()

st.caption("Spam Detection using NLP + TF-IDF + Multinomial Naive Bayes")