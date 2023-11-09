import streamlit as st
import joblib
import pandas as pd
import re

# Create Filtering Text Function to remove unnecessery character in data
def filtering_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # Remove the characters inside the square brackets
    text = re.sub('https?://\S+|www\.\S+', "", text) # removes all URLs (both those starting with "http" or "https" and those starting with "www")
    text = re.sub('<.*?>+', '', text) # removes all HTML tags (including tags that are HTML markup)
    text = re.sub('\n', "", text) # removes all newline characters
    return text

# Loaded Model
lr = joblib.load("model/LogisticRegression().joblib")
dt = joblib.load("model/DecisionTreeClassifier().joblib")
rf = joblib.load("model/RandomForestClassifier().joblib")
tfidf = joblib.load("model/TfidfVectorizer(stop_words='english').joblib")

def output_label(output):
    if output == 0:
        return "This is a Fake News"
    elif output == 1:
        return "This is a Not a Fake News"

def predict(news):
    news_txt = {"text": [news]}
    news_txt_df = pd.DataFrame(news_txt)
    x_news_txt = news_txt_df["text"].apply(filtering_text)
    xv_news_txt = tfidf.transform(x_news_txt)
    y_pred_lr = lr.predict(xv_news_txt)
    y_pred_dt = dt.predict(xv_news_txt)
    y_pred_rf = rf.predict(xv_news_txt)
    return output_label(y_pred_lr), output_label(y_pred_dt), output_label(y_pred_rf)

st.markdown("<h1 style='text-align: center; color: grey;'>"
            "Welcome to Ash_Mine News Prediction Web Apps</h1>",
            unsafe_allow_html=True)

with st.form(key="myForm", clear_on_submit=True):
    user_input = st.text_input("Please write your news below: ", "")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    st.info("News:")
    st.write(user_input)
    st.write("")
    st.info("Prediction:")
    st.write("Prediction by Logistic Regression Algorithm: \n", predict(user_input)[0])
    st.write("Prediction by Decision Tree Algorithm: \n", predict(user_input)[1])
    st.write("Prediction by Random Forest Algorithm: \n", predict(user_input)[2])

