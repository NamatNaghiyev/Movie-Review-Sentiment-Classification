import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import streamlit as st

MODEL_FILE = 'naive_bayes_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
DATASET_FILE = 'IMDB_Dataset_20MB.csv'

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        tfidf = joblib.load(VECTORIZER_FILE)
    else:
        st.info("Model tapılmadı, təlim olunur...")

        df = pd.read_csv(DATASET_FILE)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

        X = df['review']
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
        X_train_tfidf = tfidf.fit_transform(X_train)

        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        joblib.dump(tfidf, VECTORIZER_FILE)
        joblib.dump(model, MODEL_FILE)

        st.success("Model təlim olundu və yadda saxlanıldı.")
    return model, tfidf

# Load model və vectorizer
model, tfidf = load_or_train_model()

# -------- Streamlit Tətbiqi --------
st.title("Film Rəyi Təhlili - Naive Bayes Modeli ilə (Fast Version)")

st.markdown("""
Bu tətbiqə film rəyi yazın və rəyin **müsbət** (positive) ya da **mənfi** (negative) olduğunu öyrənin.  
Model: **Naive Bayes**, Təlim edilmiş versiya
""")

user_review = st.text_area("Film rəyi daxil edin:")

if st.button("Təhlil et"):
    if user_review.strip() == "":
        st.warning("Zəhmət olmasa rəy yaz, qaqa.")
    else:
        review_tfidf = tfidf.transform([user_review])
        prediction = model.predict(review_tfidf)
        prediction_proba = model.predict_proba(review_tfidf)

        if prediction[0] == 1:
            st.success(f"✅ Bu rəy **MÜSBƏT** qiymətləndirilib! ({prediction_proba[0][1]*100:.2f}% əminliklə)")
        else:
            st.error(f"❌ Bu rəy **MƏNFI** qiymətləndirilib! ({prediction_proba[0][0]*100:.2f}% əminliklə)")
