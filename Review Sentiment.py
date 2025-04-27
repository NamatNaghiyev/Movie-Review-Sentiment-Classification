# Lazımi kitabxanalar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Dataset-i oxuyuruq
df = pd.read_csv('IMDB Dataset.csv')  # Faylın adını uyğun yaz
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train-test bölürük
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model: Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# -------- Streamlit Tətbiqi --------
st.title("🎬 Film Rəyi Təhlili - Naive Bayes Modeli ilə")

st.write("""
Bu tətbiqə film rəyi yazın və rəyin **müsbət** (positive) ya da **mənfi** (negative) olduğunu öyrənin.  
Model: **Naive Bayes**
""")

user_review = st.text_area("✍️ Film rəyi daxil edin:")

if st.button("Təhlil et"):
    if user_review.strip() == "":
        st.warning("Zəhmət olmasa bir rəy daxil edin.")
    else:
        review_tfidf = tfidf.transform([user_review])
        prediction = model.predict(review_tfidf)
        prediction_proba = model.predict_proba(review_tfidf)
        
        if prediction[0] == 1:
            st.success(f"✅ Bu rəy **MÜSBƏT** (positive) qiymətləndirilib! ({prediction_proba[0][1]*100:.2f}% əminliklə)")
        else:
            st.error(f"❌ Bu rəy **MƏNFI** (negative) qiymətləndirilib! ({prediction_proba[0][0]*100:.2f}% əminliklə)")
