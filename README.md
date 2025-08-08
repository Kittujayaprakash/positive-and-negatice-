import streamlit as st
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


nltk.download('stopwords')

# Load and preprocess data
df = pd.read_csv(r"C:\Users\atlur\OneDrive\Desktop\covid-19_vaccine_tweets_with_sentiment.csv", encoding='latin1')

def clean_tweet_text(tweet_text):
    if pd.isnull(tweet_text):
        return ''
    tweet_text = tweet_text.lower()
    tweet_text = re.sub(r"http\S+|www\S+|https\S+", '', tweet_text)
    tweet_text = re.sub(r"[^a-zA-Z\s]", '', tweet_text)
    tokens = tweet_text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_tweet_text'] = df['tweet_text'].apply(clean_tweet_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_tweet_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# ...existing code...

model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Streamlit UI
st.title("COVID-19 Vaccine Tweet Sentiment Analysis")

tweet_input = st.text_area("Enter a tweet to analyze sentiment:")

if st.button("Predict Sentiment"):
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    cleaned = clean_tweet_text(tweet_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    st.write(f"Predicted Sentiment Label: {prediction}")
