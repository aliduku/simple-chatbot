import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import pairwise_distances
import pandas as pd

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
conversation_df = pd.read_excel("dialog_talk_agent.xlsx")
conversation_df.ffill(axis=0, inplace=True)

# Normalization
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]', " ", text)
    tokens = word_tokenize(text)
    lemmatizer = wordnet.WordNetLemmatizer()
    
    tagged_tokens = pos_tag(tokens, tagset=None)
    token_lemmas = []
    
    for (token, pos_token) in tagged_tokens:
        if pos_token.startswith("V"):  # verb
            pos_val = "v"
        elif pos_token.startswith("J"):  # adjective
            pos_val = "a"
        elif pos_token.startswith("R"):  # adverb
            pos_val = "r"
        else:
            pos_val = 'n'  # noun
        token_lemmas.append(lemmatizer.lemmatize(token, pos_val))
        
    return " ".join(token_lemmas)

conversation_df["normalized_text"] = conversation_df["Context"].apply(normalize_text)

# Remove stopwords
def remove_stopwords(text):
    stop = stopwords.words("english")
    text = [word for word in text.split() if word not in stop]
    return " ".join(text)

# Bag of Words
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(conversation_df["normalized_text"]).toarray()
bow_features = bow_vectorizer.get_feature_names_out()
bow_df = pd.DataFrame(bow_matrix, columns=bow_features)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(conversation_df["normalized_text"]).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_vectorizer.get_feature_names_out())

# Cosine Similarity using BOW
def chat_bow(user_input):
    normalized_input = normalize_text(user_input)
    without_stopwords = remove_stopwords(normalized_input)
    bow_input = bow_vectorizer.transform([without_stopwords]).toarray()
    cosine_similarity = 1 - pairwise_distances(bow_df, bow_input, metric="cosine")
    index_of_most_similar = cosine_similarity.argmax()
    return conversation_df["Text Response"].loc[index_of_most_similar]

# Cosine Similarity using TF-IDF
def chat_tfidf(user_input):
    normalized_input = normalize_text(user_input)
    tfidf_input = tfidf_vectorizer.transform([normalized_input]).toarray()
    cosine_similarity = 1 - pairwise_distances(tfidf_df, tfidf_input, metric="cosine")
    index_of_most_similar = cosine_similarity.argmax()
    return conversation_df["Text Response"].loc[index_of_most_similar]

# Streamlit app
def main():
    st.title("Chatbot")

    st.write("Welcome to the Chatbot! Type your message below.")

    user_input = st.text_area("You:", "")
    submit_button = st.button("Submit")

    if submit_button and user_input:
        bow_response = chat_bow(user_input)
        tfidf_response = chat_tfidf(user_input)
        
        st.write("BOW Response:", bow_response)
        st.write("TF-IDF Response:", tfidf_response)

if __name__ == "__main__":
    main()
