import streamlit as st
import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

# Load or preprocess wikitext-2 data
f = open("wiki_train_raw.txt", "r", errors="ignore")
raw_doc = f.read()
raw_doc = raw_doc.lower()

sent_token = nltk.sent_tokenize(raw_doc)
word_token = nltk.word_tokenize(raw_doc)

print(f"Number of sentence : {len(sent_token)}")
print(f"Number of Words in : {len(word_token)}")

lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = {ord(punct):None for punct in string.punctuation}
def lem_token(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def lem_normalize(text):
    return lem_token(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

preprocessed_sentences = [lem_normalize(sentence) for sentence in sent_token]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

# Greeting responses
GREET_INPUT = ("hello", "hi", "sup")
GREET_RESPONSE = ["Hi", "Hello", "I am glad you are talking to me!"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUT:
            return random.choice(GREET_RESPONSE)
        
def response(user_response):
    sent_token.append(user_response)
    tfidfvec = TfidfVectorizer(tokenizer = lem_normalize,stop_words = 'english')
    tfidf = tfidfvec.fit_transform(sent_token)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    sent_token.remove(user_response)
    if (req_tfidf == 0):
        return " I am Sorry! I dont understand you"
    else:
        return str(sent_token[idx])

# Streamlit app
def main():
    st.title("WikiChat - Your Wikipedia Chatbot")

    st.write("My name is BOT. Let's have a conversation! If you want to exit, just type 'Bye'.")

    user_input = st.text_area("You:", "")
    submit_button = st.button("Submit")  # Add a "Submit" button

    if submit_button and user_input:  # Wait for button click and non-empty input
        if user_input.lower() == "bye":
            st.write("BOT: Goodbye! Take care.")
        else:
            if greet(user_input) is not None:
                st.write("BOT:", greet(user_input))
            else:
                st.write("BOT:", response(user_input))

if __name__ == "__main__":
    main()
