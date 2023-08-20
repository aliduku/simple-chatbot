import random
import streamlit as st
import string
import re, string, unicodedata
import wikipedia as wk
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load or preprocess Wikipedia data
data = open('wiki_train_raw.txt', 'r', errors='ignore')
raw = data.read()
raw = raw.lower()
sent_tokens = nltk.sent_tokenize(raw)

def Normalize(text):
    remove_punct_dict = {ord(punct): None for punct in string.punctuation}
    word_token = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    new_words = []
    for word in word_token:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    rmv = []
    for w in new_words:
        text = re.sub("&lt;/?.*?&gt;", "&lt;&gt;", w)
        rmv.append(text)
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    lmtzr = WordNetLemmatizer()
    lemma_list = []
    rmv = [i for i in rmv if i]
    for token, tag in nltk.pos_tag(rmv):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        lemma_list.append(lemma)
    return lemma_list

welcome_input = ("hello", "hi", "greetings", "sup", "what's up","hey",)
welcome_response = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
def welcome(user_response):
    for word in user_response.split():
        if word.lower() in welcome_input:
            return random.choice(welcome_response)

def generateResponse(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = linear_kernel(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    return (
        wikipedia_data(user_response)
        if (req_tfidf == 0) or "tell me about" in user_response
        else robo_response + sent_tokens[idx]
    )

# Wikipedia search
def wikipedia_data(input):
    reg_ex = re.search('tell me about (.*)', input)
    try:
        if reg_ex:
            topic = reg_ex[1]
            return wk.summary(topic, sentences=3)
    except Exception as e:
        return "No content has been found"

# Streamlit app
def main():
    st.title("WikiChat - Your Wikipedia Chatbot")

    st.write("My name is Chatterbot. Let's have a conversation! If you want to exit, just type 'Bye'.")

    user_input = st.text_area("You:", "")
    submit_button = st.button("Submit")  # Add a "Submit" button

    if submit_button and user_input:  # Wait for button click and non-empty input
        user_input = user_input.lower()
        if user_input in ['bye', 'shutdown', 'exit', 'quit']:
            st.write("Chatterbot: Bye!!!")

        elif(welcome(user_input) != None):
            st.write(f"Chatterbot: {welcome(user_input)}")
        
        else:
            response = generateResponse(user_input)
            st.write("Chatterbot:", response)

if __name__ == "__main__":
    main()
