# Conversational Chatbot using NLTK and Streamlit

This repository contains a simple conversational chatbot implemented using NLTK (Natural Language Toolkit) and Streamlit. The chatbot is built to respond to user input by comparing the similarity of the input with pre-defined responses using two different text representation techniques: Bag of Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF).

## Overview

The chatbot performs the following steps:

1. **Data Loading:** The conversation data is loaded from an Excel file named `dialog_talk_agent.xlsx`. The dataset is pre-processed to fill missing values using forward fill.

2. **Text Normalization:** The user input is normalized by converting it to lowercase, removing special characters, and lemmatizing the tokens.

3. **Stopword Removal:** Common stopwords are removed from the normalized text to improve the quality of input for further processing.

4. **Bag of Words (BOW):** The normalized and stopwords-removed text is transformed into a numerical vector using the CountVectorizer from Scikit-learn. The BOW matrix is created, and the cosine similarity is calculated between the user input and each conversation in the dataset.

5. **TF-IDF:** Similarly, the TF-IDF vectorization technique is applied to the normalized text. The TF-IDF matrix is created, and cosine similarity is calculated.

6. **Chatbot Responses:** Based on the cosine similarity, the chatbot selects a response from the conversation dataset. It returns the response obtained from both the BOW and TF-IDF methods.

## Dependencies

- Python 3.x
- Streamlit
- NLTK
- Scikit-learn
- Pandas

## Usage

1. Install the required libraries using `pip install streamlit nltk scikit-learn pandas`.

2. Download the NLTK resources by running the code for downloading resources in the code file.

3. Place your conversation data in an Excel file named `dialog_talk_agent.xlsx`.

4. Run the Streamlit app by executing the code provided in the code file.

5. Input your message in the Streamlit app and click the "Submit" button to get responses from the chatbot.

## Notes

- This chatbot demonstrates a basic approach to building a conversational agent using text similarity. Advanced techniques and larger datasets could lead to more accurate and engaging chatbots.

- Feel free to customize and expand upon this code to further improve the chatbot's capabilities and interaction with users.

For questions or suggestions, please feel free to reach out.
