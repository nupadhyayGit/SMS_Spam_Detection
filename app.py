import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
#For stemming
from nltk.stem.porter import PorterStemmer
from pyarrow import nulls

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))
#process_func = pickle.load(open('message_processing.pkl', 'rb'))

#Function for preprocessing sms
def textmessage_processing(text):
    #Lowercase Conversion
    text = text.lower()

    #Tokenization
    text = nltk.word_tokenize(text)

    #Removing special characters and retaining numbers and alphabets
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    #Removing stopwords and punctuations
    text = y.copy()
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    #Stemming
    text = y.copy()
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#Creating UI for the website

st.title("SMS Spam Classifier")

#Input message from the user
input_msg = st.text_area('Enter the message')

if input_msg:
    #step1 - Preprocessing the message received from the user
    transformed_sms = textmessage_processing(input_msg)
    #step2 - Vectorize the transformed sms message
    vectorized_sms = tfidf.transform([transformed_sms])
    if st.button('Spam or Not?'):
        #step3 - predict if spam or not using Naive Bayes Model
        result = model.predict(vectorized_sms)[0]

        #step4 - display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
else:
    #st.header("No input message provided")
    st.button('Spam or Not?', disabled=True)
