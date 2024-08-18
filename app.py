import streamlit as st
import pickle

from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


tfidf= pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # converting to lower case
    text = nltk.word_tokenize(text)  # tokanizing the text to words

    # removing the special character
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y.copy()
    y = []

    # removing punctuation and stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y.copy()
    y = []

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title('SMS Spam Detection Classifier')

input_sms=st.text_area('Enter a message to classify')

if st.button('Predict'):
    #step 1 : pre process
    transformed_sms=transform_text(input_sms)

    #Step 2: vectorise the sms
    tfidf_sms=tfidf.transform([transformed_sms])

    # step 3: Predition
    prediction=model.predict(tfidf_sms)[0]

    #step 4: display the prediction:
    if prediction==1:
        st.header('SPAM')
    else:
        st.header('Not SPAM')
