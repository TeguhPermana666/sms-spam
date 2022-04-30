import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps =PorterStemmer()

# 1 preprocess
def transform_text(text):
    # lower case
    text = text.lower()
    #tokenization
    text = nltk.word_tokenize(text)
    # removing the spesial charcter
    y=[]
    for i in text:
        if i.isalnum():#text content aplhanumeric 
            y.append(i)
    text=y[:]
    y.clear()# remoce all value on list
    
    #removing stop wors and punctuation
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:# yang ada di stop words maka ga include dengan y
            y.append(i)
    text=y[:]
    y.clear()
    
    #steemming => only basic word
    for i in text :
        y.append(ps.stem(i))
    # text=y[:]
    # y.clear()
    return " ".join(y)

tfdif = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("Email \ SMS Spam Classifier")

input_sms = st.text_area("Enter The Message")
if st.button("Classification"):
    transform_sms=transform_text(input_sms)
    # 2. vectorize
    vector_input=tfdif.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("SPAM !!!")
    else :
        st.header("NOT SPAM !!")

