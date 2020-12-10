import streamlit as st
import joblib
from spacy.lang.en import English
import string
from sklearn.base import TransformerMixin
from spacy.lang.en.stop_words import STOP_WORDS
import time


class Predictors(TransformerMixin):

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


stopwords = list(STOP_WORDS)
parser = English()
punctuation = string.punctuation


def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    # lemmatize
    tokens = [word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in tokens]
    # remove punctuations and stop words.
    tokens = [word for word in tokens if word not in stopwords and word not in punctuation]
    return tokens


# basic function to clean the text
def clean_text(text):
    return text.lower().strip()


# The about page
def about():
    st.header('About app')
    st.write('1. This sentiment analysis app can be used by organizations to analyse the customer reviews for \
         better decision making.')
    st.write('2. It relieves businesses off the tasks of having to sort the customer reviews manually.')
    st.write(' ')
    st.write(' ')
    st.write('Model accuracy => 80%')
    st.write("Written by - Regan Muthomi")
    st.subheader('Contact Me.')
    st.write('Email: regansomi@gmail.com')


def main():
    model = joblib.load('sentiment_model')
    st.title('Product Reviews')
    st.subheader('Enter your reviews or texts, let the app do the rest')
    dropdowns = ['HOME', 'ABOUT']
    options = st.sidebar.selectbox('Select action', dropdowns)
    if options == 'HOME':
        st.write('For more information about the app, go to about page')
        text = st.text_area('Enter Text here', max_chars=2000)
        if text is not None:
            st.sidebar.header('Sentiment')
            st.sidebar.write(text.upper())
            if st.button('Analyse'):
                prediction = model.predict([text])[0]
                if prediction == 1:
                    pred = 'POSITIVE'
                    with st.spinner('Loading ...'):
                        time.sleep(5)
                        st.write('Prediction:', pred)
                elif prediction == 0:
                    pred = 'NEGATIVE'
                    with st.spinner('Loading ...'):
                        time.sleep(5)
                        st.write('Prediction:', pred)

    elif options == 'ABOUT':
        return about()


if __name__ == "__main__":
    main()
