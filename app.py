#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import os
import spacy
import en_core_web_sm
# NLP Pkgs
from textblob import TextBlob
from IPython import get_ipython
#import spacy
from gensim.summarization import summarize

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function for Sumy Summarization
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# Function to Analyse Tokens and Lemma

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    # tokens = [ token.text for token in docx]
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in docx]
    return allData


# Function For Extracting Entities

def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    allData = ['"Token":{},\n"Entities":{}'.format(tokens, entities)]
    return allData


def main():
    """ NLP Based App with Streamlit """

    # Title
    st.title("NLP with Spacy and Streamlit")
    st.subheader("Natural Language Processing for Learning")
    st.markdown("""
        #### Description
        + This is a Natural Language Processing(NLP) Based App useful for basic NLP task
        Tokenization,NER,Sentiment,Summarization
        """)

    # Tokenization
    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize Your Text")

        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # Entity Extraction
    if st.checkbox("Show Named Entities"):
        st.subheader("Analyze Your Text")

        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Extract"):
            entity_result = entity_analyzer(message)
            st.json(entity_result)

    # Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Analyse Your Text")

        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Analyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize Your Text")

        message = st.text_area("Enter Text", "Type Here ..")
        summary_options = st.selectbox("Choose Summarizer", ['sumy', 'gensim'])
        if st.button("Summarize"):
            if summary_options == 'gensim':
                st.text("Using Gensim Summarizer ..")
                summary_result = summarize(message)
            else:
                st.warning("Using Default Summarizer")
                st.text("Using Gensim Summarizer ..")
                summary_result = summarize(message)

            st.success(summary_result)

    st.sidebar.subheader("About App")
    st.sidebar.text("NLP App with Streamlit")

    st.sidebar.subheader("By")
    st.sidebar.text("Mohnish")


if __name__ == '__main__':
    main()


# In[4]:


get_ipython().system('jupyter nbconvert   --to script app.ipynb')


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[6]:


get_ipython().system('pip freeze')


# In[ ]:




