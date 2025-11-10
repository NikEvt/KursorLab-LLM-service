import sys
import os
import nltk
import sentence_transformers
import chromadb
from chromadb import Client,Collection
from chromadb.utils import embedding_functions

def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

if __name__ == '__main__':
    setup_nltk()