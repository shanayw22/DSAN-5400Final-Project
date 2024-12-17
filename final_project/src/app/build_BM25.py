import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from math import log
import os
import nltk
import pickle
import streamlit as st

# Download the punkt tokenizer (if not already downloaded)
nltk.download("punkt")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
import pickle

def save_bm25_with_df(bm25, slim_df, file_path):
    model_data = {
        "bm25": {
            "corpus": bm25.corpus,
            "doc_lengths": bm25.doc_lengths,
            "avgdl": bm25.avgdl,
            "N": bm25.N,
            "doc_freqs": bm25.doc_freqs,
            "k1": bm25.k1,
            "b": bm25.b,
        },
        "slim_df": slim_df,
    }
    with open(file_path, "wb") as f:
        pickle.dump(model_data, f)

class BM25:
    def __init__(self, corpus=None, k1=1.2, b=0.75, precomputed=None):
        self.k1 = k1
        self.b = b
        self.stop_words = set(stopwords.words("english"))  # Stopwords set
        self.lemmatizer = WordNetLemmatizer()  # Lemmatizer
        
        if precomputed:
            self.corpus = precomputed["corpus"]
            self.doc_lengths = precomputed["doc_lengths"]
            self.avgdl = precomputed["avgdl"]
            self.N = precomputed["N"]
            self.doc_freqs = precomputed["doc_freqs"]
        else:
            self.corpus = [self.preprocess(doc) for doc in corpus]
            self.doc_lengths = [len(doc) for doc in self.corpus]
            self.avgdl = np.mean(self.doc_lengths)
            self.N = len(self.corpus)
            self.doc_freqs = self._calculate_doc_frequencies()

    def preprocess(self, text):
        """Tokenizes, removes stopwords, and lemmatizes the input text."""
        if isinstance(text, str):
            tokens = word_tokenize(text.lower())  # Tokenize and lowercase
            tokens = [t for t in tokens if t.isalpha()]  # Keep only alphabetic tokens
            tokens = [t for t in tokens if t not in self.stop_words]  # Remove stopwords
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize tokens
            return tokens
        return []

    def _calculate_doc_frequencies(self):
        """Calculate document frequencies for each term in the corpus."""
        doc_freqs = Counter()
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] += 1
        return doc_freqs

    def idf(self, term):
        """Calculate the IDF of a term."""
        n_t = self.doc_freqs.get(term, 0)
        return log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)

    def bm25_score(self, query, doc_index):
        """Calculate BM25 score for a single document and query."""
        doc = self.corpus[doc_index]
        doc_length = self.doc_lengths[doc_index]
        score = 0
        for term in query:
            f_t_d = doc.count(term)
            numerator = f_t_d * (self.k1 + 1)
            denominator = f_t_d + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
            score += self.idf(term) * (numerator / denominator)
        return score

    def query(self, query_text, top_n=10):
        """Rank documents based on BM25 score for a query."""
        query = self.preprocess(query_text)
        scores = [(idx, self.bm25_score(query, idx)) for idx in range(self.N)]
        top_results = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        
        return top_results

def read_and_concat_csv(folder_path):
    """Reads and concatenates all CSV files in a folder."""
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

def load_bm25_model(file_path):
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)
    return BM25(precomputed=model_data)

folder_path = "scrapedata"
final_df = read_and_concat_csv(folder_path)
slim_df = final_df.dropna(subset=["content"])[["DocumentIdentifier", "content"]]
slim_df["content"] = slim_df["content"].astype(str)

bm25 = BM25(slim_df["content"].tolist())
save_bm25_with_df(bm25, slim_df, "bm25_with_df.pkl")
