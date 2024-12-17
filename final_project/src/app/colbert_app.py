import warnings
warnings.filterwarnings('ignore')
import nltk
import streamlit as st
from ragatouille import RAGPretrainedModel
import pandas as pd
import time
import datetime
from sentence_transformers import SentenceTransformer
from voyager import Index, Space
import json 
import pickle 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from math import log
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Media Metrics",
    page_icon="🌍",
    layout="wide", 
    initial_sidebar_state="expanded",
)


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


class MyExistingRetrievalPipeline:
    index: Index
    embedder: SentenceTransformer

    def __init__(self, embedder_name: str = "BAAI/bge-small-en-v1.5"):
        self.embedder = SentenceTransformer(embedder_name)
        self.collection_map = {}
        self.index = Index(
            Space.Cosine,
            num_dimensions=self.embedder.get_sentence_embedding_dimension(),
        )

    def index_documents(self, documents: list[str]) -> None:
        # There's very few documents in our example, so we don't bother with batching
        for document in documents:
            self.collection_map[self.index.add_item(self.embedder.encode(document['content']))] = document['content']

    def query(self, query: str, k: int = 10) -> list[str]:
        query_embedding = self.embedder.encode(query)
        to_return = []
        for idx in self.index.query(query_embedding, k=k)[0]:
            to_return.append(self.collection_map[idx])
        return to_return


existing_pipeline = MyExistingRetrievalPipeline()

def query_and_display_strings(query_string, map, k=20):

    query_embedding = existing_pipeline.embedder.encode(query_string)
    results = q1.query(query_embedding, k=k) 
    retrieved_strings = []  
    for idx in results[0]:
        try:
          retrieved_string =map[str(idx)]
          retrieved_strings.append(retrieved_string)
        except KeyError:
          print(f"Warning: No string found for index {idx}")

    return retrieved_strings 


def load_bm25_with_df(file_path):
    """Load a precomputed BM25 model and its associated DataFrame."""
    with open(file_path, "rb") as f:
        model_data = pickle.load(f)
    bm25 = BM25(precomputed=model_data["bm25"])  # Reconstruct BM25 from saved attributes
    slim_df = model_data["slim_df"]  # Retrieve the DataFrame
    return bm25, slim_df




q1 = existing_pipeline.index.load("../.ragatouille/colbert/indexes/voyager/test1.bin")

with open("../.ragatouille/colbert/indexes/voyager/test1_map.json", "r") as file:
  map = json.load(file)

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")



if 'q1' not in st.session_state:

    st.session_state.q1 = q1

q1 = st.session_state.q1

if 'map' not in st.session_state:

    st.session_state.map = map

map = st.session_state.map

if 'RAG' not in st.session_state:

    st.session_state.RAG = RAG

RAG = st.session_state.RAG



# Load colbert index 
path_to_index = "../.ragatouille/colbert/indexes/b00_split2/"
RAG1 = RAGPretrainedModel.from_index(path_to_index)




if 'bm25' not in st.session_state:
    bm25, slim_df = load_bm25_with_df("bm25_with_df.pkl")
    st.session_state.bm25 = bm25
    st.session_state.slim_df = slim_df

bm25 = st.session_state.bm25
slim_df = st.session_state.slim_df


st.title("Informed Retrieval")

if 'index' not in st.session_state:

    st.session_state.index = RAG1

index = st.session_state.index



with st.sidebar:
    st.write("**DSAN 5400 Project**")
    st.write("**Group members:**")



tab1, tab2, tab3, tab4, tab5 = st.tabs(["Colbert" , "Voyager","Colbert Rerank" ,"Chatbot", "BM25"])

with tab1:

    query_text = st.text_input("Enter your query:",  key="1")


    if query_text:

        start_time = time.time()
        top_results = index.search(query= query_text, k=10)
        end_time = time.time()
        st.write("Most relevant results generated in:", round(end_time - start_time, 4), "seconds")
        
        #st.write(f"Most relevant documents:\n\n")
        for i in top_results:
            #st.write(f"**Content**: {i['content']}")
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Source**: {i['document_metadata']['source']}")
            col1.write(f"**Country**: {i['document_metadata']['country']}")

            col2.write(f"**Publisher Bias**: {i['document_metadata']['bias']}")
            d = datetime.datetime.strptime(str(i['document_metadata']['DATE'])[:8], '%Y%m%d')
            
            col2.write(f"**Date**:{d}")
            
            if i['document_metadata']['factual_reporting'] == 'low':
                col3.write(f"**Publisher Factuality**: :red[{i['document_metadata']['factual_reporting']}]")
            elif i['document_metadata']['factual_reporting'] == 'mixed': 
                col3.write(f"**Publisher Factuality**: :orange[{i['document_metadata']['factual_reporting']}]")
            else:
                col3.write(f"**Publisher Factuality**: :green[{i['document_metadata']['factual_reporting']}]")
            
            if i['document_metadata']['Avg_Tone'] < 0:
                col3.write(f"**Document Tone**: :red[{i['document_metadata']['Avg_Tone']}]")
            else:
                col3.write(f"**Document Tone**: :green[{i['document_metadata']['Avg_Tone']}]")
                
            #st.write(f"**Country**: {i['document_metadata']['country']}")
            #st.write(f"**Publisher Bias**: {i['document_metadata']['bias']}")
            #st.write(f"**Publisher Factuality**: {i['document_metadata']['factual_reporting']}")
            #st.write(f"**Document Tone**: {i['document_metadata']['Avg_Tone']}")
            st.write(f"**Document Link**: {i['document_metadata']['DocumentIdentifier']}")
            st.write(f"**Content**: {i['content'][:280]} ... ")
            
            with st.expander("expand to read more:"):
                st.write(i['content'])
            
            st.divider()
            
            #st.write("="*90)
            
            
with tab3:

    query_text = st.text_input("Enter your query:",  key="2")

    if query_text:

        start_time = time.time()
        l = query_and_display_strings(query_text, map)
        results = RAG.rerank(query=query_text, documents=l, k=5)
        end_time = time.time()
        st.write("Most relevant results generated in:", round(end_time - start_time, 4), "seconds")
        
        for i in results: 
            st.write(f"**Content**: {i['content'][:280]} ... ")
            
            with st.expander("expand to read more:"):
                st.write(i['content'])
            
            st.divider()
            
            #st.write("="*90)
            
with tab2:

    query_text = st.text_input("Enter your query:",  key="3")

    if query_text:

        start_time = time.time()
        l = query_and_display_strings(query_text, map, 10)
        #results = RAG.rerank(query=query_text, documents=l, k=5)
        end_time = time.time()
        st.write("Most relevant results generated in:", round(end_time - start_time, 4), "seconds")
        
        for i in l: 
            st.write(f"**Content**: {i[:280]} ... ")
            
            with st.expander("expand to read more:"):
                st.write(i)
            
            st.divider()
            
            #st.write("="*90)
            
            
with tab4: 
    
    
    #col1, col2 = st.columns(2)
    
    messages = st.container(height=500)
    prompt = st.chat_input("How can I help you?")
    if prompt:
        messages.chat_message("user").write(prompt)
        top_results = index.search(query= prompt, k=1)
        
        for i in top_results:
            messages.chat_message("assistant").write(f"Informed Bot: {i['content']}")
            
with tab5:
    
    query_text = st.text_input("Enter your query:",  key="4")
    
    if query_text:
        # Query the BM25 model
        top_results = bm25.query(query_text)
        
        # Display the results
        st.write(f"Top {len(top_results)} results:")
        for idx, score in top_results:
            st.write(f"**Document ID**: {slim_df.iloc[idx]['DocumentIdentifier']}")
            st.write(f"**Score**: {score}")
            st.write(f"**Content**: {slim_df.iloc[idx]['content'][:300]}...")
            st.divider()
            
            
# @st.cache_data
# def load_data():
#     df = pd.read_csv("../english_mbfc.csv")
#     dfx = pd.read_csv("../mbfc.csv")
#     df = df.merge(dfx, left_on='base_url', right_on='source')
#     df['year'] = df['DATE'].astype(str).str[:4]
#     df['year'] = pd.to_numeric(df['year'])
#     bias_hierarchy = {'left': -1, 'left-center': -0.5, 'neutral': 0, 'right-center': 0.5, 'right': 1}
#     df['bias_encoded'] = df['bias'].map(bias_hierarchy)
#     df['country'] = df['country'].replace({'usa (44/180 press freedom)':'united states', 'usa (45/180 press freedom)':'united states', 'usa': 'united states','guam (us territory)':'united states',
#                                         'united kingdom (scotland)':'united kingdom', 'united kingsom':'united kingdom', 'northern ireland (uk)': 'united kingdom',
#                                          'italy (vatican city)':'italy'})
#     df['country'] = df['country'].str.title()
#     factual_mapping = {'low': 0, 'mixed': 1, 'high': 2}
#     df['factual_numeric'] = df['factual_reporting'].map(factual_mapping)
#     return df

# df = load_data()

# with tab6:
#     st.header("Worldview of Media Metrics")

#     min_tonality, max_tonality = df['Avg_Tone'].min(), df['Avg_Tone'].max()
#     tonality_range = st.slider(
#         "Filter Tonality",
#         min_value=float(df['Avg_Tone'].min()),
#         max_value=float(df['Avg_Tone'].max()),
#         value=(float(df['Avg_Tone'].min()), float(df['Avg_Tone'].max())),
#         step=0.1
#     )

#     factual_reporting_slider = st.slider(
#         "Filter Factual Reporting", 
#         min_value=0, max_value=2, value=(0, 2), step=1
#     )
    
#     factual_mapping = {0: 'low', 1: 'mixed', 2: 'high'}
#     factual_reporting_category = [factual_mapping[val] for val in range(factual_reporting_slider[0], factual_reporting_slider[1] + 1)]

#     st.write("Factual Reporting Categories: 0 = low , 1 = mixed, 2 = high")

#     filtered_df = df[
#         (df['Avg_Tone'] >= tonality_range[0]) & 
#         (df['Avg_Tone'] <= tonality_range[1])
#     ]

#     # aggregate data by country for the selected feature
#     aggregated_df = filtered_df.groupby('country').agg(
#         avg_bias=('bias_encoded', 'mean'),
#         avg_tonality=('Avg_Tone', 'mean'),
#         avg_factual_reporting=('factual_numeric', 'mean'), 
#         count=('bias_encoded', 'count')
#     ).reset_index()

#     # filter aggregated data based on factual reporting average
#     aggregated_df = aggregated_df[
#         (aggregated_df['avg_factual_reporting'] >= factual_reporting_slider[0]) &
#         (aggregated_df['avg_factual_reporting'] <= factual_reporting_slider[1])
#     ]

#     # create map
#     fig = px.choropleth(
#         aggregated_df,
#         locations="country",
#         locationmode="country names",
#         color="avg_bias",
#         hover_name="country",
#         color_continuous_scale="Plasma",
#         title="Map",
#         labels={"color": "Bias"}
#     )

#     fig.update_layout(
#         geo=dict(
#             showframe=True,
#             showcoastlines=True,
#             projection_type="natural earth",
#             coastlinecolor='#FFFFFF',
#             framecolor='#FFFFFF',
#             showcountries=True,
#             countrycolor='#FFFFFF'),
#         title=dict(
#             text=f"<b>Bias Map</b>",
#             x=0.5, 
#             xanchor="center",
#             font=dict(size=20)),
#         coloraxis_colorbar=dict(
#             title="Bias",
#             tickvals=[-1, -0.5, 0, 0.5, 1], 
#             ticktext=["Left", "Left Center", "Neutral", "Right Center", "Right"]),
#         paper_bgcolor="#374151",
#         font_color="#f9fafb",
#         margin={'r':0,'t':0,'l':0,'b':0},
#         geo_bgcolor= "#374151"
#     )

#     st.plotly_chart(fig, use_container_width=True, height=800)  