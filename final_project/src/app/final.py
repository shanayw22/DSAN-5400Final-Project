import warnings
warnings.filterwarnings('ignore')
import torch
if hasattr(torch, "_classes"):
    torch._classes.__path__ = []
import streamlit as st
from ragatouille import RAGPretrainedModel
import pandas as pd
import time
import datetime
import plotly.express as px
import os
import gdown
import zipfile
import shutil
import watchdog
os.environ["STREAMLIT_LOG_LEVEL"] = "error"
# Google Drive file settings
GDRIVE_URL = 'https://drive.google.com/uc?id=1FGVx4jFMLf6ijxwkgqFyLeCU3yPWGvAe'
ZIP_FILE_PATH = 'final_project/src/ragatouille.zip'
EXTRACT_TO_PATH = 'final_project/src/'
INDEX_PATH = "final_project/src/ragatouille/colbert/indexes/b00_split2/"

st.set_page_config(
    page_title="Media Metrics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def get_ragatouille_index():
    """Ensure the Google Drive ZIP file is downloaded and extracted, then return the RAG model."""
    if not os.path.exists(ZIP_FILE_PATH):
        #st.info("Downloading RAG model files from Google Drive...")
        gdown.download(GDRIVE_URL, ZIP_FILE_PATH, quiet=False)

    if not os.path.exists(EXTRACT_TO_PATH):
        os.makedirs(EXTRACT_TO_PATH, exist_ok=True)

    if zipfile.is_zipfile(ZIP_FILE_PATH):
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if '__MACOSX' not in file:
                    zip_ref.extract(file, EXTRACT_TO_PATH)

    try:
        return RAGPretrainedModel.from_index(INDEX_PATH)
    except ValueError as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    """Load and preprocess dataset."""
    df = pd.read_csv("final_project/src/english_mbfc_reduced.csv")

    bias_mapping = {'left': -1, 'left-center': -0.5, 'neutral': 0, 'right-center': 0.5, 'right': 1}
    factual_mapping = {'low': 0, 'mixed': 1, 'high': 2}
    
    df['bias_encoded'] = df['bias'].map(bias_mapping)
    df['factual_numeric'] = df['factual_reporting'].map(factual_mapping)

    country_map = {
        'usa (44/180 press freedom)': 'united states',
        'usa (45/180 press freedom)': 'united states',
        'usa': 'united states',
        'guam (us territory)': 'united states',
        'united kingdom (scotland)': 'united kingdom',
        'united kingsom': 'united kingdom',
        'northern ireland (uk)': 'united kingdom',
        'italy (vatican city)': 'italy'
    }

    df['country'] = df['country'].replace(country_map).str.title()
    return df

df = load_data()
rag_model = get_ragatouille_index()

st.title("Informed Retrieval")

# Store model in session state
if 'index' not in st.session_state:
    st.session_state.index = rag_model
index = st.session_state.index

with st.sidebar:
    st.write("**DSAN 5400 Project**")
    st.write("**Group members:**")
    st.write("Manav Arora")
    st.write("Jessica Joy")
    st.write("Shanay Wadhwani")
    st.write("Ofure Udabor")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Colbert", "Instructions"])

# **Colbert Search Tab**
with tab2:
    query_text = st.text_input("Enter your query:", key="query_input")

    if query_text:
        start_time = time.time()
        top_results = index.search(query=query_text, k=10)
        elapsed_time = round(time.time() - start_time, 4)

        st.write(f"Most relevant results generated in: {elapsed_time} seconds")

        for result in top_results:
            metadata = result['document_metadata']
            col1, col2, col3 = st.columns(3)

            col1.write(f"**Source**: {metadata['source']}")
            col1.write(f"**Country**: {metadata['country']}")
            col2.write(f"**Publisher Bias**: {metadata['bias']}")

            # Convert date to readable format
            date_str = str(metadata['DATE'])[:8]
            formatted_date = datetime.datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            col2.write(f"**Date**: {formatted_date}")

            # Publisher factuality with color coding
            factuality_color = {
                'low': 'red',
                'mixed': 'orange',
                'high': 'green'
            }.get(metadata['factual_reporting'], 'black')

            col3.write(f"**Publisher Factuality**: :{factuality_color}[{metadata['factual_reporting']}]")

            # Tone color coding
            tone_color = 'red' if metadata['Avg_Tone'] < 0 else 'green'
            col3.write(f"**Document Tone**: :{tone_color}[{metadata['Avg_Tone']}]")

            st.write(f"**Document Link**: {metadata['DocumentIdentifier']}")
            st.write(f"**Content**: {result['content'][:280]} ... ")

            with st.expander("Expand to read more:"):
                st.write(result['content'])

            st.divider()

# **Dashboard Tab**
with tab1:
    st.header("Worldview of Media Metrics")

    tonality_range = st.slider(
        "Filter Tonality",
        min_value=float(df['Avg_Tone'].min()),
        max_value=float(df['Avg_Tone'].max()),
        value=(float(df['Avg_Tone'].min()), float(df['Avg_Tone'].max())),
        step=0.1
    )

    factual_slider = st.slider(
        "Filter Factual Reporting",
        min_value=0, max_value=2, value=(0, 2), step=1
    )

    factual_categories = [val for val in range(factual_slider[0], factual_slider[1] + 1)]
    
    filtered_df = df[
        (df['Avg_Tone'].between(*tonality_range)) & 
        (df['factual_numeric'].isin(factual_categories))
    ]

    # Aggregate data by country
    aggregated_df = filtered_df.groupby('country').agg(
        avg_bias=('bias_encoded', 'mean'),
        avg_tonality=('Avg_Tone', 'mean'),
        avg_factual_reporting=('factual_numeric', 'mean'),
        count=('bias_encoded', 'count')
    ).reset_index()

    # Map visualization
    fig = px.choropleth(
        aggregated_df,
        locations="country",
        locationmode="country names",
        color="avg_bias",
        hover_name="country",
        color_continuous_scale="Plasma",
        title="Media Bias Map",
        labels={"color": "Bias"}
    )

    fig.update_layout(
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type="natural earth",
            coastlinecolor='#FFFFFF',
            framecolor='#FFFFFF',
            showcountries=True,
            countrycolor='#FFFFFF'),
        title=dict(
            text=f"<b>Bias Map</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20)),
        coloraxis_colorbar=dict(
            title="Bias",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Left", "Left Center", "Neutral", "Right Center", "Right"]),
        paper_bgcolor="#374151",
        font_color="#f9fafb",
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        geo_bgcolor="#374151"
    )

    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("How to Use This App")
    st.write("""
        This application aims to explore biases and factuality in global media coverage, particularly focusing on 
        the Palestinian-Israeli and Russian-Ukrainian conflicts.
	
        ### Features:
	
        1. **Colbert Search Tab**:  
            - Enter a query to search for relevant articles.  
            - The top 10 results are displayed along with information such as source, country, publisher bias, factuality, and tone.  
            - Links to the full articles are also provided for more in-depth exploration.  

        2. **Dashboard Tab**:  
            - Filter articles by tone and factual reporting to analyze global media coverage.  
            - View a world map of media bias across different countries with a color-coded scale.  
            - Gain insights into the average bias, factuality, and tone for each country.  
	    
        ### How to Use:
	
        1. **Colbert Search Tab**: Enter a query into the text box to retrieve articles that match your search.  
        2. **Dashboard Tab**: Use the sliders to filter articles by tonality and factuality.  
        3. Analyze the data visualizations and use the interactive map to explore how media coverage varies by country and bias.  
        
        Enjoy exploring media metrics!
	
        ### About ColBERT
	
        ColBERT (Contextualized Late Interaction over BERT) is a state-of-the-art information retrieval model that combines the power of BERT, a transformer-based language model, with efficient retrieval techniques. Unlike traditional retrieval methods that rely solely on keyword matching or shallow semantic understanding, ColBERT utilizes BERT’s deep contextualized embeddings to capture the nuanced meaning of words and phrases in context. This enables the model to better understand the intent behind user queries and retrieve documents that are more semantically relevant, even if they don’t contain exact keyword matches.  
        
        #### How ColBERT Works:
	        
        1. **Contextualized Embeddings**:  
           At the core of ColBERT is BERT, a model pre-trained on vast amounts of text data, which allows it to generate rich, context-aware embeddings for each word in a query or document. These embeddings capture the meaning of words based on their surrounding context, helping the model understand synonyms, word relationships, and complex language structures.  

        2. **Late Interaction**:  
           ColBERT employs a “late interaction” approach, where the embeddings of both the query and the documents are generated separately. These embeddings are then compared at a later stage using efficient similarity measures (like MaxSim), significantly speeding up the retrieval process while preserving the semantic richness of the embeddings.  

        3. **Efficient Search**:  
           ColBERT’s approach enables it to scale well for large datasets. By storing document embeddings and utilizing approximate nearest neighbor (ANN) search algorithms, ColBERT can quickly find the most relevant documents for a given query, making it suitable for real-time applications like ours.  
	 
        #### Why ColBERT is Effective for This Project:
        In this project, our goal is to build an information retrieval system that analyzes biases in media coverage of global conflicts. ColBERT’s ability to understand the nuanced tone, regional focus, and thematic content of news articles makes it ideal for retrieving relevant documents based on complex user queries.  
	        
        1. **Handling Complex Queries**:  
           Users may submit queries that are not simply keyword-based but involve more abstract concepts, such as “media bias in the Russian-Ukrainian conflict” or “tonality of Palestinian-Israeli coverage.” ColBERT’s deep contextual understanding of queries ensures that it can retrieve documents that are semantically aligned with the user’s intent, even if the exact terms aren’t explicitly mentioned in the text.  

        2. **Semantic Relevance**:  
           By leveraging BERT’s rich embeddings, ColBERT can find documents that are contextually similar to the query, even when they use different wording or phrasing. This improves the quality of the search results, as the model can identify articles that discuss the same issues from different angles or with varying word choices.  

        3. **Scalability and Efficiency**:  
           With the use of approximate nearest neighbor search, ColBERT can scale to handle large volumes of news articles while maintaining fast retrieval times, which is critical for our interactive visualization dashboard where users need immediate feedback on their queries.  

        By combining deep learning with efficient retrieval mechanisms, ColBERT offers an effective and scalable solution for analyzing complex, context-dependent queries about media bias and tone, ultimately helping users gain deeper insights into global news coverage.  
    """)
