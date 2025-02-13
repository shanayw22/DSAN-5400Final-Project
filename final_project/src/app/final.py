import warnings
warnings.filterwarnings('ignore')
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

tab1, tab2 = st.tabs(["Dashboard", "Colbert"])

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
