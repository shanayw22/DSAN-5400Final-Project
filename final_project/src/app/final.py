import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from ragatouille import RAGPretrainedModel
import pandas as pd
import time
import datetime
import plotly.express as px
import zipfile
import os

zip_file_path = "../ragatouille-20250118T231710Z-001.zip"
extract_to_path = "../.ragatouille"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

path_to_index = "../.ragatouille/colbert/indexes/b00_split2/"

st.set_page_config(
    page_title="Media Metrics",
    page_icon="🌍",
    layout="wide", 
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    df = pd.read_csv("../english_mbfc_reduced.csv")
    #dfx = pd.read_csv("../mbfc.csv")
    #df = df.merge(dfx, left_on='base_url', right_on='source')
    #df['year'] = df['DATE'].astype(str).str[:4]
    #df['year'] = pd.to_numeric(df['year'])
    bias_hierarchy = {'left': -1, 'left-center': -0.5, 'neutral': 0, 'right-center': 0.5, 'right': 1}
    df['bias_encoded'] = df['bias'].map(bias_hierarchy)
    df['country'] = df['country'].replace({'usa (44/180 press freedom)':'united states', 'usa (45/180 press freedom)':'united states', 'usa': 'united states','guam (us territory)':'united states',
                                        'united kingdom (scotland)':'united kingdom', 'united kingsom':'united kingdom', 'northern ireland (uk)': 'united kingdom',
                                         'italy (vatican city)':'italy'})
    df['country'] = df['country'].str.title()
    factual_mapping = {'low': 0, 'mixed': 1, 'high': 2}
    df['factual_numeric'] = df['factual_reporting'].map(factual_mapping)
    return df

df = load_data()


# Load colbert index 
path_to_index = "../.ragatouille/colbert/indexes/b00_split2/"
RAG1 = RAGPretrainedModel.from_index(path_to_index)


st.title("Informed Retrieval")

if 'index' not in st.session_state:

    st.session_state.index = RAG1

index = st.session_state.index



with st.sidebar:
    st.write("**DSAN 5400 Project**")
    st.write("**Group members:**")
    st.write("Manav Arora")
    st.write("Jessica Joy")
    st.write("Shanay Wadhwani")
    st.write("Ofure Udabor")



tab1, tab2= st.tabs(["Dashboard", "Colbert" ])

with tab2:

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
            
 
            
with tab1:
    st.header("Worldview of Media Metrics")

    min_tonality, max_tonality = df['Avg_Tone'].min(), df['Avg_Tone'].max()
    tonality_range = st.slider(
        "Filter Tonality",
        min_value=float(df['Avg_Tone'].min()),
        max_value=float(df['Avg_Tone'].max()),
        value=(float(df['Avg_Tone'].min()), float(df['Avg_Tone'].max())),
        step=0.1
    )

    factual_reporting_slider = st.slider(
        "Filter Factual Reporting", 
        min_value=0, max_value=2, value=(0, 2), step=1
    )
    
    factual_mapping = {0: 'low', 1: 'mixed', 2: 'high'}
    factual_reporting_category = [factual_mapping[val] for val in range(factual_reporting_slider[0], factual_reporting_slider[1] + 1)]

    st.write("Factual Reporting Categories: 0 = low , 1 = mixed, 2 = high")

    filtered_df = df[
        (df['Avg_Tone'] >= tonality_range[0]) & 
        (df['Avg_Tone'] <= tonality_range[1])
    ]

    # aggregate data by country for the selected feature
    aggregated_df = filtered_df.groupby('country').agg(
        avg_bias=('bias_encoded', 'mean'),
        avg_tonality=('Avg_Tone', 'mean'),
        avg_factual_reporting=('factual_numeric', 'mean'), 
        count=('bias_encoded', 'count')
    ).reset_index()

    # filter aggregated data based on factual reporting average
    aggregated_df = aggregated_df[
        (aggregated_df['avg_factual_reporting'] >= factual_reporting_slider[0]) &
        (aggregated_df['avg_factual_reporting'] <= factual_reporting_slider[1])
    ]

    # create map
    fig = px.choropleth(
        aggregated_df,
        locations="country",
        locationmode="country names",
        color="avg_bias",
        hover_name="country",
        color_continuous_scale="Plasma",
        title="Map",
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
        margin={'r':0,'t':0,'l':0,'b':0},
        geo_bgcolor= "#374151"
    )

    st.plotly_chart(fig, use_container_width=True, height=800)  