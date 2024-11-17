import pandas as pd
from google.cloud import bigquery
import pandas as pd
from newspaper import Article
from tqdm import tqdm

# Replace with your service account key file path
key_path = 'seraphic-ripple-279410-67dd9300e993.json'

client = bigquery.Client.from_service_account_json(key_path)


def get_gkg_data(date, themes=None, countries=None, min_date='20220101', table='gkg'):
    # Build the search query dynamically based on themes and countries
    theme_filter = " OR ".join([f"V2Themes LIKE '%{theme}%'" for theme in themes]) if themes else ''
    country_filter = " OR ".join([f"V2Themes LIKE '%{country}%'" for country in countries]) if countries else ''
    
    
    where_clause = f"({theme_filter})" if theme_filter else ""
    if country_filter:
        where_clause += f" AND ({country_filter})"
    where_clause += f" AND CAST(SUBSTR(CAST(Date AS STRING), 1, 8) AS BIGNUMERIC) >= {min_date}"
    
    
    query = f"""
  SELECT
    *,
    CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS BIGNUMERIC) AS Avg_Tone,
    CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(1)] AS BIGNUMERIC) AS PositiveS,
    CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(2)] AS BIGNUMERIC) AS NegativeS
FROM
    `gdelt-bq.gdeltv2.gkg`
WHERE
    (
        Themes LIKE '%ARMED CONFLICT%'
        OR Themes LIKE '%WAR%'
        OR Themes LIKE '%MILITARY%'
        OR Themes LIKE '%VIOLENCE%'
        OR V2Themes LIKE '%ARMED CONFLICT%'
        OR V2Themes LIKE '%WAR%'
        OR V2Themes LIKE '%MILITARY%'
        OR V2Themes LIKE '%VIOLENCE%'
    )
    AND (
        Themes LIKE '%%ISRAEL%%'
        OR Themes LIKE '%%PALESTINE%%'
        OR V2Themes LIKE '%%ISRAEL%%'
        OR V2Themes LIKE '%%PALESTINE%%'
        OR Themes LIKE '%%RUSSIA%%'
        OR Themes LIKE '%%UKRAINE%%'
        OR V2Themes LIKE '%%RUSSIA%%'
        OR V2Themes LIKE '%%UKRAINE%%'
    )
    AND CAST(SUBSTR(CAST(Date AS STRING), 1, 8) AS BIGNUMERIC) >= 20220101
    LIMIT 1000;
    """
    
    
    query_job = client.query(query)
    
    
    results = query_job.result()  
    
    
    df = pd.DataFrame([dict(row) for row in results])
    
    
    return df


themes = ['ARMED CONFLICT', 'WAR', 'MILITARY', 'VIOLENCE']
countries = ['ISRAEL', 'PALESTINE', 'RUSSIA', 'UKRAINE']
date = '2023-01-01'


gkg_data = get_gkg_data(date, themes=themes, countries=countries)


print(gkg_data.head())

#This takes 25 mins per 1000 rows to run. The sample output after the article content search is saved as content_sample.csv
# - Shanay.

#Feel free to build on this

def fetch_article_content(df, url_column, content_column='content'):
    """
    Fetches article content from URLs and adds a new column to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the URLs.
    - url_column (str): The column name with the URLs.
    - content_column (str): The name of the new column for article content.

    Returns:
    - pd.DataFrame: The updated DataFrame with the new content column.
    """
    contents = [] 
    total_rows = len(df) 

    for i, url in enumerate(tqdm(df[url_column], desc="Fetching article content")):
        try:
            
            article = Article(url)
            article.download()
            article.parse()
            contents.append(article.text)
        except Exception as e:
            print(f"Error processing URL at index {i}: {url} - {e}")
            contents.append(None)  

    
    df[content_column] = contents
    return df


df = fetch_article_content(gkg_data, url_column='DocumentIdentifier')

df.to_csv("content_sample.csv")

df.columns

