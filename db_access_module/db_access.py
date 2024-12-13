import pandas as pd
from google.cloud import bigquery
import pandas as pd
from newspaper import Article
from tqdm import tqdm
import tldextract

# Replace with your service account key file path
key_path = 'seraphic-ripple-279410-67dd9300e993.json'

client = bigquery.Client.from_service_account_json(key_path)


def get_gkg_data(min_date, max_date,date=None, themes=None, countries=None, table='gkg'):
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
        AND CAST(SUBSTR(CAST(Date AS STRING), 1, 8) AS BIGNUMERIC) >= {min_date}
        AND CAST(SUBSTR(CAST(Date AS STRING), 1, 8) AS BIGNUMERIC) < {max_date}
    LIMIT 333333
    """
    
    # Execute the query
    query_job = client.query(query)
    
    # Use tqdm to show progress as rows are processed
    results = query_job.result()
    print("Query run successful!")
    
    # Convert to DataFrame with progress bar
    df = results.to_arrow(progress_bar_type="tqdm").to_pandas()
    
    return df


#gkg_data_batch_1 = get_gkg_data('20220101', '20230101')

#gkg_data_batch_2 = get_gkg_data('20230101', '20240101')
#gkg_data_batch_3 = get_gkg_data('20240101', '20250101')

mbfc = pd.read_csv('mbfc_scrape/mbfc.csv')


#result = pd.concat([gkg_data_batch_1, gkg_data_batch_2, gkg_data_batch_3], axis=0) 

#result.to_csv("all_gkg_data.csv")



#english_df = result[result['TranslationInfo'].isnull()]

def get_base_url(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"  # Extract domain and suffix


#english_df['base_url'] = english_df['DocumentIdentifier'].apply(get_base_url)
#base_url_list = [get_base_url(url) for url in mbfc["source"]]

# Subset: Exact matches of base URLs
#matching_rows = english_df[english_df['base_url'].isin(base_url_list)]


#matching_rows.to_csv("english_mbfc.csv")

# Run this on english_mbfc.csv 

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


df = fetch_article_content(gkg_data.head(), url_column='DocumentIdentifier')

df.to_csv("content_sample11.csv")

df.columns

