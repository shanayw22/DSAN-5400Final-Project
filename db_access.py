import pandas as pd
from google.cloud import bigquery

# Replace with your service account key file path
key_path = 'seraphic-ripple-279410-67dd9300e993.json'

# Initialize the BigQuery client
client = bigquery.Client.from_service_account_json(key_path)

# Define the function to get data from BigQuery
def get_gkg_data(date, themes=None, countries=None, min_date='20220101', table='gkg'):
    # Build the search query dynamically based on themes and countries
    theme_filter = " OR ".join([f"V2Themes LIKE '%{theme}%'" for theme in themes]) if themes else ''
    country_filter = " OR ".join([f"V2Themes LIKE '%{country}%'" for country in countries]) if countries else ''
    
    # Construct the complete WHERE clause
    where_clause = f"({theme_filter})" if theme_filter else ""
    if country_filter:
        where_clause += f" AND ({country_filter})"
    where_clause += f" AND CAST(SUBSTR(CAST(Date AS STRING), 1, 8) AS BIGNUMERIC) >= {min_date}"
    
    # Build the full query
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