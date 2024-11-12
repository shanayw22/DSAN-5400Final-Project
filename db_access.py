from google.cloud import bigquery

# Initialize the BigQuery client
client = bigquery.Client()

# Define your SQL query
query = """
SELECT Actor1Name, EventCode, AvgTone, ActionGeo_CountryCode
FROM `gdelt-bq.gdeltv2.events`
WHERE EventCode = '191'
AND ActionGeo_CountryCode = 'US'
ORDER BY AvgTone DESC
LIMIT 10
"""

# Run the query
query_job = client.query(query)
results = query_job.result()

# Convert results to a DataFrame
df = results.to_dataframe()
print(df)