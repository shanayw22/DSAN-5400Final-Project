import pandas as pd
df = pd.read_csv("../english_mbfc.csv")

dfx = pd.read_csv("../mbfc.csv")
   
df = df.merge(dfx, left_on='base_url', right_on='source')

subset = df[['country','bias', 'factual_reporting', "Avg_Tone"]]
subset.to_csv("../english_mbfc_reduced.csv")