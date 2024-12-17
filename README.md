# Analyzing Media Bias in Global News Coverage of the Palestinian-Israeli and Russian-Ukrainian Conflicts: Tone, Factuality, and Regional Focus

Georgetown University

DSAN 5400 Final Project

Team: Manav Arora, Jessica Joy, Shanay Wadhwani, Ofure Udabor

## Analyzing Media Bias in Global News Coverage of the Palestinian-Israeli and Russian-Ukrainian Conflicts: Tone, Factuality, and Regional Focus



## Data Source: 

See Fetched Data Below
https://drive.google.com/file/d/1LD4XFWOPv3ePgA6sxkqHLxq-mboYPTSm/view?usp=sharing

@misc{gdelt_gkg,
  author = {{GDELT Project}},
  year = {2024},
  title = {Global Knowledge Graph (GKG) Dataset},
  url = {https://www.gdeltproject.org},
  note = {Accessed: June 2024}
}

Up to date data can be fetched using big query apis through the data_fetch.ipynb file.


## NLP Objective

Build an Information Retrieval system to address how biases in global news coverage of Palestinian-Israeli and Russian-Ukrainian conflicts manifest across different media outlets, specifically through variations in tone, factuality, and regional focus. By uncovering these patterns, this project seeks to improve our users’ understanding of their media consumption and news source biases.

### Data Sources

**GDELT (Global Database of Events, Language, and Tone):** a global database that collects and analyses news articles from around the world to track events, locations, people, and themes, aiming to study human society.

**MBFC (Media Bias/Fact Check):** a website that rates the bias and factual accuracy of news sources, categorizing outlets across a spectrum from left-leaning to right-leaning, and assessing the reliability of information. It’s a resource for understanding media bias and source credibility.

**News Articles:** Scrape relevant articles using APIs or scraping libraries like Beautiful Soup to build our corpus.

#### Data Files and Indices
The data used and the indices created for this project exceeded the size limits on GitHub and are included as google drive links here:
- Link to consolidated Dataset:
- Link to BM25 index: [pickle file](https://drive.google.com/file/d/1-yNSzuZtZwlhwia3X5Seu1scwPmjbve3/view?usp=sharing)
- Link to Colbert and Voyager Indices: [Folder titled .ragatouille](https://drive.google.com/drive/folders/11g6RW9nKCl7SJ_niU_pYIIq7oLcQwxRi?usp=sharing)

### Methods

**Information Retrieval System:** Perform indexing and query processing and then use ranking algorithms to create an efficient retrieval system, displaying metrics such as bias, tone, factuality reporting.

**Visualization Dashboard:** Create an interactive map that displays regions alongside filtering for bias scores, tone, and factuality reporting allowing for comparative analysis across regions and outlets.

#### Models Tested:

**Baseline:** TF-IDF, BM25

**Neural Models:** Voyager, ColBERT, ColBERT Rerank

### Current Relevant Research

* Leetaru, Kalev et al (2013) GDELT: Global Data on Events, Location and Tone, 1979-2012.
* Watanabe, Kohei (2017) Measuring bias in international news: a large-scale analysis of news agency coverage of the Ukraine crisis. PhD thesis, London School of Economics and Political Science.	
* Hamborg, Felix (2023) Revealing Media Bias in News Articles: NLP Techniques for Automated Frame Analysis
* Kwak, H., An, J. (2014) A First Look at Global News Coverage of Disasters by Using the GDELT Dataset. In: Aiello, L.M., McFarland, D. (eds) Social Informatics. SocInfo 2014.
* Francisco-Javier Rodrigo-Ginés et al. (2024) A systematic review on media bias detection: What is media bias, how it is expressed, and how to detect it, Expert Systems with Applications, Volume 237, Part C. 


