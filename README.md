# DSAN-5400Final-Project

DSAN 5400 Final Project
Team: Manav Arora, Jessica Joy, Shanay Wadhwani

What is the NLP problem your team will address?

Build an Information Retrieval system to address how biases in global news coverage of Palestinian-Israeli and Russian-Ukrainian conflicts manifest across different media outlets, specifically through variations in tone, polarity, and regional focus. By uncovering these patterns, this project seeks to analyze whether and how media coverage may contribute to public perception, polarization, and the formation of biased or incomplete understandings of these global conflicts.

What recent work has been done in the area? You do not need to complete a thorough literature review, but you should identify two or three key papers and developments relevant to your experimental question.

Leetaru, Kalev et al (2013) GDELT: Global Data on Events, Location and Tone, 1979-2012.
Watanabe, Kohei (2017) Measuring bias in international news: a large-scale analysis of news agency coverage of the Ukraine crisis. PhD thesis, London School of Economics and Political Science.	
Hamborg, Felix (2023) Revealing Media Bias in News Articles: NLP Techniques for Automated Frame Analysis
Kwak, H., An, J. (2014) A First Look at Global News Coverage of Disasters by Using the GDELT Dataset. In: Aiello, L.M., McFarland, D. (eds) Social Informatics. SocInfo 2014.
Francisco-Javier Rodrigo-Ginés et al. (2024) A systematic review on media bias detection: What is media bias, how it is expressed, and how to detect it, Expert Systems with Applications, Volume 237, Part C. 


What datasets will you be using?

GDELT (Global Database of Events, Language, and Tone): a global database that collects and analyses news articles from around the world to track events, locations, people, and themes, aiming to study human society.

MBFC (Media Bias/Fact Check): a website that rates the bias and factual accuracy of news sources, categorizing outlets across a spectrum from left-leaning to right-leaning, and assessing the reliability of information. It’s a resource for understanding media bias and source credibility.

News Articles: Scrape relevant articles using APIs or scraping libraries like Beautiful Soup to build out our corpus.

What methods will you use? These should include some of what we’ve covered in class, but you may optionally include approaches outside class.

Information Retrieval System: Perform indexing and query processing and then use ranking algorithms like BM25 or deep learning-based models to create an efficient retrieval system.

Topic Modelling:  Apply topic modeling techniques like LDA (Latent Dirichlet Allocation) to uncover the main narratives. Cluster articles by topics to observe differences in framing, examining word choice and thematic focus.

Visualization Dashboard: Create an interactive map that displays regions alongside, bias scores, tone, and themes and enables users to view article summaries by clicking on areas. Allow for filtering by source, region, or specific conflicts to facilitate comparative analysis across outlets.


How will you evaluate the performance of your system?
	
Information Retrieval System:
Precision, Recall, and F1-Score: These metrics assess how accurately the system retrieves relevant documents while minimizing irrelevant results.
Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR): MAP provides an average measure of precision across all relevant documents, while MRR focuses on the ranking position of the first relevant document.
Normalized Discounted Cumulative Gain (nDCG): Measures the relevance of retrieved documents at different ranks, with a discount applied for lower-ranked documents.


