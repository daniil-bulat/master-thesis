## ReadMe: NLP Master Thesis

The following document aims to provide an overview of the code used in this thesis.

This repository contains all code, figures and data-sets that were used for the analysis. Not all data-sets were uploaded due to capacity constraints, nevertheless all data can be reproduced using the provided code.

This document outlines the following points:
1. Data - Webscraping
2. Exploratory Data Analysis
3. Natural Languge Processing
4. Pedictive Analysis
   - Random Forest
   - Feature Selection
   - Grid Search
5. Final Results



## 1. Data - Webscraping

#### Webscraping Data from Tripadvisor.com
All data was scraped from [tripadvisor.com]. The ```tripadvisor_review_scraper.py``` file contains all the code to scrape relevant information needed for negative taste-driven review analysis. Additionally, the ```functions_tripadvisor_scraping.py``` contains all functions used in the scraper. Information scraped includes hotel name, hotel adresse, number of reviews, average hotel rating, written reviews and respective review rating, tridadvisor ranking and the distribution of reviews.


## 2. Exploratory Data Analysis - EDA
The script ```eda_tripadvisor.py``` contains the code needed for EDA.

## 3. Natural Languge Processing
The Natural Language Processing model can be found in the following scripts:

```nlp_sentiment_analysis.py``` and ```sentiment_and_nlp_functions.py```



## 4. Pedictive Analysis
Predictive analysis scripts are: ```predictive_analysis.py``` and ```random_forest_functions.py``` and ```parameterization_taste_reviews.py```
#### Random Forest
#### Feature Selection
#### Grid Search

## 5. Final Results
Final Results can be computed with the ```final_result_calculation.py``` script.





