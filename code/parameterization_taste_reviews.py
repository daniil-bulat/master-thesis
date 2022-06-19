# Parameterizatio of taste review prediction
import pandas as pd
import os
import numpy as np
import seaborn as sns


# Directory
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/code')
from sentiment_and_nlp_functions import clean_text, show_wordcloud, sentiment_analysis_for_reviews,add_descriptive_variables,parameterization_rf_tatse_pred



# Read in data
hotel_review_df = pd.read_csv('data/UK_hotel_reviews.csv')
nlp_review_df = pd.read_parquet("data/full_nlp_review_df.parquet", engine="fastparquet")




# Set Variables
bad_review_threshold = 3.5
variance_threshold = 1.4
dtm_lower = 1.0
dtm_upper = 2.4


result_df = parameterization_rf_tatse_pred(hotel_review_df, nlp_review_df, bad_review_threshold, variance_threshold, dtm_lower, dtm_upper)
# estimated computation time: 12min
    
    

most_probable = result_df.sort_values('y_probability')


most_probable['review_text'].head(10)


# Some Means
np.mean(most_probable['average_rating'])
np.mean(most_probable['review_rating'])
np.mean(most_probable['average_rating'] - most_probable['review_rating'])
np.mean(most_probable['y_probability'])


