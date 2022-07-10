# Parameterizatio of taste review prediction
import pandas as pd
import os
import numpy as np
import seaborn as sns


# Directory
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/code')
from sentiment_and_nlp_functions import clean_text, show_wordcloud, sentiment_analysis_for_reviews,add_descriptive_variables,parameterization_rf_tatse_pred



# Read in data
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')
hotel_review_df = pd.read_parquet('data/UK_hotel_reviews.parquet')
nlp_review_df = pd.read_parquet("data/full_nlp_review_df.parquet", engine="fastparquet")




# Set Variables
bad_review_threshold = 3.5
variance_threshold = 1.0
dtm_lower = 1.0
dtm_upper = 2.0
bad_month_threshold = 2.1


result_df = parameterization_rf_tatse_pred(hotel_review_df,
                                           nlp_review_df,
                                           bad_review_threshold,
                                           variance_threshold,
                                           dtm_lower, dtm_upper, bad_month_threshold)
# estimated computation time: 12min
    
    

most_probable = result_df.sort_values('y_probability')


most_probable['review_text'].head(10)


# Some Means
np.mean(most_probable['average_rating'])
np.mean(most_probable['review_rating'])
np.mean(most_probable['average_rating'] - most_probable['review_rating'])
np.mean(most_probable['y_probability'])

most_probable.to_csv("test_taste.csv")












# Statistical Methodology of defining the sample sd's

distribution_list = []
for i in range(0,len(hotel_review_df['review_rating'])):
    distribution_list.extend([5] * int(hotel_review_df['excellent'].iloc[i]))
    distribution_list.extend([4] * int(hotel_review_df['very_good'].iloc[i]))
    distribution_list.extend([3] * int(hotel_review_df['average'].iloc[i]))
    distribution_list.extend([2] * int(hotel_review_df['poor'].iloc[i]))
    distribution_list.extend([1] * int(hotel_review_df['terrible'].iloc[i]))
    


np.mean(distribution_list)
np.var(distribution_list)
np.median(distribution_list)



# A convenient definition of an outlier is a point which falls more than
# 1.5 times the interquartile range above the third quartile or below the first quartile.
# https://mathworld.wolfram.com/Outlier.html


below_median = list(filter(lambda x: x < 4, distribution_list))
above_median = list(filter(lambda x: x >=  4, distribution_list))

q_three = np.median(above_median)
q_one = np.median(below_median)


IQR = q_three - q_one

print("Outlier Threshold: " + str(1.5 * IQR))




#for each review compute wether the review is outlier / in mean /else or taste diff in its hotel dist














