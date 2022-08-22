##############################################################################
##############################################################################
#                                                                            #
#                             Sentiment Analysis                             #
#                            positive / negative                             #
#                            TripAdvisor Reviews                             #
#                               Master Thesis                                #
#                                                                            #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


directory_path = ''
directory_functions_path = ''
directory_figure_path = ''


# Directory
os.chdir(directory_functions_path)
from sentiment_and_nlp_functions import clean_text, show_wordcloud, sentiment_analysis_for_reviews,add_descriptive_variables
from vader_lexicon import import_adj_vader




##############################################################################
# Data Cleaning and Applying NLP
##############################################################################
os.chdir(directory_path)
hotel_review_df = pd.read_parquet('data/UK_hotel_reviews.parquet')
hotel_review_df = hotel_review_df.reset_index()
hotel_review_df.rename(columns = {'index':'ID'}, inplace = True)


## NLP
import_adj_vader() # Vader Lexicon
nlp_review_df = sentiment_analysis_for_reviews(hotel_review_df, clean_text)


# Save NLP as parquet
nlp_review_df.to_parquet("data/full_nlp_review_df.parquet", compression=None)
nlp_review_df = nlp_review_df.reset_index()
nlp_review_df.rename(columns = {'index':'ID'}, inplace = True)



# Add some additional Variables to the initial data set
full_hotel_review_df = add_descriptive_variables(hotel_review_df, 3.1, 2.1)


# Save Hotel Information as parquet
full_hotel_review_df.to_parquet("data/full_hotel_review_df.parquet", compression=None)




##############################################################################
# Select Sample
##############################################################################

# select only relevant columns
sample_reviews_df = full_hotel_review_df[['ID',
                                          'bad_review_dummy',
                                          'review_rating']]

slim_nlp_review_df = nlp_review_df[['ID',
                                    'review',
                                    'review_clean',
                                    'compound',
                                    'nb_words',
                                    'pos',
                                    'neg']]


# for future use
joined_hotel_review_df = pd.merge(full_hotel_review_df,nlp_review_df[['ID','compound']],on='ID', how='left')
sample_reviews_df = joined_hotel_review_df.drop_duplicates(subset='review_text', keep="last")






##############################################################################
# Wordcloud
##############################################################################

show_wordcloud(sample_reviews_df["review_clean"])
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/review_wordcloud.png')




##############################################################################
# Tables of most pos and neg reviews
##############################################################################

# highest positive sentiment reviews (with more than 5 words)
sample_reviews_df[sample_reviews_df["nb_words"] >= 5].sort_values("pos", ascending = False)[["review", "pos"]].head(10)

# lowest negative sentiment reviews (with more than 5 words)
sample_reviews_df[sample_reviews_df["nb_words"] >= 5].sort_values("neg", ascending = False)[["review", "neg"]].head(10)





##############################################################################
# Density Plot of Reviews
##############################################################################
sns.set_theme(style="whitegrid", palette="pastel")

for x in [0, 1]:
    subset = sample_reviews_df[sample_reviews_df['bad_review_dummy'] == x]
    
    # Draw Density Plot
    if x == 0:
        lab = "Positive Reviews"
    else:
        lab = "Negative Reviews"
        
    sns.distplot(subset['compound'], hist = False, label = lab)

# Lables
plt.title('Density Plot of Review Sentiment')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Density')
plt.legend()
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/density_plot_reviews.png')











