##############################################################################
##############################################################################
#                                                                            #
#                             Sentiment Analysis                             #
#                                 good / bad                                 #
#                            TripAdvisor Reviews                             #
#                               Master Thesis                                #
#                                                                            #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################



import pandas as pd
import os
import nltk
from nltk.corpus import wordnet
import string
from sklearn import metrics
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from nltk.metrics import ConfusionMatrix
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from funcsigs import signature
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import time
from dask import dataframe as dd
import dask.multiprocessing
import pyarrow
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC



# Directory
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')
from functions_nlp import get_wordnet_pos, clean_text, show_wordcloud





##############################################################################
# Data Prearation
##############################################################################

hotel_review_df = pd.read_csv('data/UK_hotel_reviews.csv')


# append the positive and negative text reviews
hotel_review_df["review"] = hotel_review_df["review_title"] +" "+ hotel_review_df["review_text"]

# Bad Review Dummy
hotel_review_df["bad_review_dummy"] = hotel_review_df["review_rating"].apply(lambda x: 1 if x < 3.5 else 0)


# clean text data
hotel_review_df["review"] = hotel_review_df["review"].astype(str)
hotel_review_df["review_clean"] = hotel_review_df["review"].apply(lambda x: clean_text(x))


# add number of characters column
hotel_review_df["nb_chars"] = hotel_review_df["review_clean"].apply(lambda x: len(x))

# add number of words column
hotel_review_df["nb_words"] = hotel_review_df["review_clean"].apply(lambda x: len(x.split(" ")))



# Save to CSV
hotel_review_df.to_csv("data/clean_tripadvisor_review_table.csv", encoding='utf8', index=False)

# Read CSV
hotel_review_df = pd.read_csv("data/clean_tripadvisor_review_table.csv")

# Add Variance of Reviews Column
mu = []
var = []

for i in range(0,len(hotel_review_df['review_rating'])):
    distribution_list = []
    distribution_list.extend([5] * int(hotel_review_df['excellent'].iloc[i]))
    distribution_list.extend([4] * int(hotel_review_df['very_good'].iloc[i]))
    distribution_list.extend([3] * int(hotel_review_df['average'].iloc[i]))
    distribution_list.extend([2] * int(hotel_review_df['poor'].iloc[i]))
    distribution_list.extend([1] * int(hotel_review_df['terrible'].iloc[i]))
    
    mu.append(np.mean(distribution_list))
    var.append(np.var(distribution_list))



hotel_review_df['mu'] = mu
hotel_review_df['var'] = var



# Add a dummy variable for reviews of high variance hotels
hotel_review_df['high_var_dummy'] = hotel_review_df['var'].apply(lambda x: 1 if x > 1.0 else 0)
hotel_review_df['var_to_mu'] = 0
hotel_review_df.loc[hotel_review_df['high_var_dummy'] == 1, 'var_to_mu'] = abs(hotel_review_df['average_rating'] - hotel_review_df['review_rating'])


# If the variance in review ratings and the number of reviews is high, then we
# assume that a part of the variance can be explained by taste differences in customers.
# The rest might be e.g. due to timing.
# Therefore we add a varibale 'taste_diff_dummy' that is 1 if the variance is larger 1
# and 0 otherwise.

hotel_review_df['taste_diff_dummy'] = hotel_review_df['var_to_mu'].apply(lambda x: 1 if x > 2 else 0)





##############################################################################
# Use a Sample for Feature Selection
##############################################################################

bad_reviews = hotel_review_df[hotel_review_df["bad_review_dummy"]==1].iloc[0:5000,:]
good_reviews = hotel_review_df[hotel_review_df["bad_review_dummy"]==0].iloc[0:5000,:]

sample_frames = [bad_reviews, good_reviews]
sample_reviews_df = pd.concat(sample_frames)



##############################################################################
# Sentiment Analysis
##############################################################################

# Vader Lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

new_words = {
    'exceptional': 0.9,
    'overprice': -3.3,
    'bed bug': -6.6,
    'small': -3.0,
}

sid.lexicon.update(new_words)


# add sentiment anaylsis columns
sample_reviews_df["sentiments"] = sample_reviews_df["review_clean"].apply(lambda x: sid.polarity_scores(str(x)))
sample_reviews_df = pd.concat([sample_reviews_df.drop(['sentiments'], axis=1),
                               sample_reviews_df['sentiments'].apply(pd.Series)], axis=1)

# create doc2vec vector columns
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sample_reviews_df["review_clean"].apply(lambda x: str(x).split(" ")))]


# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4) # 26.5 sec


# transform each document into a vector data  time: 33.54 sec
doc2vec_df = sample_reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
sample_reviews_df = pd.concat([sample_reviews_df, doc2vec_df], axis=1)



# add tf-idfs columns
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(sample_reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = sample_reviews_df.index
sample_reviews_df = pd.concat([sample_reviews_df, tfidf_df], axis=1)



# Save DF
sample_reviews_df.to_parquet("sample_sentiment_analysis.parquet", compression=None)


# Read csv with parquet
sample_reviews_df = pd.read_parquet("sample_sentiment_analysis.parquet", engine="fastparquet")








# select only relevant columns
reviews_df = sample_reviews_df[['bad_review_dummy',
                              'review_rating',
                              'review',
                              'review_clean']]

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

for x in [0, 1]:
    subset = sample_reviews_df[sample_reviews_df['bad_review_dummy'] == x]
    
    # Draw Density Plot
    if x == 0:
        label = "Positive Reviews"
    else:
        label = "Negative Reviews"
    sns.distplot(subset['compound'], hist = False, label = label)

# Lables
plt.title('Density Plot of Review Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/density_plot_reviews.png')






















