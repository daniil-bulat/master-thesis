##############################################################################
##############################################################################
#                                                                            #
#                                    EDA                                     #
#                            TripAdvisor Reviews                             #
#                               Master Thesis                                #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import os
import math
import matplotlib.pyplot as plt
import string
import pandas as pd
import numpy as np
import seaborn as sns


# Data
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')
df_new_hotel_reviews = pd.read_csv('FINAL_UK_hotel_reviews.csv')

df_new_hotel_reviews = df_new_hotel_reviews[df_new_hotel_reviews['average_rating']<5.1]
df_new_hotel_reviews = df_new_hotel_reviews[df_new_hotel_reviews['review_rating']<5.1]
df_new_hotel_reviews['poor'] = df_new_hotel_reviews['poor'].apply(pd.to_numeric)


###############################################################################
# EDA
###############################################################################


# Basic EDA on data
len(df_new_hotel_reviews['review_text']) # number of reviews
len(df_new_hotel_reviews['hotel_name'].unique()) # number of hotels

print(np.mean(df_new_hotel_reviews['average_rating']))
print(np.min(df_new_hotel_reviews['average_rating']))
print(np.max(df_new_hotel_reviews['average_rating']))

print(np.mean(df_new_hotel_reviews['review_rating']))
print(np.min(df_new_hotel_reviews['review_rating']))
print(np.max(df_new_hotel_reviews['review_rating']))

y_1 = np.mean(df_new_hotel_reviews['excellent'] / df_new_hotel_reviews['num_reviews'])
y_2 = np.mean(df_new_hotel_reviews['very_good'] / df_new_hotel_reviews['num_reviews'])
y_3 = np.mean(df_new_hotel_reviews['average'] / df_new_hotel_reviews['num_reviews'])
y_4 = np.mean(df_new_hotel_reviews['poor'] / df_new_hotel_reviews['num_reviews'])
y_5 = np.mean(df_new_hotel_reviews['terrible'] / df_new_hotel_reviews['num_reviews'])

x = ['terrible', 'poor', 'average', 'very good', 'excellent']
y = [y_5, y_4, y_3, y_2, y_1]

sns.barplot(x,y)


# BOXPLOT
import seaborn as sns
sns.set_theme(style="whitegrid")

ax = sns.boxplot(x="average_rating", y="num_reviews", data=df_new_hotel_reviews)



# Total_Number_of_Reviews vs Average_Score
sns.set_theme(color_codes=True)

sns.regplot(x='average_rating', y="num_reviews", data=df_new_hotel_reviews,
            scatter_kws={"color": "blue"}, line_kws={"color": "red"})



plt.title('Total Number of Reviews vs Average Score')
plt.xlabel('Average Score')
plt.ylabel('Total Number of Reviews')
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/trip_advisor/figures/number_of_reviews_v_avScore.png')




# Distribution of Average_Score
df_new_hotel_reviews['average_rating'].value_counts(normalize = True)

# Histogram
sns.distplot(df_new_hotel_reviews['average_rating'], hist=True, kde=False, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'})

# Lables
plt.title('Histogram of Average Score')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/figures/Histogram_of_AvScore.png')



###############################################################################

## Review_Total_Negative_Word_Counts
np.mean(reviews_df['Review_Total_Negative_Word_Counts'])
np.std(reviews_df['Review_Total_Negative_Word_Counts'])

# Histogram
sns.distplot(reviews_df['Review_Total_Negative_Word_Counts'], hist=True, kde=False, 
             bins=int(20), color = 'red',
             hist_kws={'edgecolor':'black'})
# Lables
plt.title('Negative Review Word Count')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/figures/negative_review_word_count.png')


## Review_Total_Positive_Word_Counts
np.mean(reviews_df['Review_Total_Positive_Word_Counts'])
np.std(reviews_df['Review_Total_Positive_Word_Counts'])

# Histogram
sns.distplot(reviews_df['Review_Total_Positive_Word_Counts'], hist=True, kde=False, 
             bins=int(20), color = 'green',
             hist_kws={'edgecolor':'black'})
# Lables
plt.title('Positive Review Word Count')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/figures/positive_review_word_count.png')






## Mean Characters
print(np.mean(reviews_df["nb_chars"]))
print(np.std(reviews_df['nb_chars']))

## Mean Words
print(np.mean(reviews_df["nb_words"]))
print(np.std(reviews_df['nb_words']))






##############################################################################
# Heat Map
##############################################################################
# https://www.storybench.org/how-to-build-a-heatmap-in-python/
import gmaps 
import gmaps.datasets 

hotel_review_df




























