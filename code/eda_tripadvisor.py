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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Data
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')
df_new_hotel_reviews = pd.read_csv('data/UK_hotel_reviews.csv')


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
sns.set_theme(style="whitegrid", palette="pastel")

ax = sns.boxplot(x="average_rating", y="num_reviews", data=df_new_hotel_reviews)
plt.title('Number of Reviews vs Average Score')
plt.xlabel('Average Score')
plt.ylabel('Number of Reviews')
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/boxplot.png')




# Total_Number_of_Reviews vs Average_Score
sns.regplot(x='average_rating', y="num_reviews", data=df_new_hotel_reviews,
            scatter_kws={"color": (0.5529411764705883, 0.6274509803921569, 0.796078431372549)},
            line_kws={"color": (0.9058823529411765, 0.5411764705882353, 0.7647058823529411)})



plt.title('Total Number of Reviews vs Average Score')
plt.xlabel('Average Score')
plt.ylabel('Total Number of Reviews')
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/number_of_reviews_v_avScore.png')




# Distribution of Average_Score
df_new_hotel_reviews['average_rating'].value_counts(normalize = True)

# Histogram
sns.distplot(df_new_hotel_reviews['average_rating'], hist=True, kde=False, 
             bins=int(180/5), color = (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
             hist_kws={'edgecolor':'black'})

# Lables
plt.title('Histogram of Average Score')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/Histogram_of_AvScore.png')









##############################################################################
# Heat Map
##############################################################################
# https://www.storybench.org/how-to-build-a-heatmap-in-python/
import gmaps 
import gmaps.datasets 





























