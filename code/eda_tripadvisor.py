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

# Figure Directory
figure_dir = '/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/'

# Data
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')
df_hotel_reviews = pd.read_parquet('data/UK_hotel_reviews.parquet')

# Add a dummy variable for ratings less than 3.6
df_hotel_reviews['bad_review_dummy'] = df_hotel_reviews['review_rating'].apply(lambda x: 1 if x<3.6 else 0)


###############################################################################
# EDA
###############################################################################

# Scraped Data (considering only reviews that were actually scraped)

review_column = pd.DataFrame(df_hotel_reviews['review_text'].apply(lambda x: len(str(x).split(' '))))

print(len(review_column['review_text']))
print(np.mean(review_column['review_text']))
print(np.std(review_column['review_text']))
print(max(review_column['review_text']))
print(min(review_column['review_text']))


# Word count vs Average_Score
df_hotel_reviews['word_count'] = review_column['review_text']

sns.set_style("white")
sns.regplot(x='review_rating', y="word_count", data=df_hotel_reviews,
            scatter_kws={"color": (0.5529411764705883, 0.6274509803921569, 0.796078431372549)},
            line_kws={"color": (0.9058823529411765, 0.5411764705882353, 0.7647058823529411)})



plt.title('Word Count vs Average Score')
plt.xlabel('Review Rating')
plt.ylabel('Word Count')
plt.savefig(figure_dir + 'word_count_v_avScore.png')


print(np.mean(df_hotel_reviews['review_rating']))
print(np.std(df_hotel_reviews['review_rating']))
print(np.min(df_hotel_reviews['review_rating']))
print(np.max(df_hotel_reviews['review_rating']))


print(np.mean(df_hotel_reviews['average_rating']))
print(np.std(df_hotel_reviews['average_rating']))
print(np.min(df_hotel_reviews['average_rating']))
print(np.max(df_hotel_reviews['average_rating']))


print(len(df_hotel_reviews.groupby('hotel_name')))




# Negative / Positive Reviews

neg_reviews = df_hotel_reviews[df_hotel_reviews['bad_review_dummy']==1]
pos_reviews = df_hotel_reviews[df_hotel_reviews['bad_review_dummy']==0]

len(neg_reviews['review_rating'])
print(np.mean(neg_reviews['review_rating']))
print(np.std(neg_reviews['review_rating']))

len(pos_reviews['review_rating'])
print(np.mean(pos_reviews['review_rating']))
print(np.std(pos_reviews['review_rating']))




len(neg_reviews) / len(df_hotel_reviews['bad_review_dummy'])
perc_of_neg_reviews = round((1-neg_reviews)*100,2)

# Negative Review Distribution
sns.set_style("white")
sns.distplot(neg_reviews['review_rating'], hist=True, kde=False, 
             bins=int(3), color = (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
             hist_kws={'edgecolor':'black'})

## Lables
plt.title('Distribution of Negative Reviews')
plt.ylim(0, 55e3)
plt.xlabel('Review Rating')
plt.ylabel('Frequency')
plt.savefig(figure_dir + 'Distribution_Neg_Reviews.png')



# Positive Review Distribution
sns.distplot(pos_reviews['review_rating'], hist=True, kde=False, 
             bins=int(2), color = (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
             hist_kws={'edgecolor':'black'})

## Lables
plt.title('Distribution of Positive Reviews')
plt.ylim(0, 55e3)
plt.xlabel('Review Rating')
plt.ylabel('Frequency')
plt.savefig(figure_dir + 'Distribution_Pos_Reviews.png')




# Extrapolated Data (extrapolating the underlying distribution data like number of reviews or number
# of excellent reviews, that were not neccessarily scraped, but the information is availbale)

hotel_ranking_dist = df_hotel_reviews.groupby('hotel_name')['num_reviews', 'average_rating', 'excellent', 'very_good', 'average', 'poor', 'terrible', 'tripadv_ranking'].mean()

print(np.mean(hotel_ranking_dist['num_reviews']))
print(np.std(hotel_ranking_dist['num_reviews']))
max(hotel_ranking_dist['num_reviews'])
min(hotel_ranking_dist['num_reviews'])



np.mean(hotel_ranking_dist['tripadv_ranking'])
np.std(hotel_ranking_dist['tripadv_ranking'])
max(hotel_ranking_dist['tripadv_ranking'])
min(hotel_ranking_dist['tripadv_ranking'])






# Barplot How much do categories account for num reviews
y_1 = np.mean(hotel_ranking_dist['excellent'] / hotel_ranking_dist['num_reviews'])
y_2 = np.mean(hotel_ranking_dist['very_good'] / hotel_ranking_dist['num_reviews'])
y_3 = np.mean(hotel_ranking_dist['average'] / hotel_ranking_dist['num_reviews'])
y_4 = np.mean(hotel_ranking_dist['poor'] / hotel_ranking_dist['num_reviews'])
y_5 = np.mean(hotel_ranking_dist['terrible'] / hotel_ranking_dist['num_reviews'])

x = ['terrible', 'poor', 'average', 'very good', 'excellent']
y = [y_5, y_4, y_3, y_2, y_1]

sns.set_theme(style="whitegrid", palette="pastel")
sns.barplot(x,y)
plt.savefig(figure_dir + 'barplot_cat_num_rev.png')




# BOXPLOT
ax = sns.boxplot(x="average_rating", y="num_reviews", data=hotel_ranking_dist)
plt.title('Number of Reviews vs Average Score')
plt.xlabel('Average Score')
plt.ylabel('Number of Reviews')
plt.savefig(figure_dir + 'boxplot.png')




# Total_Number_of_Reviews vs Average_Score
sns.set_style("white")
sns.regplot(x='average_rating', y="num_reviews", data=hotel_ranking_dist,
            scatter_kws={"color": (0.5529411764705883, 0.6274509803921569, 0.796078431372549)},
            line_kws={"color": (0.9058823529411765, 0.5411764705882353, 0.7647058823529411)})



plt.title('Total Number of Reviews vs Average Score')
plt.xlabel('Average Score')
plt.ylabel('Total Number of Reviews')
plt.savefig(figure_dir + 'number_of_reviews_v_avScore.png')




# Distribution of Average_Score
hotel_ranking_dist['average_rating'].value_counts(normalize = True)

# Histogram
sns.distplot(hotel_ranking_dist['average_rating'], hist=True, kde=False, 
             bins=int(180/5), color = (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
             hist_kws={'edgecolor':'black'})

# Lables
plt.title('Histogram of Average Score')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
plt.savefig(figure_dir + 'Histogram_of_AvScore.png')








##############################################################################
# Pair Plot
##############################################################################

pairplot_df = df_hotel_reviews[['bad_review_dummy', 'num_reviews','average_rating', 'tripadv_ranking']]
sns.pairplot(pairplot_df,hue='bad_review_dummy', palette='husl')

plt.savefig(figure_dir + 'pairplot.png')




##############################################################################
# Review Months
##############################################################################

df_hotel_reviews['review_date'] = df_hotel_reviews['review_date'].replace('Date of stay: ', '', regex=True)
df_hotel_reviews['review_date'] = df_hotel_reviews['review_date'].str.replace('\d+', '')


month_dist = df_hotel_reviews.groupby('review_date').agg(count=('review_date', 'size'))
month_dist = month_dist.reset_index(level=0)

month_order = ['January ', 'February ', 'March ', 'April ', 'May ', 'June ', 'July ', 'August ', 'September ', 'October ', 'November ', 'December ']

month_dist['review_date'] = pd.Categorical(month_dist['review_date'], categories=month_order, ordered=True)
month_dist.sort_values(by='review_date',inplace=True)



# Plot
sns.set_theme(style="whitegrid", palette="pastel")
bp = sns.barplot(month_dist['review_date'], month_dist['count'],color = (0.5529411764705883, 0.6274509803921569, 0.796078431372549))
bp.set_xticklabels(bp.get_xticklabels(),rotation = 45)

## Lables
plt.title('Reviews per Month')
plt.ylabel('')
plt.savefig(figure_dir + 'reviews_per_month.png')












