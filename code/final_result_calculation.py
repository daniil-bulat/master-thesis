##############################################################################
##############################################################################
#                                                                            #
#                               Final Results                                #
#                               Master Thesis                                #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# Directory
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')

# Import Data
hotel_review_df = pd.read_parquet('data/UK_hotel_reviews.parquet')
result_df = pd.read_csv('result_final_test_31_08.csv')




# Taste Reviews
taste_reviews = result_df[result_df['y_predicted']==1]

# EDA Taste Reviews
np.mean(taste_reviews['average_rating'])
np.std(taste_reviews['average_rating'])

np.mean(taste_reviews['review_rating'])
np.std(taste_reviews['review_rating'])


# EDA Full Data Set
np.mean(hotel_review_df['average_rating'])
np.std(hotel_review_df['average_rating'])

np.mean(hotel_review_df['review_rating'])
np.std(hotel_review_df['review_rating'])





# Scraped Reviews ONLY
grouped_full_df = hotel_review_df.groupby('hotel_name').agg(average_rating=('average_rating', 'mean'), review_rating=('review_rating', 'mean'), count = ('review_rating', 'size'), num_reviews = ('num_reviews', 'mean'))
grouped_taste_df = taste_reviews.groupby('hotel_name').agg(taste_average_rating=('average_rating', 'mean'), taste_review_rating=('review_rating', 'mean'), taste_count = ('review_rating', 'size'))

final_comparison_df = grouped_taste_df.merge(grouped_full_df, on='hotel_name', how='left')

final_comparison_df['adj_review_rating_scraped'] = (final_comparison_df['review_rating'] * final_comparison_df['count'] -  final_comparison_df['taste_review_rating'] * final_comparison_df['taste_count']) / (final_comparison_df['count'] - final_comparison_df['taste_count'])

final_comparison_df['diff_in_review_rating_scraped'] = final_comparison_df['review_rating'] - final_comparison_df['adj_review_rating_scraped']



# Extrapolation to full data set
final_comparison_df['adj_review_rating_full'] = (final_comparison_df['average_rating'] * final_comparison_df['num_reviews'] -  final_comparison_df['taste_review_rating'] * final_comparison_df['taste_count']) / (final_comparison_df['num_reviews'] - final_comparison_df['taste_count'])

final_comparison_df['diff_in_review_rating_full'] = final_comparison_df['review_rating'] - final_comparison_df['adj_review_rating_full']



# EDA Results
np.mean(final_comparison_df['diff_in_review_rating_scraped'])
np.mean(final_comparison_df['diff_in_review_rating_full'])

# What types of hotels are impacted most
taste_impact_scraped = final_comparison_df.groupby('average_rating').agg(diff_scraped=('diff_in_review_rating_scraped', 'mean'))
final_comparison_df.groupby('average_rating').agg(diff_full=('diff_in_review_rating_full', 'mean'))

#plot
plt.bar(taste_impact_scraped.index, abs(taste_impact_scraped['diff_scraped']), alpha=0.5, width=0.2)



# What months have most taste reviews

y_pred = taste_reviews['y_predicted']



joined_hotel_review_df = hotel_review_df.join(y_pred)
joined_hotel_review_df = joined_hotel_review_df[joined_hotel_review_df['y_predicted']==1]

joined_hotel_review_df = joined_hotel_review_df.reset_index(level=0)
joined_hotel_review_df['review_date'] = joined_hotel_review_df['review_date'].replace('Date of stay: ', '', regex=True)
joined_hotel_review_df['review_date'] = joined_hotel_review_df['review_date'].str.replace('\d+', '')


month_dist = joined_hotel_review_df.groupby('review_date').agg(count=('review_date', 'size'))
month_dist = month_dist.reset_index(level=0)

month_order = ['January ', 'February ', 'March ', 'April ', 'May ', 'June ', 'July ', 'August ', 'September ', 'October ', 'November ', 'December ']

month_order_2 = ['July ', 'August ', 'September ', 'October ', 'November ', 'December ', 'January ', 'February ', 'March ', 'April ', 'May ', 'June ']


month_dist['review_date'] = pd.Categorical(month_dist['review_date'], categories=month_order, ordered=True)
month_dist.sort_values(by='review_date',inplace=True)



#plot
plt.bar(month_dist['review_date'], month_dist['count'], alpha=0.5, width=0.2)
plt.xticks(fontsize=7, rotation=90)










