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
import seaborn as sns


# Directory
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')


# Figure Directory
figure_dir = '/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/'


# Import Data
#hotel_review_df = pd.read_parquet('data/UK_hotel_reviews.parquet')
#result_df = pd.read_csv('result_final_test_31_08.csv')

result_df = pd.read_csv('data/result_data_wo_train.csv')
result_df = result_df.drop_duplicates(subset='review_text', keep="last")


# Taste Reviews
taste_reviews = result_df[(result_df['y_predicted']==1) & (result_df['review_rating']<3.1)]

# EDA Taste Reviews
np.mean(taste_reviews['average_rating'])
np.std(taste_reviews['average_rating'])

np.mean(taste_reviews['review_rating'])
np.std(taste_reviews['review_rating'])

len(taste_reviews) / len(result_df['y_predicted'])


#groupby
group_tatse = taste_reviews.groupby('hotel_name').agg(
    num_reviews=('num_reviews', np.mean),
    mu=('mu_hotels', np.mean),
    av_hotel_rat=('average_rating', np.mean),
    num_taste=('y_predicted', 'count'))

group_tatse['perc_of_taste'] = group_tatse['num_taste'] / group_tatse['num_reviews']
max(group_tatse['perc_of_taste']) #percentage of tatse-driven reviews per hotel


group_tatse['av_hotel_rat'] = group_tatse['av_hotel_rat'].apply(lambda x: 3.0 if (x > 2.8)  & (x < 3) else x)
av_rating_perc_taste = pd.DataFrame(group_tatse.groupby('av_hotel_rat')['perc_of_taste'].mean())
av_rating_perc_taste = av_rating_perc_taste.reset_index()

sns.set_theme(style="whitegrid")
sns.barplot(av_rating_perc_taste['av_hotel_rat'],av_rating_perc_taste['perc_of_taste']*100, color = (0.5529411764705883, 0.6274509803921569, 0.796078431372549))

plt.title('Fraction of Taste-Driven Reviews per Hotel Class')
plt.xlabel('Average Hotel Rating')
plt.ylabel('% Taset-Driven Reviews')
plt.savefig(figure_dir + 'perc_taste_per_cat.png')







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





########################################################################

taste_driven = result_df[(result_df['y_predicted']==1) & (result_df['review_rating']<3.1)]

len(taste_driven) / len(result_df['y_predicted'])

np.mean(taste_driven['review_rating'])
np.std(taste_driven['review_rating'])

np.mean(taste_driven['average_rating'])
np.std(taste_driven['average_rating'])

#groupby
group_tatse = taste_driven.groupby('hotel_name').agg(
    num_reviews=('num_reviews', np.mean),
    mu=('mu_hotels', np.mean),
    av_hotel_rat=('average_rating', np.mean),
    num_taste=('y_predicted', 'count'))

group_tatse['perc_of_taste'] = group_tatse['num_taste'] / group_tatse['num_reviews']
max(group_tatse['num_taste'] / group_tatse['num_reviews'])


########################################################################


most_probable = result_df.sort_values('y_probability')

most_probable['review_text'].head(10)


# Some Means
np.mean(most_probable['average_rating'])
np.mean(most_probable['review_rating'])
np.mean(most_probable['average_rating'] - most_probable['review_rating'])
np.mean(most_probable['y_probability'])

most_probable.to_csv("test_taste.csv")


