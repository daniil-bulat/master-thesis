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
hotel_review_df = pd.read_parquet('data/UK_hotel_reviews.parquet')

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
np.mean(group_tatse['perc_of_taste'])

group_tatse['av_hotel_rat'] = group_tatse['av_hotel_rat'].apply(lambda x: 3.0 if (x > 2.8)  & (x < 3) else x)
av_rating_perc_taste = pd.DataFrame(group_tatse.groupby('av_hotel_rat')['perc_of_taste'].mean())
av_rating_perc_taste = av_rating_perc_taste.reset_index()

# plot
sns.set_theme(style="whitegrid")
sns.barplot(av_rating_perc_taste['av_hotel_rat'],av_rating_perc_taste['perc_of_taste']*100, color = (0.5529411764705883, 0.6274509803921569, 0.796078431372549))

plt.title('Fraction of Taste-Driven Reviews per Hotel Class')
plt.xlabel('Average Hotel Rating')
plt.ylabel('% Taset-Driven Reviews')
plt.savefig(figure_dir + 'perc_taste_per_cat.png')


####################################

scraped_rev_total = result_df.groupby('hotel_name')['average_rating', 'review_rating'].mean()
np.mean(scraped_rev_total)

non_taste_rev = result_df[result_df['y_predicted']==0]
scraped_rev_non_taste = non_taste_rev.groupby('hotel_name')['review_rating'].mean()
np.mean(scraped_rev_non_taste)

# plot
joined_taste_diff = pd.merge(scraped_rev_total,scraped_rev_non_taste,on='hotel_name',how='left')

joined_taste_diff['difference'] = joined_taste_diff['review_rating_y'] - joined_taste_diff['review_rating_x']
joined_taste_diff['average_rating'] = joined_taste_diff['average_rating'].apply(lambda x: 3.0 if (x > 2.8)  & (x < 3) else x)
av_rat_vs_diff = pd.DataFrame(joined_taste_diff.groupby('average_rating')['difference'].mean())
av_rat_vs_diff = av_rat_vs_diff.reset_index()

sns.set_theme(style="whitegrid")
sns.barplot(av_rat_vs_diff['average_rating'],av_rat_vs_diff['difference'], color = (0.5529411764705883, 0.6274509803921569, 0.796078431372549))

plt.title('Impact of Taste-Driven Reviews per Hotel Class')
plt.xlabel('Average Hotel Rating')
plt.ylabel('Impact of Taste-Driven Reviews')
plt.savefig(figure_dir + 'taste_impact_on_ratings.png')



########################################################################
# What months have most taste reviews

joined_hotel_review_df = result_df[(result_df['y_predicted']==1) & (result_df['review_rating']<3.1)]
joined_hotel_review_df.rename(columns = {'Unnamed: 0':'ID'}, inplace = True)

hotel_review_df.index = hotel_review_df.index.set_names(['ID'])
hotel_review_df_test = hotel_review_df.reset_index()

joined_hotel_review_df = pd.merge(joined_hotel_review_df,hotel_review_df_test[['ID','review_date']],on='ID', how='left')

#joined_hotel_review_df = joined_hotel_review_df.reset_index(level=0)
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

plt.title('Taste-Driven Reviews by Month')
plt.savefig(figure_dir + 'by_month_dist.png')



########################################################################
# Highest Probability
most_probable = result_df[(result_df['y_predicted']==1) & (result_df['review_rating']<3.1)]
most_probable = most_probable.sort_values('y_probability')
most_probable['review_text'].head(10)

most_probable.to_csv("most_probable_taste_reviews.csv")












