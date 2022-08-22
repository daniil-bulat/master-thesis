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

# add directories
directory_path = ''
directory_functions_path = ''
directory_figure_path = ''

# Directory
os.chdir(directory_path)


# Figure Directory
figure_dir = directory_figure_path


# Import Data
hotel_review_df = pd.read_parquet('data/UK_hotel_reviews.parquet')

result_df = pd.read_csv('data/result_data_wo_train.csv')


# Taste Reviews
len(result_df[result_df['y_predicted']==1]) / len(result_df['y_predicted'])
taste_reviews = result_df[(result_df['y_predicted']==1) & (result_df['review_rating']<3.1)]

# result df
np.mean(result_df['review_rating'])
np.std(result_df['review_rating'])
len(result_df['review_rating'])

np.mean(result_df['average_rating'])
np.std(result_df['average_rating'])
len(result_df['hotel_name'].unique())



# EDA Taste Reviews
np.mean(taste_reviews['review_rating'])
np.std(taste_reviews['review_rating'])
len(taste_reviews['review_rating'])
min(taste_reviews['review_rating'])
max(taste_reviews['review_rating'])

np.mean(taste_reviews['average_rating'])
np.std(taste_reviews['average_rating'])
len(taste_reviews['hotel_name'].unique())
min(taste_reviews['average_rating'])
max(taste_reviews['average_rating'])

len(taste_reviews) / len(result_df['y_predicted'])


# groupby hotel name
group_tatse = taste_reviews.groupby('hotel_name').agg(
    num_reviews=('num_reviews', np.mean),
    mu=('mu_hotels', np.mean),
    av_hotel_rat=('average_rating', np.mean),
    num_taste=('y_predicted', 'count'))



group_tatse['perc_of_taste'] = group_tatse['num_taste'] / group_tatse['num_reviews']
min(group_tatse['perc_of_taste'])
max(group_tatse['perc_of_taste']) #percentage of tatse-driven reviews per hotel
np.mean(group_tatse['perc_of_taste'])
np.std(group_tatse['perc_of_taste'])

group_tatse['av_hotel_rat'] = group_tatse['av_hotel_rat'].apply(lambda x: 3.0 if (x > 2.8)  & (x < 3) else x)
av_rating_perc_taste = pd.DataFrame(group_tatse.groupby('av_hotel_rat')['perc_of_taste'].mean())
av_rating_perc_taste = av_rating_perc_taste.reset_index()

# plot
sns.set_theme(style="whitegrid")
sns.barplot(av_rating_perc_taste['av_hotel_rat'],av_rating_perc_taste['perc_of_taste']*100, color = (0.788558246828143, 0.8066897347174163, 0.8948558246828143))

plt.title('Fraction of Taste-Driven Reviews per Hotel Class')
plt.xlabel('Average Hotel Rating')
plt.ylabel('% Taset-Driven Reviews')
plt.savefig(figure_dir + 'perc_taste_per_cat.png')


##############################################################################

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
sns.barplot(av_rat_vs_diff['average_rating'],av_rat_vs_diff['difference'], color = (0.788558246828143, 0.8066897347174163, 0.8948558246828143))

plt.title('Impact of Taste-Driven Reviews per Hotel Class')
plt.xlabel('Average Hotel Rating')
plt.ylabel('Impact of Taste-Driven Reviews')
plt.savefig(figure_dir + 'taste_impact_on_ratings.png')


##############################################################################
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



##############################################################################


#take result df exclude taste and compute all averages and sd again, what is overall impact


non_taste_reviews = result_df[(result_df['y_predicted']==0)]

np.mean(non_taste_reviews['review_rating'])
np.std(non_taste_reviews['review_rating'])
len(non_taste_reviews['review_rating'])

np.mean(non_taste_reviews['average_rating'])
np.std(non_taste_reviews['average_rating'])
len(non_taste_reviews['hotel_name'].unique())





taste_reviews.rename(columns = {'Unnamed: 0':'ID'}, inplace = True)

join_res_hot = pd.merge(taste_reviews,hotel_review_df_test[['ID', 'excellent', 'very_good', 'average', 'poor', 'terrible']],on='ID', how='left')


after_clean_join_res_hot = join_res_hot


excellent = []
very_good = []
average = []
poor = []
terrible = []

for i in range(0, len(after_clean_join_res_hot)):
    
    if after_clean_join_res_hot.iloc[i,5] == 5:
        excellent.append(1)
    else:
        excellent.append(0)
    
    if after_clean_join_res_hot.iloc[i,5] == 4:
        very_good.append(1)
    else:
        very_good.append(0)
    
    if after_clean_join_res_hot.iloc[i,5] == 3:
        average.append(1)
    else:
        average.append(0)
    
    if after_clean_join_res_hot.iloc[i,5] == 2:
        poor.append(1)
    else:
        poor.append(0)
    
    if after_clean_join_res_hot.iloc[i,5] == 1:
        terrible.append(1)
    else:
        terrible.append(0)
        


after_clean_join_res_hot['excellent_sub'] = excellent
after_clean_join_res_hot['very_good_sub'] = very_good
after_clean_join_res_hot['average_sub'] = average
after_clean_join_res_hot['poor_sub'] = poor
after_clean_join_res_hot['terrible_sub'] = terrible



grouped_final = after_clean_join_res_hot.groupby('hotel_name').agg(
    average_rating =('average_rating', np.mean),
    mu_hotels = ('mu_hotels', np.mean),
    std_hotels = ('std_hotels', np.mean),
    excellent = ('excellent', np.mean),
    very_good = ('very_good', np.mean),
    average = ('average', np.mean),
    poor = ('poor', np.mean),
    terrible = ('terrible', np.mean),
    excellent_sub = ('excellent_sub', np.sum),
    very_good_sub= ('very_good_sub', np.sum),
    average_sub = ('average_sub', np.sum),
    poor_sub = ('poor_sub', np.sum),
    terrible_sub = ('terrible_sub', np.sum),
    num_taste_rev = ('y_predicted', np.sum),
    num_rev = ('num_reviews', np.mean))



grouped_final['new_excellent'] = grouped_final['excellent'] - grouped_final['excellent_sub']
grouped_final['new_very_good'] = grouped_final['very_good'] - grouped_final['very_good_sub']
grouped_final['new_average'] = grouped_final['average'] - grouped_final['average_sub']
grouped_final['new_poor'] = grouped_final['poor'] - grouped_final['poor_sub']
grouped_final['new_terrible'] = grouped_final['terrible'] - grouped_final['terrible_sub']



# Add Average Skewness of ratings to df
mu_hotels = []
std_hotels = []

for i in range(0, len(grouped_final['excellent'])):
    distribution_list = []

    distribution_list.extend([5] * int(grouped_final['new_excellent'].iloc[i]))
    distribution_list.extend([4] * int(grouped_final['new_very_good'].iloc[i]))
    distribution_list.extend([3] * int(grouped_final['new_average'].iloc[i]))
    distribution_list.extend([2] * int(grouped_final['new_poor'].iloc[i]))
    distribution_list.extend([1] * int(grouped_final['new_terrible'].iloc[i]))
    
    mu_hotels.append(np.mean(distribution_list))
    std_hotels.append(np.std(distribution_list))

grouped_final['new_mu_hotels'] = mu_hotels
grouped_final['new_std_hotels'] = std_hotels




grouped_final = grouped_final[['average_rating','mu_hotels', 'std_hotels','new_mu_hotels','new_std_hotels', 'num_taste_rev', 'num_rev']]
grouped_final['diff_to_old'] = grouped_final['new_mu_hotels'] - grouped_final['mu_hotels']
grouped_final['rev_taste_ratio'] = grouped_final['num_taste_rev'] / grouped_final['num_rev']
    
    
    
np.mean(grouped_final['new_mu_hotels'])
np.std(grouped_final['new_mu_hotels'])
len(grouped_final['new_mu_hotels'])
min(grouped_final['new_mu_hotels'])
max(grouped_final['new_mu_hotels'])


np.mean(grouped_final['mu_hotels'])
np.std(grouped_final['mu_hotels'])
len(grouped_final['mu_hotels'])
min(grouped_final['mu_hotels'])
max(grouped_final['mu_hotels'])


np.mean(grouped_final['diff_to_old'])
np.std(grouped_final['diff_to_old'])
len(grouped_final['diff_to_old'])
max(grouped_final['diff_to_old'])
min(grouped_final['diff_to_old'])


sns.set_theme(style="white")
sns.regplot(x='diff_to_old', y="rev_taste_ratio", data=grouped_final,
            scatter_kws={"color": (0.788558246828143, 0.8066897347174163, 0.8948558246828143)},
            line_kws={"color": (0.2823529411764706, 0.47058823529411764, 0.8156862745098039)})

plt.title('Taste-Driven Review Impact on Rating Improvement')
plt.xlabel('Rating Improvement')
plt.ylabel('Ratio')
plt.savefig(figure_dir + 'taste_review_impact.png')



##############################################################################
# Highest Probability
most_probable = result_df[(result_df['y_predicted']==1) & (result_df['review_rating']<3.1)]
most_probable = most_probable.sort_values('y_probability')
most_probable['review_text'].head(10)

most_probable.to_csv("data/most_probable_taste_reviews.csv")







