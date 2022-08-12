##############################################################################
##############################################################################
#                                                                            #
#                    Parameterization of Taste Reviews                       #
#                               Master Thesis                                #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import pandas as pd
import os
import numpy as np
import seaborn as sns
import scipy
from scipy.stats import norm, skewnorm, lognorm, stats
import math
import matplotlib.pyplot as plt

# Figure Directory
figure_dir = '/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/'

# Directory
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/code')
from sentiment_and_nlp_functions import parameterization_rf_tatse_pred
from parameterization_functions import normal, skew_normal


# Read in data
os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')
hotel_review_df = pd.read_parquet('data/UK_hotel_reviews.parquet')
nlp_review_df = pd.read_parquet('data/full_nlp_review_df.parquet')

# Group By Hotel Names
gk = hotel_review_df.groupby('hotel_name')['excellent','very_good','average', 'poor', 'terrible'].mean()


# Add Average Skewness of ratings to df
mu_hotels = []
std_hotels = []
skew_hotels = []

for i in range(0, len(gk['excellent'])):
    distribution_list = []

    distribution_list.extend([5] * int(gk['excellent'].iloc[i]))
    distribution_list.extend([4] * int(gk['very_good'].iloc[i]))
    distribution_list.extend([3] * int(gk['average'].iloc[i]))
    distribution_list.extend([2] * int(gk['poor'].iloc[i]))
    distribution_list.extend([1] * int(gk['terrible'].iloc[i]))
    
    mu_hotels.append(np.mean(distribution_list))
    std_hotels.append(np.std(distribution_list))
    skew_hotels.append(scipy.stats.skew(distribution_list, axis = 0, bias = True))


gk['mu_hotels'] = mu_hotels
gk['std_hotels'] = std_hotels
gk['skew_hotels'] = skew_hotels

np.mean(gk['mu_hotels'])
np.mean(gk['std_hotels'])
np.mean(gk['skew_hotels'])

min(gk['skew_hotels'])
max(gk['skew_hotels'])



# Total skewness of ratings

distribution_list = []

for i in range(0,len(gk['excellent'])):
    distribution_list.extend([5] * int(gk['excellent'].iloc[i]))
    distribution_list.extend([4] * int(gk['very_good'].iloc[i]))
    distribution_list.extend([3] * int(gk['average'].iloc[i]))
    distribution_list.extend([2] * int(gk['poor'].iloc[i]))
    distribution_list.extend([1] * int(gk['terrible'].iloc[i]))
    
mu = np.mean(distribution_list)
sd = np.std(distribution_list)
skew = scipy.stats.skew(distribution_list, axis = 0, bias = True)
num_rev = len(distribution_list)


# create some random data from a skewnorm
data = skewnorm.rvs(skew, loc=mu, scale=sd, size=num_rev)

# draw a histogram and kde of the given data
ax = sns.distplot(data, kde_kws={'label':'kde of given data'}, label='histogram')

# 70%
sd_mod = sd * 1.037
scipy.stats.skewnorm.cdf(mu+sd_mod,skew,mu,sd) - scipy.stats.skewnorm.cdf(mu-sd_mod,skew,mu,sd)

# 5%
upper = 1.969
lower = 0
scipy.stats.skewnorm.cdf(upper,skew,mu,sd)


##############################################################################


# Full Distribution
full_rr_distribution = pd.DataFrame(distribution_list, columns=['review_rating'])

mu = np.mean(full_rr_distribution)
sd = np.std(full_rr_distribution)
skew = float(scipy.stats.skew(full_rr_distribution, axis = 0, bias = True))*(-1)
num_rev = len(full_rr_distribution)

ax = sns.histplot(full_rr_distribution, stat="density", 
             bins=int(5), color = (0.5529411764705883, 0.6274509803921569, 0.796078431372549))

skew_normal(mu, sd, skew, num_rev)
normal(mu, sd, num_rev)


## Lables
plt.title('Skew-Normal Approximation')
plt.xlabel('Review Rating')
plt.ylabel('pdf')
plt.legend(labels=["skew normal distribution","normal distribution", "Review Rating"])
plt.savefig(figure_dir + 'skew_norm_approx.png')



##############################################################################

# Set Variables
bad_review_threshold = 3.6 # or 3.1 [mean is 3.97, 3.5 and lower can be considered bad]
variance_threshold = 1.1407
dtm_lower = 1.037
dtm_upper = 1.969
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























