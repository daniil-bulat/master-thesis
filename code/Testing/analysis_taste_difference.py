##############################################################################
##############################################################################
#                                                                            #
#                       Analysis of Taste Difference                         #
#                               Master Thesis                                #
#                                                                            #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# 1 ### import data set with all predictions of taste reviews
# and see what it predicts well, what not, can adjustments be made?
##############################################################################

# Read with parquet



taste_diff_result_df = pd.read_parquet("data/taste_diff_result_df_TEST.parquet", engine="fastparquet")

full_sample_reviews_df = pd.read_parquet("data/sample_sentiment_analysis_taste_diff.parquet", engine="fastparquet")


##The Variance Threshold was set to 1 for identifying high variance / taste reviews
## Do false predictions have a close to 1 variance?

false_predictions = taste_diff_result_df[(taste_diff_result_df['y_predicted'] == taste_diff_result_df['taste_diff_dummy'])==False]

# Histogram of false predictions
sns.set(style="darkgrid")

sns.histplot(data=false_predictions, x="var", kde=True)
plt.show()

# mean
np.mean(false_predictions['var'])







## False Negative
false_negative = taste_diff_result_df[(taste_diff_result_df['y_predicted'] - taste_diff_result_df['taste_diff_dummy'])==-1]

# Histogram of false predictions
sns.histplot(data=false_negative, x="neg", kde=True)
plt.show()

# mean
np.mean(false_negative['neg'])

#compared to full data set
full_sample_reviews_df = full_sample_reviews_df[full_sample_reviews_df['taste_diff_dummy']==1]
sns.histplot(data=full_sample_reviews_df, x="neg", kde=True)
plt.show()



##############################################################################
# 2 ### IMPACT
##############################################################################









# Hypothesis is that the bad/good algo doesn't recognize bad taste-reviews
# as bad reviews bc its a matter of taste.
# Purely from the language they seem good, but the person didnt enjoy stay
# therefore, look at reviews that the algo predicted to be good but are in 
# fact bad.
# look if we can find smth in variance in those cases



false_positives_good_bad = test_join[test_join['y_pred']==0]

false_negatives_good_bad = test_join[test_join['y_pred']==1]



## Review Rating
# Mean review_rating
print("False Positive Mean: " + str(np.mean(false_positives_good_bad['review_rating'])))
print("False Negative Mean: " + str(np.mean(false_negatives_good_bad['review_rating'])))
print("Overall Mean: " + str(np.mean(hotel_review_df['review_rating'])))
print("     ")

# SD review_rating
print("False Positive Var: " + str(np.var(false_positives_good_bad['review_rating'])))
print("False Negative Var: " + str(np.var(false_negatives_good_bad['review_rating'])))
print("Overall Var: " + str(np.var(hotel_review_df['review_rating'])))
print("     ")



## Average Hotel  Rating
# Mean review_rating
print("False Positive Mean: " + str(np.mean(false_positives_good_bad['average_rating'])))
print("False Negative Mean: " + str(np.mean(false_negatives_good_bad['average_rating'])))
print("Overall Mean: " + str(np.mean(hotel_review_df['average_rating'])))
print("     ")

# SD review_rating
print("False Positive Var: " + str(np.var(false_positives_good_bad['average_rating'])))
print("False Negative Var: " + str(np.var(false_negatives_good_bad['average_rating'])))
print("Overall Var: " + str(np.var(hotel_review_df['average_rating'])))
print("     ")




# look at one hotel, and see how many of which review category
# make distribution and look at variance

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



# large variance indicates spread

np.min(hotel_review_df['num_reviews'])
np.max(hotel_review_df['num_reviews'])
np.mean(hotel_review_df['num_reviews'])


high_variance = hotel_review_df[hotel_review_df['var']>1.0]
many_revies = hotel_review_df[hotel_review_df['num_reviews']>1900]


high_var_hotels = high_variance['hotel_name'].unique()
high_num_revies_hotels = many_revies['hotel_name'].unique()

comp_list = false_positives_good_bad['hotel_name'].apply(lambda x: any([k in x for k in high_var_hotels]))
comp_list_num = false_positives_good_bad['hotel_name'].apply(lambda x: any([k in x for k in high_num_revies_hotels]))



false_positives_good_bad['high_variance_match'] = comp_list
false_positives_good_bad['high_numRev_match'] = comp_list_num

sub = false_positives_good_bad[false_positives_good_bad['high_variance_match']==True]
sub_numRev = false_positives_good_bad[false_positives_good_bad['high_numRev_match']==True]


len(sub['high_variance_match']) / len(high_var_hotels) #28%
len(sub_numRev['high_numRev_match']) / len(high_num_revies_hotels) #34%


# 28% of false clasified false positives are from high variance hotels


