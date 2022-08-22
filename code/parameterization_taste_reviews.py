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
import scipy
from scipy.stats import norm, skewnorm, lognorm, stats


# add directories
directory_path = '/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis'
directory_functions_path = '/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/code'
directory_figure_path = ''


# Figure Directory
figure_dir = directory_figure_path

# Directory
os.chdir(directory_functions_path)
from sentiment_and_nlp_functions import parameterization_rf_tatse_pred, add_descriptive_variables
from parameterization_functions import normal, skew_normal


# Read in data
os.chdir(directory_path)
hotel_review_df = pd.read_parquet('data/full_hotel_review_df.parquet')
nlp_review_df = pd.read_parquet('data/full_nlp_review_df.parquet')
nlp_review_df = nlp_review_df.drop_duplicates(subset='review_text', keep="last") # drop duplicates



# Set Variables
bad_review_threshold = 3.1
bad_month_threshold = 2.1


result_df = parameterization_rf_tatse_pred(hotel_review_df,
                                           nlp_review_df,
                                           bad_review_threshold,
                                           bad_month_threshold)
# estimated computation time: 12min


###############################################################################
# FINAL DATA FRAME WITH PREDICTION RESULTS
###############################################################################
result_df.to_csv("result_data_wo_train.csv")

    


