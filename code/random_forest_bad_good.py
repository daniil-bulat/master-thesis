##############################################################################
##############################################################################
#                                                                            #
#                          Random Forrest Prediction                         #
#                                good vs bad                                 #
#                               Master Thesis                                #
#                                                                            #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from funcsigs import signature
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc, roc_auc_score


data_dir = '/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/data/'
figure_dir = '/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis/figures/'


##############################################################################
# Data Prearation
##############################################################################

os.chdir('/Users/danielbulat/Desktop/Uni/Master Thesis/python/master-thesis')

# Read csv with parquet
nlp_review_df = pd.read_parquet("data/full_nlp_review_df.parquet", engine="fastparquet")
full_hotel_review_df = pd.read_parquet("data/full_hotel_review_df.parquet", engine="fastparquet")





##############################################################################
# Select Sample
##############################################################################

# select only relevant columns
sample_reviews_df = full_hotel_review_df[['bad_review_dummy']]

slim_nlp_review_df = nlp_review_df.drop(['review_title','review_text','review', 'review_clean'], 1)


# join with nlp df
sample_reviews_df = sample_reviews_df.join(slim_nlp_review_df)



bad_reviews = sample_reviews_df[sample_reviews_df["bad_review_dummy"]==1].iloc[0:5000,:]
good_reviews = sample_reviews_df[sample_reviews_df["bad_review_dummy"]==0].iloc[0:5000,:]

sample_frames = [bad_reviews, good_reviews]
sample_reviews_df = pd.concat(sample_frames)


# free up memory from unneccessary variables / df's
del(bad_reviews, good_reviews, sample_frames, slim_nlp_review_df)
gc.collect()



##############################################################################
# Train / Test Sets
##############################################################################


label = "bad_review_dummy"
ignore_cols = [label]
features = [c for c in sample_reviews_df.columns if c not in ignore_cols]



# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(sample_reviews_df[features], sample_reviews_df[label], test_size = 0.30, random_state = 77)






##############################################################################
# Feature Elimination
##############################################################################


pipeline = Pipeline(
    [('selector',SelectKBest(f_regression)), #score variables according to F-score
     ('model',RandomForestRegressor(random_state = 77))])




search = GridSearchCV(
    estimator = pipeline,
    param_grid = {
        'selector__k':[500,1000,5000,9762], 
        'model__n_estimators': [50,90,120,200,500]}, #np.arange(90,250,20)},
    n_jobs=2,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=3)


search.fit(sample_reviews_df[features],sample_reviews_df[label])
search.best_params_ # n_est=200, k=9762
search.best_score_ #neg_mean_squared_error: -0.1428 (-0.079)

# The grid search results say that all features should be used with 200 estimators.



##############################################################################
# Random Forest
##############################################################################

# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 90, random_state = 77)
rf.fit(X_train, y_train)

# predictive power
y_preds = rf.predict(X_test)
print(rf.score(X_train, y_train)) # 1.0
print(rf.score(X_test, y_test))   # 0.893

# Confusion Matrix
print(metrics.confusion_matrix(y_test, y_preds))
#[[1338  135]
#[ 186 1341]]

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(10)






##############################################################################
# ROC curve
##############################################################################

y_pred = [x[1] for x in rf.predict_proba(X_test)]

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
plt.savefig(figure_dir + 'good_bad_ROC_curve.png')









##############################################################################
# PR curve
##############################################################################

average_precision = average_precision_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})

plt.figure(1, figsize = (15, 10))
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.savefig(figure_dir + 'precision_recall_good_bad.png')








##############################################################################
# Random Grid Search for new RF Clasifier
##############################################################################


## Tuning the Random forest 

# Number of trees in random forest
n_estimators = [200,250]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [10, 50, 75, 100]

# Minimum number of samples required to split a node
min_samples_split = [1, 2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Criterion
criterion=['gini', 'entropy']
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion}



rf_base = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf_base,
                               param_distributions = random_grid,
                               n_iter = 30, cv = 3,
                               verbose=2,
                               random_state=77, n_jobs = 4)

rf_random.fit(X_train, y_train)

rf_random.best_params_


# Reevaluate Model
y_preds = rf_random.predict(X_test)
print(rf_random.score(X_train, y_train))  #0.9748571428571429
print(rf_random.score(X_test, y_test))   #0.8956666666666667

# Confusion Matrix
print(metrics.confusion_matrix(y_test, y_preds))




##############################################################################
# Grid Search via SVM Support Vector Machine
##############################################################################

param_grid = { 
  'C': [100,500,1000,2500,5000], # C is the penalty parameter, which represents misclassification or error term
  'gamma': [0.1,0.01,0.001,0.0001],  # Gamma defines how far influences the calculation of plausible line of separation
  'kernel': ['rbf', 'sigmoid']}


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train,y_train) # 5CV

grid.best_params_
# RESULT: grid.best_params_ = {'C': 5000, 'gamma': 0.0001,'kernel': 'rbf'}

param_grid = { 
  'C': [5000], 
  'gamma': [0.0001],
  'kernel': ['rbf']}


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train,y_train) 


print(grid.score(X_train, y_train)) #0.9801428571428571
print(grid.score(X_test, y_test)) #0.9063333333333333

###
y_preds_grid = grid.predict(X_test)
grid_result_df = pd.DataFrame(y_test)
grid_result_df['y_pred'] = y_preds_grid

grid_result_df["wrong_prediction"] = grid_result_df['bad_review_dummy'] != grid_result_df['y_pred']

grid_false_predictions = grid_result_df[grid_result_df["wrong_prediction"] == True]








##############################################################################
# Analysis
##############################################################################


false_predictions
grid_false_predictions


hotel_review_df = hotel_review_df.drop('bad_review_dummy', 1)

test_join = grid_false_predictions.join(hotel_review_df)


# Hypothesis is that the algo doesn't recognize bad taste-reviews
# Purely from the language they seem good, but the person didnt enjoy stay
# therefore, look at reviews that the algo predicted to be good but are in 
# fact bad.
# look if we can find smth in variance in those cases



test_join.columns


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










