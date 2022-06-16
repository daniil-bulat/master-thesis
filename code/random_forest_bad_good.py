##############################################################################
##############################################################################
#                                                                            #
#                      Random Forrest Prediction of NLP                      #
#                                good vs bad                                 #
#                               Master Thesis                                #
#                                                                            #
##############################################################################
##############################################################################

import pandas as pd
import os
import string
from sklearn import metrics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from nltk.metrics import ConfusionMatrix
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from funcsigs import signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import time
from dask import dataframe as dd
import dask.multiprocessing
import pyarrow
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC



##############################################################################
# Data Prearation
##############################################################################

hotel_review_df = pd.read_csv('FINAL_UK_hotel_reviews.csv')

hotel_review_df = hotel_review_df[hotel_review_df['average_rating']<5.1]
hotel_review_df = hotel_review_df[hotel_review_df['review_rating']<5.1]
hotel_review_df['poor'] = hotel_review_df['poor'].apply(pd.to_numeric)
hotel_review_df = hotel_review_df.reset_index(drop=True)




# Read csv with parquet
sample_reviews_df = pd.read_parquet("sample_sentiment_analysis_1_TA.parquet", engine="fastparquet")



##############################################################################
# Random Forest
##############################################################################


label = "bad_review_dummy"
ignore_cols = [label, 'review', 'review_clean', 'review_rating']
features = [c for c in sample_reviews_df.columns if c not in ignore_cols]

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(sample_reviews_df[features], sample_reviews_df[label], test_size = 0.30, random_state = 77)


# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 90, random_state = 77)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(20)

top_feat = feature_importances_df['feature'][:100].tolist()

# predictive power
y_preds = rf.predict(X_test)
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))

# Confusion Matrix
print(metrics.confusion_matrix(y_test, y_preds))

result_df = pd.DataFrame(y_test)
result_df['y_pred'] = y_preds








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
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/trip_advisor/figures/good_bad_ROC_curve.png')









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
plt.savefig('/Users/danielbulat/Desktop/Uni/Master Thesis/python/trip_advisor/figures/precision_recall.png')




##############################################################################
#                       Tuning the Random forest                             #
##############################################################################

#features = top_feat





# Number of trees in random forest
n_estimators = np.linspace(10, 300, int((300-10)/20) + 1, dtype=int)

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [1, 5, 10, 20, 50, 75, 100]

# Minimum number of samples required to split a node
min_samples_split = [1, 2, 5, 10, 15, 20]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4]

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

##############################################################################
# Random Grid Search for new RF Clasifier
##############################################################################

rf_base = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf_base,
                               param_distributions = random_grid,
                               n_iter = 30, cv = 5,
                               verbose=2,
                               random_state=77, n_jobs = 4)

rf_random.fit(X_train, y_train)
print("done")

rf_random.best_params_

# Reevaluate Model
y_preds = rf_random.predict(X_test)
print(rf_random.score(X_train, y_train))  #0.9748571428571429
print(rf_random.score(X_test, y_test))   #0.8956666666666667

# Confusion Matrix
print(metrics.confusion_matrix(y_test, y_preds))


y_preds_rand = rf_random.predict(X_test)
random_gs_result_df = pd.DataFrame(y_test)
random_gs_result_df['y_pred'] = y_preds_rand

random_gs_result_df["wrong_prediction"] = random_gs_result_df['bad_review_dummy'] != random_gs_result_df['y_pred']

false_predictions = random_gs_result_df[random_gs_result_df["wrong_prediction"] == True]




##############################################################################
# Grid Search
##############################################################################

param_grid = { 
  'C': [0.1, 1, 10, 100, 500], 
  'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train,y_train) 

grid.best_params_


#grid.best_params_ = {'C': 500, 'gamma': 0.001}

param_grid = { 
  'C': [500], 
  'gamma': [0.001]
}


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train,y_train) 


print(grid.score(X_train, y_train)) #0.9852857142857143
print(grid.score(X_test, y_test)) #0.8903333333333333

###
y_preds_grid = grid.predict(X_test)
grid_result_df = pd.DataFrame(y_test)
grid_result_df['y_pred'] = y_preds_grid

grid_result_df["wrong_prediction"] = grid_result_df['bad_review_dummy'] != grid_result_df['y_pred']

grid_false_predictions = grid_result_df[grid_result_df["wrong_prediction"] == True]




##############################################################################
# Feature Elimination
##############################################################################
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


data = sample_reviews_df.drop(['bad_review_dummy','review', 'review_clean'],1)
target = sample_reviews_df['bad_review_dummy']


pipeline = Pipeline(
    [
     ('selector',SelectKBest(f_regression)), #score variables according to F-score
     ('model',RandomForestRegressor(random_state = 77))
    ]
)




search = GridSearchCV(
    estimator = pipeline,
    param_grid = {
  'selector__k':[1000,2500,3205] , 
  'model__n_estimators':np.arange(90,250,20)   
 },
    n_jobs=-1,
    scoring="neg_mean_squared_error",
    cv=4,
    verbose=3
)


search.fit(data,target)
search.best_params_
search.best_score_

#[CV 4/4] END model__n_estimators=90, selector__k=1000;, score=-0.150 total time= 3.1min
#[CV 4/4] END model__n_estimators=90, selector__k=3205;, score=-0.150 total

# search.best_params_ = 'model__n_estimators': 230, 'selector__k': 2500}

final_pipeline = search.best_estimator_
final_classifier = final_pipeline.named_steps['selector']

select_indices = final_pipeline.named_steps['selector'].transform(
    np.arange(len(data.columns)).reshape(1, -1))


feature_names = X_train.columns[select_indices]

feature_names = feature_names.tolist()
dds = pd.DataFrame(feature_names)
dds.to_csv("feature_list.csv")


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

np.min(hotel_review_df['var'])
np.max(hotel_review_df['var'])


high_variance = hotel_review_df[hotel_review_df['var']>1.0]


high_var_hotels = high_variance['hotel_name'].unique()


comp_list = false_positives_good_bad['hotel_name'].apply(lambda x: any([k in x for k in high_var_hotels]))



false_positives_good_bad['high_variance_match'] = comp_list

sub = false_positives_good_bad[false_positives_good_bad['high_variance_match']==True]














