##############################################################################
##############################################################################
#                                                                            #
#                         Random Forrest Prediction                          #
#                         Taste Differences Overall                          #
#                               Master Thesis                                #
#                                                                            #
#                               Daniil Bulat                                 #
#                                                                            #
##############################################################################
##############################################################################

import pandas as pd
import os
import nltk
from nltk.corpus import wordnet
import string
from sklearn import metrics
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from dask import dataframe as dd
import dask.multiprocessing
import pyarrow
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


directory_path = ''
directory_functions_path = ''
directory_figure_path = ''


# Import functions
os.chdir(directory_functions_path)
from random_forest_functions import roc_curve_custom, pr_curve_custom


# Directory
figure_dir = directory_figure_path
os.chdir(directory_path)

# Data
full_hotel_review_df = pd.read_parquet("data/full_hotel_review_df.parquet", compression=None)

nlp_review_df = pd.read_parquet('data/full_nlp_review_df.parquet')


# Read CSV
#hotel_review_df = pd.read_csv("data/clean_tripadvisor_review_table.csv")



##############################################################################
# Use a Sample for Feature Selection
##############################################################################

full_hotel_review_df.index = full_hotel_review_df.index.set_names(['ID'])
taste_df_rf = full_hotel_review_df.reset_index()
taste_df_rf = taste_df_rf[['ID', 'taste_diff_dummy']]

nlp_review_df.index = nlp_review_df.index.set_names(['ID'])

joined_hotel_review_df = pd.merge(taste_df_rf,nlp_review_df,on='ID',how='left')
joined_hotel_review_df = joined_hotel_review_df.drop(['ID', 'review_title', 'review_text', 'review','review_clean'],axis=1)


ntdr = joined_hotel_review_df[joined_hotel_review_df['taste_diff_dummy']==1].iloc[0:3000]
n_ntdr = joined_hotel_review_df[joined_hotel_review_df['taste_diff_dummy']==0].iloc[0:3000]

frames = [ntdr, n_ntdr]
joined_df = pd.concat(frames)

##############################################################################
# Train / Test Split
##############################################################################

label = "taste_diff_dummy"
ignore_cols = [label]
features = [c for c in joined_df.columns if c not in ignore_cols]

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(joined_df[features], joined_df[label], test_size = 0.30, random_state = 77)



##############################################################################
# Feature Elimination
##############################################################################

pipeline = Pipeline(
    [
     ('selector',SelectKBest(f_regression)), #score variables according to F-score
     ('model',RandomForestRegressor(random_state = 77))
    ]
)



search = GridSearchCV(
    estimator = pipeline,
    param_grid = {
  'selector__k':[100,1000,2500,3000,3339] , 
  'model__n_estimators':np.arange(50,350,50)   
 },
    n_jobs=-1,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=3
)


target = full_hotel_review_df[['taste_diff_dummy']]

search.fit(nlp_review_df,target)
search.best_params_ # {'model__n_estimators': 250, 'selector__k': 3339} all features
search.best_score_ # -0.1327006496



final_pipeline = search.best_estimator_
final_classifier = final_pipeline.named_steps['selector']

select_indices = final_pipeline.named_steps['selector'].transform(np.arange(len(data.columns)).reshape(1, -1))

feature_names = X_train.columns[select_indices]

feature_names = feature_names.tolist()
dds = pd.DataFrame(feature_names)
dds.to_csv("feature_list_taste.csv")

##############################################################################
# Random Forest
##############################################################################


# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 90, random_state = 77)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(10)


# predictive power
y_preds = rf.predict(X_test)
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))

# Confusion Matrix
print(metrics.confusion_matrix(y_test, y_preds))



# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_preds)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
               'Cottonwood/Willow', 'Aspen', 'Douglas-fir',    
               'Krummholz']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()



# View the classification report for test data and predictions
print(classification_report(y_test, y_preds))


# what we see is that if the algo works,
# we can predict taste reviews in any size sample
# so we look at the total of taste diff
# and can compute what the effect is on the overall rating of that hotel




##############################################################################
# Random Grid Search for new RF Clasifier
##############################################################################


# Tuning the Random forest 

# Number of trees in random forest
n_estimators = np.linspace(100, 500, int((500-50)/50) + 1, dtype=int)

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


rf_base = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf_base,
                               param_distributions = random_grid,
                               n_iter = 30, cv = 5,
                               verbose=2,
                               random_state=77, n_jobs = 4)

rf_random.fit(X_train, y_train)
print("done")

rf_random.best_params_
#{'n_estimators': np.linspace,
#'min_samples_split': 10,
#'min_samples_leaf': 1,
#'max_features': 'auto',
#'max_depth': 100,
#'criterion': 'entropy',
#'bootstrap': False}

# Reevaluate Model
y_preds = rf_random.predict(X_test)
print(rf_random.score(X_train, y_train))
print(rf_random.score(X_test, y_test))

# Confusion Matrix
print(metrics.confusion_matrix(y_test, y_preds))


# Evaluate Results

# Base Model
print('Train Accuracy = {:0.2f}%.'.format(rf.score(X_train, y_train)*100))
print('Test Accuracy = {:0.2f}%.'.format(rf.score(X_test, y_test)*100))
base_accuracy = rf.score(X_test, y_test)


# Random Grid Model
print('Train Accuracy = {:0.2f}%.'.format(rf_random.score(X_train, y_train)*100))
print('Test Accuracy = {:0.2f}%.'.format(rf_random.score(X_test, y_test)*100))
random_accuracy = rf_random.score(X_test, y_test)

# Improvement
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))




##############################################################################
# Non-Random Grid Search for new RF Clasifier
##############################################################################

# Grid Search based on above results
from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [False],
    'max_depth': [90, 100, 110],
    'max_features': ['auto'],
    'min_samples_leaf': [1, 2, 3, 4],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [250, 500, 1000]
}



# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
grid_search.best_params_


## Reevaluate Results
# Random Grid Model
print('Train Accuracy = {:0.2f}%.'.format(rf_random.score(X_train, y_train)*100))
print('Test Accuracy = {:0.2f}%.'.format(rf_random.score(X_test, y_test)*100))
random_accuracy = rf_random.score(X_test, y_test)

# non_random Grid Search Model
print('Train Accuracy = {:0.2f}%.'.format(grid_search.score(X_train, y_train)*100))
print('Test Accuracy = {:0.2f}%.'.format(grid_search.score(X_test, y_test)*100))
grid_search_accuracy = grid_search.score(X_test, y_test)

# Improvement
print('Improvement of {:0.2f}%.'.format( 100 * (grid_search_accuracy - random_accuracy) / random_accuracy))




##############################################################################
# Other Grid Search
##############################################################################

joined_hotel_review_df = pd.merge(full_hotel_review_df,nlp_review_df,on='ID', how='left')

param_grid = { 
  'C': [1000, 5000, 10000], 
  'gamma': [0.001, 0.0001]
}


other_grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
other_grid.fit(X_train,y_train) 

other_grid.best_params_


#other_grid.best_params_ = {'C': 1000, 'gamma': 0.001}

param_grid = { 
  'C': [5000], 
  'gamma': [0.0001]
}


other_grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3)
other_grid.fit(X_train,y_train)


# Confusion Matrix
y_preds = other_grid.predict(X_test)
print(metrics.confusion_matrix(y_test, y_preds))



## Reevaluate Results
# non_random Grid Search Model
print('Train Accuracy = {:0.2f}%.'.format(grid_search.score(X_train, y_train)*100))
print('Test Accuracy = {:0.2f}%.'.format(grid_search.score(X_test, y_test)*100))
grid_search_accuracy = grid_search.score(X_test, y_test)

# other Grid Search Model
print('Train Accuracy = {:0.2f}%.'.format(other_grid.score(X_train, y_train)*100))
print('Test Accuracy = {:0.2f}%.'.format(other_grid.score(X_test, y_test)*100))
other_grid_search_accuracy = other_grid.score(X_test, y_test)

# Improvement
print('Improvement of {:0.2f}%.'.format( 100 * (other_grid_search_accuracy - grid_search_accuracy) / grid_search_accuracy))


# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": other_grid.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(10)



# Curves ROC and PR


# ROC Curve
roc_curve_custom(other_grid, X_test, y_test, y_preds, 'taste_SVM_ROC_curve.png', figure_dir)


# PR Curve
pr_curve_custom(y_test, y_preds, 'taste_SVM_PR_curve.png', figure_dir)








