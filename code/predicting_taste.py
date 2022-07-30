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
import time
from dask import dataframe as dd
import dask.multiprocessing
import pyarrow
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC



# Read CSV
hotel_review_df = pd.read_csv("data/clean_tripadvisor_review_table.csv")



##############################################################################
# Use a Sample for Feature Selection
##############################################################################

# select only relevant columns
sample_reviews_df = hotel_review_df[['taste_diff_dummy',
                                       'review',
                                       'review_clean']]



bad_reviews = hotel_review_df[hotel_review_df["taste_diff_dummy"]==1].iloc[0:5000,:]
good_reviews = hotel_review_df[hotel_review_df["taste_diff_dummy"]==0].iloc[0:5000,:]

sample_frames = [bad_reviews, good_reviews]
sample_reviews_df = pd.concat(sample_frames)



##############################################################################
# Sentiment Analysis
##############################################################################

# Vader Lexicon
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

new_words = {
    'exceptional': 0.9,
    'overprice': -3.3,
    'bed bug': -6.6,
    'small': -3.0,
}

sid.lexicon.update(new_words)


# add sentiment anaylsis columns
sample_reviews_df["sentiments"] = sample_reviews_df["review_clean"].apply(lambda x: sid.polarity_scores(str(x)))
sample_reviews_df = pd.concat([sample_reviews_df.drop(['sentiments'], axis=1),
                               sample_reviews_df['sentiments'].apply(pd.Series)], axis=1)

# create doc2vec vector columns
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sample_reviews_df["review_clean"].apply(lambda x: str(x).split(" ")))]


# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4) # 26.5 sec


# transform each document into a vector data  time: 33.54 sec
doc2vec_df = sample_reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
sample_reviews_df = pd.concat([sample_reviews_df, doc2vec_df], axis=1)



# add tf-idfs columns
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(sample_reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = sample_reviews_df.index
sample_reviews_df = pd.concat([sample_reviews_df, tfidf_df], axis=1)



# Save DF
sample_reviews_df.to_parquet("data/TASTE_sample_sentiment_analysis.parquet", compression=None)


# Read csv with parquet
sample_reviews_df = pd.read_parquet("data/TASTE_sample_sentiment_analysis.parquet", engine="fastparquet")












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


data = sample_reviews_df.drop(['taste_diff_dummy','review', 'review_clean'],1)
target = sample_reviews_df['taste_diff_dummy']


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


search.fit(data,target)
search.best_params_ # {'model__n_estimators': 250, 'selector__k': 3339} all features
search.best_score_ # -0.1327006496



final_pipeline = search.best_estimator_
final_classifier = final_pipeline.named_steps['selector']

select_indices = final_pipeline.named_steps['selector'].transform(
    np.arange(len(data.columns)).reshape(1, -1))


feature_names = X_train.columns[select_indices]

feature_names = feature_names.tolist()
dds = pd.DataFrame(feature_names)
dds.to_csv("feature_list_taste.csv")

##############################################################################
# Random Forest
##############################################################################


label = "taste_diff_dummy"
ignore_cols = [label, 'review_clean', 'review']
features = [c for c in sample_reviews_df.columns if c not in ignore_cols]

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(sample_reviews_df[features], sample_reviews_df[label], test_size = 0.30, random_state = 77)


# train a random forest classifier
rf = RandomForestClassifier(n_estimators = 90, random_state = 77)
rf.fit(X_train, y_train)

# show feature importance
feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)
feature_importances_df.head(10)


# predictive power
y_preds = rf.predict(X_test)
print(rf.score(X_train, y_train))  # 1 (overfitting)
print(rf.score(X_test, y_test)) # 0.8726666666666667

# Confusion Matrix
print(metrics.confusion_matrix(y_test, y_preds))
#[[1239  234]
#[ 139 1388]]














# Get and reshape confusion matrix data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
#{'n_estimators': 500,
#'min_samples_split': 10,
#'min_samples_leaf': 1,
#'max_features': 'auto',
#'max_depth': 100,
#'criterion': 'entropy',
#'bootstrap': False}

# Reevaluate Model
y_preds = rf_random.predict(X_test)
print(rf_random.score(X_train, y_train))  #1.0
print(rf_random.score(X_test, y_test))   #0.8773333333333333

# Confusion Matrix
print(metrics.confusion_matrix(y_test, y_preds))
#[[1240  233]
#[ 135 1392]]


# Evaluate Results

# Base Model
print('Train Accuracy = {:0.2f}%.'.format(rf.score(X_train, y_train)*100))  # 1 (overfitting)
print('Test Accuracy = {:0.2f}%.'.format(rf.score(X_test, y_test)*100)) # 0.8726666666666667
base_accuracy = rf.score(X_test, y_test)


# Random Grid Model
print('Train Accuracy = {:0.2f}%.'.format(rf_random.score(X_train, y_train)*100)) #1.0
print('Test Accuracy = {:0.2f}%.'.format(rf_random.score(X_test, y_test)*100))   #0.8773333333333333
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
print('Train Accuracy = {:0.2f}%.'.format(rf_random.score(X_train, y_train)*100)) #1.0
print('Test Accuracy = {:0.2f}%.'.format(rf_random.score(X_test, y_test)*100))   #0.8773333333333333
random_accuracy = rf_random.score(X_test, y_test)

# non_random Grid Search Model
print('Train Accuracy = {:0.2f}%.'.format(grid_search.score(X_train, y_train)*100)) #
print('Test Accuracy = {:0.2f}%.'.format(grid_search.score(X_test, y_test)*100))   #
grid_search_accuracy = grid_search.score(X_test, y_test)

# Improvement
print('Improvement of {:0.2f}%.'.format( 100 * (grid_search_accuracy - random_accuracy) / random_accuracy))




##############################################################################
# Other Grid Search
##############################################################################

param_grid = { 
  'C': [0.1, 100, 500, 1000], 
  'gamma': [1, 0.1,  0.001]
}


other_grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
other_grid.fit(X_train,y_train) 

other_grid.best_params_


#other_grid.best_params_ = {'C': 500, 'gamma': 0.001}

param_grid = { 
  'C': [500], 
  'gamma': [0.01]
}


other_grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
other_grid.fit(X_train,y_train) 



## Reevaluate Results
# non_random Grid Search Model
print('Train Accuracy = {:0.2f}%.'.format(grid_search.score(X_train, y_train)*100)) #
print('Test Accuracy = {:0.2f}%.'.format(grid_search.score(X_test, y_test)*100))   #
grid_search_accuracy = grid_search.score(X_test, y_test)

# other Grid Search Model
print('Train Accuracy = {:0.2f}%.'.format(other_grid.score(X_train, y_train)*100)) #0.9341428571428572
print('Test Accuracy = {:0.2f}%.'.format(other_grid.score(X_test, y_test)*100))   #0.8883333333333333
other_grid_search_accuracy = other_grid.score(X_test, y_test)

# Improvement
print('Improvement of {:0.2f}%.'.format( 100 * (other_grid_search_accuracy - grid_search_accuracy) / grid_search_accuracy))





##############################################################################
# Result DF
##############################################################################

y_pred_other_grid = other_grid.predict(X_test)


taste_diff_result_df = pd.DataFrame(y_test)
taste_diff_result_df['y_predicted'] = y_pred_other_grid


full_sample_reviews_df = full_sample_reviews_df.drop(['taste_diff_dummy',
                                                      'Unnamed: 0.1',
                                                      'Unnamed: 0',
                                                      'hotel_id',
                                                      'review_title',
                                                      'review_text',
                                                      'review_clean'], 1)


taste_diff_result_df = taste_diff_result_df.join(full_sample_reviews_df)



# Full Data Set

data = full_sample_reviews_df.drop(['review_rating',
                                    'hotel_name',
                                    'review_clean'],1)



target = sample_reviews_df['taste_diff_dummy']

X_test = 
y_pred_other_grid = other_grid.predict(X_test)

full_sample_reviews_df.columns





# Save DF
taste_diff_result_df.to_parquet("data/taste_diff_result_df_TEST.parquet", compression=None)


# Read with parquet
taste_diff_result_df = pd.read_parquet("data/taste_diff_result_df_TEST.parquet", engine="fastparquet")















