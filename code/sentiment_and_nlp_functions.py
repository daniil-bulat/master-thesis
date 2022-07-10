# Sentiment Analysis and NLP Functions Script

import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


sid = SentimentIntensityAnalyzer()

def clean_text(text):
    
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english') 
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)






def sentiment_analysis_for_reviews(df, clean_text):

    # select only relevant columns
    df = df[['review_title','review_text']]
    
    # append title and review text
    df["review"] = df["review_title"] +" "+ df["review_text"]
    df["review"] = df["review"].astype(str)
    
    # clean text data
    df["review_clean"] = df["review"].apply(lambda x: clean_text(x))
    
    # add number of characters column
    df["nb_chars"] = df["review_clean"].apply(lambda x: len(x))

    # add number of words column
    df["nb_words"] = df["review_clean"].apply(lambda x: len(x.split(" ")))

    # add sentiment anaylsis columns
    df["sentiments"] = df["review_clean"].apply(lambda x: sid.polarity_scores(str(x)))
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)
    
    # create doc2vec vector columns
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["review_clean"].apply(lambda x: str(x).split(" ")))]
    
    # train a Doc2Vec model with our text data
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    
    # transform each document into a vector data
    doc2vec_df = df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    df = pd.concat([df, doc2vec_df], axis=1)

    # add tf-idfs columns
    tfidf = TfidfVectorizer(min_df = 10)
    tfidf_result = tfidf.fit_transform(df["review_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = df.index
    df = pd.concat([df, tfidf_df], axis=1)

    return df






def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    



def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    



def add_descriptive_variables(df, upper_bad_review_threshold, high_var_threshold, distance_threshold_lower, distance_threshold_upper, bad_month_threshold):

    df = df.drop(['review_title'],1)
    
    # Add Bad Review Dummy
    df["bad_review_dummy"] = df["review_rating"].apply(lambda x: 1 if x < upper_bad_review_threshold else 0)
    
    # Add Variance of Reviews Column
    mu = []
    var = []
    
    for i in range(0,len(df['review_rating'])):
        distribution_list = []
        distribution_list.extend([5] * int(df['excellent'].iloc[i]))
        distribution_list.extend([4] * int(df['very_good'].iloc[i]))
        distribution_list.extend([3] * int(df['average'].iloc[i]))
        distribution_list.extend([2] * int(df['poor'].iloc[i]))
        distribution_list.extend([1] * int(df['terrible'].iloc[i]))
        
        mu.append(np.mean(distribution_list))
        var.append(np.var(distribution_list))
    
    df['mu'] = mu
    df['var'] = var
    
    # Add a dummy variable for reviews of high variance hotels
    df['many_reviews_dummy'] = df['num_reviews'].apply(lambda x: 1 if x > np.mean(df['num_reviews']) else 0)
    df['high_var_dummy'] = df['var'].apply(lambda x: 1 if  x > high_var_threshold else 0)
    df['dist_to_mu'] = 0
    df.loc[(df['high_var_dummy'] == 1) & (df['many_reviews_dummy'] == 1), 'dist_to_mu'] = (df['average_rating'] - df['review_rating'])
    
    # If the variance in review ratings and the number of reviews is high, then we
    # assume that a part of the variance can be explained by taste differences in customers.
    # The rest might be e.g. due to timing.
    # Therefore we add a varibale 'taste_diff_dummy' that is 1 if the variance is larger 1
    # and 0 otherwise.
    
    df['taste_diff_dummy'] = df['dist_to_mu'].apply(lambda x: 1 if distance_threshold_lower < x < distance_threshold_upper  else 0)
    
    # bad month dummy
    av_rating_group = df.groupby(['hotel_name'])['average_rating'].mean()
    av_rating_group = av_rating_group.reset_index() # df


    bad_month_group = df.groupby(['hotel_name', 'review_date'])['review_rating'].mean()
    bad_month_group = bad_month_group.reset_index() #df

    bad_month = pd.merge(bad_month_group, av_rating_group, on='hotel_name', how='left')

    bad_months = bad_month[ (bad_month['average_rating'] - bad_month['review_rating']) > bad_month_threshold]
    bad_months['bad_month_dummy'] = 1


    merged_df = df.merge(bad_months, on=["hotel_name","review_date"], how='left')
    merged_df['bad_month_dummy'] = merged_df['bad_month_dummy'].fillna(0)
    df = merged_df.drop(['review_rating_y', 'average_rating_y'], 1)
    
    df.rename(columns = {'hotel_id': 'hotel_id',
                         'review_text': 'review_text',
                         'review_rating_x': 'review_rating',
                         'review_date': 'review_date',
                         'hotel_name': 'hotel_name',
                         'num_reviews': 'num_reviews',
                         'average_rating_x': 'average_rating',
                         'excellent': 'excellent',
                         'very_good': 'very_good',
                         'average': 'average',
                         'poor': 'poor',
                         'terrible': 'terrible',
                         'tripadv_ranking': 'tripadv_ranking',
                         'bad_review_dummy': 'bad_review_dummy',
                         'mu': 'mu',
                         'var': 'var',
                         'many_reviews_dummy': 'many_reviews_dummy',
                         'high_var_dummy': 'high_var_dummy',
                         'dist_to_mu': 'dist_to_mu',
                         'taste_diff_dummy': 'taste_diff_dummy',
                         'bad_month_dummy': 'bad_month_dummy'}, inplace=True)

    
    return df, distribution_list




def parameterization_rf_tatse_pred(info_df, nlp_df, bad_review_threshold, variance_threshold, dtm_lower, dtm_upper, bad_month_threshold):
    
    # Add some additional Variables to the initial data set
    full_hotel_review_df = add_descriptive_variables(info_df, bad_review_threshold, variance_threshold, dtm_lower, dtm_upper, bad_month_threshold)
  
    # select only relevant columns
    sample_reviews_df = full_hotel_review_df[(full_hotel_review_df['dist_to_mu'] >0) & (full_hotel_review_df['bad_month_dummy'] == 0)]
    sample_reviews_df = sample_reviews_df[['taste_diff_dummy']]
  
    slim_nlp_review_df = nlp_df.drop(['review_title','review_text','review', 'review_clean'], 1)
  
  
    # join with nlp df
    sample_reviews_df = sample_reviews_df.join(slim_nlp_review_df)
  
    
    taste_reviews = sample_reviews_df[sample_reviews_df["taste_diff_dummy"]==1].iloc[0:400,:]
    print(str(len(taste_reviews)) + " Taste Reviews")
    non_taste_reviews = sample_reviews_df[sample_reviews_df["taste_diff_dummy"]==0].iloc[0:500,:]
    print(str(len(non_taste_reviews)) + " Non-Taste Reviews")
  
    sample_frames = [taste_reviews, non_taste_reviews]
    sample_reviews_df = pd.concat(sample_frames)
  
  
  
    # Training / Test Splitt
    ##############################################################################
  
  
    label = "taste_diff_dummy"
    ignore_cols = [label]
    features = [c for c in sample_reviews_df.columns if c not in ignore_cols]
  
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(sample_reviews_df[features], sample_reviews_df[label], test_size = 0.30, random_state = 77)
    
  
  
      # Rado Forest
    param_grid = {
        'C': [5000], 
        'gamma': [0.0001],
        'kernel': ['rbf']}
  
  
    other_grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3)
    other_grid.fit(X_train,y_train)
    
    print('Train Accuracy = {:0.2f}%.'.format(other_grid.score(X_train, y_train)*100))
    print('Test Accuracy = {:0.2f}%.'.format(other_grid.score(X_test, y_test)*100))
  
    ##############################################################################
    # Result of Prediction on Full Set
    ##############################################################################
  
    x_samp = slim_nlp_review_df.sample(frac=0.2)
  
    data_full = x_samp.join(full_hotel_review_df)
  
    # If the algorithm works, we now predict the whole data set,
    # and look if it found sound bad taste reviews in low var hotels
  
    full_y_preds = other_grid.predict(x_samp)
    print("predictions computed") 
    full_y_preds_proba = other_grid.predict_proba(x_samp)
    print("prediction probability computed")
  
    # Predicted Results
    data_full['y_predicted'] = full_y_preds
    data_full['y_probability'] = full_y_preds_proba[:,1]
  
    low_num_reviews = data_full[(data_full['many_reviews_dummy']==0) & (data_full['y_predicted']==1)]
    low_num_review_results = low_num_reviews[['y_predicted', 'y_probability', 'hotel_name','average_rating','review_rating','review_text','dist_to_mu','var','mu']]
    print("DONE!")
  
    return low_num_review_results
 
  
  































