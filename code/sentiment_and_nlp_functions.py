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
    



def add_descriptive_variables(df, upper_bad_review_threshold, high_var_threshold, distance_threshold):

    df.drop(['review_title', 'review_text', 'Unnamed: 0.1', 'Unnamed: 0'],1)
    
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
    df['high_var_dummy'] = df['var'].apply(lambda x: 1 if x > high_var_threshold else 0)
    df['dist_to_mu'] = 0
    df.loc[(df['high_var_dummy'] == 1) & (df['many_reviews_dummy'] == 1), 'dist_to_mu'] = abs(df['average_rating'] - df['review_rating'])
    
    # If the variance in review ratings and the number of reviews is high, then we
    # assume that a part of the variance can be explained by taste differences in customers.
    # The rest might be e.g. due to timing.
    # Therefore we add a varibale 'taste_diff_dummy' that is 1 if the variance is larger 1
    # and 0 otherwise.
    
    df['taste_diff_dummy'] = df['dist_to_mu'].apply(lambda x: 1 if x > distance_threshold  else 0)

    return df 

























