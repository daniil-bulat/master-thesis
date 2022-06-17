# Vader Lexicon

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def import_adj_vader():
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    
    new_words = {
        'exceptional': 0.9,
        'overprice': -3.3,
        'bed bug': -6.6,
        'small': -3.0,
    }
    
    sid.lexicon.update(new_words)
    
    return print("Imported and Updated Vader Lexicon")


