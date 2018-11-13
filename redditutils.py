# This is a utility file used to preprocess text for a reddit.com/r/showerthoughts
# NLP project

# USE RANDOM STATE 2325 FOR ALL TRAIN TEST SPLITS

import numpy as np
import re

from string import punctuation
from imblearn.over_sampling import SMOTE

def lower_string(s):
    return s.lower()

def strip_punc(s):
    return ''.join(c for c in s if c not in punctuation)

def strip_nums(s):
    return re.sub('[0-9]', '', s)

def make_labels(score):
    if score > 1:
        return 1
    else:
        return 0

def remove_bad_indices(features, response):
    # Remove document-response pairs that failed to be embedded in vector space
    bad_indices = list(np.unique(np.where(np.isnan(features))[0]))
        
    features = np.delete(features, bad_indices, axis=0)
    response = np.delete(np.array(response), bad_indices)
    
    return features, response

def upsample(features, response):
    # Returns SMOTE-upsample features and response

    sm = SMOTE()

    return sm.fit_sample(features, response)

def make_w2v_features(words, model, num_features):
    features = np.zeros(num_features)
    
    model_vocab = set(model.index2word)
    
    num_words = 0
    
    # Loop over words in documents. If the word is in model's vocabulary,
    # generate its feature vector
    for w in words:
        if w in model_vocab:
            num_words += 1
            features = np.add(features, model[w])
            
    # Normalize the feature vector
    features = np.divide(features, num_words)
    
    return features