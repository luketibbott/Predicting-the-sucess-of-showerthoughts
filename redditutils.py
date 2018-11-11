# This is a utility file used to preprocess text for a reddit.com/r/showerthoughts
# NLP project

# USE RANDOM STATE 2325 FOR ALL TRAIN TEST SPLITS

import numpy as np
import re

from string import punctuation
from imblearn import SMOTE as sm

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
    bad_indices = list(np.unique(np.where(np.isnan(features, axis=0)[0])))
    
    features = np.delete(features, bad_indices, axis=0)
    response = np.delete(np.array(response), bad_indices)

def upsample(features, response):
    # Returns SMOTE-upsample features and response

    return sm.fit_sample(features, response)