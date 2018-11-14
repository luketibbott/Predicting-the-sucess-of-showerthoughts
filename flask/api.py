import pickle
import numpy as np
import re

from gensim.models import KeyedVectors
from string import punctuation

w2v = KeyedVectors.load('../word2vec_mod.pkl', mmap='r')
model = pickle.load(open('../final_model.pkl', 'rb'))


example = {'showerthought': """It's ironic how "Finding Nemo" has made clownfish into one of
     the most popular fish for saltwater aquariums despite the bad guys
     in the movie being the ones who catch Nemo and put him in a tank"""}

def lower_string(s):
    return s.lower()

def strip_punc(s):
    return ''.join(c for c in s if c not in punctuation)

def strip_nums(s):
    return re.sub('[0-9]', '', s)

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

def make_features(showerthought):
    showerthought = strip_punc(strip_nums(lower_string(showerthought['showerthought'])))

    features = make_w2v_features(showerthought.split(' '), w2v, 400)

    return features

def predict_success(showerthought):
    # Convert showerthought to 400 dimensional vector space
    features = make_features(showerthought)

    prob_success = round(model.predict_proba(features.reshape(1, -1))[0, 1], 2)

    if prob_success > 0.5:
        pred = True
    else:
        pred = False

    result = {'prediction': pred,
              'prob_good': prob_success}

    return result

if __name__ == '__main__':
    print(predict_success(example))