import pickle
import numpy as np
import redditutils as ru

model = pickle.load(open('../random_forest.pkl', 'rb'))
w2v = pickle.load(open('../w2v_mod.pkl'))
make_features = pickle.load(open('../w2v.pkl', 'rb'))
cv = pickle.load(open('../cv.pkl', 'rb'))


example = """It's ironic how "Finding Nemo" has made clownfish into one of
     the most popular fish for saltwater aquariums despite the bad guys
     in the movie being the ones who catch Nemo and put him in a tank"""

def make_features(showerthought):
    showerthought = ru.strip_punc(ru.strip_nums(ru.lower_string(s)))

    features = ru.make_w2v_features(showerthought.split(' '), w2v, 400)

    return features

def predict_success(showerthought):
    # Convert showerthought to 400 dimensional vector space
    features = make_features(showerthought)

    prob_success = model.predict_proba(features)[0, 1]

    result = {'prediction': int(prob_success > 0.5),
              'prob_success': prob_success}

    return result

if __name__ == '__main__':
    print(predict_success(example))