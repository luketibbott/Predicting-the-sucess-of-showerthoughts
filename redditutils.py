# This is a utility file used to preprocess text for a reddit.com/r/showerthoughts
# NLP project

def lower_string(s):
    return s.lower()

def strip_punc(s):
    return ''.join(c for c in s if c not in punctuation)

def strip_nums(s):
    return re.sub('[0-9]', '', s)