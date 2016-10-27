"""
calc TF-IDF for each (word, document) in brown corpus
"""
from collections import defaultdict
from operator import add
from nltk.corpus import brown
from collections import Counter
import math

# load data
sents = brown.sents()[:500]
sents = [[w.lower() for w in sent] for sent in sents]
# build foolish docs
sents_in_a_doc = 20
all_docs = [reduce(add, sents[i: i + sents_in_a_doc]) for i in range(0, len(sents), sents_in_a_doc)]
all_words = set(reduce(add, all_docs))

# TF
tf_mapping = {} # (w, doc-id): tf
for i, doc in enumerate(all_docs):
    for w, count in Counter(doc).items():
        tf_mapping[(w, i)] = float(count) / len(doc)

# IDF
idf_mapping = defaultdict(int) # w: idf
for w in all_words:
    df = sum([(w in doc) for doc in all_docs])
    idf_mapping[w] = 1. + math.log(len(all_docs) / df)

# printing
for i, doc in enumerate(all_docs):
    for w in doc:
        tf = tf_mapping[(w, i)]
        idf = idf_mapping[w]
        print 'tf-idf of {} in doc{}: {}'.format(w, i, tf * idf)
