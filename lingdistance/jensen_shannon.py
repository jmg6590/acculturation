
import os
import re
import glob
import sys
import random
import logging
import cPickle as pickle
from operator import itemgetter
from collections import defaultdict, Counter
import numpy as np

from tokenizers import TwitterTokenizer as Tokenizer

############

DIR = os.path.split(__file__)[0]
LIWC = pickle.load(file(os.path.join(DIR, 'LIWC2007dictionary_regex.pickle')))

TOKENIZER = Tokenizer(preserve_case=False,
                      preserve_all_caps=False,
                      filter_html_tags=True,
                      filter_twitter_usernames=True, 
                      filter_twitter_hashtags=True,
                      filter_urls=True,
                      filter_dates=True,
                      normalize_dates=True,
                      mark_negation_scope=False)

############

def jensen_shannon_distances(documents, doc_segmentation_fnc=lambda x: [x['frm']], 
                                liwc_map=False, vocabsize=1000,
                                sampling=False, sampsize=1000,
                                min_segment_size=100):
    """
    Workhorse function for measuring JS distance between two sets of documents.
    
    args:
        documents- list of documents. Docs must support accessing text 
            via brackets-- d['text']
        doc_segmentation_fnc - callable - given a document, return a list of strings,
            each string a representation of a segment the document belongs to

    kargs:


    return value:
        for each pair of segments of documents (as defined by what 
        doc_segmentation_fnc returns), we compute the JS-distance. 
        Return format is a dict--
            {('segment1', 'segment2'): JS distance (float)}
    """

    # Put documents into buckets that we want to compare:
    segments2docs = defaultdict(list)
    for i, msg in enumerate(documents):
        sys.stderr.write('\r') ; sys.stderr.write('msg %s' % i) ; sys.stderr.flush()
        keys = doc_segmentation_fnc(msg)
        if not isinstance(keys, list):
            keys = [keys]
        for k in keys:
            segments2docs[k].append(msg)
    sys.stderr.write('\n')

    # Reduce size of buckets, if desired:
    if sampling:
        # First get rid of any buckets with insufficient docs
        segments = segments2docs.keys()
        for key in segments:
            if len(segments2docs[key]) < sampsize:
                print "Too few messages for ", key
                del segments2docs[key]
        # Now randomly sample
        for key, messages in segments2docs.iteritems():
            random.shuffle(messages)
            segments2docs[key] = random.sample(messages, sampsize)   

    # Throw out segments with too few messages, if desired:
    if min_segment_size:
        segments = segments2docs.keys()
        for key in segments:
            if len(segments2docs[key]) < min_segment_size:
                print "Too few messages for ", key
                del segments2docs[key]

    # For each segment, get count distribution over terms:
    dists = {}
    for key, docs in segments2docs.iteritems():
        dists[key] = get_term_count_distribution(docs, vocabsize=vocabsize, liwc_map=liwc_map)

    # Now measure pairwise distances:
    distances = {}
    segments = sorted(segments2docs.keys())
    for i, seg1 in enumerate(segments):
        for seg2 in segments[i+1:]:
            distances[(seg1, seg2)] = jensen_shannon(dists[seg1], dists[seg2])
    return distances



######################################################################
# General utilities        


def get_term_count_distribution(messages, vocabsize=1000, liwc_map=False):
    # Flatten to a single list of words:
    words = [w for msg in messages for w in tokenize(msg['text'])]
    if liwc_map:
        # Perform the LIWC transformation:
        words = collapse_by_liwc(words)
    # Create count dictionary:
    countdict = Counter(words)
    # Vocab size restriction based on frequency:
    countdict = dict(sorted(countdict.items(), key=itemgetter(1), reverse=True)[ : min(len(countdict), vocabsize)])
    # Distribution:
    dist = counts2dist(countdict)
    return dist

def tokenize(text):
    words = TOKENIZER.tokenize(text)
    return words

def collapse_by_liwc(words):
    cats = []
    for cat, regex in LIWC.items():
        cats += [cat for w in words if regex.search(w)]
    return cats

def counts2dist(countdict):
    total = float(sum(countdict.values()))
    return {key:val/total for key, val in countdict.iteritems()}

def jensen_shannon(f, g):
    vocab = sorted(set(f.keys()) | set(g.keys()))
    p = np.zeros(len(vocab))
    q = np.zeros(len(vocab))
    for i, w in enumerate(vocab):
        p[i] = f.get(w, 0.0)
        q[i] = g.get(w, 0.0)
    pq = (p + q) / 2.0                
    a = 0.5 * kl(p, pq)
    b = 0.5 * kl(q, pq)
    return np.sqrt(a + b)

def kl(p, q):
    return np.sum(p * safelog2(p/q))

def safelog2(x):
    with np.errstate(divide='ignore'):
        x = np.log2(x)
        x[np.isinf(x)] = 0.0
        return x


######################################################################



