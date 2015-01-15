import cPickle as pickle
import re
import glob
import sys
from operator import itemgetter
from collections import defaultdict, Counter
import numpy as np
import random
import corpcorp
import logging

LIWC = pickle.load(file('LIWC2007dictionary_regex.pickle'))

######################################################################
# Month-level by-user analysis

def monthly_user_distances(filenames, min_monthly_messages=20):
    """This is the main function for doing analysis by month.

    -- filenames: a list of the names of the pickle files
       representing individual users

    -- min_monthly_messages: a threshold that can be set to avoid
       issues relating to sparsity

    The function returns a dictionary mapping each filename to
    a dictionary mapping (YYYY, MM) pairs to distance values.
    """    
    all_dists = {}
    for i, filename in enumerate(filenames):
        try:
            print '%d of %d: %s' % (i + 1, len(filenames), filename)
            member = corpcorp.Member(filename)
            dists = member2monthly_dists(member, min_monthly_messages=min_monthly_messages)
            # Can print results to the screen -- if something goes wrong, they
            # can be copied and pasted from stdout
            # print member.username, dists 
            all_dists[member.filename] = dists
        except Exception:
            logging.exception('Exception processing %s' % filename)
    return all_dists

def member2monthly_dists(member, min_monthly_messages=20, liwc_map=True):
    """The workhorse for the monthly analysis.

    -- member: a corpcorp.Member instance

    -- min_monthly_messages: a threshold that can be set to avoid
       issues relating to sparsity

    -- liwc_map: True means map each word to its LIWC category or categories
       if there are any, else ignore that word; False means no remapping
    
    The result is a dictionary mapping (YYYY, MM) pairs to distance values.
    """
    # First, we get the words for the members and for their interlocutors:    
    fromto_messages = {'member_counts': defaultdict(list), 'other_counts': defaultdict(list)}
    for i, msg in enumerate(member.messages):
        sys.stderr.write('\r') ; sys.stderr.write('msg %s' % i) ; sys.stderr.flush()
        if msg.frm and msg.date:
            key = 'member_counts' if member.username in msg.frm else 'other_counts'
            date = (msg.date.year, msg.date.month)
            fromto_messages[key][date].append(msg)
    sys.stderr.write('\n')
    # Now convert the word lists into count dictionaries, then probability distributions,
    # and then perform the distance analysis:
    distances = {}
    for date, member_messages in sorted(fromto_messages['member_counts'].items()):
        sys.stderr.write('\r') ; sys.stderr.write('date: %s-%s' % date) ; sys.stderr.flush()
        if len(member_messages) >= min_monthly_messages:
            other_messages = fromto_messages['other_counts'][date]
            if len(other_messages) >= min_monthly_messages:
                # Member's wordlist and then distribution:
                member_words = tokenizer([w for msg in member_messages for w in msg.words()], liwc_map=liwc_map)
                member_dist = counts2dist(Counter(member_words))
                # Interlocutors' single word list and then distribution:
                other_words = tokenizer([w for msg in other_messages for w in msg.words()], liwc_map=liwc_map)
                other_dist = counts2dist(Counter(other_words))
                # Distance:
                distances[date] = jensen_shannon(member_dist, other_dist)
    sys.stderr.write('\n')
    return distances

######################################################################
# User-level analysis (no temporal structure)

member_counts_key = 'member_counts'
other_counts_key = 'other_counts'

def user_distances(filenames, sampsize=1000, sampling=False, liwc_map=True, vocabsize=1000, employee_classifier_func=(lambda x : [other_counts_key])):
    """Main function for doing distances without paying attention to temporal structure:

    -- filenames: should be a list of the names of the pickle files
       representing individual users

    -- sampsize: number of interlocutors to sample

    -- sampling: whether to sample interlocutors (False means include all of them)

    -- liwc_map=True means map each word to its LIWC category or categories
       if there are any, else ignore that word

    -- vocabsize: pay attention to only this many vocab items, ordered by frequency; 0 means use all vocab

    -- employee_classifier_func: optional function for mapping interlocutors to groups based on some external criteria

    The return value is a dictionary mapping each filename to a dictionary mapping each group name to its distance from the
    individual represented by the filename.
    """
    all_user_distances = {}
    for filename in filenames:
        member = corpcorp.Member(filename)
        distances = member2distributions(member, sampsize=sampsize, sampling=sampling, liwc_map=liwc_map, vocabsize=vocabsize, employee_classifier_func=employee_classifier_func)
        if distances:
            all_user_distances[member.filename] = distances
            print member.username, distances
    return all_user_distances

def member2distributions(member, sampsize=1000, sampling=False, liwc_map=True, vocabsize=1000, employee_classifier_func=(lambda x : [other_counts_key])):
    """Workhorse function for the user-level non-temporal analysis:

    member: a corpcorp.Member instance

    all the other keyword parameters are set by  user_distances

    The return value is a dictionary mapping each group name to its distance from the
    individual represented by member.
    """    
    fromto_messages = defaultdict(list)
    for i, msg in enumerate(member.messages):
        sys.stderr.write('\r') ; sys.stderr.write('msg %s' % i) ; sys.stderr.flush()
        keys = []
        if msg.frm:
            if member.username in msg.frm:
                keys = [member_counts_key]
            else:
                keys = employee_classifier_func(msg.frm)
            for key in keys:
                fromto_messages[key].append(msg)
            ##### if i >= 200: break ## for debugging
    sys.stderr.write('\n')
    if sampling:
        for key, messages in fromto_messages.items():
            # Don't keep anyone who has too few messages:
            if len(messages) < sampsize:
                print "Too few messages for" % member.filename
                return None
            random.shuffle(messages)
            fromto_messages[key] = messages[ : sampsize]        
    dists = {}
    for key, messages in fromto_messages.items():
        # Flatten to a single list of words:
        words = [w for msg in messages for w in msg.words()]
        # Perform the LIWC transformation and create count dictionary:
        countdict = Counter(tokenizer(words, liwc_map=liwc_map))
        # Vocab size restriction based on frequency:
        countdict = dict(sorted(countdict.items(), key=itemgetter(1), reverse=True)[ : vocabsize])
        # Distribution:
        dists[key] = counts2dist(countdict)
    distances = {}
    for key, val in dists.items():
        if key != member_counts_key:
            distances[key] = jensen_shannon(dists[member_counts_key], val)
    return distances

######################################################################
# General utilities        

def tokenizer(words, liwc_map=True):
    if not liwc_map:
        return words
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

def email2groups(s):    
    for key, groups in email2group_dict.iteritems():
        if key in s.lower():
            return groups
    # Last resort is an "other" classification:
    return [other_counts_key]
    
