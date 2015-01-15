import os
import datetime
import re
import cPickle as pickle
import json
import glob
from collections import defaultdict
import dateutil.parser
import sys
from tokenizers import TwitterTokenizer as Tokenizer

TOKENIZER = Tokenizer(preserve_case=False,
                      preserve_all_caps=False,
                      filter_html_tags=True,
                      filter_twitter_usernames=True, 
                      filter_twitter_hashtags=True,
                      filter_urls=True,
                      filter_dates=True,
                      normalize_dates=True,
                      mark_negation_scope=False)

class CorpCorpCorpus:
    def __init__(self, filenames):
        self.filenames = filenames

    def iter_members(self, display=False):
        for i, filename in enumerate(self.filenames): 
            if display:
                 print (i+1), filename
            yield Member(filename)

    def iter_messages(self, display=False):
        for mem in self.iter_members(display=display):
            for msg in mem.messages:
                yield msg

    def __len__(self):
        return len(self.filenames)


def json_lines_load_streaming(infile):
    return (json.loads(line.strip()) for line in infile)


class Member:
    def __init__(self, filename):
        self.filename = filename
        self.username = os.path.basename(filename).replace(".p", "").replace(".jsons", "")
        # Will M 2014-11-24: support jsons as well as pickles
        with open(self.filename, 'rb') as infile:
            if self.filename.endswith('.jsons'):
                msgs = list(json_lines_load_streaming(infile))
            else:
                msgs = pickle.load(infile)
        self.messages = [Message(m) for m in msgs]

    def __len__(self):
        return len(self.messages)

                                
class Message:
    """
    Initialization sets these attributes:
    
    'body', 'x-gmail-labels', 'delivered-to', 'from',
    'sender', 'cc', 'bcc', 'to', 'references',
    'in-reply-to', 'date', 'reply-to',
    'message-id', 'importance', 'subject'
    """    
    def __init__(self, m):
        for key, val in m.items():
            key = key.replace("-", "_")
            if key == 'date':
                try:
                    val = dateutil.parser.parse(val)
                except:
                    val = None
            elif key in ('to', 'cc', 'bcc') and val:
                val = [x.strip(',').strip() for x in val.splitlines()]
            elif key == 'from':
                key = 'frm'
            setattr(self, key, val)
            
    def words(self):
        w = []
        if self.body:
            w = TOKENIZER.tokenize(self.body)
        return w
