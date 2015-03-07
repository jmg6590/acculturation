import os
import sys
import datetime
import re
import cPickle as pickle
import json
import glob
from collections import defaultdict
import dateutil.parser

"""
Objects for wrangling data in the 
corpcorp json email format
"""

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


class Member:

    def __init__(self, filename):
        self.filename = filename
        self.username = os.path.basename(filename).replace(".p", "").replace(".jsons", "").replace(".json", "")
        self.username = self.username[:self.username.rfind("_")]
        with open(self.filename, 'rb') as infile:
            if self.filename.endswith('.jsons') or self.filename.endswith(".json"):
                msgs = json.loads(infile.read())
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
            
    def __getitem__(self, key):
        # Want to let users index into Message
        # to access its text
        if key == "text":
            if self.body:
                return self.body
        if hasattr(self, key):
            return getattr(self, key)
        return False

