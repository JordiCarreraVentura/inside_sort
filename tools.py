import nltk
import re

from nltk import sent_tokenize as splitter


WORDS = re.compile('[a-z]+', re.IGNORECASE)
FLOAT = re.compile('^(1\.0|0\.[0-9]+)$')

def tokenize(string):
    return [
        word.lower() for word in WORDS.findall(string)
        if word and word.isalpha()
    ]


def average(values):
    if not values:
        return 0.0
    return sum(values) / len(values)
    
        
def e(string):
    try:
        string.encode('utf-8')
    except Exception:
       return string


def groups(clusters):
    _groups = deft(list)
    for key, bowid, doc in clusters:
        _groups[key].append((bowid, doc))
    return _groups


def union_set(items):
    u = set([])
    for item in items:
        u.update(item)
    return u


def interset(items):
    if not items:
        return set([])
    inter = items[0]
    for item in items[1:]:
        inter = inter.intersection(item)
    return inter


def cat_dedup(keys):
    if not keys:
        return tuple([])
    deduped = list(keys[0])
    for key in keys[1:]:
        deduped += [item for item in key if item not in deduped]
        #deduped += [item for item in key]
    return tuple(deduped)


def encode(string):
    try:
        return string.encode('utf-8')
    except Exception:
        return string


def format_min_wfreq(freq_par):
    if FLOAT.match(freq_par):
        return float(freq_par)
    else:
        return int(freq_par)
