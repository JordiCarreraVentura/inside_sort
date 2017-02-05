
import random

import sys

from random import (
    choice as choose,
    random as randfloat,
    randint,
    shuffle
)


a = 'a'
b = 'b'
c = 'c'
d = 'd'
p = 'p'
q = 'q'
r = 'r'
x = 'x'
y = 'y'
z = 'z'
u = 'u'
v = 'v'

A = [(a, 0.4), (b, 0.3), (c, 0.2), (d, 0.1)]
P = [(p, 0.55), (q, 0.4), (r, 0.05)]
X = [(x, 0.4), (y, 0.35), (z, 0.15), (u, 0.05), (v, 0.05)]
APX = [(A, 0.3), (P, 0.2), (X, 0.5)]
LENS = [(2, 0.2), (3, 0.4), (4, 0.25), (5, 0.1), (6, 0.04), (7, 0.01)]


def sample_one(items):
    roll = randfloat()
    curr = 0.0
    for (item, prob) in items:
        #print roll, curr, '\t', prob
        curr += prob
        if roll <= curr: return item
    return item


if __name__ == '__main__':
    
    n = int(sys.argv[1])
    test_set = 'test_set.txt'
    samples = []
    while len(samples) < n:
        g = sample_one(APX)
        l = sample_one(LENS)
        sample = []
        while len(sample) < l:
            item = sample_one(g)
            sample.append(item)
        sample += [randint(1, 50000) for i in range(l * 2)]
        shuffle(sample)
        text = ' '.join([str(i) for i in sample])
        #print text
        samples.append(text)
    
    text = '\n'.join(samples)
    with open(test_set, 'wb') as wrt:
        wrt.write(text)
    
