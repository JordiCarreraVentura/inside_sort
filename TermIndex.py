from collections import (
    defaultdict as deft
)

# from Tools import to_csv as _to_csv


class TermIndex:
    
    def __init__(self, name):
        self.name = name
        self.seen = deft(bool)
        self.words = dict([])
        self.ids = dict([])
        self.n = 0
    
    def drop(self, _id):
        word = self[_id]
        del self.seen[word]
        del self.ids[word]

    def __call__(self, item):
        seen = self.seen[item]
        if not seen:
            _id = self.n
            self.ids[item] = _id
            self.words[_id] = item
            self.seen[item] = True
            self.n += 1
            return _id
        else:
            return self.ids[item]
    
    def known(self, w):
        return self.seen[w]
    
    def __getitem__(self, _id):
        if _id < self.n:
            return self.words[_id]
        else:
            return None
    
    def __iter__(self):
        for w, boolean in self.seen.items():
            if boolean:
                yield w
# 
#     def to_csv(self):
#         path = '%s.csv' % self.name
#         _to_csv(sorted(self.ids.items(), key=lambda x: x[1]), path)
