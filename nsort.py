from __future__ import division

from collections import (
    Counter,
    defaultdict as deft
)

from copy import deepcopy as cp

from itertools import permutations

from random import (
    sample,
    shuffle
)

from TermIndex import TermIndex

from tools import (
    average,
    cat_dedup,
    e,
    interset,
    tokenize,
    union_set
)

from tqdm import tqdm



#    [1]    3x the baseline

class nsort:

    def __init__(
        self,
        min_cluster=3,
        min_wfreq=3,
        max_wfreq=0.2,
        baselines=10,
        window=10,
        novel_ratio=0.75,
        doc_overlap=0.25,
        min_assoc_multiplier=1,            # [2]  ToDo: seems useless, look into it.
        verbose=False,
        #unigram_fallback='most_freq/near_third'
    ):
        self.min_cluster = min_cluster
        self.min_wfreq = min_wfreq
        self.max_wfreq = max_wfreq
        self.baselines = baselines
        self.min_assoc = min_assoc_multiplier
        self.novel_ratio = novel_ratio
        self.doc_overlap = doc_overlap
        self.window = window
        self.bows = []
        self.window_by_wid = deft(set)
        self.is_feature = deft(bool)
        self.wfreq = Counter()
        self.ti = TermIndex('')
        self.bows_by_wid = deft(list)
        self.wids_by_bow = deft(list)
        self.assignments = deft(list)
        self.bowid_by_docid = dict([])
        self.verbose = verbose
        if self.verbose: print self


    def clear(self):
        self.bows = []
        self.window_by_wid = deft(set)
        self.is_feature = deft(bool)
        self.wfreq = Counter()
        self.ti = TermIndex('')
        self.bows_by_wid = deft(list)
        self.wids_by_bow = deft(list)
        self.assignments = deft(list)
        self.bowid_by_docid = dict([])


    def __str__(self):
        return """<%s with parameters <
    min_cluster=%d
    baselines=%d
    min_assoc=%d
>>""" % ('nsort.nsort instance', self.min_cluster, self.baselines, self.min_assoc)


    def __hapax(self, rewritten):
        if self.verbose: print 'counting words and searching for low frequency words...'

        if self.verbose:
            gtor = tqdm(rewritten)
        else:
            gtor = rewritten

        for bowid, doc in gtor:
            self.wfreq.update(set(doc))
        if isinstance(self.max_wfreq, float):
            t = len(rewritten) * self.max_wfreq
        else:
            t = self.max_wfreq
        for w, f in self.wfreq.items():
            if f >= self.min_wfreq and f < t:
                self.is_feature[w] = True


    def __call__(self, documents):
        rewritten = self.word2int(documents)
        hapax = self.__hapax(rewritten)
        self.make_index(rewritten)
        return self.cluster(documents)

    
    def word2int(self, documents):
        """Create an index of all the words in the input documents. Each word's in-
dex will be a unique integer denoting the iteration number when it was first 
encountered as the call iterated through the sequence of every token in the do-
cuments. Then, transform the input documents from lists of strings to lists of integers."""
        rewritten = []
        for bowid, doc in documents:
            rwrtn = [self.ti(w) for w in doc]
            rewritten.append((bowid, rwrtn))
        return rewritten


    def make_index(self, rewritten):

        if self.verbose:
            gtor = tqdm(rewritten)
        else:
            gtor = rewritten

        for docid, (bowid, doc) in enumerate(gtor):
            self.bowid_by_docid[docid] = bowid
            self.__window_update(doc)
            bow = Counter([w for w in doc if self.is_feature[w]]).items()
            self.bows.append(bow)
            for wid, freq in bow:
                self.bows_by_wid[wid].append(docid)
                self.wids_by_bow[bowid].append(wid)
        self.__compile()
    
    
    def __window_update(self, doc):
        for i, wid in enumerate(doc):
            window = self.__window(i, doc)
            self.window_by_wid[wid].update(window)


    def __window(self, i, doc):
        j = i + 1
        minim = i - self.window
        maxim = j + self.window
        if minim < 0:
            minim = 0
        if maxim > len(doc):
            maxim = len(doc)
        return doc[minim:i] + doc[j:maxim]

    
    def __compile(self):
        for wid, bowids in self.bows_by_wid.items():
            self.bows_by_wid[wid] = set(bowids)
        for bowid, wids in self.wids_by_bow.items():
            self.wids_by_bow[bowid] = set(wids)
    
    
    def cluster(self, documents):
        space = [
            (tuple(sorted([wid])), set([wid]), cp(bowids))
            for wid, bowids in self.bows_by_wid.items()
        ]
        while True:
            _space = []
            assocs = []
            taken = set([])
            
            if self.verbose:
                gtor = tqdm(space)
            else:
                gtor = space
            
            for wid, cluster, subspace in gtor:
                content_words = set([i for i in cluster if self.is_feature[i]])
                if content_words < cluster:
                    continue
                candidates = interset([self.window_by_wid[i] for i in cluster])
                
                for _wid in candidates:
                    if _wid in wid:
                        continue
                    prev = cp(cluster)
                    prev.add(_wid)
                    bowids = self.bows_by_wid[_wid]
                    common = subspace.intersection(bowids)
                    if not common:
                        continue
                    ratio = len(common) / len(bowids)
                    if len(common) < self.min_cluster:
                        continue
                    elif ratio < self.doc_overlap:    # parameter
                        continue

                    relation = (
                        tuple(sorted(prev)),
                        prev,
                        common
                    )
                    assocs.append((ratio, relation))
                    #_space.append(relation)
            
            assocs.sort(reverse=True, key=lambda x: x[0] * len(x[1][2]))
            for ratio, relation in assocs:
                key, cluster, subspace = relation
                new = subspace - taken

                if len(new) / len(subspace) < self.novel_ratio:    # parameter
                    continue
                taken.update(subspace)
                if self.verbose: print '\t+', ratio, tuple([self.ti[i] for i in key])
                _space.append((ratio, relation))

            space = self.__dissoc_best_relations_above_thresh(_space)
            self.__assignments(space)
            if self.verbose: print 'went in %d, came out %d\n\n' % (
                len(_space), len(space)
            )
            if not _space:
                break
        return self.__sort()
    
    
    def __assignments(self, space):
        if not space:
            return
        for key, cluster, subspace in space:
            for bowid in subspace:
                self.assignments[bowid].append(key)
    
    
    def __sort(self):
        out = []
        scores = deft(set)
        for key, value in self.assignments.items():
            scores[value[0]].add(key)
            feature_set = tuple(
                [self.ti[feature] for feature in cat_dedup(tuple(value))]
            )
            triple = (feature_set, key, value)
            out.append(triple)
        if not scores.values():
            upperbound = 0
        else:
            upperbound = max([len(x) for x in scores.values()])
        out.sort(key=lambda x: (upperbound - len(scores[x[-1][0]]), x[0]))
        remainder = self.__get_remainder(set(self.assignments.keys()))
        return [(len(scores[z[0]]), x, self.bowid_by_docid[y]) for x, y, z in out] +\
                remainder
    
    
    def __get_remainder(self, covered):
        domain = set(range(len(self.bows)))
        diff = domain - covered
        remainder = []
        for i in diff:
            candidate_features = sorted(
                [(self.wfreq[wid], wid)
                for wid, f in self.bows[i] if self.is_feature[wid]],
                reverse=True,
                key=lambda x: x[0]
            )
            if not candidate_features:
                feature_set = tuple(['None'])
            else:
                feature_set = tuple([self.ti[candidate_features[-1][1]]])
            triple = (feature_set, self.bowid_by_docid[i])
            remainder.append(triple)
        freqDist = Counter(zip(*remainder)[0])
        remainder = [(freqDist[x], x, y) for x, y in remainder]
        return remainder


    def __dissoc_best_relations_above_thresh(self, space):
        baseline = self.__baseline(space)
        return [
            relation
            for ratio, relation in space
            if ratio > (baseline * self.min_assoc)
        ]
    
    
    def __baseline(self, space):
        if len(space) < self.baselines:
            k = len(space)
        else:
            k = self.baselines
        samples = sample(space, k)
        baselines = [ratio for ratio, _ in samples]
        return average(baselines)




class nSort:

    def __init__(
        self,
# 		min_cluster=2,
# 		min_wfreq=1,
# 		window=5,
# 		max_wfreq=400,
# 		baselines=20,
# 		min_assoc_multiplier=1,
# 		novel_ratio=0.2,
# 		doc_overlap=0.1
		min_cluster=3,
		min_wfreq=3,
		window=20,
		max_wfreq=0.2,
		baselines=20,
		min_assoc_multiplier=1,
		novel_ratio=0.2,
		doc_overlap=0.1,
		verbose=False,
    ):
        self.ns = nsort(
            min_cluster=min_cluster,
            min_wfreq=min_wfreq,
            window=window,
            max_wfreq=max_wfreq,
            baselines=baselines,
            min_assoc_multiplier=min_assoc_multiplier,
            novel_ratio=novel_ratio,
            doc_overlap=doc_overlap,
            verbose=verbose
        )
        self.verbose = verbose


    def __call__(self, documents):
        out = []
        space = [(i, tokenize(doc)) for i, doc in enumerate(documents)]
        if self.verbose: print 'in', len(documents)
        while True:
            clusters = self.ns(space)
            positives, negatives = self.__unzip(documents, clusters)
            if self.verbose: print 'at', len(positives), len(negatives)
#             out = positives + negatives
#             break
            if not positives:
                out += negatives
                break
            else:
                self.ns.clear()
                out += positives
                if negatives:
                    a, b, c, d = zip(*negatives)
                    space = zip(c, [tokenize(text) for text in d])
        if self.verbose: print 'out', len(out)
        upperbound = max([x for x, _, _, _ in out])
        self.data = sorted(out, key=lambda x: (upperbound - x[0], x[1]))
    
    
    def __unzip(self, documents, clusters):
        positives, negatives = [], []
        for cluster in clusters:
            cluster_size, feature_set, doc_id = cluster
            _cluster = (cluster_size, feature_set, doc_id, documents[doc_id])
            if cluster_size > self.ns.min_cluster:
                positives.append(_cluster)
            else:
                negatives.append(_cluster)
        return positives, negatives

    
    def __iter__(self):
        for count, feature_set, bowid, text in self.data:
            yield feature_set, text
    
    
    def texts(self):
        for features, text in self:
            yield text
    
    
    def features(self):
        for features, text in self:
            yield features
    
    
    def tuples(self):
        for features, text in self:
            yield features, text
            

