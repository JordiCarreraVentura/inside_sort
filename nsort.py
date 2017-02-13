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
    format_min_wfreq,
    interset,
    tokenize,
    union_set
)

from tqdm import tqdm

import sys


class nsort:
    """
Main inside-sorting class. The intended meaning of "inside sorting"
is a process by which records (strings of textual data, by
assumption) are sorted not alphabetically but rather by the words
they contain and, more specifically, significant word co-occurrence
patterns (often reflecting semantic relationships, but not
exclusively).


"""
    def __init__(
        self,
        min_cluster=3,
        min_wfreq=3,
        max_wfreq=0.2,
        baselines=10,
        window=10,
        novel_ratio=0.75,
        doc_overlap=0.25,
        min_assoc_multiplier=1,
        verbose=False,
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
        self.log = sys.stderr
        if self.verbose: self.log.write('%s\n' % str(self))


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
    max_wfreq=%f
    min_wfreq=%d
    min_cluster=%d
    baselines=%d
    min_assoc=%d
>>""" % ('nsort.nsort instance', self.max_wfreq, self.min_wfreq, self.min_cluster, self.baselines, self.min_assoc)


    def __hapax(self, rewritten):
        if self.verbose: self.log.write(
            'counting words and searching for low frequency words...\n'
        )

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
        """
The call to the nsort class takes as input a list of tokenized
documents (i.e., a list of strings, where each string is intended as
a word and the list should generally be viewed as a full sentence or
document).

If called from within the nSort wrapper, tokenization will normally
have  taken place within the latter by the time the call is made.

The output is the list of clusters into which the incoming documents
have been grouped. More specifically, the output consists of a list
where  each item is a triple with the form <x,y,z> and where each
element in the triple is defined as follows:

	a    int             Number of documents in a cluster.

	b    tuple:string    Features that triggered a cluster.

	c    int             An index to an input document. Over the
	                     the entire list, this amounts to the indexes
	                     of all the input documents.

The list is sorted by 'a' in descending order (larger clusters first)
and then by 'c' in ascending order (features in alphabetical order).

The call to the class goes through two normalization steps, one
indexation step, and lastly the clustering loop.


NORMALIZATION

In the first normalization step, input documents are transformed into
lists of integers. The integers are biunivocally mapped to words,
i.e., each integer is an identifier that references some word and
that word only. The mappings are stored in a TermIndex instance and
can be recovered at any point; likewise, translating from words into
integers or integers into words can be done as a simple lookup
operation. During this pass, frequency counts are also calculated
and they are kept for word indexes.


DIMENSIONALITY REDUCTION

Secondly, the frequency counts are used to reduce the dimensionality
of the input. Any words occurring less than 'min_wfreq' times or
more than 'max_wfreq' times (which can be set to either an integer
denoting a frequency or a floating-point number denoting a ratio
over the total number of documents) are removed at this point and do
not undergo further processing.


INDEXATION

Thirdly, the output from the previous step (i.e., lists of integers
with reduced dimensionality) is indexed, with instances being mapped
to the words/integers they contain, so that, e.g., given any word and
the set of document instances it appears in, any co-occurring words
across all documents can be easily retrieved.


CLUSTERING

Lastly, the output from the previous step is passed as input for the
clustering process. During clustering, the system iteratively merges
any pair of smaller clusters sharing some word if that word  appears
in a 'doc_overlap' ratio of the total number of documents currently
in the cluster (so, each cluster is increasingly sub-divided into the
 largest possible cluster as determined by the next highest-
frequency word available within the cluster).

As an additional condition, the words triggering a cluster must co-
occur with the potential candidate for sub-dividing that cluster. The
co-occurrence must happen within a window 'window' of words around
each other.

However, not every word meeting the previous conditions is used yet
for subdividing a cluster. Instead, they are kept as candidates and
sorted by the strength of their 'doc_overlap' association to the
cluster (in descending order, i.e., stronger associations first).
Candidates must also significantly outperform a number 'baselines' of
baselines according to this score (i.e., the 'doc_overlap' for any
candidate must be higher than for all of a 'baselines' number of
words taken at random).

The system then iterates over the sorted list and keeps the best
candidate as long as it selects documents not previously selected by
some higher-scoring candidate. In particular, a ratio 'novel_ratio'
of all the documents selected by a new candidate cannot have been
selected previously by any other candidate for the current one to be
accepted.

The process continues until there are no features left that select
a cluster with a cardinality of at least 'min_cluster' items.
"""
        rewritten = self.word2int(documents)
        hapax = self.__hapax(rewritten)
        self.make_index(rewritten)
        return self.cluster(documents)


    def word2int(self, documents):
        """
Create an index of all the words in the input documents. Each word's
index will be a unique integer denoting the iteration number when it
was first encountered as the call iterated through the sequence of
every token in the documents. Then, transform the input documents
from lists of strings to lists of integers."""
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
                if self.verbose: self.log.write(
                    '\t+%f %s\n' % (ratio, ' | '.join(tuple([self.ti[i] for i in key])))
                )
                _space.append((ratio, relation))

            space = self.__dissoc_best_relations_above_thresh(_space)
            self.__assignments(space)
            if self.verbose: self.log.write('went in %d, came out %d\n\n' % (
                len(_space), len(space)
            ))
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
        if remainder:
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
    """
Wrapper over the nsort class for iterative sorting. Each iteration's
output is passed as input for the next iteration. In this way, the
sorting follows a cascade process whereby previous clusters are
further subdivided and previously undetected clusters become
increasingly clearer.

The nSort class takes the same parameters as the nsort class. The
loop stops when no more clusters can be found with the sorting
parameters specified.

For a detailed description of the parameters, run:

'python inside_sort.py --help'."""
    def __init__(
        self,
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
        max_wfreq = format_min_wfreq(max_wfreq)
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


    def __str__(self):
        return str(self.ns)


    def __call__(self, documents):
        """
Applies inside-sorting to the items stored in the variable
'documents'. The variable is assumed to point to a Python iterator
and the value returned with each iteration is assumed to be a string
denoting an input text of varying length but does not have to be
necessarily limited to strings.
"""
        out = []
        space = [(i, tokenize(doc)) for i, doc in enumerate(documents)]
        if self.verbose: self.ns.log.write('in %d\n' % len(documents))
        while True:
            clusters = self.ns(space)
            positives, negatives = self.__unzip(documents, clusters)
            if self.verbose: self.ns.log.write(
                'at %d %d\n' % (len(positives), len(negatives))
            )
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
                else:
                    space = []
        if self.verbose: self.ns.log.write('out %d\n' % len(out))
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
        """
Allows to iterate over the sorted records. In each iteration, a tuple
of the form <a, b> is returned, where 'a' denotes the feature that
triggered a cluster and 'b' denotes an item of the current cluster.

The elements of each cluster are presented sequentially (i.e., there
is no output such that a record contains all the members of a
cluster. However, 'b' elements can be grouped by their correlated 'a'
elements; in that case, the full list of 'b's associated with each
'a' now does represent all the members of the cluster defined by 'a')
"""
        for count, feature_set, bowid, text in self.data:
            yield feature_set, text


    def texts(self):
        """Returns the 'b' element from the class iterator method."""
        for features, text in self:
            yield text


    def features(self):
        """Returns the 'a' element from the class iterator method."""
        for features, text in self:
            yield features


    def tuples(self):
        """Synonym of the class iterator method."""
        for features, text in self:
            yield features, text


