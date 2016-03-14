# -*- encoding: utf-8 -*-
import math
import re
import sys

from collections import (
    Counter,
    defaultdict as deft
)

from copy import deepcopy as cp

from Tests import TOY as examples


LETTER = re.compile('[a-záéíóúñ ]'.decode('utf-8'), re.IGNORECASE)
BLANKS = re.compile(' {2,}')


def mdn(values):
    if not values:
        return 0
    else:
        values.sort()
        l = len(values)
        return values[int(l / 2)]


def remove_punct(string):
    new = ''
    for char in string:
        if LETTER.match(char):
            new += char
    return BLANKS.sub(' ', new)


class InsideSort:

    def __init__(
        self,
        window=6,
#         min_part_size=1
        min_part_size=10
    ):
        self.min_part_size = min_part_size
        self.window = window
        self.original = []
        self.inverted_index = deft(set)

    def __set(self, tokenized):
        self.original = tokenized
        for i, instance in enumerate(tokenized):
            self.inverted_index[tuple(instance)].add(i)

    def __call__(self, tokenized):
        self.__set(tokenized)
        partitions = [(True, [], cp(tokenized))]
        while True:
            _partitions = self.__partition(partitions)
            if len(partitions) == len(_partitions):
                partitions = _partitions
                break
            else:
                partitions = _partitions
        return self.__serialize(partitions)

    def __serialize(self, partitions):
        rows = []
        to_cover = set(range(len(self.original)))
        covered = set([])
        for i, (state, history, p) in enumerate(partitions):
            for instance in p:
                covered.update(self.inverted_index[tuple([w.lower() for w in instance])])
                rows.append((i, history, instance))

        for j in to_cover - covered:
            rows.append((i + 1, ['*'], self.original[j]))

        return rows

    def __partition(self, partitions):
        new = []
        for state, history, p in partitions:
            if state:
                new += self.__partition_one(history, p)
            else:
                new.append((state, history, p))
        return new

    def __partition_one(self, history, p):
        centers, index, freq = self.__find_centers(p)
        self.__to_sets(index)
        gravities = self.__to_gravity(centers, freq)
        parts = self.__sort(history, p, index, gravities)
        return parts

    def __sort(self, history, p, index, gravities):
        corr = self.__assign_all(p, index, gravities)
        self.__auto_sort(corr)
        tagged = zip(p, corr)
        return self.__grouped(history, tagged)

    def __grouped(self, history, tagged):

        groups = deft(list)
        for tokens, tag in tagged:
            _tokens = []
            for w in tokens:
                if w == tag:
                    _tokens.append('%s' % w.upper())
                else:
                    _tokens.append(w)
            groups[tag].append(_tokens)

        for tag, group in groups.items():
            if len(group) < self.min_part_size:
                groups[None] += group
                del groups[tag]

        grouped = []
        for tag, group in groups.items():
            tree = cp(history)

            if tag:
                tree.append(tag)

            l = len(group)
            if l == self.min_part_size and tag:
                grouped.append((False, tree, group))
            elif l == self.min_part_size and not tag:
                grouped.append((False, cp(history), group))
            elif tag and l > self.min_part_size:
                grouped.append((True, tree, group))
            elif not tag and l > self.min_part_size:
                grouped.append((True, cp(history) + ['*'], group))

        return grouped


    def __auto_sort(self, corr):
        unique_index = dict([])
        unique_dist = Counter()
        self.__sort_non_unique(unique_index, unique_dist, corr)
        self.__sort_unique(unique_index, unique_dist, corr)

    def __sort_non_unique(self, unique_index, unique_dist, corr):
        for i, seq in enumerate(corr):
            #
            if not seq:
                corr[i] = None
                continue

            unique, non_unique = self.__dissociate(seq)
            #
            if non_unique:
                corr[i] = sorted(
                    non_unique,
                    key=lambda x: x[1],
                    reverse=True
                )[0][0]
            elif unique:
                unique_dist.update(zip(*unique)[0])
                unique_index[i] = unique
            else:
                corr[i] = None

    def __sort_unique(self, unique_index, unique_dist, corr):
        for i, unique in unique_index.items():
            corr[i] = sorted(
                unique,
                key=lambda x: unique_dist[x[0]],
                reverse=True
            )[0][0]

    def __dissociate(self, seq):
        non_unique = []
        unique = []
        for w, f in seq:
            if f == 1:
                unique.append((w, f))
            else:
                non_unique.append((w, f))
        return unique, non_unique

    def __assign_all(self, p, index, gravities):
        corr = [[] for x in p]
        to_cover = set(range(len(p)))
        covered = set([])
        for i, (w, grav, ww) in enumerate(gravities):
            idist = self.__recall(index, ww)
            if len(gravities) >= 100 and not i % 100:
                print i, w, len(gravities), round(i / float(len(gravities)), 2)
            for j, freq in idist:
                corr[j].append((w, freq))
                covered.add(j)
        for missing in to_cover - covered:
            corr[missing].append((None, 0))
        return corr

    def __recall(self, index, ww):
        recalled = Counter()
        for w in ww:
            recalled.update(index[w])
        return recalled.items()

    def __find_centers(self, p):
        centers = deft(Counter)
        index = deft(list)
        freq = deft(float)
        for i, instance in enumerate(p):
            for j, w in enumerate(instance):
                if w.isupper():
                    continue
                index[w].append(i)
                ctxt = self.__frame(j, instance)
                centers[w].update(ctxt)
                freq[w] += 1

        for w, f in freq.items():
            if f >= 200:
                del centers[w]

        return centers, index, freq

    def __frame(self, j, instance):
        start = j - self.window
        end = j + 1 + self.window
        if start < 0:
            start = 0
        if end > len(instance):
            end = len(instance)
        ctxt = [_w for _w in instance[start:j] + instance[j + 1:end]
                if not _w.isupper()]
        return ctxt

    def __to_sets(self, index):
        for w, ii in index.items():
            index[w] = set(ii)

    def __to_gravity(self, centers, freq):
        gravities = dict([])
        mass = float(sum(freq.values()))
        for w, coocs in centers.items():
            codist = coocs.most_common()
            if not codist:
                continue
            most_freq = codist[0][1]
#             topdist = [(w, f) for w, f in codist if f >= (most_freq * 0.1)]
            topdist = [(w, f) for w, f in codist if f >= (most_freq * 0.8)]
            fxy_mass = freq[w]
            if fxy_mass < self.min_part_size or not topdist:
                continue
            gravities[w] = mdn(
                [(f / fxy_mass) * math.log((f / fxy_mass) / (freq[_w] / mass))
                 for _w, f in topdist]
            ) * sum(zip(*topdist)[1])
        gdist = [(w, grav, [w] + centers[w].keys())
                 for w, grav in gravities.items()]
        if not gdist:
            return []
        return sorted(
            gdist,
            key=lambda x: x[1],
            reverse=True
        )



if __name__ == '__main__':

    file_from = sys.argv[1]
    sorter = InsideSort()

#     tokenized = [remove_punct(e.lower()).split() for e in examples]
    tokenized = [remove_punct(e.lower()).split()
                 for e in open(file_from, 'rb').readlines()]

    _sorted = sorter(tokenized)

    for cluster, feature_tree, instance in _sorted:
        print cluster, '\t', '/'.join(feature_tree), '\t', ' '.join(instance)

