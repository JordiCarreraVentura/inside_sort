#!/usr/bin/env python
import argparse
import sys

from nsort import nSort

HELP = {
    'min_cluster': 'Minimum size of any clusters hypothesized by the system.',
    'min_wfreq': 'Minimum word frequency.',
    'max_wfreq': 'Maximum word frequency.',
    'window': 'Size of the co-occurrence window within which potential semantic relations will be computed. For any word, it denotes a range of neighboring words both to its left and to its right. Thus, the value provided by this parameter is effectively doubled. Any words within this area will be considered terms potentially semantically related to the target of the window.',
    'format': '\'txt\' (sorted texts), \'tsv\' (tab-separated pairs of each feature and an input text selected by that feature)',
    'baselines': 'How many baselines will be used to measure the relevance of the statistical association between two words. Each baseline is a word chosen at random from the whole vocabulary. The goal is for any pair of semantically-related words to have scores higher than all the baselines by some factor (denoted by the \'min_assoc_multiplier\' parameter.',
    'min_assoc_multiplier': 'For any pair of words with a given statistical association, this parameter denotes how many times higher than the baseline that association must be over the baseline(s) for the two words to be considered semantically related.',
    'novel_ratio': 'For any cluster, minimum ratio of new documents (over total documents) it must account for for that cluster to be deemed useful. During cluster assignment, and out of all the documents selected by a given cluster/set of features, some of those documents may have already been assigned to a higher-scoring cluster by the time another cluster is considered. In those cases, this parameter denotes the ratio of documents in the cluster that must not have been assigned to a previous cluster if the current one is to be kept.',
    'doc_overlap': 'Given some set of features defining a cluster and a new potential feature, the minimum ratio of documents in which the latter must co-occur with every word in the former.'
}


class Argparser:
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def parse(self):
            #
            #    Input file
        self.parser.add_argument(
            '-r', '--read',
            nargs='?',
            type=argparse.FileType('r'),
            default=sys.stdin
        )
            #
            #    Output file
        self.parser.add_argument(
            '-w', '--write',
            nargs='?',
            type=argparse.FileType('w'),
            default=sys.stdout
        )
            #
            #    Verbosity
        self.parser.add_argument(
            '-v', '--verbose',
            action='store_true'
        )
            #
            #    Format of the output file
        self.parser.add_argument(
            '-f', "--format",
            help=HELP['format'],
            default='tsv'    # txt, tsv
        )
            #
            #    Parameter: minimum cluster size
        self.parser.add_argument(
            '--min_cluster',
            help=HELP['min_cluster'],
            type=int,
            default=3
        )
            #
            #    Parameter: minimum word frequency
        self.parser.add_argument(
            '--min_wfreq',
            help=HELP['min_wfreq'],
            type=int,
            default=3
        )
            #
            #    Parameter: maximum word frequency
        self.parser.add_argument(
            '--max_wfreq',
            help=HELP['max_wfreq'],
            #type=int,
            default=0.2
        )
            #
            #    Parameter: co-occurrence window size
        self.parser.add_argument(
            '--window',
            help=HELP['window'],
            type=int,
            default=10
        )
            #
            #    Parameter: number of baselines
        self.parser.add_argument(
            '--baselines',
            help=HELP['baselines'],
            type=int,
            default=20
        )
            #
            #    Parameter: minimum association multiplier
        self.parser.add_argument(
            '--min_assoc_multiplier',
            help=HELP['min_assoc_multiplier'],
            type=float,
            default=1.3
        )
            #
            #    Parameter: cluster novelty ratio
        self.parser.add_argument(
            '--novel_ratio',
            help=HELP['novel_ratio'],
            type=float,
            default=0.75
        )
            #
            #    Parameter: document overlap
        self.parser.add_argument(
            '--doc_overlap',
            help=HELP['doc_overlap'],
            type=float,
            default=0.25
        )

        #    Parse and initialize arguments
        args = self.parser.parse_args()
        return args



if __name__ == '__main__':

    ap = Argparser()
    args = ap.parse()

    ns = nSort(
        min_cluster=args.min_cluster,
        min_wfreq=args.min_wfreq,
        window=args.window,
        max_wfreq=args.max_wfreq,
        baselines=args.baselines,
        min_assoc_multiplier=args.min_assoc_multiplier,
        novel_ratio=args.novel_ratio,
        doc_overlap=args.doc_overlap,
        verbose=args.verbose,
    )
    
    ns([line for line in args.read])

    with args.write as wrt:
        for feature_set, text in ns:
            if args.format == 'txt':
                line = '%s' % text
            elif args.format == 'tsv':
                line = '%s\t%s' % ('_'.join(feature_set), text)
            wrt.write(line)
