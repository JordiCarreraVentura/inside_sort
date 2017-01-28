
from nsort import (
    nsort,
    nSort
)

from tools import (
    splitter
)

from tqdm import tqdm


def read(newlines_corpus, n=500, sentences=False):
    texts = []
    with open(newlines_corpus, 'rb') as rd:
        for i, line in enumerate(tqdm(rd)):
            text = line.decode('utf-8').strip()
            if not sentences:
                texts.append(text)
            else:
                for sentence in splitter(text):
                    texts.apppend(text)
            if n and i >= n:
                break
    return texts


if __name__ == '__main__':
    
    #	initialize an nsort instance:
    ns = nsort(
        min_cluster=3,
        min_wfreq=3,
#         min_wfreq=1,		# for test
        window=3,
        max_wfreq=0.2,		# for documents
#         max_wfreq=0.01,		# for sentences
#         max_wfreq=1.0,		# for test
        baselines=100,
        #unigram_fallback='most_freq/near_third',
        min_assoc_multiplier=1.3,
        novel_ratio=0.75,
        doc_overlap=0.25
    )
    
    #	read documents:
    newlines_corpus = '/Users/jordi/Laboratorio/corpora/raw/blogs/blogs.1.500.txt'
    newlines_corpus = '/Users/jordi/Laboratorio/corpora/raw/blog2008.txt'
    newlines_corpus = '/Users/jordi/Laboratorio/WebInterpret/data/UK.ES.txt'
    newlines_corpus = '/Users/jordi/Laboratorio/corpora/raw/Kaggle Billion word imputation corpus/phone.txt'
#     documents = read(newlines_corpus, n=500, sentences=True)
    texts = read(newlines_corpus, n=10000)
    
#     documents = [
#         'a b c d e f g h i j k a b a a b',
#         'a b c a b c',
#         'c u v w',
#         'x y z',
#         'p q r',
#         's t u',
#         'a b',
#         'q f g h',
#         'f g k y z',
#         'a b c',
#         'f g'
#     ]
#     documents = [document.split() for document in documents]

    #	run nsorting:
#     clusters = ns(documents)
    
    ns = nSort()
    ns(texts)
    
    for a, b in ns.tuples():
        print a, b

    

