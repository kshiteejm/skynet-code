# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging

import graph
import walks as serialized_walks
from gensim.models import Word2Vec
from .skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

from scipy.sparse import coo_matrix

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

NUMBER_WALKS = 10
WALK_LENGTH = 40
REPRESENTATION_SIZE = 64
WINDOW_SIZE = 5
MAX_MEMORY_DATA_SIZE = 1000000000
WORKERS = 1
SEED = 0

__author__ = 'Bryan Perozzi'
__email__ = 'bperozzi@cs.stonybrook.edu'
__version__ = '1.0.0'


def get_deepwalk_representation(adj_matrix, number_walks=NUMBER_WALKS, walk_length=WALK_LENGTH, 
                                representation_size=REPRESENTATION_SIZE, window_size=WINDOW_SIZE, 
                                max_memory_data_size=MAX_MEMORY_DATA_SIZE, workers=WORKERS, seed=SEED,
                                vertex_freq_degree=False):
    G = graph.from_numpy(coo_matrix(adj_matrix))

    num_walks = len(G.nodes()) * number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * walk_length

    print("Data size (walks*length): {}".format(data_size))

    if data_size < max_memory_data_size:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                            path_length=walk_length, alpha=0, rand=random.Random(seed))
        print("Training...")
        model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=workers)
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, max_memory_data_size))
        print("Walking...")

        walks_filebase = "filebase.walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                                         path_length=walk_length, alpha=0, rand=random.Random(seed),
                                                         num_workers=workers)

        print("Counting vertex frequency...")
        if not vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(walk_files, workers)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                                         size=representation_size,
                                         window=window_size, min_count=0, trim_rule=None, workers=workers)

    return model.wv.vectors
