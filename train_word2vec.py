from __future__ import print_function

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    #    # from paper "SECTOR: A Neural Model for Coherent Topic Segmentation and Classification"
    # We pretrained word embeddings with 256 dimensions for the specific tasks using word2vec on lowercase English and German Wikipedia documents using a window size of 7.
    # python train_word2vec_model.py small.txt wikimodelsave german_wiki.txt
    if len(sys.argv) < 3:
        print("Useing: python train_word2vec_model.py input_text "
              "output_gensim_model")
        sys.exit(1)
    inp, outp1 = sys.argv[1:3]

    model = Word2Vec(LineSentence(inp), size=300, window=7, min_count=10,
                     workers=multiprocessing.cpu_count())

    model.wv.save_word2vec_format(outp1, binary=True)
    # model.save(outp1)
    # model.wv.save_word2vec_format(outp2, binary=False)
