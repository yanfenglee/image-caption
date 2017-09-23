from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json

class ImgCapData(object):
    # special tokens
    START = '<START>'
    END = '<END>'
    NULL = '<NULL>'

    def __init__(self, basedir, anno_file, max_length=15):
        self.basedir = basedir
        self.data = None
        self.image_idx = None
        self.image_files = None
        self.caption_vecs = None
        self.w2idx = None
        self.max_length = max_length
        self.anno_file = anno_file
        self.vocabs = None

    def process_annotations(self):
        with open(self.anno_file) as f:
            anno_data = json.load(f)

        data = []
        image_files = {}
        vocabs = Counter()
        for i, ann in enumerate(anno_data):
            image_id = ann['image_id']
            image_files[image_id] = i
            for cap in ann['caption']:
                words = seg(cap)
                if len(words) <= self.max_length:
                    for w in words:
                        vocabs[w] += 1
                    data += [{'image':image_id,'caption':cap}]

        self.data = data
        self.image_files = image_files
        self.vocabs = vocabs

    def build_vocabulary_idx(self):

        vocab = [w for w in self.vocabs]
        w2idx = {self.NULL: 0, self.START: 1, self.END: 2}
        idx = 3
        for word in vocab:
            w2idx[word] = idx
            idx += 1

        self.w2idx = w2idx

    def build_img_cap_vec(self):
        n_examples = len(self.data)
        captions = np.ndarray((n_examples,self.max_length+2)).astype(np.int32)
        image_idxs = np.ndarray(n_examples, dtype=np.int32)

        for idx, anno in enumerate(self.data):
            image_idxs[idx] = self.image_files[anno['image']]
            words = seg(anno['caption'])
            vec = []
            vec.append(self.w2idx[self.START])
            for w in words:
                vec.append(self.w2idx[w])
            vec.append(self.w2idx(self.END))

            pad_n = self.max_length + 2 - len(vec)
            vec.extend([self.w2idx[self.NULL] * pad_n])

            captions[idx,:] = np.asarray(vec)

        self.image_idx = image_idxs
        self.caption_vecs = captions


    def build_all_and_save(self):
        self.process_annotations()
        self.build_vocabulary_idx()
        self.build_img_cap_vec()

        

    def load_data():
        pass

    def extract_features():
        pass



def test():
    basedir = '/Users/lyfpcy/ml/aichallenge'
    save(basedir)
    
if __name__ == "__main__":
    test()