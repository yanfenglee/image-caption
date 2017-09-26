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
from feature_extractor import *

class ImgCapData(object):
    # special tokens
    START = '<START>'
    END = '<END>'
    NULL = '<NULL>'

    N_START = 1
    N_END = 2
    N_NULL = 0

    def __init__(self, basedir, anno_file="caption.json", max_length=15, max_sample = None):
        self.basedir = basedir
        self.annotations = None
        self.image_idx_vec = None
        self.image_file2idx = None
        self.caption_vecs = None
        self.w2idx = None
        self.max_length = max_length
        self.anno_file = basedir+anno_file
        self.vocabs = None
        self.features = None
        self.max_sample = max_sample
        self.model_path = basedir + "/model"

        # extra
        self.idx2w = None
        self.image_idx2file = None

    def post_init(self):
        tmp = {i: img for img, i in self.image_file2idx.items()}
        n_imgs = len(tmp)
        imgs = []
        for idx in range(n_imgs):
            imgs.append(tmp[idx])

        self.image_idx2file = np.array(imgs)
        self.idx2w = {i: w for w, i in self.w2idx.items()}

    def get_image_path(self, image_id):
        return self.basedir + '/images/' + image_id

    def process_annotations(self):
        with open(self.anno_file) as f:
            anno_data = json.load(f)

        data = []
        image_file2idx = {}
        vocabs = Counter()
        for i, ann in enumerate(anno_data):
            if self.max_sample != None and i >= self.max_sample:
                break

            image_id = ann['image_id']
            image_file2idx[image_id] = i
            for cap in ann['caption']:
                words = seg(cap)
                if len(words) <= self.max_length and len(words) > 1:
                    for w in words:
                        vocabs[w] += 1
                    data += [{'image':image_id,'caption':cap}]

        self.annotations = data
        self.image_file2idx = image_file2idx
        self.vocabs = vocabs

    def build_vocabulary_idx(self):

        vocab = [w for w in self.vocabs]
        w2idx = {self.NULL: self.N_NULL, self.START: self.N_START, self.END: self.N_END}
        idx = 3
        for word in vocab:
            w2idx[word] = idx
            idx += 1

        self.w2idx = w2idx
        self.idx2w = {i: w for w, i in w2idx.items()}

    def build_img_cap_vec(self):
        n_examples = len(self.annotations)
        captions = np.ndarray((n_examples,self.max_length+2)).astype(np.int32)
        image_idx_vecs = np.ndarray(n_examples, dtype=np.int32)

        for idx, anno in enumerate(self.annotations):
            image_idx_vecs[idx] = self.image_file2idx[anno['image']]
            words = seg(anno['caption'])
            vec = []
            vec.append(self.w2idx[self.START])
            for w in words:
                vec.append(self.w2idx[w])
            vec.append(self.w2idx[self.END])

            pad_n = self.max_length + 2 - len(vec)
            vec.extend([self.w2idx[self.NULL]] * pad_n)
            captions[idx,:] = np.asarray(vec)

        self.image_idx_vec = image_idx_vecs
        self.caption_vecs = captions

    def decode_caption_vec(self, caption_vec):
        if caption_vec.ndim != 2:
            raise RuntimeError('caption dim must be 2')

        N, T = caption_vec.shape

        decoded = []
        for i in range(N):
            words = []
            for t in range(T):
                word = self.idx2w[caption_vec[i, t]]
                if word == ImgCapData.END:
                    words.append(' .')
                if word != ImgCapData.NULL:
                    words.append(word)
            decoded.append(''.join(words))

        return decoded


    def _save(self, data, path):
        save_pickle(data, self.basedir+path)

    def _load(self, path):
        return load_pickle(self.basedir + path)

    def build_all_and_save(self):
        self.process_annotations()
        self.build_vocabulary_idx()
        self.build_img_cap_vec()

        self.post_init()

        self._save(self.annotations, '/annotations.pkl')
        self._save(self.image_file2idx, '/image_file2idx.pkl')
        self._save(self.image_idx_vec, '/image_idx_vec.pkl')
        self._save(self.w2idx, '/w2idx.pkl')
        self._save(self.caption_vecs, '/caption_vecs.pkl')

        
    def load_data(self):
        self.annotations = self._load('/annotations.pkl')
        self.image_file2idx = self._load('/image_file2idx.pkl')
        self.image_idx_vec = self._load('/image_idx_vec.pkl')
        self.w2idx = self._load('/w2idx.pkl')
        self.caption_vecs = self._load('/caption_vecs.pkl')

        self.post_init()

    def extract_feature(self):
        fe = FeatureExtractor(self)
        self.features = fe.extract_vgg()
        self._save(self.features, '/features.pkl')

    def load_feature(self):
        self.features = self._load('/features.pkl')

def test_build():
    basedir = '/Users/lyfpcy/ml/aichallenge/val/'
    data = ImgCapData(basedir,'caption.json')
    data.build_all_and_save()

def test_load():
    basedir = '/Users/lyfpcy/ml/aichallenge/val/'
    data = ImgCapData(basedir,'caption.json')
    data.load_data()
    
    n_sample = data.caption_vecs.shape[0]
    sample = [0,1,3,5,10,20,40,100,150,300,500,1000,10000,20000,40000,n_sample-1]
    vecs = data.caption_vecs[sample,:]

    decoded = data.decode_caption_vec(vecs)

    imgs = data.image_idx2file[data.image_idx_vec[sample]]

    for img,cap in zip(imgs,decoded):
        print(img,' ==> ', cap)
    
if __name__ == "__main__":
    test_load()