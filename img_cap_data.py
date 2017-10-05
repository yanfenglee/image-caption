from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *

import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
import json
import hashlib

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
        self.anno_file = basedir+'/'+anno_file
        self.features = None
        self.max_sample = max_sample

        # extra
        self.idx2w = None
        self.image_idx2file = None

    def get_image_path(self, image_id):
        return self.basedir + '/images/' + image_id

    def process_images(self):
        image_files = os.listdir(self.basedir + '/images/')

        idx2img = []
        for i, img in enumerate(image_files):
            idx2img.append(img)

        self.image_idx2file = idx2img

    def flatten_annotations(self):
        with open(self.anno_file) as f:
            anno_data = json.load(f)

        data = []
        for i, ann in enumerate(anno_data):
            if self.max_sample != None and i >= self.max_sample:
                break

            image_id = ann['image_id']
            for cap in ann['caption']:
                words = seg(cap)
                if len(words) <= self.max_length and len(words) > 1:
                    data += [{'image_id':image_id,'caption':cap}]

        self.annotations = data

    def save_ref_for_eval_score(self):
        self.flatten_annotations()
        ref = {"annotations":[],"images":[],"type": "captions"}
        for i, ann in enumerate(self.annotations):
            caption = ' '.join(seg(ann['caption']))
            img = ann['image_id'].strip('.jpg')
            image_id = int(int(hashlib.sha256(img).hexdigest(), 16) % sys.maxint)
            
            ref["annotations"].append({"id":i+1, "caption":caption, "image_id":image_id})
            ref["images"].append({"file_name":img, "id":image_id})

        with open(self.basedir+"/caption_ref.json", 'wb') as f:
            json.dump(ref, f)


    def build_image_idx(self):
        imgs = Counter()
        for ann in self.annotations:
            imgs[ann['image_id']] += 1

        imgs_vec = [img for img in imgs]

        self.image_idx2file = np.array(imgs_vec)
        self.image_file2idx = {img: idx for idx, img in enumerate(imgs_vec)}

    def build_vocabulary_idx(self):
        vocabs = Counter()
        for ann in self.annotations:
            words = seg(ann['caption'])
            for w in words:
                vocabs[w] += 1

        vocab_vec = [self.NULL, self.START, self.END]
        for word in vocabs:
            vocab_vec.append(word)

        self.idx2w = vocab_vec
        self.w2idx = {w: i for i, w in enumerate(vocab_vec)}

    def build_img_cap_vec(self):
        n_examples = len(self.annotations)
        captions = np.ndarray((n_examples,self.max_length+2)).astype(np.int32)
        image_idx_vecs = np.ndarray(n_examples, dtype=np.int32)

        for idx, anno in enumerate(self.annotations):
            image_idx_vecs[idx] = self.image_file2idx[anno['image_id']]
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

        print('n_examples: ', n_examples)
        print('captions shape: ', captions.shape)

    def decode_caption_vec(self, caption_vec):
        if caption_vec.ndim != 2:
            raise RuntimeError('caption dim must be 2')

        N, T = caption_vec.shape

        decoded = []
        for i in range(N):
            words = []
            for t in range(T):
                word = self.idx2w[caption_vec[i, t]]
                if word == ImgCapData.START:
                    continue
                if word == ImgCapData.END:
                    #words.append(' .')
                    break
                if word != ImgCapData.NULL:
                    words.append(word)
            decoded.append(''.join(words))

        return decoded


    def _save(self, data, path):
        save_pickle(data, self.basedir+path)

    def _load(self, path):
        return load_pickle(self.basedir + path)

    def _anno_exist(self):
        return os.path.isfile(self.anno_file)

    def build_all_and_save(self):
        if not self._anno_exist():
            self.process_images()
            self._save(self.image_idx2file, '/image_idx2file.pkl')

        else:
            self.flatten_annotations()
            self.build_image_idx()
            self.build_vocabulary_idx()
            self.build_img_cap_vec()

            self._save(self.annotations, '/annotations.pkl')
            self._save(self.image_idx2file, '/image_idx2file.pkl')
            self._save(self.image_idx_vec, '/image_idx_vec.pkl')
            self._save(self.w2idx, '/w2idx.pkl')
            self._save(self.idx2w, '/idx2w.pkl')
            self._save(self.caption_vecs, '/caption_vecs.pkl')

        
    def load_data(self):
        if not self._anno_exist():
            self.image_idx2file = self._load('/image_idx2file.pkl')

        else:
            self.annotations = self._load('/annotations.pkl')
            self.image_idx2file = self._load('/image_idx2file.pkl')
            self.image_idx_vec = self._load('/image_idx_vec.pkl')
            self.w2idx = self._load('/w2idx.pkl')
            self.idx2w = self._load('/idx2w.pkl')
            self.caption_vecs = self._load('/caption_vecs.pkl')

def test_build():
    basedir = '/Users/lyfpcy/ml/challenge/test/'
    data = ImgCapData(basedir,'caption.json')
    data.build_all_and_save()

def test_load():
    basedir = '/Users/lyfpcy/ml/challenge/test/'
    data = ImgCapData(basedir,'caption.json')
    data.load_data()
    
    n_sample = data.caption_vecs.shape[0]
    sample = [0,1,3,5,10,20,40,100,150,300,500,1000,10000,20000,20003,n_sample-1]
    vecs = data.caption_vecs[sample,:]

    decoded = data.decode_caption_vec(vecs)

    imgs = data.image_idx2file[data.image_idx_vec[sample]]

    for img,cap in zip(imgs,decoded):
        print img,' ==> ', cap
    
if __name__ == "__main__":
    test_build()
    test_load()