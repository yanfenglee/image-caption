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

    def process_annotations(self):
        with open(self.anno_file) as f:
            anno_data = json.load(f)

        data = []
        for ann in anno_data:
            image_id = ann['image_id']
            for cap in ann['caption']:
                if len(seg(cap)) <= self.max_length:
                    data += [{'image':image_id,'caption':cap}]
        self.data = data
        return data

    def build_vocabulary(self):
        counter = Counter()
        for anno in self.data:
            words = seg(anno['caption'])
            for w in words:
                counter[w] += 1

        vocab = [w for w in counter]
        w2idx = {NULL: 0, START: 1, END: 2}
        idx = 3
        for word in vocab:
            w2idx[word] = idx
            idx += 1

        self.w2idx = w2idx
        return w2idx

    def build_caption_vec(self):
        n_examples = len(self.data)
        captions = np.ndarray((n_examples,self.max_length+2)).astype(np.int32)
        
        for idx, anno in enumerate(annotations):
            words = seg(anno['caption'])
            vec = []
            vec.append(w2idx[START])
            for w in words:
                vec.append(w2idx[w])
            vec.append(w2idx(END))

            pad_n = self.max_length + 2 - len(vec)
            vec.extend([w2idx[NULL] for _ in range(0,pad_n)])

            captions[idx,:] = np.asarray(vec)

        self.caption_vecs = captions

        return captions

    def build_image_file(self):
        image_files = {}

        idx = 0
        for anno in self.data:
            img_id = anno['image']
            if not img_id in image_files:
                image_files[img_id] = idx
                idx += 1
        
        self.image_files = image_files

    def build_imgage_idx(self):
        image_idxs = np.ndarray(len(self.data), dtype=np.int32)
        
        for i, anno in enumerate(self.data):
            image_idxs[i] = self.image_files[anno['image']]
        
        self.image_idx = image_idxs


    def build_all_and_save(self):
        batch_size = 100
        max_length = 15

        train_data = process_annotations(basedir+'/train/caption.json',max_length)
        val_data = process_annotations(basedir+'/val/caption.json',max_length)

        # about 4000 images and 20000 captions for val / test dataset
        val_cutoff = int(0.1 * len(val_data))
        test_cutoff = int(0.2 * len(val_data))

        save_pickle(train_data, basedir+'/train/train.annotations.pkl')
        save_pickle(val_data[:val_cutoff], basedir+'/val/val.annotations.pkl')
        save_pickle(val_data[val_cutoff:test_cutoff], basedir+'/test/test.annotations.pkl')

        for d in ['train', 'val', 'test']:
            annotations = load_pickle(basedir+'/%s/%s.annotations.pkl' % (d, d))

            w2idx = build_vocabulary(annotations)
            save_pickle(w2idx, basedir+'/%s/w2idx.pkl' % d)
            
            captions = build_caption_vec(annotations, w2idx, max_length)
            save_pickle(captions, basedir+'/%s/%s.captions.pkl' % (d, d))

    def load_data():
        pass

    def extract_features():
        pass



def test():
    basedir = '/Users/lyfpcy/ml/aichallenge'
    save(basedir)
    
if __name__ == "__main__":
    test()