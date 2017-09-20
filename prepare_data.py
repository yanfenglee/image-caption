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

# special tokens
START = '<START>'
END = '<END>'
NULL = '<NULL>'

def seg(sentence):
    return sentence.split(' ')

def process_annotations(anno_file, max_length):
    with open(anno_file) as f:
        anno_data = json.load(f)

    data = []
    for ann in anno_data:
        image_id = ann['image_id']
        for cap in ann['caption']:
            if len(seg(cap)) <= max_length:
                data += [{'image':image_id,'caption':cap}]

    return data

def build_vocabulary(annotations):
    counter = Counter()
    for anno in annotations:
        words = seg(anno['caption'])
        for w in words:
            counter[w] += 1

    vocab = [w for w in counter]
    w2id = {NULL: 0, START: 1, END: 2}
    idx = 3
    for word in vocab:
        w2id[word] = idx
        idx += 1

    return w2id

def build_caption_vec(annotations, w2id, max_length=15):
    n_examples = len(annotations)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)
    
    for idx, anno in enumerate(annotations):
        words = seg(anno['caption'])
        vec = []
        vec.append(w2id[START])
        for w in words:
            vec.append(w2id[w])
        vec.append(w2id(END))

        pad_n = max_length + 2 - len(vec)
        vec.extend([w2id[NULL] for _ in range(0,pad_n)])

        captions[idx,:] = np.asarray(vec)

    return captions

def save(basedir):
    batch_size = 100
    max_length = 15

    train_data = process_annotations(basedir+'/annotations/train.json',max_length)
    val_data = process_annotations(basedir+'/annotations/val.json',max_length)

    # about 4000 images and 20000 captions for val / test dataset
    val_cutoff = int(0.1 * len(val_data))
    test_cutoff = int(0.2 * len(val_data))

    save_pickle(train_data, basedir+'/train/train.annotations.pkl')
    save_pickle(val_data[:val_cutoff], basedir+'/val/val.annotations.pkl')
    save_pickle(val_data[val_cutoff:test_cutoff], basedir+'/test/test.annotations.pkl')

    for d in ['train', 'val', 'test']:
        annotations = load_pickle(basedir+'/%s/%s.annotations.pkl' % (d, d))

        w2id = build_vocabulary(annotations)
        save_pickle(w2id, basedir+'/%s/w2id.pkl' % d)
        
        captions = build_caption_vec(annotations, w2id, max_length)
        save_pickle(captions, basedir+'/%s/%s.captions.pkl' % (d, d))

