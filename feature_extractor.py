from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *
from img_cap_data import ImgCapData

import tensorflow as tf
import numpy as np
import pandas as pd
from core.h5utils import H5Tensor
from core.utils import *
import os
import json
import sys
import time

class FeatureExtractor(object):

    def __init__(self, path, name="features.h5"):
        self.save_path = path+"/"+name

    def extract_vgg(self, imgcapdata=None):
        # extract conv5_3 feature vectors
        data = imgcapdata
        vggnet = Vgg19(os.environ['ML_DATA'] + "/challenge/imagenet-vgg-verydeep-19.mat")
        vggnet.build()

        batch_size = 128

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            n_examples = len(data.image_idx2file)

            #all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            all_feats = H5Tensor(self.save_path,maxshape=(n_examples, 196, 512),dtype=np.float32)
            start = all_feats.get_shape()[0]

            print('extract begin from idx: ', start)

            for idx in range(start, n_examples, batch_size):
                end = idx+batch_size
                image_batch_file = data.image_idx2file[idx:end]
                read_batch = [ndimage.imread(data.get_image_path(x), mode='RGB') for x in image_batch_file]
                image_batch = np.stack(read_batch)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats.append(feats)

                print ("Processed %d features.." %(end), time.time())

                all_feats.flush()
                print ('flushed data... ', time.time())

    def load_feature(self):
        tensor = H5Tensor(self.save_path)
        return tensor.data()


if __name__ == "__main__":
    
    basedir = os.environ['ML_DATA']+"/challenge/"
    folder = sys.argv[1]

    data = ImgCapData(basedir=basedir+folder)
    data.load_data()


    savepath = basedir + folder
    fe = FeatureExtractor(path=savepath)

    fe.extract_vgg(data)