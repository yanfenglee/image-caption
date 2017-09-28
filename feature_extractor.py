from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *
from img_cap_data import ImgCapData

import tensorflow as tf
import numpy as np
import pandas as pd
#import hickle
from core.utils import *
import os
import json

class FeatureExtractor(object):

    def __init__(self, save_path=None):
        self.save_path = save_path
        self.features = None

    def extract_vgg(self, imgcapdata=None):
        # extract conv5_3 feature vectors
        data = imgcapdata
        vggnet = Vgg19(os.environ['ML_DATA'] + "/challenge/imagenet-vgg-verydeep-19.mat")
        vggnet.build()

        batch_size = 128

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            n_examples = len(data.image_idx2file)

            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for idx in range(0, n_examples, batch_size):
                end = idx+batch_size

                image_batch_file = data.image_idx2file[idx:end]
                read_batch = [ndimage.imread(data.get_image_path(x), mode='RGB') for x in image_batch_file]
                image_batch = np.stack(read_batch)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[idx:end, :] = feats
                print ("Processed %d features.." %(end))

            self.features = all_feats
            save_pickle(self.features, self.save_path+'/features.pkl')


    def load_feature(self):
        self.features = load_pickle(self.save_path+'/features.pkl')
        return self.features


if __name__ == "__main__":
    
    basedir = os.environ['ML_DATA']+"/challenge/"
    folder = sys.argv[1]

    data = ImgCapData(basedir=basedir+folder, anno_file="caption.json")
    data.load_data()


    savepath = basedir + folder
    fe = FeatureExtractor(save_path=None)

    fe.extract_vgg(data)