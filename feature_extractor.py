from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *
import img_cap_data

import tensorflow as tf
import numpy as np
import pandas as pd
#import hickle
import os
import json

class FeatureExtractor(object):

    def __init__(self, imgcapdata):
        self.data = imgcapdata

    def extract_vgg(self):
        # extract conv5_3 feature vectors
        
        vggnet = Vgg19(os.environ['ML_DATA'] + "/challenge/imagenet-vgg-verydeep-19.mat")
        vggnet.build()

        batch_size = 128

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            n_examples = len(self.data.image_idx2file)

            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for idx in range(0, n_examples, batch_size):
                end = idx+batch_size

                image_batch_file = self.data.image_idx2file[idx:end]
                read_batch = [ndimage.imread(self.data.get_image_path(x), mode='RGB') for x in image_batch_file]
                image_batch = np.stack(read_batch)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[idx:end, :] = feats
                print ("Processed %d features.." %(end))

            return all_feats