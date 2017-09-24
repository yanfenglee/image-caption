from scipy import ndimage
from collections import Counter
from core.vggnet import Vgg19
from core.utils import *
import img_cap_data

import tensorflow as tf
import numpy as np
import pandas as pd
import hickle
import os
import json

class FeatureExtractor(object):

    def __init__(self, imgcapdata):
        self.data = imgcapdata

    def extract_vgg(self, vgg_model_path, save_path):
        # extract conv5_3 feature vectors
        vggnet = Vgg19(vgg_model_path)
        vggnet.build()

        with tf.Session() as sess:
            tf.initialize_all_variables().run()

            n_examples = len(self.data.image_idx2file)

            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for start, end in zip(range(0, n_examples, batch_size),
                                range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = self.data.image_idx2file[start:end]
                image_batch = np.array(map(lambda x: ndimage.imread(self.data.get_image_path(x), mode='RGB'), image_batch_file)).astype(
                    np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print ("Processed %d features.." %(end)

            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path)
            print ("Saved %s.." % (save_path))