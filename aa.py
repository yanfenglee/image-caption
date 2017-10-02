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


basedir = os.environ["ML_DATA"] + '/challenge/'
vocabs = load_pickle(basedir+"train/w2idx.pkl")

print len(vocabs)