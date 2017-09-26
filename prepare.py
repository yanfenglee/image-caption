from img_cap_data import ImgCapData

import sys
import os

basedir = os.environ["ML_DATA"] + "/challenge/"

datadir = sys.argv[1]

if len(sys.argv) <= 2:
    sample_num = None
else:
    sample_num = int(sys.argv[2])

data = ImgCapData(basedir=basedir+datadir, anno_file='/caption.json', max_sample = sample_num)

data.build_all_and_save()
data.extract_feature()
