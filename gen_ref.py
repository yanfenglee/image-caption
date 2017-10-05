from img_cap_data import ImgCapData

import sys
import os

basedir = os.environ["ML_DATA"] + "/challenge/"

datadir = sys.argv[1]

data = ImgCapData(basedir=basedir+datadir)

data.save_ref_for_eval_score()
