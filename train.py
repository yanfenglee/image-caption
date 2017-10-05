from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import *
import os
import sys
from img_cap_data import ImgCapData
from feature_extractor import FeatureExtractor
import argparse

def main(folder,epoch,pretrained_model):
    basedir = os.environ['ML_DATA']+"/challenge/"
    # load train dataset
    train_data = ImgCapData(basedir=basedir+folder, anno_file="caption.json")
    train_data.load_data()

    train_feature = FeatureExtractor(basedir+folder).load_feature()

    model = CaptionGenerator(imgcap=train_data, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, train_data=train_data, features=train_feature, n_epochs=epoch, batch_size=128,
                                          learning_rate=0.001, print_every=100, save_every=1, image_path=basedir+'/image/',
                                    pretrained_model=pretrained_model, model_path=basedir+'/model/',
                                     print_bleu=True, log_path=basedir+'/log/')

    solver.train()

if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument("--folder", required=True, help="train data folder")
    ps.add_argument("--model", type=int, default=-1, help="continue with pretrained model")
    ps.add_argument("--epoch", type=int, default=100, help="epoch number")

    args = ps.parse_args()

    main(folder=args.folder,epoch=args.epoch,pretrained_model=args.model)