from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import *
import os
from img_cap_data import ImgCapData

def main():
    basedir = os.environ['ML_DATA']+"/challenge/"
    # load train dataset
    train_data = ImgCapData(basedir=basedir+'train', anno_file="caption.json")
    train_data.load_data()
    train_data.load_feature()

    val_data = ImgCapData(basedir=basedir+'val', anno_file="caption.json")
    val_data.load_data()
    val_data.load_feature()

    model = CaptionGenerator(imgcap=train_data, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, train_data=train_data, val_data=val_data, n_epochs=20, batch_size=128, update_rule='adam',
                                          learning_rate=0.001, print_every=1, save_every=10, image_path=basedir+'/image/',
                                    pretrained_model=None, model_path=basedir+'/model/',
                                     print_bleu=True, log_path=basedir+'/log/')

    solver.train()

if __name__ == "__main__":
    main()