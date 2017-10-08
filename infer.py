from core.model import CaptionGenerator
from core.utils import *
import os
from img_cap_data import ImgCapData
import tensorflow as tf
import json
import sys
from feature_extractor import FeatureExtractor
import argparse


def save_json(data, path):
    with open(path, 'wb') as f:
        json.dump(data, f)
        print 'save json finished: ', path

def inference(sample_dir="val",model_num=1, maxlen=15):

    basedir = os.environ['ML_DATA']+"/challenge/"
    batch_size = 128
    save_path = basedir+sample_dir+"/infer.json"

    model_data = ImgCapData(basedir=basedir+"train")
    model_data.load_data()


    data = ImgCapData(basedir=basedir+sample_dir)
    data.load_data()

    features = FeatureExtractor(basedir+sample_dir).load_feature()


    model = CaptionGenerator(imgcap=model_data, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    # build a graph to sample captions
    alphas, betas, sampled_captions = model.build_inference(max_len=maxlen)    # (N, max_len, L), (N, max_len)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.train.Saver().restore(sess, basedir+'/model/imgcap-model/model.ckpt-'+str(model_num))

        nexamples = features.shape[0]
        all_sam_cap = np.ndarray((nexamples, maxlen),dtype=np.int32)

        for i in range(0, nexamples, batch_size):
            end = i+batch_size
            features_batch = features[i:end]
            feed_dict = { model.features: features_batch }
            all_sam_cap[i:end] = sess.run(sampled_captions, feed_dict)  

        all_decoded = model_data.decode_caption_vec(all_sam_cap)

        caption = []
        for idx, cap in enumerate(all_decoded):
            img = data.image_idx2file[idx]
            caption.append({"image_id": img.strip('.jpg'), "caption": cap})
        
        save_json(caption, save_path)



if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument("--folder", required=True, help="inference data folder")
    ps.add_argument("--model", required=True, help="model number used for inference")
    ps.add_argument("--maxlen", type=int, default=15, help="sentence max length")
    args = ps.parse_args()

    inference(args.folder, model_num=args.model,maxlen=args.maxlen)