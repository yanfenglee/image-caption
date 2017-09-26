from core.model import CaptionGenerator
from core.utils import *
import os
from img_cap_data import ImgCapData
import tensorflow as tf
import json


def save_json(data, path):
    with open(path, 'wb') as f:
        json.dump(data, f)

def test(sample_dir="val"):

    basedir = os.environ['ML_DATA']+"/challenge/"
    batch_size = 128
    save_path = basedir+sample_dir+"/test.json"

    model_data = ImgCapData(basedir=basedir+"train")
    model_data.load_data()


    data = ImgCapData(basedir=basedir+sample_dir)
    data.load_data()
    data.load_feature()



    model = CaptionGenerator(imgcap=data, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    features = data.features

    # build a graph to sample captions
    alphas, betas, sampled_captions = model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        tf.train.Saver().restore(sess, basedir+'/model/imgcap-model/model.ckpt')

        num_iter = int(np.ceil(float(features.shape[0]) / batch_size))
        all_sam_cap = np.ndarray((num_iter*batch_size, 20))

        for i in range(num_iter):
            features_batch = features[i*batch_size:(i+1)*batch_size]
            feed_dict = { model.features: features_batch }
            all_sam_cap[i*batch_size:(i+1)*batch_size] = sess.run(sampled_captions, feed_dict)  

        all_decoded = model_data.decode_caption_vec(all_sam_cap)

        caption = []
        for idx, cap in enumerate(all_decoded):
            img = data.image_idx2file[idx]
            caption.append({"image_id": img, "caption": cap})
        
        save_json(caption, save_path)



if __name__ == "__main__":
    test("val")