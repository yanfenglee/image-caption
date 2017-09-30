import tensorflow as tf
#import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
from scipy import ndimage
from core.utils import *
from core.bleu import evaluate


class CaptioningSolver(object):
    def __init__(self, model, features, train_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17) 
                - image_idxs: Indices for mapping caption to image of shape (400000, ) 
                - word_to_idx: Mapping dictionary from word to index 
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.train_data = train_data
        self.features = features
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)


        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def get_feature(self, idxs):
        t0 = time.time()

        features = []
        for i in idxs:
            feat = self.features[i]
            features.append(feat)

        features = np.stack(features,axis=0)

        t1 = time.time()
        #print 'read features spend time: ', t1-t0

        return features

    def train(self):
        # train/val dataset
        n_examples = self.train_data.caption_vecs.shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        #features = self.features
        captions = self.train_data.caption_vecs
        image_idxs = self.train_data.image_idx_vec

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        tf.get_variable_scope().reuse_variables()
        _, _, generated_captions = self.model.build_inference(max_len=20)


        print("Data size: %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations per epoch: %d" %n_iters_per_epoch)
        
        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            #summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            epoch = 0
            while epoch < self.n_epochs:
                rand_idxs = np.random.permutation(n_examples)
                captions = captions[rand_idxs]
                image_idxs = image_idxs[rand_idxs]

                for i in range(n_iters_per_epoch):
                    t0 = time.time()

                    captions_batch = captions[i*self.batch_size:(i+1)*self.batch_size]
                    image_idxs_batch = image_idxs[i*self.batch_size:(i+1)*self.batch_size]
                    features_batch = self.get_feature(image_idxs_batch)
                    feed_dict = {self.model.features: features_batch, self.model.captions: captions_batch}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l

                    t1 = time.time()
                    print "1 batch spend time: ", t1-t0

                    # if (i+1) % self.print_every == 0:
                    #     print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(epoch+1, i+1, l)
                    #     ground_truths = captions[image_idxs == image_idxs_batch[0]]
                    #     decoded = self.train_data.decode_caption_vec(ground_truths)
                    #     #for j, gt in enumerate(decoded):
                    #     #    print "Ground truth: " ,gt
                    #     gen_caps = sess.run(generated_captions, feed_dict)
                    #     decoded = self.train_data.decode_caption_vec(gen_caps)
                    #     #t2 = time.time()
                    #     print image_idxs_batch[0]
                    #     print decoded[0]

                print ("Previous epoch loss: ", prev_loss)
                print ("Current epoch loss: ", curr_loss)
                print ("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0
                
                # # print out BLEU scores and file write
                # if self.print_bleu:
                #     all_gen_cap = np.ndarray((n_iters_val*self.batch_size, 20),dtype=np.int32)
                #     for i in range(n_iters_val):
                #         features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                #         feed_dict = {self.model.features: features_batch}
                #         gen_cap = sess.run(generated_captions, feed_dict=feed_dict)  
                #         all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                    
                #     all_decoded = self.train_data.decode_caption_vec(all_gen_cap)
                #     save_pickle(all_decoded, self.basedir + "/val/val.candidate.captions.pkl")
                #     scores = evaluate(data_path=self.basedir, split='val', get_scores=True)
                #     write_bleu(scores=scores, path=self.model_path, epoch=e)

                epoch += 1
                # save model's parameters
                if epoch % self.save_every == 0 or epoch == self.n_epochs:
                    saver.save(sess, os.path.join(self.model_path, 'imgcap-model/model.ckpt'), global_step=epoch)
                    print ("model.ckpt-%s saved." % epoch)
            
         
    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17) 
            - image_idxs: Indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_inference(max_len=20)    # (N, max_len, L), (N, max_len)
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.model.features: features_batch }
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = self.train_data.decode_caption_vec(sam_cap)

            # if attention_visualization:
            #     for n in range(10):
            #         print "Sampled Caption: %s" %decoded[n]

            #         # Plot original image
            #         img = ndimage.imread(image_files[n])
            #         plt.subplot(4, 5, 1)
            #         plt.imshow(img)
            #         plt.axis('off')

            #         # Plot images with attention weights 
            #         words = decoded[n].split(" ")
            #         for t in range(len(words)):
            #             if t > 18:
            #                 break
            #             plt.subplot(4, 5, t+2)
            #             plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
            #             plt.imshow(img)
            #             alp_curr = alps[n,t,:].reshape(14,14)
            #             alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
            #             plt.imshow(alp_img, alpha=0.85)
            #             plt.axis('off')
            #         plt.show()

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)  
                all_decoded = self.train_data.decode_caption_vec(all_sam_cap)
                #save_pickle(all_decoded, self.basedir + "/%s/%s.candidate.captions.pkl" %(split,split))