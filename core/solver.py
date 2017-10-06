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

    def get_model_path(self, no=-1):
        path = os.path.join(self.model_path, 'imgcap-model/model.ckpt')
        if no >= 0:
            path += '-'+str(no)
        return path

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
        #_, _, generated_captions = self.model.build_inference(max_len=20)

        # summary op   
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        
        summary_op = tf.summary.merge_all() 


        print("Data size: %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations per epoch: %d" %n_iters_per_epoch)
        
        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model >= 0:
                trained_mode = self.get_model_path(self.pretrained_model)
                print("Start training with pretrained Model: ", trained_mode)
                saver.restore(sess, trained_mode)

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
                    #print "1 batch spend time: ", t1-t0

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        print 'current batch loss: ', l
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, epoch*n_iters_per_epoch + i)

                print ("Previous epoch loss: ", prev_loss)
                print ("Current epoch loss: ", curr_loss)
                print ("Elapsed time: ", time.time() - start_t)
                prev_loss = curr_loss
                curr_loss = 0

                epoch += 1
                # save model's parameters
                if epoch % self.save_every == 0 or epoch == self.n_epochs:
                    saver.save(sess, self.get_model_path(), global_step=epoch)
                    print ("model.ckpt-%s saved." % epoch)
            