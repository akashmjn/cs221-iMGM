import os
import time
import tensorflow as tf

from tensorflow.contrib.training import HParams
from collections import namedtuple
from basic_rnn_akash import RNNMusic

hparams = HParams(input_len=1,rnn_layer_size=64,lr=0.01,num_epochs=500)

FOLDER = '../data/sampleChorale/'
saved_models_folder = './models/shitty_rnn/'
# saved_models_folder = './models/shitty_rnn/epoch_500'
# os.makedirs(saved_models_folder,exist_ok=True)

# rnn_music = RNNMusic(hparams)
# graph = rnn_music.build_graph()

# with graph.as_default():
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         sess.run(init)
#         rnn_music.fit(sess, saver, FOLDER, saved_models_folder)

def run_training(graph,train_dir,hparams):
    with graph.as_default():

with tf.Graph().as_default():
    rnn_music = RNNMusic(hparams)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(saved_models_folder, 'epoch_500.ckpt'))
        rnn_music.generate(sess, 100, './models/shitty_rnn/test_output.mid')
        
