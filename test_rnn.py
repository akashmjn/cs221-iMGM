import tensorflow as tf
from src.basic_rnn import RNNMusic
import os

FOLDER = './data/bach/'
saved_models_folder = './models/shitty_rnn/'
with tf.Graph().as_default():
    rnn_music = RNNMusic(input_len=1, lr=0.01)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        rnn_music.fit(sess, saver, FOLDER, saved_models_folder)

# saved_models_folder = './models/shitty_rnn/epoch_10/'
# with tf.Graph().as_default():
    # rnn_music = RNNMusic(input_len=1, lr=0.01)
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
        # saver.restore(sess, os.path.join(saved_models_folder, 'epoch_10.ckpt'))
        # rnn_music.generate(sess, 100, './models/shitty_rnn/test_output.mid')