import tensorflow as tf
from src.basic_rnn import RNNMusic
import os

FOLDER = './data/rnn_music/'
saved_models_folder = './models/shitty_rnn/'
with tf.Graph().as_default():
    rnn_music = RNNMusic(num_epochs=10)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        rnn_music.fit(sess, saver, FOLDER, saved_models_folder)

# saved_models_folder = './models/shitty_rnn/epoch_10/'
# with tf.Graph().as_default():
    # rnn_music = RNNMusic()
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
        # saver.restore(sess, os.path.join(saved_models_folder, 'epoch_10.ckpt'))
        # rnn_music.generate(sess, 100, './models/shitty_rnn/test_output.mid', saved_models_folder)