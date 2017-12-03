import tensorflow as tf
from src.basic_rnn import RNNMusic
import os
import glob
import numpy as np
from src.midi import midi_to_matrix
from collections import defaultdict
import pdb

input_len = 5
num_units = 64
lr = 0.005
num_epochs = 10
FOLDER = './data/overfit/'
saved_models_folder = './models/shitty_rnn/'

# Some stats about the folder
stats = defaultdict(int)
files = glob.glob(FOLDER + '*.mid')
for file in files:
    matrix = midi_to_matrix(file)
    notes = np.argmax(matrix, axis=1)
    for note in notes:
        stats[note] += 1
keys = [key for key in stats]
values = [stats[key] for key in keys]
most_occurring_note = keys[values.index(max(values))]
with open(os.path.join(saved_models_folder, 'train_stats.txt'),'w') as f:
    f.write('Number of notes used to predict next note: ' + str(input_len) + '\n')
    f.write('Hidden vector dimension: ' + str(num_units) + '\n')
    f.write('Learning rate: ' + str(lr) + '\n')
    f.write('Number of epochs: ' + str(num_epochs) + '\n')
    f.write('Stats about the frequency of notes\n')
    f.write('Most occurring note: ' + str(most_occurring_note) + '\n')
    for key in stats:
        f.write(str(key) + ': ' + str(stats[key]) + '\n')

# Train
with tf.Graph().as_default():
    rnn_music = RNNMusic(input_len=input_len, num_units=num_units, lr=lr, num_epochs=num_epochs)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        rnn_music.fit(sess, saver, FOLDER, saved_models_folder)

# Test
saved_models = './models/shitty_rnn/epoch_' + str(num_epochs) + '/'
with tf.Graph().as_default():
    rnn_music = RNNMusic(input_len=input_len, num_units=num_units, lr=lr, num_epochs=num_epochs)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(saved_models, 'epoch_' + str(num_epochs) + '.ckpt'))
        rnn_music.generate(sess, 100, './models/shitty_rnn/test_output.mid', os.path.join(saved_models_folder, 'test_stats.txt'))