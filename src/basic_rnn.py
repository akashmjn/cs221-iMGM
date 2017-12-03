import glob
import numpy as np
import os
import tensorflow as tf

from .midi import midi_to_matrix, matrix_to_midi

###############################################################################

class RNNMusic:
    def __init__(self, input_len=5, lr=0.001, num_epochs=10):
        self.input_len = input_len
        self.lr = lr
        self.num_epochs = num_epochs
        self.build()
    
    def build(self):
        self.add_placeholders()
        self.newstate = self.forward_prop()
        self.loss = self.add_loss_op(newstate)
        self.train_op = self.add_train_op()
    
    def add_placeholders(self):
        self.inputs = tf.placeholder(shape=(None,self.input_len,128), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(None,128), dtype=tf.float32)
    
    def create_feed_dict(self, inputs, labels=None):
        feed_dict = {}
        feed_dict[self.inputs] = inputs
        if not labels is None:
            feed_dict[self.labels] = labels
        return feed_dict
    
    def forward_prop(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=16, activation=tf.nn.relu)
        output, state = tf.nn.dynamic_rnn(lstm_cell, self.inputs, dtype=tf.float32)
        final_output = output[:,-1,:]
        self.newstate = tf.contrib.layers.fully_connected(final_output, 128, activation_fn=None)
    
    def add_loss_op(self, newstate):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=newstate))
        return loss
    
    def add_train_op(self):
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        return train_op
    
    def train_batch(self, sess, inputs, labels):
        feed_dict = self.create_feed_dict(inputs=inputs, labels=labels)
        batch_loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return batch_loss
    
    def run_epoch(self, sess, files):
        num_files, epoch_loss = 0, 0
        for file in files:
            matrix = midi_to_matrix(file, join_chords=False)
            if matrix is None or len(matrix)<self.input_len+1:
                continue
            else:
                num_files += 1
                inputs = []
                labels = []
                for i in range(len(matrix)-self.input_len):
                    inputs.append(matrix[i:i+self.input_len,:])
                    labels.append(matrix[i+self.input_len,:])
                inputs = np.stack(inputs)
                labels = np.vstack(labels)
                epoch_loss += self.train_batch(sess, inputs, labels)
        epoch_loss /= num_files
        return epoch_loss
    
    def fit(self, sess, saver, midi_folder, saved_models_folder):
        files = glob.glob(midi_folder + '*.mid')
        loss_file = os.path.join(saved_models_folder, 'loss.txt')
        epoch_losses = []
        for i in range(self.num_epochs):
            print('***** Epoch ' + str(i) + ' *****')
            epoch_losses.append(self.run_epoch(sess, files))
            print('Average loss for this epoch: ' + str(epoch_losses[-1]))
            print('Saving model and loss progression for the epoch')
            with open(loss_file, 'a') as f:
                f.write(str(epoch_losses[-1]) + '\n')
            epoch_folder = 'epoch_'+str(i+1)
            os.mkdir(os.path.join(saved_models_folder, epoch_folder))
            saver.save(sess, os.path.join(saved_models_folder, epoch_folder, 'epoch_'+str(i+1)+'.ckpt'))
            print('Model saved')
    
    def generate(self, sess, start_notes, num_notes, save_midi_path, model_folder):
        notes = start_notes
        for _ in range(num_notes):
            input = notes[:,-self.input_len:,:]
            newstate = self.forward_prop()
            feed_dict = 
            newstate_np = sess.run()

###############################################################################

if __name__ == '__main__':
    FOLDER = '../data/rnn_music/'
    saved_models_folder = '../models/shitty_rnn/'
    with tf.Graph().as_default():
        rnn_music = RNNMusic(num_epochs=2)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            rnn_music.fit(sess, saver, FOLDER, saved_models_folder)

###############################################################################

if __name__ == '__main__':
    tf.reset_default_graph()
    rnn_music = RNNMusic()
    saver = tf.train.saver()
    saved_models_folder = '../models/shitty_rnn/epoch_10/'
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(saved_models_folder, '*.ckpt'))
        