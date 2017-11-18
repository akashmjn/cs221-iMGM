import glob
import numpy as np
import tensorflow as tf

from midi import midi_to_matrix, matrix_to_midi

###############################################################################

class RNNMusic:
    def __init__(self, input_len=5, lr=0.001, num_epochs=10):
        self.input_len = input_len
        self.lr = lr
        self.num_epochs = num_epochs
        self.build()
    
    def build(self):
        self.add_placeholders()
        newstate_prob = self.forward_prop()
        self.loss = self.add_loss_op(newstate_prob)
        self.train_op = self.add_train_op()
    
    def add_placeholders(self):
        self.inputs = tf.placeholder(shape=(None,self.input_len,128), dtype=tf.int32)
        self.labels = tf.placeholder(shape=(None,128), dtype=tf.int32)
    
    def create_feed_dict(self, inputs, labels=None):
        feed_dict = {}
        feed_dict[self.inputs] = inputs
        if not labels is None:
            feed_dict[self.labels] = labels
        return feed_dict
    
    def forward_prop(self):
        cells = [tf.contrib.rnn.BasicLSTMCell(100) for _ in range(self.input_len)]
        multi_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
        output, state = tf.nn.dynamic_rnn(cells, self.inputs, dtype=tf.float32)
        newstate = tf.contrib.layers.fully_connected(output, 128, activation_fn=None)
        newstate_prob = tf.contrib.layers.softmax(newstate)
        return newstate_prob
    
    def add_loss_op(self, newstate_prob):
        newstate_prob = self.forward_prop()
        loss = tf.reduce_mean(tf.softmax_cross_entropy_with_logits(labels=self.labels, logits=newstate_prob))
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
            matrix = midi_to_matrix(file)
            if matrix is None or len(matrix)<self.input_len+1:
                continue
            else:
                num_files += 1
                inputs = []
                labels = []
                for i in range(len(matrix)-1):
                    inputs.append(matrix[i:i+self.input_len,:])
                    labels.append(matrix[i+self.input_len,:])
                inputs = np.stack(inputs)
                labels = np.vstack(labels)
                epoch_loss += self.train_batch(sess, inputs, labels)
        epoch_loss /= num_files
        return epoch_loss
    
    def fit(self, sess, saver, folder):
        if folder[-1] != '/':
            folder += '/'
        files = glob.glob(folder)
        epoch_losses = []
        for i in range(self.num_epochs):
            print('***** Epoch ' + str(i) + ' *****')
            epoch_losses.append(self.run_epoch(sess, files))
            print('Saving model')
            