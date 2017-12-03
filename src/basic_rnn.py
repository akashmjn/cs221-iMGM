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
        self.loss = self.add_loss_op()
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
        newstate = tf.contrib.layers.fully_connected(final_output, 128, activation_fn=None)
        return newstate
    
    def add_loss_op(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.newstate))
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
    
    def generate(self, sess, num_notes, save_midi_path):
        notes = np.zeros((1,self.input_len,128))
        for i in range(self.input_len):
            notes[0,i,np.random.randint(60,72)] = 1
        for _ in range(num_notes):
            input = notes[:,-self.input_len:,:]
            feed_dict = self.create_feed_dict(inputs=input)
            newstate_np = sess.run(self.newstate, feed_dict=feed_dict)
            newstate_onehot = np.zeros(newstate_np.shape)
            newstate_onehot[0,np.argmax(newstate_np)] = 1
            notes_temp = np.vstack((notes[0,:,:], newstate_onehot))
            notes = notes_temp.reshape(1,notes_temp.shape[0],notes_temp.shape[1])
        matrix_to_midi(notes.reshape(notes.shape[1],notes.shape[2]), save_midi_path)

###############################################################################