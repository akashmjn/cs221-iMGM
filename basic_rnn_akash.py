import glob
import numpy as np
import os
import tensorflow as tf
import time
import pdb

from src.midi import midi_to_matrix, matrix_to_midi

###############################################################################

class RNNMusic:
    """
    Simple model where number of steps required for output can be varied 
    Methods: initialize (either empty or from saved model), train_op, generate 
    Init method requires: hparams object - (input_len,lr,num_epochs)
    Exposes objects: output_h, output_y, droput (train/val) flag, 
    Internal objects: LSTM/RNN cells/model, 
    """
    def __init__(self,hparams):
        self.hparams = hparams # number of previous time steps to consider

    ## functions that initiaize model graph ## 
    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            self.add_placeholders()
            self.newstate = self.forward_prop()
            self.loss = self.add_loss_op()
            self.train_op = self.add_train_op()
        return graph
    
    def add_placeholders(self):
        self.inputs = tf.placeholder(shape=(None,self.hparams.input_len,128), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(None,128), dtype=tf.float32)
    
    def forward_prop(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hparams.rnn_layer_size,
         activation=tf.nn.relu)
        output, state = tf.nn.dynamic_rnn(lstm_cell, self.inputs, dtype=tf.float32)
        final_output = output[:,-1,:]
        newstate = tf.contrib.layers.fully_connected(final_output, 128, activation_fn=None)
        return newstate
    
    def add_loss_op(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.newstate))
        return loss
    
    def add_train_op(self):
        train_op = tf.train.AdamOptimizer(self.hparams.lr).minimize(self.loss)
        return train_op
    ## end initialization ##

    def create_feed_dict(self, inputs, labels=None):
        feed_dict = {}
        feed_dict[self.inputs] = inputs
        if not labels is None:
            feed_dict[self.labels] = labels
        return feed_dict   

    def train_batch(self, sess, inputs, labels):
        feed_dict = self.create_feed_dict(inputs=inputs, labels=labels)
        # pdb.set_trace()
        batch_loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        tf.summary.scalar('loss',batch_loss)
        return batch_loss
    
    def run_epoch(self, sess, files):
        num_files, epoch_loss = 0, 0
        for file in files:
            matrix = midi_to_matrix(file, join_chords=False)
            if matrix is None or len(matrix)<self.hparams.input_len+1:
                continue
            else:
                num_files += 1
                inputs = []
                labels = []
                for i in range(len(matrix)-self.hparams.input_len):
                    inputs.append(matrix[i:i+self.hparams.input_len,:])
                    labels.append(matrix[i+self.hparams.input_len,:])
                inputs = np.stack(inputs)
                labels = np.vstack(labels)
                epoch_loss += self.train_batch(sess, inputs, labels)
        epoch_loss /= num_files
        return epoch_loss
    
    def fit(self, sess, saver, midi_folder, saved_models_folder):
        # tensorboard initialization
        # log_dir = os.path.join(saved_models_folder,'logs')
        # summaries = tf.summary.merge_all()
        # writer = tf.summary.FileWriter(
        #         os.path.join(log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        # writer.add_graph(sess.graph)              

        files = glob.glob(midi_folder + '*.mid')
        loss_file = os.path.join(saved_models_folder, 'loss.txt')
        epoch_losses = []
        for i in range(self.hparams.num_epochs):
            print('***** Epoch ' + str(i) + ' *****')
            epoch_losses.append(self.run_epoch(sess, files))
            print('Average loss for this epoch: ' + str(epoch_losses[-1]))
            print('Saving model and loss progression for the epoch')
            with open(loss_file, 'a') as f:
                f.write(str(epoch_losses[-1]) + '\n')

            # # instrument for tensorboard
            # summ = sess.run(summaries,{}) 
            # writer.add_summary(summ, i)                  

            epoch_folder = 'epoch_'+str(i+1)
            if i==(self.hparams.num_epochs-1):
                # os.mkdir(os.path.join(saved_models_folder, epoch_folder))
                saver.save(sess,os.path.join(saved_models_folder,epoch_folder,'checkpoint.ckpt'))
                print('Model saved')
    
    def generate(self, sess, num_notes, save_midi_path):
        notes = np.zeros((1,self.hparams.input_len,128))
        for i in range(self.hparams.input_len):
            notes[0,i,np.random.randint(60,72)] = 1

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))       

        for _ in range(num_notes):
            input = notes[:,-self.hparams.input_len:,:]
            feed_dict = self.create_feed_dict(inputs=input)
            newstate_np = sess.run(self.newstate, feed_dict=feed_dict)

            # generate next state by sampling from output distribution
            logit = np.squeeze(newstate_np)
            probs = np.exp(logit)/sum(np.exp(logit)) # computing softmax
            sampled_state_idx = weighted_pick(probs)
            sampled_state = np.zeros(newstate_np.shape)
            sampled_state[0,sampled_state_idx] = 1
            notes_temp = np.vstack((notes[0,:,:], sampled_state))
            notes = notes_temp.reshape(1,notes_temp.shape[0],notes_temp.shape[1])

        matrix_to_midi(notes.reshape(notes.shape[1],notes.shape[2]), save_midi_path)

###############################################################################