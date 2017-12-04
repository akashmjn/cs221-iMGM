import glob
import numpy as np
import os
import tensorflow as tf
import time
import pdb

from src.midi import midi_to_matrix, matrix_to_midi
from src.sequence import Sequence

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
            self.newstate = self.add_forward()
            self.loss = self.add_loss_op()
            self.train_op = self.add_train_op()
        return graph
    
    def add_placeholders(self):
        self.inputs = tf.placeholder(shape=(None,self.hparams.input_len,self.hparams.input_size), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(None,self.hparams.input_size), dtype=tf.float32)
    
    def add_forward(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hparams.rnn_layer_size,
         activation=tf.nn.relu)
        output, state = tf.nn.dynamic_rnn(lstm_cell, self.inputs, dtype=tf.float32)
        final_output = output[:,-1,:]
        newstate = tf.contrib.layers.fully_connected(final_output, self.hparams.input_size, activation_fn=None)
        return newstate
    
    def add_loss_op(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.newstate))
        return loss
    
    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.hparams.lr)
        gradvs = optimizer.compute_gradients(self.loss)
        clipped_grads = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gradvs] 
        train_op = optimizer.apply_gradients(clipped_grads) 
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
        # iterate over files, read in as sequence object
        # create batches in RNN representation (one/many hot) train
        for file in files:
            sequence = Sequence()
            sequence.load_midi(file,join_tracks=False)
            if sequence is None or len(sequence)<self.hparams.input_len+1:
                continue
            else:
                num_files += 1
                inputs = []
                labels = []
                for i in range(len(sequence)-self.hparams.input_len):
                    inputs.append(sequence[i:i+self.hparams.input_len])
                    labels.append(sequence[i+self.hparams.input_len])
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

        # create folder, files for saving checkpoints/recording stats
        files = glob.glob(midi_folder + '*.mid')
        loss_file = os.path.join(saved_models_folder, 'loss.txt')
        os.makedirs(saved_models_folder,exist_ok=True)
        os.system("touch {}".format(loss_file))
        epoch_losses = []
        for i in range(1,self.hparams.num_epochs+1):
            # Each epoch iterates over all files in a folder, a batch is one file
            true_i = i+self.hparams.epoch_offset
            print('***** Epoch ' + str(true_i) + ' *****')
            epoch_losses.append(self.run_epoch(sess, files)) # runs the actual training
            print('Average loss for this epoch: ' + str(epoch_losses[-1]))
            with open(loss_file, 'a') as f:
                f.write(str(epoch_losses[-1]) + '\n')

            # # instrument for tensorboard
            # summ = sess.run(summaries,{}) 
            # writer.add_summary(summ, i)                  

            epoch_folder = 'epoch_'+str(true_i)
            if true_i%5==0:
                print('Saving model and loss progression for the epoch')
                saver.save(sess,os.path.join(saved_models_folder,epoch_folder,'checkpoint.ckpt'))
                print('Model saved')
    
    def generate(self, sess, num_notes, save_midi_path):

        # initialize sequence object with starting sequence (required for next state)
        # notes = np.zeros((1,self.hparams.input_len,hparams.input_size))
        note_sequence = Sequence()
        for i in range(self.hparams.input_len):
            row = np.zeros(self.hparams.input_size)
            row[np.random.randint(60,72)] = 1
            note_sequence.add(row)

        def weighted_pick(weights): # because np.random.choice behaves messed up
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))       

        ## TODO AKASH: Incorporate beam-search type method for generation
        for _ in range(num_notes):
            input = note_sequence[-self.hparams.input_len:]
            input = np.expand_dims(np.vstack(input),axis=0) 
            feed_dict = self.create_feed_dict(inputs=input)
            newstate_np = sess.run(self.newstate, feed_dict=feed_dict)

            # generate next state by sampling from output distribution
            logit = np.squeeze(newstate_np)
            probs = np.exp(logit)/sum(np.exp(logit)) # computing softmax
            sampled_state_idx = weighted_pick(probs)
            sampled_state = np.zeros(self.hparams.input_size)
            sampled_state[sampled_state_idx] = 1

            # convert, and add state to sequence object
            note_sequence.add(sampled_state)
            # notes_temp = np.vstack((notes[0,:,:], sampled_state))
            # notes = notes_temp.reshape(1,notes_temp.shape[0],notes_temp.shape[1])

        # return sequence object / write to file 

        note_sequence.many_hot_to_midi(save_midi_path)
        # matrix_to_midi(notes.reshape(notes.shape[1],notes.shape[2]), save_midi_path)

###############################################################################