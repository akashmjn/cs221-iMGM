import glob
import numpy as np
import os
import tensorflow as tf
import time
import pdb
import collections 
import heapq
import copy

from .midi import midi_to_matrix, matrix_to_midi
from .sequence import Sequence

###############################################################################

BeamEntry = collections.namedtuple("BeamEntry",["score","sequence","state"])

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
            self.output = self.add_forward()
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
        output = tf.contrib.layers.fully_connected(final_output, self.hparams.input_size, activation_fn=None)
        return output
    
    def add_loss_op(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.output))
        return loss
    
    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.hparams.lr)
        gradvs = optimizer.compute_gradients(self.loss)
        clipped_grads = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gradvs] 
        train_op = optimizer.apply_gradients(clipped_grads) 
        return train_op
    ## end initialization ##

    ## functions for running epochs 
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

    def _batchify(self,sequence):
        inputs = []
        labels = []
        if len(sequence)>self.hparams.input_len:
            for i in range(len(sequence)-self.hparams.input_len):
                inputs.append(sequence[i:i+self.hparams.input_len])
                labels.append(sequence[i+self.hparams.input_len])
            inputs = np.stack(inputs)
            labels = np.vstack(labels)       
        else:
            inputs = np.expand_dims(np.stack(sequence),axis=0)
        return (inputs,labels)
    
    def run_epoch(self, sess, files):
        num_files, epoch_loss = 0, 0
        # iterate over files, read in as sequence object
        # create batches in RNN representation (one/many hot) train
        for file in files:
            sequence = Sequence(epsilon=self.hparams.epsilon)
            sequence.load_midi(file,join_tracks=False)
            if sequence is None or len(sequence)<self.hparams.input_len+1:
                continue
            else:
                num_files += 1
                inputs,labels = self._batchify(sequence)
                epoch_loss += self.train_batch(sess, inputs, labels)
        epoch_loss /= num_files
        return epoch_loss

    def _write_to_line_file(self,metric_file,metric,epoch_num):
        line_no = epoch_num-1
        with open(metric_file, 'r+') as f:
           lines = f.readlines()
           # check if line encountered and overwrite from that point onwards
           if len(lines)!=0 and line_no <= (len(lines)-1):
               lines.insert(line_no,str(metric)+'\n')
               del lines[line_no+1:]   # truncates the file
           else:
               lines.append(str(metric)+'\n')
           # if line not overwritten, append to end
           f.seek(0)   # moves the pointer to the start of the file
           f.writelines(lines)        

    def fit(self, sess, saver, midi_folder, saved_models_folder):
        # tensorboard initialization
        log_dir = os.path.join(saved_models_folder,'logs')
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
            # append loss to file
            self._write_to_line_file(loss_file,epoch_losses[-1],true_i)
            # with open(loss_file, 'a') as f:
            #     f.write(str(epoch_losses[-1]) + '\n')

            # # instrument for tensorboard
            # summ = sess.run(summaries,{}) 
            # writer.add_summary(summ, i)                  

            epoch_folder = 'epoch_'+str(true_i)
            if true_i%5==0:
                print('Saving model and loss progression for the epoch')
                saver.save(sess,os.path.join(saved_models_folder,epoch_folder,'checkpoint.ckpt'))
                print('Model saved')
    ## end training / epoch running code

    ## functions for generating sequence
    def get_next_output(self,sess,note_sequence,return_probs=False):
        """
        Returns a PDF over possible states, conditioned on the sequence so far
        """
        input,labels = self._batchify(note_sequence)
        # input = note_sequence[-self.hparams.input_len:]
        # input = np.expand_dims(np.vstack(input),axis=0) 
        # input = note_sequence.sequence
        # input = np.expand_dims(np.vstack(input),axis=1) 
        feed_dict = self.create_feed_dict(inputs=input)
        output_np = sess.run(self.output, feed_dict=feed_dict)       
        output_np = np.squeeze(output_np[-1,:]) # original size (len(sequence),input_size)
        if return_probs==True:
            logits = output_np - np.max(output_np) # well behaved softmax
            probs = np.exp(logits)/sum(np.exp(logits))
            return probs
        else:
            return output_np

    def sample_sequence(self,sess,note_sequence,num_notes,argmax=False): 
        """
        Uses RNNMusic (model) object, Sequence object, number of notes
        """
        def weighted_pick(weights): # because np.random.choice behaves messed up
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))              

        for _ in range(num_notes):
            # generate next state by sampling from output distribution
            probs = self.get_next_output(sess,note_sequence,return_probs=True) 
            if argmax:
                sampled_state_idx = np.argmax(probs)
            else:
                sampled_state_idx = weighted_pick(probs) # random sampling
            sampled_state = np.zeros(self.hparams.input_size)
            sampled_state[sampled_state_idx] = 1       
            # convert, and add state to sequence object
            note_sequence.add(sampled_state)
            # notes_temp = np.vstack((notes[0,:,:], sampled_state))
            # notes = notes_temp.reshape(1,notes_temp.shape[0],notes_temp.shape[1])           

    def _generate_branches(self,sess,beam_entry,beam_entries,beam_size):
        """
        Generates prediction from note_sequence, adds all possible states to beam_entries
        """
        score,sequence,state = beam_entry
        probs = self.get_next_output(sess,sequence,return_probs=True)
        # for each option, push to fixed size heap
        for i in range(len(probs)):
            new_score = score + np.log(probs[i])
            # to avoid extra copying, first check if next_score > min element in heap
            if len(beam_entries)==0 or new_score > (beam_entries[0].score):
                new_state = np.zeros(self.hparams.input_size)
                new_state[i] = 1
                new_sequence = copy.deepcopy(sequence)
                new_sequence.add(new_state)
                new_entry = BeamEntry(new_score,new_sequence,new_state)
                # add to fixed size heap
                if len(beam_entries) < beam_size:
                    heapq.heappush(beam_entries,new_entry)
                else:
                    try:
                        heapq.heappushpop(beam_entries,new_entry)
                    except:
                        pdb.set_trace()
                        print("yo")

    def beam_search_sequence(self,sess,note_sequence,num_notes,beam_size):
        """
        beam entry : (score,sequence,state)
        score - cumulative log probability of sequence log(s0) + log(s1) .. 
        sequence - Sequence object accumulated so far
        state - last state in sequence object used to generate next branches 
        """
        beam_entries = []
        initial_entry = BeamEntry(0,note_sequence,note_sequence[-1])
        # initialize beam of size k from top k of branches from inital sequence
        self._generate_branches(sess,initial_entry,beam_entries,beam_size)
        # repeat for num_notes
        for i in range(num_notes):
            # pop all candidates (score gets smaller every iteration p1*p2, etc. 
            # would be added)
            new_entries = []
            for curr_entry in beam_entries: 
                # generate branches for 
                # curr_entry = heapq.heappop(beam_entries)
                self._generate_branches(sess,curr_entry,new_entries,beam_size)
            beam_scores = [ round(_.score,2) for _ in new_entries]
            print("Note {}, Scores {}".format(i,beam_scores))
            beam_entries = new_entries
        # prune beam to top 1 and return 
        best_entry = sorted(beam_entries,key=lambda x: x.score)[-1]
        return best_entry

    def generate(self, sess, num_notes, save_midi_path):

        # initialize sequence object with starting sequence (required for next state)
        # notes = np.zeros((1,self.hparams.input_len,hparams.input_size))
        # TODO: initialize sequence of notes from a file 
        note_sequence = Sequence(epsilon=self.hparams.epsilon)
        for i in range(self.hparams.input_len):
            row = np.zeros(self.hparams.input_size)
            row[np.random.randint(60,72)] = 1
            note_sequence.add(row)

        # # ## Random sampling
        # self.sample_sequence(sess,note_sequence,num_notes,argmax=False)
        # # return sequence object / write to file 
        # note_sequence.many_hot_to_midi(save_midi_path)

        # Beam search implementation
        result = self.beam_search_sequence(sess,note_sequence,num_notes,2)
        score,sequence,state = result
        print("Beam searched sequence of length {} with log likelihood {}".format(
            len(sequence), score
        ))
        sequence.many_hot_to_midi(save_midi_path)

        # matrix_to_midi(notes.reshape(notes.shape[1],notes.shape[2]), save_midi_path)

###############################################################################