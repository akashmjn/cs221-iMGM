import numpy as np
from collections import defaultdict
import pdb
from . import constants

## TODO: Handle all the license stuff, when taking code from magenta/models/rl_tuner

## Idea of rough evaluation rules that can be used (search for Music Theory Rewards):
# https://magenta.tensorflow.org/2016/11/09/tuning-recurrent-networks-with-reinforcement-learning/
# https://github.com/tensorflow/magenta/blob/master/magenta/models/rl_tuner/rl_tuner_eval_metrics.py

class MelodyEvaluator(object):
    # Holds a dict object with stats after evaluating a melody
    # @param root: MIDI int representation of root note, used to calculate major scale  
    def __init__(self,root):
        self.root = root
        self.eval_stats = defaultdict(float)
        self.constants = constants

    def evaluate_melody(self,melody_matrix):
        self.eval_in_key_stat(melody_matrix)
        self.eval_interval_stat(melody_matrix)

    """
    Computes a score of fraction of notes played in key
    @param melody_matrix: assumed to be matrix of one-hot vectors (nx128)
    @param root: MIDI int representation of root note, used to calculate major scale  
    """
    def eval_in_key_stat(self,melody_matrix,nOctaves=1):
        # number of valid notes encountered 
        numNotes = 0
        scale = self.compute_scale(nOctaves)
        # pdb.set_trace()
        for i in range(melody_matrix.shape[0]):
            noteVec = melody_matrix[i,:]
            noteIdx = np.argmax(noteVec)
            # not a blank note, update total count 
            if noteVec[noteIdx]!=0:
                numNotes += 1

            if noteIdx in scale:
                self.eval_stats['frac_notes_in_key'] += 1
        self.eval_stats['frac_notes_in_key'] /= numNotes

    def compute_scale(self,nOctaves,type='major'):
        """
        Returns a list with MIDI int notes for scale starting at root
        NOTE: currently by default uses major, can add more patterns if we want
        @param nOctaves: int total spread of Octaves required on either side i.e. for nOctaves=1 root-12 : root+12
        """       
        scale = set() 
        if type=='major':
            pattern = self.constants.MAJOR_PATTERN*nOctaves*2
            # get lowest note, iterate len(pattern)*nOctaves*2 times
            currNote = self.root-nOctaves*self.constants.OCTAVE
            scale.add(currNote)
            # pdb.set_trace()
            for i in range(len(pattern)):
                currNote += pattern[i]
                scale.add(currNote)
        return scale

    def eval_interval_stat(self, melody_matrix):
        """
        Computes the melodic interval just played and adds it to a stat dict.

        A dictionary of composition statistics with fields updated to include new
        intervals.
        """
        self.eval_stats['interval_stats'] = defaultdict(int) 
        self.eval_stats['raw_intervals'] = defaultdict(int)

        nIntervalJumps = 0
        for i in range(1,melody_matrix.shape[0]):
            currNoteVec,prevNoteVec = (melody_matrix[i,:], melody_matrix[i-1,:] )
            currNote,prevNote = (np.argmax(currNoteVec), np.argmax(prevNoteVec) )
            # pdb.set_trace()

            # skip all the interval updating if either are blank, or no jump
            interval = abs(currNote-prevNote)
            if currNoteVec[currNote]==0 or prevNoteVec[prevNote]==0 or interval == 0:
                continue
            else: nIntervalJumps += 1

            # update the stats of raw intervals
            self.eval_stats['raw_intervals'][interval] += 1

            # update stats for commonly used intervals
    
            # not using magenta special intervals for now (not sure what they mean)
            # if interval == REST_INTERVAL:
            #     interval_stats['num_rest_intervals'] += 1
            # elif interval == REST_INTERVAL_AFTER_THIRD_OR_FIFTH:
            #     interval_stats['num_special_rest_intervals'] += 1
            if interval >= self.constants.OCTAVE:
                self.eval_stats['interval_stats']['num_octave_jumps'] += 1
                # elif interval == (IN_KEY_FIFTH or
                #     interval == IN_KEY_THIRD):
                # self.eval_stats['interval_stats']['num_in_key_preferred_intervals'] += 1
            elif interval == self.constants.FIFTH:
                self.eval_stats['interval_stats']['num_fifths'] += 1
            elif interval == self.constants.THIRD:
                self.eval_stats['interval_stats']['num_thirds'] += 1
            elif interval == self.constants.SIXTH:
                self.eval_stats['interval_stats']['num_sixths'] += 1
            elif interval == self.constants.SECOND:
                self.eval_stats['interval_stats']['num_seconds'] += 1
            elif interval == self.constants.FOURTH:
                self.eval_stats['interval_stats']['num_fourths'] += 1
            elif interval == self.constants.SEVENTH:
                self.eval_stats['interval_stats']['num_sevenths'] += 1

        # Normalizing as % of total jumps made
        # pdb.set_trace()
        for key,value in self.eval_stats['interval_stats'].items():
            self.eval_stats['interval_stats'][key] = value/nIntervalJumps
        for key,value in self.eval_stats['raw_intervals'].items():
            self.eval_stats['raw_intervals'][key] = value/nIntervalJumps

    ### Chord-based evaluation scores ###

    def major_minor_chord_evaluation(midi_matrix, step_size = 5):
    	pass
    
    def major_minor_chord_jump_evaluation(midi_matrix, start_index, end_index):
    	longest_major_chain, longest_minor_chain = 0, 0
    	current_major_chain, current_minor_chain = 0, 0
    
    	for chord in midi_matrix[start_index:end_index]:
    		pass
    
    def is_major_chord(chord):
        # get a list of unique base notes (ranges from 0 - 11 for C up to B)
        base_notes = sorted(set([note_pitch % 12 for note_pitch in np.nonzero(chord)[0]]))
    
        # same note or only one note is major (I chord)
        if len(base_notes) == 1:
            return True
    
        # two notes must be IV, or V intervals
        if len(base_notes) == 2:
            return base_notes[1] - base_notes[0] in [5, 7]
    
        # three or more notes must contain 135 chord or 145 chord to be major
        for note in [0, 7]:
            if note not in base_notes:
                return False
        return (4 in base_notes) ^ (5 in base_notes)
