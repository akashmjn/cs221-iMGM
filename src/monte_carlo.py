import glob
import numpy as np
import random

from collections import defaultdict
from midi import midi_to_matrix

###############################################################################

class MonteCarlo:
    def __init__(self, folder):
        if folder[-1] == '/':
            self.FOLDER = folder
        else:
            self.FOLDER = folder + '/'
        self.states = defaultdict(float)
        self.transitions = defaultdict(float)
        self.T = defaultdict(float)
    
    def read_midi(self, path):
        midi_matrix = midi_to_matrix(path, join_chords=False)
        return midi_matrix
    
    def update_transition_counts(self, midi_matrix):
        for i in range(len(midi_matrix) - 1):
            state = np.argmax(midi_matrix[i,:])
            state_pair = (np.argmax(midi_matrix[i,:]), np.argmax(midi_matrix[i+1,:]))
            self.states[state] += 1
            self.transitions[state_pair] += 1
    
    def get_transition_probabilities(self):
        for state_pair in self.transitions:
            state = state_pair[0]
            self.T[state_pair] = self.transitions[state_pair]/self.states[state]
    
    def train(self):
        train_files = glob.glob(self.FOLDER + '*.mid')
        for path in train_files:
            midi_matrix = self.read_midi(path)
            self.update_transition_counts(midi_matrix)
        self.get_transition_probabilities()
    
    def generate_music(self, note1=None, num_notes=100):
        '''
        Generate "music (:P)" from a starting note. If no starting note is given,
        pick a random note in the middle octave and start from there. Use a note
        generated at every time step to generate more notes. Self sustaining.
        
        Parameter
        ---------
        note1: int
                Any number between 60 and 71, both inclusive
        num_notes: int
                Any number greater than 0
        '''
        if note1 is None:
            note1 = random.randint(60, 71)
        state = note1
        
        # Time to generate some shit!
        generated_notes = [state]
        for _ in range(num_notes):
            potential_new_states = [key[1] for key in self.T if key[0]==state]
            prob_distribution = [self.T[(state, new_state)] for new_state in potential_new_states]
            next_state = np.random.choice(potential_new_states, p=prob_distribution)
            generated_notes.append(next_state)
            state = next_state
        
        # Make the generated shit into one-hot vectors
        kickass_melody = []
        for val in generated_notes:
            one_hot = np.zeros(128,)
            one_hot[val] = 1
            kickass_melody.append(one_hot)
        kickass_melody = np.vstack(kickass_melody)
        