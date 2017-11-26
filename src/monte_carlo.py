import os,sys
import glob
import numpy as np
import random
import pdb

from collections import Counter, defaultdict, namedtuple
from .midi import midi_to_matrix, matrix_to_midi, midi_to_matrix_quantized
from . import pretty_midi
from . import constants

###############################################################################

# State object
# State tuples defined as: 
# (duration,pitch)
# duration: float expressed as fraction of quarter note
# pitch: int MIDI pitch 0-127, or -1 (if silence)
MarkovState = namedtuple('MarkovState',['duration','pitch'])

class MarkovSequence(object):
    """
    Maintains an internal list of MarkovState objects. 
    Functions to convert to and from MIDI format
    """
    def __init__(self,epsilon=1.0/4,maxPause=4):
        self.sequence = []
        self.epsilon = epsilon # default 16th note i.e. 1/4 quarter notes 
        self.maxPause = maxPause # default 1 bar i.e. 4 quarter notes

    def __len__(self):
        return len(self.sequence)

    def __iter__(self):
        for state in self.sequence:
            yield state

    def __getitem__(self,i):
        return self.sequence[i]

    def add(self,state):
        """
        @param state: MarkovState object
        """
        self.sequence.append(state)

    def __round_duration(self,duration):
        """
        Rounds off a duration as a multiple of epsilon (smallest beat)
        @param duration: expressed as a ratio of the beat/quarter note
        """
        return int(duration/self.epsilon)*self.epsilon

    def from_midi(self,midi_path):
        """
        Similar to functions in .midi, picks track with most notes and processes
        However, the representation here is a list of states. Each state can represent
        a (note/pause) along with its duration.
        Durations are calculated as a ratio of the beat (a quarter note), by roughy matching notes
        to beats (from pretty_midi) in case tempo changes midway 
        Checks for gaps between notes and adds those as states. If there are many concurrent notes/chords
        it will arbitrarily pick notes and make it monophonic. #TODO: this might be improvable

        @param midi_path: namesake, input file
        """
        # get data from MIDI file
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
        except:
            print ("Error processing file, MIDI output path: %s" % midi_path)
            return None
    
        # get the instruments that are not drums (if option is selected)
        note_list = map(lambda i: i.notes, midi_data.instruments)
    
        # pick the track with the most notes
        # notes: list of pretty_midi.Note objects
        notes = max(note_list, key=lambda notes:len(notes))       

        # list of times of beats throughout the piece. use to compute beat lengths
        # (may not be constant due to tempo changes)
        beats = midi_data.get_beats()

        # iterate through notes, and add states to sequence
        prev_note = None
        beat_idx = 1
        for curr_note in notes:
            # find closest beat 
            while beat_idx<(len(beats)-1) and beats[beat_idx]<curr_note.start:
                beat_idx += 1 
            beat_length = beats[beat_idx]-beats[beat_idx-1]

            if prev_note is None: # first note
                new_state = MarkovState(self.__round_duration((curr_note.end-curr_note.start)/beat_length),curr_note.pitch)
            else:
                gap = (curr_note.start-prev_note.end)/beat_length
                if gap >= self.epsilon: # check for a gap/pause, add to sequence
                    duration = self.__round_duration(gap) # rounding off in terms of epsilon
                    duration = min(self.maxPause,duration) # clip long pauses (date error from intermittent parts) 
                    pause_state = MarkovState(duration,-1) # use -1 to encode a pause
                    self.add(pause_state)
                if gap >= -self.epsilon: # skips concurrent notes
                    duration = self.__round_duration((curr_note.end-curr_note.start)/beat_length)
                    new_state = MarkovState(duration,curr_note.pitch)

            # ignore if duration goes down to 0. Can happen if there are notes smaller than epsilon 
            if new_state.duration>0:
                self.add(new_state)
            prev_note = curr_note

    def write_midi(self,midi_path,instrument_name = "Cello", beat_length = 0.5, velocity = 100):
        """
        As in .midi, converts sequence back to MIDI for output. 
        """
        midi_data = pretty_midi.PrettyMIDI()
        instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
        track = pretty_midi.Instrument(program=instrument_program)
    
        starting_time = 0
        for state in self.sequence:
            note_length = beat_length*state.duration # normalizing times w.r.t. quarter notes
            if state.pitch != -1:
                note = pretty_midi.Note(
                    velocity=velocity, pitch=state.pitch, start=starting_time, end=(starting_time+note_length))
                track.notes.append(note)
            starting_time += note_length 
    
        midi_data.instruments.append(track)
        midi_data.write(midi_path)       

class MonteCarlo(object):
    """
    Object is trained on a folder of MIDI files 
    Models p(s_{i+1} | s_i,s_{i-1},...)
    conditional - tuple of states after the |
    self.order: can set the number of states to include
    self.transitions: internally maintains transition counts as a dict of dicts 
                      key: conditional, value: dict{following_state:transition_count} 
    self.conditionals: counts number of times transition was made from conditional
                        used to normalize probability, for all possibilites in transitions

    Includes functions to train on a folder, and generate a melody random sampling from transitions
    """
    def __init__(self, folder,order=1,epsilon=1.0/4):
        self.FOLDER = os.path.join(folder,'')
        self.order = order
        self.epsilon = epsilon
        self.conditionals = defaultdict(float)
        self.transitions = defaultdict(Counter)

    def update_transition_counts(self,markov_sequence):
        for i in range(len(markov_sequence)-self.order):
            curr_state = markov_sequence[i+self.order] # curr state i+order
            # construct conditionals based on order e.g. order 2: (i,i+1) 
            conditional = tuple([markov_sequence[i+j] for j in range(self.order)])
            # update transitions dict
            self.conditionals[conditional] += 1
            self.transitions[conditional][curr_state] += 1

    def train(self):
        train_files = glob.glob(self.FOLDER + '*.mid')
        for path in train_files:
            print("Training on: {}".format(os.path.split(path)[1]))
            markov_sequence = MarkovSequence(self.epsilon)
            markov_sequence.from_midi(path)
            if markov_sequence is None:
                continue
            else:
                self.update_transition_counts(markov_sequence)
        # self.get_transition_probabilities()

    def generate_melody(self, output_path, note1=None, num_notes=100, flag=None):
        '''
        Generate "music (:P)" from a starting note. If no starting note is given,
        pick a random note in the middle octave and start from there. Use a note
        generated at every time step to generate more notes. Self sustaining.

        @param output_path: writes out the generated sequence to MIDI file if present
        @param note1: int
                Any number between 60 and 71, both inclusive (not using this for now, can be added)
        @param num_notes: int
                Any number greater than 0
        '''

        # Initialize with a randomly chosen conditional
        # TODO: Improve this initialization maybe to conditional with most possibilites
        conditional = random.choice(list(self.conditionals.keys()))
        # if note1 is None:
        #     note1 = random.randint(60, 71)
        # state = MarkovState(1.0,note1) # starting with a quarter note by default 

        # Create MarkovSequence, initialize with starting states
        kickass_melody = MarkovSequence(self.epsilon)
        for state in conditional: kickass_melody.add(state)

        # Random walk on transitions and append to sequence, time to generate!
        count = 0
        while count<num_notes:
            # Randomly sample a next state from transition distribution
            # TODO - akash: check if conditional was never seen before (below values will all be None) 
            potential_new_states = list(self.transitions[conditional].keys())
            nTransitions = self.conditionals[conditional]
            prob_distribution = [ _/nTransitions for _ in self.transitions[conditional].values() ]

            # check if only one transition has been seen so far
            if len(prob_distribution)==1:
                print("Warning: Picked next state deterministically. Check counts.")
                next_state = potential_new_states[0]
            else:
                # pdb.set_trace()
                idxList = range(len(prob_distribution))
                next_state = potential_new_states[np.random.choice(idxList,p=prob_distribution)] # some numpy issue workaround

            if flag == 'restrict': # check if constraints satisfied
                prev_state = conditional[self.order-1]
                chosenFlag = (abs(next_state.pitch-prev_state.pitch) <= constants.OCTAVE)
            else:
                chosenFlag = True

            if chosenFlag: # if satisfactory, update conditional and iterate next
                kickass_melody.add(next_state)
                count += 1
                # shifting left by 1 and adding in next_state
                conditional = [ conditional[1+j] for j in range(self.order-1) ]
                conditional.append(next_state)
                conditional = tuple(conditional)

        # Based on optional flag, write the sequence to midi file
        if output_path is not None:
            kickass_melody.write_midi(output_path)

        return kickass_melody
    
    # def read_midi(self, path):
    #     midi_matrix = midi_to_matrix(path, join_chords=False)
    #     # midi_matrix = midi_to_matrix_quantized(path, join_chords=False)
    #     return midi_matrix
    
    # def update_transition_counts(self, midi_matrix):
    #     for i in range(len(midi_matrix) - 1):
    #         state = np.argmax(midi_matrix[i,:])
    #         state_pair = (np.argmax(midi_matrix[i,:]), np.argmax(midi_matrix[i+1,:]))
    #         self.states[state] += 1
    #         self.transitions[state_pair] += 1
    
    # def get_transition_probabilities(self):
    #     for state_pair in self.transitions:
    #         state = state_pair[0]
    #         self.T[state_pair] = self.transitions[state_pair]/self.states[state]   

    # def output_midi(self, kickass_melody, output_path):
    #     matrix_to_midi(matrix=kickass_melody, out_midi_path=output_path)
    
    # def get_sample_music(self, output_path, note1=None, num_notes=100):
    #     kickass_melody = self.generate_melody(note1=note1, num_notes=num_notes)
    #     self.output_midi(kickass_melody, output_path)

################################################################################