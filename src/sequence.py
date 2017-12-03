from .constants import NUM_POSSIBLE_NOTES
import numpy as np
from . import pretty_midi

class Sequence(object):
    def __init__(self, epsilon = 1.0 / 4):
        self.sequence = []
        self.epsilon = epsilon # default 16th note i.e. 1/4 quarter notes

    def __len__(self):
        return len(self.sequence)

    def __iter__(self):
        for state in self.sequence:
            yield state

    def __getitem__(self,i):
        return self.sequence[i]

    def reset(self):
        self.sequence = []

    def add(self, row):
        self.sequence.append(row)

    def get_beat_range(self, note_data, beat_length):
        unit_length = beat_length * self.epsilon
        start_index = int(note_data.start / unit_length)
        end_index = int(note_data.end / unit_length)
        return (start_index, end_index)

    def from_midi(self, midi_path, join_tracks = True):
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
            return False

        # get all instruments
        note_list = map(lambda i: i.notes, midi_data.instruments)

        # pick the track with the most notes
        # notes: list of pretty_midi.Note objects
        if join_tracks:
            notes = sum(note_list, [])
        else:
            notes = max(note_list, key=lambda notes:len(notes))
        notes.sort(key=lambda note: note.start)

        # list of times of beats throughout the piece. use to compute beat lengths
        # (may not be constant due to tempo changes)
        beats = midi_data.get_beats()

        '''
        TODO - convert note list to many-hot vector representation
        '''
        beat_idx, beat_length = 1, None
        for note_data in notes:
            while beat_idx < (len(beats) - 1) and beats[beat_idx] < note_data.start:
                beat_idx += 1 
            beat_length = beats[beat_idx] - beats[beat_idx - 1]

            start_idx, end_idx = self.get_beat_range(note_data, beat_length)

            # Fill empty space with zero vectors.
            for i in range(len(self), start_idx):
                self.add(self.note_2_vec())

            # Add note data to each of the vectors from start to end index.
            for i in range(start_idx, end_idx):
                if i < len(self):
                    self.sequence[i] = self.note_2_vec(note_data, self.sequence[i])
                else:
                    self.add(self.note_2_vec(note_data))

        return True

    def note_2_vec(self, note_data = None, vec = np.array([])):
        if not vec.any():
            vec = np.zeros(NUM_POSSIBLE_NOTES, np.int32)
        if note_data:
            vec[note_data.pitch] = 1
        return vec

    def to_midi(self, midi_path, beat_length = 0.5, instrument_name = "Cello", velocity = 100, one_hot = False):
        """
        As in .midi, converts sequence back to MIDI for output.
        """
        midi_data = pretty_midi.PrettyMIDI()
        instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
        track = pretty_midi.Instrument(program=instrument_program)

        current_pitches = dict()

        starting_time = 0
        for many_hot_state in self.sequence:

            note_length = beat_length * self.epsilon # normalizing times w.r.t. quarter notes

            timestep_pitches = set(np.nonzero(many_hot_state)[0])
            if one_hot: # Pick the highest note to get a one-hot representation.
                timestep_pitches = set(max(timestep_pitches))
            processed_pitches = set()

            # Write notes that are current but no longer appear in the current timestep.
            for current_pitch in current_pitches:
                if current_pitch not in timestep_pitches:
                    current_note_start_time = current_pitches[current_pitch]
                    note = pretty_midi.Note(
                        velocity=velocity, pitch=current_pitch, start=current_note_start_time, end=starting_time)
                    track.notes.append(note)
                    processed_pitches.add(current_pitch)

            # Remove processed notes.
            for pitch_to_remove in processed_pitches:
                current_pitches.pop(pitch_to_remove, None)

            # Add notes we haven't seen in the previous timestep to track them.
            for timestep_pitch in timestep_pitches:
                if timestep_pitch not in current_pitches:
                    current_pitches[timestep_pitch] = starting_time

            starting_time += note_length

        # Fill in the remaining unprocessed pitches.
        for current_pitch in current_pitches:
            note = pretty_midi.Note(
                velocity=velocity, pitch=current_pitch, start=current_pitches[current_pitch], end=starting_time)
            track.notes.append(note)

        midi_data.instruments.append(track)
        midi_data.write(midi_path)

    def from_matrix(self, matrix):
        self.reset()
        for row in matrix:
            self.add(row[0])

    def to_matrix(self):
        return np.asmatrix(self.sequence)

def test_sequence_class(filepath = "../data/example.mid"):
    seq = Sequence()
    seq.from_midi(filepath)
    seq.to_midi(filepath[:-4] + "-merged.mid")

if __name__ == "__main__":
    test_sequence_class()
