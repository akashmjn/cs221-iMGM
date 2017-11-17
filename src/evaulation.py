import numpy as np
import pretty_midi

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
