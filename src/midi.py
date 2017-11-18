'''
@author Nathan Dalal
A demo is commented here for example usage.

def demo():
    merge_tracks('data/example.mid', 'data/merged.mid')
    answer = midi_to_matrix('data/merged.mid')
    matrix_to_midi(answer, 'data/output.mid')
'''
import numpy as np
import pretty_midi

# There are 128 possible notes in a MIDI file.
NUM_POSSIBLE_NOTES = 128

'''
Calls merge_tracks() on each MIDI file in input_dir and places into output_dir.

@param input_dir -> path to folder containing MIDI file to convert
@param out_midi_path -> path to folder which will contain outputted MIDI files
@param recursive -> whether to recursively search through directories
@param out_instrument_name -> instrument of the output file track
@param include_drums -> allows filtering out of drum and percussion based instruments
@param num_tracks -> should be set to zero, do not change -> important for labeling files in recursion

@return number of tracks merged
'''
def merge_many_tracks(input_dir, output_dir, recursive = True, merged_filename_label = "merged" ,out_instrument_name = "Cello", include_drums = True, num_tracks = 0):
    import os
    for root, dirs, filenames in os.walk(input_dir):
        for f in filenames:
            if f.endswith(".mid"):
                input_file_path = os.path.join(input_dir, f)
                output_file_path = os.path.join(output_dir, str(num_tracks) + "_" + f[:-4] + "_" + merged_filename_label + ".mid")
                try:
                    merge_tracks(input_file_path, output_file_path, out_instrument_name, include_drums)
                    num_tracks += 1
                    print ("merged successfully, MIDI output path: %s" % output_file_path)
                except:
                    print ("FAILURE: merging failed on following MIDI input path: %s" % input_file_path)
        if recursive:
            for next_dirname in dirs:
                num_tracks += merge_many_tracks(os.path.join(input_dir, next_dirname), output_dir, \
                    recursive, merged_filename_label, out_instrument_name, include_drums, num_tracks)

    return num_tracks

'''
Merges all tracks of one MIDI file and outputs a MIDI file with just one track.

This function has no return value and puts the MIDI file 
  in the output path specified.

@param in_midi_path -> path to MIDI file to convert
@param out_midi_path -> path to MIDI file where the output is written
@param out_instrument_name -> instrument of the output file track
@param include_drums -> allows filtering out of drum and percussion based instruments
'''
def merge_tracks(in_midi_path, out_midi_path, out_instrument_name = "Cello", include_drums = True):
    midi_data = pretty_midi.PrettyMIDI(in_midi_path)
    notes = sum(map(lambda i: i.notes, filter(lambda i: include_drums or not i.is_drum, midi_data.instruments)), [])
    notes.sort(key=lambda note: note.start)

    if not notes:
        raise Exception(("empty MIDI file %s" % in_midi_path) if include_drums else ("no non-drum tracks or empty MIDI file %s" % in_midi_path))

    out_midi_data = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(out_instrument_name)
    track = pretty_midi.Instrument(program=instrument_program)

    track.notes = notes
    out_midi_data.instruments.append(track)
    out_midi_data.write(out_midi_path)

'''
Creates a matrix from a MIDI file.
Takes a MIDI file, and converts one of the tracks in the file
  to the matrix specified below.

@param midi_path -> path to MIDI file to convert
@param join_chords ->
  if true, makes the one hot np sub-arrays represent chords with multiple ones
  if false, assumes each note happens independent of chords and returns one-hot vectors

@return a numpy matrix of shape (num_chords_in_MIDI_track x NUM_POSSIBLE_NOTES)
'''
def midi_to_matrix(midi_path, join_chords = True):
    # get data from MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)

    # get the instruments that are not drums (if option is selected)
    note_list = map(lambda i: i.notes, midi_data.instruments)

    # pick the track with the most notes
    notes = max(note_list, key=lambda notes:len(notes))

    if not join_chords:
        return np.array([note_2_vec(note.pitch) for note in notes], np.int32)

    '''
    The rest of this code is now for merging chords in vectors.
    '''
    epsilon = 0.01
    chord_notes_counted = 0

    # make a matrix assuming no chords
    input_matrix = np.zeros((len(notes), NUM_POSSIBLE_NOTES), np.int32)

    # fill in matrix with chord information
    for note_index, note_data in enumerate(notes):
        current_note_index = note_index - chord_notes_counted

    # non-chord case
        if note_index == 0 or abs(previous_note_data.start - note_data.start) > epsilon:
            input_matrix[current_note_index] = note_2_vec(note_data)
            previous_note_data = note_data
        else: # chord case
            input_matrix[current_note_index] = note_2_vec(note_data, input_matrix[current_note_index - 1])
            chord_notes_counted += 1

    input_matrix = input_matrix[:input_matrix.shape[0] - chord_notes_counted]
    return input_matrix

'''
Helper function.
Creates a one-hot 1D np array of the note passed in.

@param note_data -> instance of pretty_midi.Note
@param vec a previous vector (this allows us to store chords in one vector)

@return -> one-hot vector of size 128 (or many-hot if vec is not None)
'''
def note_2_vec(note_data, vec = np.array([])):
    if not vec.any():
        vec = np.zeros(NUM_POSSIBLE_NOTES, np.int32)
    vec[note_data.pitch] = 1
    return vec

'''
Creates a MIDI file from a matrix of notes,
  where shape of input is (time_steps x NUM_POSSIBLE_NOTES).

The output of the file has no temporal knowledge of notes besides chords so
  it simply outputs a MIDI file with a 
  default instrument, default note length in seconds, and default velocity.

This function has no return value and puts the matrix 
  as a MIDI file in the path specified.

@param matrix -> input note numpy matrix of shape (time_steps x NUM_POSSIBLE_NOTES)
@param out_midi_path -> where to output the MIDI file
@param instrument_name -> instrument used in one-track MIDI output
@param note_length -> length of each note in output
@param velocity -> velocity of each output note
'''
def matrix_to_midi(matrix, out_midi_path, instrument_name = "Cello", note_length = 0.5, velocity = 100):
    midi_data = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(instrument_name)
    track = pretty_midi.Instrument(program=instrument_program)

    starting_time = 0
    for chord in matrix:
        for pitch in np.nonzero(chord)[0]:
            note = pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=starting_time, end=starting_time + note_length)
            track.notes.append(note)
        starting_time += note_length

    midi_data.instruments.append(track)
    midi_data.write(out_midi_path)

merge_many_tracks('data/raw/', 'data/merged/')
