import numpy as np
import pretty_midi

NUM_POSSIBLE_NOTES = 128

'''
Creates a matrix from a MIDI file.
Takes a MIDI file, and converts one of the tracks in the file
  to the matrix specified below.

@param midi_path -> path to midi file to convert
@param join_chords ->
  if true, makes the one hot np sub-arrays represent chords with multiple ones
  if false, assumes each note happens independent of chords and returns one-hot vectors

@return an array of size (num_chords_in_MIDI_track x NUM_POSSIBLE_NOTES)
'''
def midi_to_matrix(midi_path, join_chords = True):
  # get data from midi file
  midi_data = pretty_midi.PrettyMIDI(midi_path)
  
  # pick the instrument that is not a drum that has the most notes
  non_drum_tracks = filter(lambda i: not i.is_drum, midi_data.instruments)
  if not non_drum_tracks:
    raise Exception("midi file does not contain any non-drum tracks:\n\t %s" % midi_path)
  track = max(non_drum_tracks, key=lambda i:len(i.notes))

  if not join_chords:
    return np.array([note_2_vec(note.pitch) for note in track.notes], np.int32)

  '''
  The rest of this code is now for merging chords in vectors.
  '''
  epsilon = 1e-3
  chord_notes_counted = 0

  # make a matrix assuming no chords
  input_matrix = np.zeros((len(track.notes), NUM_POSSIBLE_NOTES), np.int32)

  # assuming the last note is not in a chord with the first note
  for note_index, note_data in enumerate(track.notes):

    # non-chord case
    if note_index == 0 or abs(previous_note_data.start - note_data.start) > epsilon:
      input_matrix[note_index - chord_notes_counted] = note_2_vec(note_data)
    else: # chord case
      chord_notes_counted += 1
      input_matrix[note_index - chord_notes_counted] = \
        note_2_vec(note_data, input_matrix[note_index - chord_notes_counted])

    previous_note_data = track.notes[note_index - chord_notes_counted]

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

@param matrix -> input note matrix of shape (time_steps x NUM_POSSIBLE_NOTES)
@param out_midi_path -> where to output the MIDI file
@param instrument_name -> instrument used in one-track MIDI output
@param note_length -> length of each note in output
@param velocity -> velocity of each output note
'''
def matrix_to_midi(matrix, out_midi_path, instrument_name="Cello", note_length = 0.5, velocity = 100):
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
