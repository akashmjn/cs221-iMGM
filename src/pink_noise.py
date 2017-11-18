import numpy as np

from midi import matrix_to_midi

###############################################################################

class PinkNoise:
    def __init__(self, num_notes=100):
        self.n = num_notes
        frequencies = [261.63]
        for i in range(35):
            frequencies.append(frequencies[-1]*1.06)
        probabilities = [1/f for f in frequencies]
        sump = sum(probabilities)
        self.prob = [p/sump for p in probabilities]
        self.notes = list(range(48,84))
    
    def generate_melody(self):
        notes = []
        for _ in range(self.n):
            note = np.random.choice(self.notes, p=self.prob)
            row = np.zeros(128,)
            row[note] = 1
            notes.append(row)
        notes = np.vstack(notes)
        return notes
    
    def output_midi(self, notes, output_path):
        matrix_to_midi(matrix=notes, out_midi_path=output_path)
    
    def get_sample_music(self, output_path):
        notes = self.generate_melody()
        self.output_midi(notes=notes, output_path=output_path)

###############################################################################