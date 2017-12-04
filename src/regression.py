import glob
import math
import numpy as np
import os
import random
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestClassifier as RFC

from .midi import midi_to_matrix, matrix_to_midi
from collections import defaultdict

###############################################################################

class Classification:
    def __init__(self, input_len=5, num_epochs=10):
        self.input_len = input_len
        self.num_epochs = 10
        self.model = RFC(warm_start=True)
    
    def fit(self, midi_folder):
        files = glob.glob(midi_folder + '*.mid')
        for _ in range(self.num_epochs):
            for file in files:
                matrix = midi_to_matrix(file, join_chords=False)
                if matrix is None or len(matrix)<self.input_len+1:
                    continue
                else:
                    X = []
                    y = []
                    for i in range(len(matrix)-self.input_len):
                        X.append(np.argmax(matrix[i:i+self.input_len,:], axis=1))
                        y.append(np.argmax(matrix[i+self.input_len,:]))
                    X = np.vstack(X)
                    y = np.hstack(y)
                    self.model.fit(X, y)
    
    def generate(self, num_notes, save_midi_path, stats_path):
        notes = np.random.randint(36, 85, self.input_len)
        for _ in range(num_notes):
            input = notes[-self.input_len:]
            label = np.around(self.model.predict(input.reshape(1, len(input))))
            notes = np.hstack((notes, label))
        midi_matrix = np.zeros((len(notes), 128))
        midi_matrix[range(len(notes)), notes] = 1
        matrix_to_midi(midi_matrix, save_midi_path)
        # Stats
        stats = defaultdict(int)
        for note in notes:
            stats[note] += 1
        with open(stats_path, 'w') as f:
            for key in stats:
                f.write(str(key) + ': ' + str(stats[key]) + '\n')

###############################################################################