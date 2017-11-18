import numpy as np

###############################################################################

class PinkNoise:
    

notes = list(range(128))

def pink_noise(num_notes=100):
    frequencies = [16.35]
    for i in range(127):
        frequencies.append(frequencies[-1]*1.06)
    
    probabilities = [1/f for f in frequencies]
    sump = sum(probabilities)
    prob = [p/sump for p in probabilities]
    