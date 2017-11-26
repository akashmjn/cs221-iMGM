import os,sys
# from .monte_carlo import *
# from .evaluation import *
import src
from collections import defaultdict

def testMonteCarlo(inpath,outpath,order=1):
    mcObj = src.monte_carlo.MonteCarlo(inpath,order=order,epsilon=1.0/8)
    mcObj.train() 
    melody = mcObj.generate_melody(outpath,note1=64,num_notes=500,flag=None)
    printSequenceStats(melody)
    return mcObj

def printSequenceStats(markov_sequence):
    pitchcounts = defaultdict(int)
    durationcounts = defaultdict(int)
    for elem in markov_sequence:
        pitchcounts[elem.pitch] += 1
        durationcounts[elem.duration] += 1    
    print("Pitch counts:")
    print(pitchcounts)
    print("Duration counts:")
    print(durationcounts)   

def testEvaluator(inpath,outpath,rootNote):
    eval  = MelodyEvaluator(rootNote)
    mcObj = MonteCarlo(inpath)
    mcObj.train()
    kickass_melody = mcObj.generate_melody(note1=rootNote,num_notes=200)
    eval.evaluate_melody(kickass_melody)
    for key,value in eval.eval_stats.items():
        print("{}: {}".format(key,value))
    matrix_to_midi(kickass_melody,outpath)

def pm_track_select_io(testpath,outpath,instrument_name = "Cello"):
    mdi = src.pretty_midi.PrettyMIDI(testpath)
    note_list = map(lambda i: i.notes, mdi.instruments)
    notes = max(note_list, key=lambda notes:len(notes))   

    outmdi = src.pretty_midi.PrettyMIDI()
    instrument_program = src.pretty_midi.instrument_name_to_program(instrument_name)
    track = src.pretty_midi.Instrument(program=instrument_program)   
    track.notes = notes
    outmdi.instruments.append(track)
    outmdi.write(outpath)

def midi_matrix_io(testpath,outpath):
    mat = src.midi.midi_to_matrix(testpath)
    src.midi.matrix_to_midi(mat,outpath)

def markov_io(testpath,outpath):
    mkv = src.monte_carlo.MarkovSequence(1.0/8)
    mkv.from_midi(testpath)
    mkv.write_midi(outpath)
    printSequenceStats(mkv)

def testAllIO(testpath,outpath):
    fname = 'pmio.mid'
    print("Running io with just pretty_midi")
    pm_track_select_io(testpath,os.path.join(outpath,fname))

    fname = 'midimat.mid'
    print("Running io with just internal functions")
    midi_matrix_io(testpath,os.path.join(outpath,fname))

    fname = 'mkvio.mid'
    print("Running io with just markov sequence reader")   
    markov_io(testpath,os.path.join(outpath,fname))

if __name__ == "__main__":

    if len(sys.argv)>=3:
        inpath,outpath = sys.argv[1],sys.argv[2]

    # assumes inpath - to a specific file, outpath - folder 
    # testAllIO(inpath,outpath)    

    # assumes inpath - to folder with files, outpath - outputfile
    testMonteCarlo(inpath,outpath,order=2)


