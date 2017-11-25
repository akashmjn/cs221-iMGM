import os,sys
from .monte_carlo import *
from .evaluation import *
from .midi import midi_to_matrix, matrix_to_midi, midi_to_matrix_quantized

def testMonteCarlo(inpath,outpath):
    mcObj = MonteCarlo(inpath)
    mcObj.train() 
    mcObj.generate_music(outpath,note1=64,num_notes=500)
    return mcObj

def testEvaluator(inpath,outpath,rootNote):
    eval  = MelodyEvaluator(rootNote)
    mcObj = MonteCarlo(inpath)
    mcObj.train()
    kickass_melody = mcObj.generate_melody(note1=rootNote,num_notes=200)
    eval.evaluate_melody(kickass_melody)
    for key,value in eval.eval_stats.items():
        print("{}: {}".format(key,value))
    matrix_to_midi(kickass_melody,outpath)

if __name__ == "__main__":

    if len(sys.argv)>=3:
        inpath,outpath = sys.argv[1],sys.argv[2]
    # testMonteCarlo(inpath,outpath)
    
    testEvaluator(inpath,outpath,60)
