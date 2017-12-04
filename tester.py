#/usr/bin/env python3

import os,sys
import glob
# from .monte_carlo import *
# from .evaluation import *
import src
import argparse
import time
import tensorflow as tf
from collections import defaultdict
from tensorflow.contrib.training import HParams
from collections import namedtuple
from basic_rnn_akash import RNNMusic


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

# input - path, suffix to search for, extension
def collectMIDIFiles(source_path,dest_path,suffix):
    if suffix:
        queryPath = os.path.join(source_path,'**','*'+suffix)
    generator = glob.iglob(queryPath,recursive=True)
    # Create subfolder for a suffix if required 
    os.system('mkdir -p '+dest_path)
    count = 0
    for file in generator:
        os.system('cp "{}" "{}"'.format(file,dest_path))
        count += 1
    print("Copied {} files into {}".format(count,dest_path))

def testRNNTrain(input_path,model_path,hparams):
    
    rnn_music = RNNMusic(hparams)
    graph = rnn_music.build_graph()
    with graph.as_default():
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            if hparams.epoch_offset==0:
                sess.run(init)
            else:
                restore_path = os.path.join(model_path,
                    "epoch_{}".format(hparams.epoch_offset),"checkpoint.ckpt")
                print("Retraining model from: "+restore_path)
                saver.restore(sess,restore_path)
            rnn_music.fit(sess, saver, input_path, model_path)   

def testRNNGenerate(model_path,output_path,hparams):
    
    rnn_music = RNNMusic(hparams)
    graph = rnn_music.build_graph()   
    with graph.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            rnn_music.generate(sess, 500, output_path)
  

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run tests on Markov Generator.')

    parser.add_argument('-m',dest='mode',default='mkv',
        help='Selects what scripts to run: io (read/write MIDI), \
        mkv (train generate melody), fmidi (to collect midi files), \
        trnn (train RNN on folder) or grnn (generate from saved RNN)')
    parser.add_argument('-i',dest='inpath',
        help='Input path: MIDI file for io, Folder for mkv')   
    parser.add_argument('-o',dest='outpath',
        help='Output path: Folder for io, MIDI file for mkv')      
    parser.add_argument('--order',dest='order',type=int,default=1,
        help='Order of markov process. Problematic beyond 2')         
    parser.add_argument('--suffix',dest='suffix',default='*.mid',
        help='File suffix to collect. Defaults to *.mid')            
    parser.add_argument('--lr',dest='lr',type=float,
        help='Initial learning rate')                     
    parser.add_argument('--inputLen',dest='input_len',type=int,default=4,
        help='Number of x steps processed to generate output')                           
    parser.add_argument('--layerSize',dest='rnn_layer_size',type=int,default=36,
        help='Size of hidden layer h in RNN')                        
    parser.add_argument('--nepochs',dest='num_epochs',type=int,default=50,
        help='Number of epochs to start training from')                  
    parser.add_argument('--epochOffset',dest='epoch_offset',type=int,default=0,
        help='Offset of epochs to start training from')               

    args = parser.parse_args()
    print(args)
    if len(vars(args))== 0:
        raise Exception('Invalid arguments')

    hparams = HParams(input_size=129,input_len=args.input_len,rnn_layer_size=args.rnn_layer_size,
        lr=args.lr,num_epochs=args.num_epochs,epoch_offset=args.epoch_offset,epsilon=1.0/4)

    if args.mode=='io':
        # assumes inpath - to a specific file, outpath - folder 
        testAllIO(args.inpath,args.outpath)    
    elif args.mode=='mkv':
        # assumes inpath - to folder with files, outpath - outputfile
        testMonteCarlo(args.inpath,args.outpath,order=args.order)
    elif args.mode=='fmidi':
        collectMIDIFiles(args.inpath,args.outpath,args.suffix)
    elif args.mode=='trnn':
        testRNNTrain(args.inpath,args.outpath,hparams)       
    elif args.mode=='grnn':
        testRNNGenerate(args.inpath,args.outpath,hparams)
