# CS 221 - Artificial Intelligence - Final Project

Music generation from public MIDI datasets using Markov chains, LSTM language models + sampling, beam search inference. 

### Project Structure:

**src/:** 
  - sequence: class for dealing with MIDI and converting to useful representation used by models
  - monte_carlo: class for markov-chain model
  - basic_rnn, bidirectional_rnn, sequence_rnn, stacked_rnn: various model classes
  - evaluation: utilities for computing basic statistics on generate sequences 
  
**runner.py:** Runs scripts for training, inference from saved checkpoints:

*Training:*
```
python3 runner.py -m trnn -i ../data/BachChorales/ -o models/sequence_rnn_BachChorales_128/ --lr 0.0005 --epochOffset 0 --inputLen 1 --layerSize 128 --nepochs 50
```
*Generation:*
```
python3 tester.py -m grnn -i models/sequence_rnn_BachChorales_128/epoch_60/checkpoint.ckpt -o ../outputs/sequence_RNN_BachChorales/sequence_rnn_128_60.mid --inputLen 1 --layerSize 128 --lr 0
```

### Reports

See our write-ups for more information.
- [Poster](reports/poster.pdf)
- [Project Report](reports/final.pdf)

### Contributors
- Akash Mahajan - Masters, Management Science and Engineering
- Suraj Heereguppe - Masters, Institute for Computational and Mathematical Engineering
- Nathan Dalal - Undergraduate, Computer Science
