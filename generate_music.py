from funcs import*

import sys
sys.path.append('..\\model')

#C:\Program Files\FFmpeg\bin

import rnn

import pretty_midi
from midiutil import MIDIFile

TRAINLIMIT = 1000 
LEARNING_RATE = 1e-2
LIMITDATA = 30
TEMPERATURE = 6


# Extract notes from MIDI file
all_musics = [extract_notes_from_midi(f"..\\data\\training_music\\{i}.mid") for i in [1,1] ]
created_music = []


# Mapping notes to indices
notes = []
for n in all_musics:
    notes += n

notes = set(notes)
note_to_index = { note:i for i,note in enumerate(notes) }
index_to_note = { i:note for i,note in enumerate(notes) }

input_size = len(notes)
output_size = len(notes)

model = rnn.model(input_size,output_size, temperature = TEMPERATURE )

for sequence_notes in all_musics:
    # Create sequences and targets
    sequence_length = 20  # Set the desired length of each sequence
    inputs, targets = generate_sequences_and_targets(sequence_notes, sequence_length)

    # Map the inputs and targets using this dictionary
    mapped_inputs = [[note_to_index[note] for note in seq] for seq in inputs][-LIMITDATA:]
    mapped_targets = [[note_to_index[note] for note in seq] for seq in targets][-LIMITDATA:]


    # Convert targets to one-hot encoding
    one_hot_targets = [one_hot_encode(seq, output_size) for seq in mapped_targets]

    # Flatten inputs and one-hot targets for training
    flattened_inputs = [note for seq in mapped_inputs for note in seq]
    flattened_targets = [target for seq in one_hot_targets for target in seq]

    # Train the model
    model.train(flattened_inputs, flattened_targets, TRAINLIMIT, LEARNING_RATE)

    #generate music
    length = 100

    np.random.seed(None)

    start_note = note_to_index[list(notes)[np.random.randint(len(notes))]]

    x = np.zeros((1, input_size))
    x[0][start_note] = 1
        
    generated_sequence = [index_to_note[start_note]]
    for _ in range(length):
        y_pred, _ = model.forward_propagation(x)
        probabilities = y_pred.flatten()
        next_note_idx = np.random.choice(len(probabilities), p=probabilities)
        next_note = index_to_note[next_note_idx]
        generated_sequence.append(next_note)
        x = np.zeros((1, input_size))
        x[0][next_note_idx] = 1
    
    created_music.append(generated_sequence)

"""Save to file""" 
instruments = []

#intruments .midi

#bass_channel = 2
#bass_instrument = 32
#midi.addProgramChange(track, bass_channel, time, bass_instrument)
#
#flute_channel = 3
#flute_instrument = 73
#midi.addProgramChange(track, flute_channel, time, flute_instrument)


save_midi_to_file(created_music[-2:])

