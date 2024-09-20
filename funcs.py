import os
import numpy as np
import librosa
from pydub import AudioSegment
from pydub import AudioSegment
from midi2audio import FluidSynth
import pretty_midi
from midiutil import MIDIFile

def get_unique_filename(filename):
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    return  new_filename

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def generate_sequences_and_targets(data, sequence_length=5):
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        sequence = data[i :i + sequence_length]
        target = data[i + 1:i + sequence_length+1]
        

        # Append -1 to the target sequence if it is shorter than the sequence length
        if len(target) < sequence_length:
            target.append(-1)
        
        sequences.append(sequence)
        targets.append(target)
    
    return sequences, targets

def extract_notes_from_midi(filename: str):
    notes = []
    midi_data = pretty_midi.PrettyMIDI(filename)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append(note.pitch)
    return notes
    
def frequency_to_midi_note(frequency):
    return int(round(69 + 12 * np.log2(frequency / 440.0)))

def extract_notes_from_mp3(filename:str):
    audio = AudioSegment.from_mp3(f"{filename}")
    audio.export("converted_file.wav", format="wav")

    y, sr = librosa.load("converted_file.wav")

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    ret = []
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            ret.append( pitch )
            
    return ret

def save_midi_to_file(note_sequences,filename = "generated_music.midi", tempo = 120, track = 0, channel = 0, time = 0, duration = .7, volume = 100):

    filename = get_unique_filename(filename)
    midi = MIDIFile(1)
    midi.addTempo(track, time, tempo)
    
    piano_channel = 0
    piano_instrument = 0  
    midi.addProgramChange(track, piano_channel, time, piano_instrument)


    violin_channel = 1
    violin_instrument = 25
    midi.addProgramChange(track, violin_channel, time, violin_instrument)
    
    
    for note1,note2 in zip(note_sequences[0],note_sequences[1]) :
        midi.addNote(track, piano_channel, note1, time, duration, volume)
            
        midi.addNote(track, violin_channel, note2, time, duration, int(volume*0.5))
        time += duration

    with open(filename, "wb") as output_file:
        midi.writeFile(output_file)   

def save_midi(note_sequences, filename="rock_guitar_music.midi", tempo=120, duration_per_note=0.7):
    """
    Save note sequences as a MIDI file with a rock guitar instrument.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=30)  # Program 30 = Distortion Guitar

    time = 0
    for note in note_sequences:
        midi_note = pretty_midi.Note(velocity=100, pitch=int(note), start=time, end=time + duration_per_note)
        instrument.notes.append(midi_note)
        time += duration_per_note

    midi.instruments.append(instrument)
    midi.write(filename)
    print(f"MIDI file saved to {filename}")  
        
def midi_to_mp3(midi_filename, mp3_filename, soundfont_path="guitar.sf2"):
    """
    Convert MIDI to MP3 using a rock guitar SoundFont.
    """
    fs = FluidSynth(soundfont_path)
    fs.midi_to_audio(midi_filename, mp3_filename)
    print(f"MP3 file saved to {mp3_filename}")
