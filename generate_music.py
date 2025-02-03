from MusicGenerationModel import MusicLSTMModel
import torch
import pretty_midi

# Load the model
model = MusicLSTMModel()
model.load_state_dict(torch.load('models/model.pth'))
model.train()

# Function to generate music
def generate_music(model, initial_notes, num_notes, sequence_length=8):
    notes_sequence = initial_notes[0]
    
    # Generate new notes until we have the required number of notes
    for i in range(num_notes - len(initial_notes)):
        # Create input tensor from the last `sequence_length` notes
        sequence = notes_sequence[i:i+sequence_length].unsqueeze(0) # add the batch dimension
        print(sequence)
        
        # Predict the next note
        with torch.no_grad():
            predicted_note = model(sequence)

        print(predicted_note)

        # Extract the predicted note (start, end, pitch, velocity)
        predicted_note_dict = {
            'start': predicted_note[0].item(),
            'end': predicted_note[1].item(),
            'pitch': int(predicted_note[2].item()),
            'velocity': int(predicted_note[3].item())
        }
        
        # Append the predicted note to the sequence
        notes_sequence.append(predicted_note_dict)
    
    return notes_sequence

# Function to convert the list of notes to a MIDI file
def notes_to_midi(notes, output_file='generated_music.mid'):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.program_to_instrument(0))  # 0 is the piano program
    
    for note in notes:
        midi_note = pretty_midi.Note(
            velocity=note['velocity'],
            pitch=note['pitch'],
            start=note['start'],
            end=note['end']
        )
        instrument.notes.append(midi_note)
    
    midi.instruments.append(instrument)
    midi.write(output_file)
    print(f'MIDI file saved to {output_file}')

# Initial notes to start the sequence (replace with your own starting notes)'
#rnd_idx = random.randint(0, len(dataset)-1)

initial_notes = dataset[872]

# Generate music with the model
num_notes = 200  # Desired number of notes to generate
generated_notes = generate_music(model, initial_notes, num_notes)

# Convert the generated notes to a MIDI file
notes_to_midi(generated_notes, 'generated_music.mid')
