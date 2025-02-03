import os
import glob
import pretty_midi
import numpy as np
import torch
import json
from tqdm import tqdm

def has_data(filename):
    return os.path.exists(filename) and os.path.getsize(filename) > 0

def get_midi_files(base_dir):
    midi_files = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):  # Check if it's a directory
            midi_files.extend(glob.glob(os.path.join(folder_path, '*.midi')))  # Get all .midi files in the folder

    return midi_files

def process_midi_file(midi_file):
    midi = pretty_midi.PrettyMIDI(midi_file)
    
    piano = midi.instruments[0]

    # Extract notes: pitch, start time, end time, and velocity
    notes = []
    for note in piano.notes:
        notes.append({
            'start': note.start,
            'end': note.end,
            'pitch': note.pitch,
            'velocity': note.velocity
        })
    return notes

def save_in_chunks(chunk_data, filename):
    """ Save notes data incrementally in chunks. """
    if not os.path.exists(filename) or not has_data(filename):
        np.save(filename, chunk_data)
    else:
        existing_data = np.load(filename, allow_pickle=True).tolist()
        existing_data.extend(chunk_data)
        np.save(filename, existing_data)

def compute_incremental_stats(t, all_notes):
    mean = 0.0
    m2 = 0.0
    count = 0
    
    for note in all_notes:
        val = note[t]
        count += 1
        delta = val - mean
        mean += delta / count
        m2 += delta * (val - mean)  # This updates variance incrementally
    
    # Calculate standard deviation (sample std)
    std = float(torch.sqrt(torch.tensor(m2 / (count - 1))) if count > 1 else torch.tensor(0.0))

    print(f"Mean: {mean}, Std: {std}")  # Convert tensor to Python float
    return mean, std  # Ensure std is a native float

# ----------------------- Collect MIDI Files -----------------------
base_dir = 'data/maestro-v3.0.0'
midi_files = get_midi_files(base_dir)

print(f"Found {len(midi_files)} MIDI files.")

# ----------------------- Extract Data from MIDI Files -----------------------
all_notes = []
chunk_size = 50000

for i, midi_file in tqdm(enumerate(midi_files), total=len(midi_files)):
    notes = process_midi_file(midi_file)
    all_notes.extend(notes)

np.save('data/notes.npy', all_notes)

# ----------------------- Extract Normalization Data -----------------------
print('Calculating Stats')
start_mean, start_std = compute_incremental_stats('start', all_notes)
end_mean, end_std = compute_incremental_stats('end', all_notes)
pitch_mean, pitch_std = compute_incremental_stats('pitch', all_notes)
velocity_mean, velocity_std = compute_incremental_stats('velocity', all_notes)

normalization_stats = {
    'start_mean': start_mean,
    'start_std': start_std,
    'end_mean': end_mean,
    'end_std': end_std,
    'pitch_mean': pitch_mean,
    'pitch_std': pitch_std,
    'velocity_mean': velocity_mean,
    'velocity_std': velocity_std
}

filename = 'data/normalization_stats.json'

with open(filename, 'w') as f:
    json.dump(normalization_stats, f, indent=4)

print(f'Saved Normalization Stats into {filename}')

# ----------------------- Normalize MIDI Data and Save It -----------------------

# TODO: Raise error if file already exists

normalized_notes = []
for i, note in tqdm(enumerate(all_notes), total=len(all_notes)):
    normalized_note = {}
    normalized_note['start'] = (note['start'] - start_mean) / start_std
    normalized_note['end'] = (note['end'] - end_mean) / end_std
    normalized_note['pitch'] = (note['pitch'] - pitch_mean) / pitch_std
    normalized_note['velocity'] = (note['velocity'] - velocity_mean) / velocity_std
    normalized_notes.append(normalized_note)
    if i % 50000 == 0 and i != 0:
        save_in_chunks(normalized_notes, 'data/normalized_notes.npy')
        normalized_notes = []

print('Normalized all notes')