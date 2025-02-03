import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class MusicSequenceDataset(Dataset):
    def __init__(self, notes, sequence_length):
        self.sequence_length = sequence_length

        # Filter sequences after checking for valid start time drops
        self.sequences = self.filter_sequences(notes)
    
    def valid_sequence(self, start, seq):
        """Check if the sequence has strictly increasing start times."""
        for note in seq:
            if note['start'] < start:  # Detected a sudden drop
                return False
            else:
                start = note['start']
        return True

    def filter_sequences(self, notes):
        """Remove entire sequences that contain a drop in start time."""
        valid_sequences = []

        for i in range(len(notes) - self.sequence_length):
            seq = notes[i:i + self.sequence_length]
            start = seq[0]['start']
            if self.valid_sequence(start, seq):
                target = notes[i + self.sequence_length]
                X = torch.tensor([[note['start'], note['end'], note['pitch'], note['velocity']] for note in seq], dtype=torch.float32)
                y = torch.tensor([target['start'], target['end'], target['pitch'], target['velocity']], dtype=torch.float32)
                valid_sequences.append((X, y))

        return valid_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return X, y
