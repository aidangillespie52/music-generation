import torch
from torch.utils.data import Dataset

class MusicSequenceDataset(Dataset):
    def __init__(self, notes, sequence_length):
        self.sequence_length = sequence_length
        self.sequences = self.filter_sequences(notes)

    def valid_sequence(self, start, seq):
        for note in seq:
            if note['start'] < start:  # Detected a sudden drop
                return False
            else:
                start = note['start']

        return True
        
    def filter_sequences(self, notes):
        """Remove entire sequences that contain a drop in start time."""
        res = []

        for i, note in enumerate(notes[:len(notes) - self.sequence_length]):
            start = note['start']
            seq = notes[i:i+self.sequence_length]
            valid_seq = self.valid_sequence(start, seq)
            
            if valid_seq:
                target = notes[i + self.sequence_length]
                X = torch.tensor([[note['start'], note['end'], note['pitch'], note['velocity']] for note in seq], dtype=torch.float32)
                y = torch.tensor([target['start'], target['end'], target['pitch'], target['velocity']], dtype=torch.float32)
                res.append((X, y))

        return res
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return X, y