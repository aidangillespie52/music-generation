import torch
from torch.utils.data import Dataset

# TODO: Remove sequences in the dataset where the start and end values suddenly drop like when a song ends and another one starts

class MusicSequenceDataset(Dataset):
    def __init__(self, notes, sequence_length):
        self.notes = notes  # list of lists of dictionaries (list of sequences of notes)
        self.sequence_length = sequence_length  # length of the input sequence

    def __len__(self):
        return len(self.notes) - self.sequence_length  # Total number of sequences available

    def __getitem__(self, idx):
        # Get a sequence of notes (list of dictionaries)
        sequence = self.notes[idx:idx + self.sequence_length]
        target = self.notes[idx + self.sequence_length]

        # The sequence should now be a flat list of dictionaries, create the tensor for X
        X = torch.tensor([[note['start'], note['end'], note['pitch'], note['velocity']] for note in sequence], dtype=torch.float32)
        
        # The target note is the next note's features
        y = torch.tensor([target['start'], target['end'], target['pitch'], target['velocity']], dtype=torch.float32)
        
        return X, y