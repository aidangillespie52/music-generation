import torch
from torch.utils.data import Dataset

class MusicSequenceDataset(Dataset):
    def __init__(self, notes, sequence_length):
        self.sequence_length = sequence_length
        self.notes = self.filter_sequences(notes)

    def filter_sequences(self, notes):
        # Remove the sequences where one song is ending and another starts
        filtered_notes = []
        prev_start = None

        for note in notes:
            if prev_start is not None and note['start'] < prev_start:
                continue  # Skip this note if a sudden drop occurs
            
            filtered_notes.append(note)
            prev_start = note['start']

        return filtered_notes

    def __len__(self):
        return len(self.notes) - self.sequence_length  

    def __getitem__(self, idx):
        sequence = self.notes[idx:idx + self.sequence_length]
        target = self.notes[idx + self.sequence_length]

        X = torch.tensor([[note['start'], note['end'], note['pitch'], note['velocity']] for note in sequence], dtype=torch.float32)
        y = torch.tensor([target['start'], target['end'], target['pitch'], target['velocity']], dtype=torch.float32)
        
        return X, y