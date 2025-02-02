import torch.nn as nn

class MusicLSTMModel(nn.Module):
    """
    input_size: refers to the start, end, pitch and velocity that the model gets as input
    hidden_size: number of neurons in the hidden layers
    num_layers: the number of hidden layers in the model
    output_size: outputs the next predicted note in the form of start, end, pitch and velocity
    """
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=4):
        super(MusicLSTMModel, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map the hidden state to the output
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Pass through LSTM layer
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Only take the output of the last time step
        last_lstm_output = lstm_out[:, -1, :]
        
        # Pass the last output through a fully connected layer
        output = self.fc(last_lstm_output)
        
        return output