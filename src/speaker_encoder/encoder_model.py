"""
File: encoder_model.py
Purpose: Defines the architecture for the speaker encoder. 
         Learns an embedding that captures speaker identity.
"""
import torch
import torch.nn as nn

class SpeakerEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Example: Simple feedforward or LSTM
        self.lstm = nn.LSTM(input_size=40, hidden_size=embed_dim, batch_first=True)
    
    def forward(self, x):
        # x shape: (batch, time, features)
        _, (h, _) = self.lstm(x)
        return h[-1]
