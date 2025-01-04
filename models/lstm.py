import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, batch_first=True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first
        )

        self.fc = nn.Linear(hidden_size, 1)

        self.criterion == nn.L1Loss()

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)

        Returns:
            predictions of shape (batch_size) or (batch_size, 1)
        """

        # hidden is tuple of (h_n, c_n) for final hidden, final cell states
        output, (h_n, c_n) = self.lstm(x)

        # Often we use the last time-step's output for prediction:
        # output[:, -1, :] is the hidden state at the final time step
        # shape: (batch_size, hidden_size)
        out = self.fc(output[:, -1, :])


        out = out.squeeze(-1)

        return out