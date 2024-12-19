import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims, dropouts=None):
        super().__init__()
        layers = []
        input_dim = input_size

        # Build the hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.BatchNorm1d(input_dim))

            if i > 0:
                layers.append(nn.ReLU())

            layers.append(nn.Linear(input_dim, hidden_dim))

            # Add dropout if specified
            if dropouts and i < len(dropouts):
                layers.append(nn.Dropout(dropouts[i]))

            input_dim = hidden_dim
            
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))

        self.model = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x).squeeze(-1)
