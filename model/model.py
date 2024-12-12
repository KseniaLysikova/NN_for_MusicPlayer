import torch
import torch.nn as nn


class DynamicNsNet2(nn.Module):
    def __init__(self, num_features=257, hidden_size=400):
        super(DynamicNsNet2, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.gru1 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, num_features)

        self.exit_layers = nn.ModuleList([
            nn.Linear(num_features, num_features),
            nn.Linear(num_features, num_features),
            nn.Linear(num_features, num_features)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.fc1(x))

        x, _ = self.gru1(x)
        mask0 = torch.sigmoid(self.exit_layers[0](self.fc2(x)))

        x, _ = self.gru2(x)
        mask1 = torch.sigmoid(self.exit_layers[1](self.fc2(x)))

        x_fc2 = torch.relu(self.fc2(x))
        mask2 = torch.sigmoid(self.exit_layers[2](x_fc2))

        return mask0, mask1, mask2


def loss_fn(predicted, target, alpha=0.3, c=0.3):
    target = target.transpose(1, 2)

    magnitude_error = torch.mean((torch.abs(target)**c - torch.abs(predicted)**c)**2)
    complex_error = torch.mean((torch.pow(target, c) - torch.pow(predicted, c))**2)
    return alpha * complex_error + (1 - alpha) * magnitude_error


