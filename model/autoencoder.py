import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.embedding_dim = 32
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, self.embedding_dim),
            )

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_size),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    # def encode(self, x):
    #     with torch.no_grad():
    #         model.eval()

