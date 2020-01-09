import torch
from torch import nn
from torch.nn import functional as F

### CLASSIFIER used in train_cnn.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3, 1)
        self.bn_1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, 3, 1)
        self.bn_2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(3240, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn_1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.bn_2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 3240)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

### TRANSFORMER model used in transform.py (however, this should be replaced with a simple backprop on the latent vector itself...)
class TRANSFORMER(nn.Module):
    def __init__(self):
        super(TRANSFORMER, self).__init__()
        self.transformer = nn.Sequential(
            nn.Linear(300,300),
        )
        
    def forward(self, x):
        x = self.transformer(x)
        return x

### Autoencoder used in train_ae.py
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),     
        )
        
        self.z_enter = nn.Linear(12800, 300)
        self.z_develop = nn.Linear(300, 12800)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, output_padding=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, output_padding=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, output_padding=0, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        output_encoder = self.encoder(x).view(-1, 12800)
        z = self.z_enter(output_encoder)
        return z

    def decode(self, z):
        reconstruction = self.decoder(self.z_develop(z).view(-1, 32, 20, 20))
        return reconstruction

    def forward(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z