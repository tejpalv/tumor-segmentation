import torch
from torch import nn
from torch.nn import functional as F

# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         self.fc1 = nn.Linear(3072, 512)
#         self.fc21 = nn.Linear(512, 20)
#         self.fc22 = nn.Linear(512, 20)
#         self.fc3 = nn.Linear(20, 512)
#         self.fc4 = nn.Linear(512, 3072)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x.view(-1, 3072)))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3)).view(-1, 3, 32, 32)

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 1024))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), z, mu, logvar


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


class TRANSFORMER(torch.nn.Module):
    def __init__(self):
        super(TRANSFORMER, self).__init__()
        self.transformer = nn.Sequential(
            nn.Linear(300,300),
        )
        
    def forward(self, x):
        x = self.transformer(x)
        return x


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
        output_encoder = self.z_enter(output_encoder)
        return output_encoder

    def decode(self, z):
        g = self.decoder(self.z_develop(z).view(-1, 32, 20, 20))
        return g

    def forward(self, x):
        output_encoder = self.encode(x)
        g = self.decode(output_encoder)
        return g, output_encoder



class LINEAR_AE(nn.Module):
    def __init__(self):
        super(LINEAR_AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(6400,1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            nn.Linear(1600,1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            nn.Linear(1600,1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            nn.Linear(1600,300),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(300,1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            nn.Linear(1600,1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            nn.Linear(1600,6400),
            nn.BatchNorm1d(6400),
            nn.ReLU(),
        )


    def forward(self, x):
        x = self.encoder(x.view(-1, 6400))
        x = self.decoder(x).view(-1, 80, 80)
        return x



