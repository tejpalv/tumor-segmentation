import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

BATCH_SIZE = 10
EPOCHS = 1
LOG_INTERVAL = 10

device = torch.device("cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, 20)
        self.fc22 = nn.Linear(512, 20)
        self.fc3 = nn.Linear(20, 512)
        self.fc4 = nn.Linear(512, 1024)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_index, (data, label) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        loss = loss_function(reconstruction, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_index % LOG_INTERVAL == 0:
            print(f"Train Epoch: {epoch} [{batch_index * len(data)} / {len(train_loader.dataset)} ] - Loss: {loss.item() / len(data)} ")

    print(f"====> Epoch: {epoch} Average Loss: {train_loss / len(train_loader.dataset)}")


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for index, (data, label) in enumerate(test_loader):
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            test_loss += loss_function(reconstruction, data, mu, logvar).item()
            if index == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      reconstruction.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss}")

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

