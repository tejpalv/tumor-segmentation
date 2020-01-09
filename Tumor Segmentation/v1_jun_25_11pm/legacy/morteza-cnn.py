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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(12500, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 12500)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss = nn.MSELoss()

def train(epoch):
    model.train()
    for batch_index, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        for i in range(len(label)):
            if label[i] % 2 == 0:
                label[i] = 0
            else:
                label[i] = 1
        label = label.float()
        optimizer.zero_grad()
        output = model(data)
        # print(f"OUTPUT: {output.data} LABEL: {label.data}")
        train_loss = loss(output, label)
        train_loss.backward()
        optimizer.step()
        if batch_index % LOG_INTERVAL == 0:
            print(f"Train Epoch: {epoch} [{batch_index * len(data)} / {len(train_loader.dataset)}] - Loss: {train_loss.item()}")
            


train(0)