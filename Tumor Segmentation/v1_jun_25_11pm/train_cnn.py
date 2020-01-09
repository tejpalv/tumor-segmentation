import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import CNN

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 1
LOG_INTERVAL = 10

POSITIVE_DIRECTORY = "/Users/john/Documents/morteza-reborn/data/ship_vae"
NEGATIVE_DIRECTORY = "/Users/john/Documents/morteza-reborn/data/water_vae"

device = torch.device("cpu")

positive = datasets.ImageFolder(root=POSITIVE_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))

negative = datasets.ImageFolder(root=NEGATIVE_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))

positive_loader = torch.utils.data.DataLoader(dataset=positive, batch_size=BATCH_SIZE, shuffle=True)
negative_loader = torch.utils.data.DataLoader(dataset=negative, batch_size=BATCH_SIZE, shuffle=True)

model = CNN().to(device)

criterion = nn.MSELoss(size_average=True, reduce=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)


for epoch in range(EPOCHS):
    print(f"EPOCH {epoch}")
    correct = 0
    total = 0
    for batch_index, ((positive, _), (negative, __)) in enumerate(zip(positive_loader, negative_loader)):
        out_positive = model(positive)
        loss = criterion(out_positive, torch.ones(1))
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if out_positive > 0.5:
            correct += 1

        out_negative = model(negative)
        loss = criterion(out_negative, torch.zeros(1))
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if out_negative < 0.5:
            correct += 1

        total += 2

        if batch_index % LOG_INTERVAL == 0:
            print(f"Train Epoch: {epoch} [{batch_index * len(positive)} / {len(positive_loader.dataset)}] - Loss: {loss.item()}")
            print(f"ACCURACY: {100 * correct / total } %")

    torch.save(model, "cnn.pt")


    

