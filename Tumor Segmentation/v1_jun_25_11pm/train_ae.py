import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import AE

BATCH_SIZE = 5
LEARNING_RATE = 1e-4  
EPOCHS = 31
LOG_INTERVAL = 60

POSITIVE_DIRECTORY = "/Users/john/Documents/morteza-reborn/data/both"

device = torch.device("cpu")

positive = datasets.ImageFolder(root=POSITIVE_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))

positive_loader = torch.utils.data.DataLoader(dataset=positive, batch_size=BATCH_SIZE, shuffle=True)

model = AE().to(device)

criterion = nn.MSELoss(size_average=True, reduce=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(EPOCHS):
    for batch_index, (positive, _1) in enumerate(positive_loader):
        if positive.shape[0] < BATCH_SIZE: break
        out_positive = model(positive)
        loss = criterion(out_positive, positive)
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()


        if batch_index % LOG_INTERVAL == 0:
            print(f"Train Epoch: {epoch} [{batch_index * len(positive)} / {len(positive_loader.dataset)}] - Loss: {loss.item()}")
            n = min(positive.size(0), 8)
            comparison = torch.cat([positive[:n],
                                    out_positive.view(BATCH_SIZE, 3, 80, 80)[:n]])
            save_image(comparison.cpu(),
                        f"results/reconstruction_positive_{str(epoch)}_{str(batch_index)}.png", nrow=n)



torch.save(model, 'vae.pt')


   