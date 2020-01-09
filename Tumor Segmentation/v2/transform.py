import torch, torch.utils.data, os, random, string
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import AE, CNN, TRANSFORMER

### HYPERPARAMETERS
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-3
SHIP_DIRECTORY = os.getcwd() + "/data/ship_vae"

### INIT DATA
ship_vae = datasets.ImageFolder(root=SHIP_DIRECTORY, transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))
ship_vae_loader = torch.utils.data.DataLoader(dataset=ship_vae, batch_size=BATCH_SIZE, shuffle=True)


### HELPER FUNCS
def gv_margin(x, margin): #guibas-virdi margin, simply minimizes the floor of how low a value can go
    return torch.max(margin, x)

def gv_loss(z_no_lesion, z_with_lesion): #MSE(Z_NL, Z_WL) + CNN(DECODE(Z_NL))
    mse_loss = nn.MSELoss(reduce=True)
    nl = vae.decode(z_no_lesion)
    wl = vae.decode(z_with_lesion)
    gv = gv_margin(mse_loss(z_no_lesion, z_with_lesion), torch.tensor([1]).float())
    cnn_loss = 0.1 * cnn(nl)
    print(f"GV: {gv.data}  CNN: {cnn_loss.data}")
    loss =  gv + cnn_loss
    return loss, nl, wl

### MODEL
device = torch.device("cpu")
transform = TRANSFORMER().to(device)
cnn = torch.load("saves/cnn.pt")
vae = torch.load("saves/vae.pt")

criterion = nn.MSELoss(size_average=True, reduce=True)
optimizer = torch.optim.Adam(transform.parameters(), lr=LEARNING_RATE, amsgrad=True)

### PROPAGATE
for (ship, _) in ship_vae_loader:
    reconstruction, z_with_lesion = vae(ship) #taking a positive, extracting the recon and latent z

    for x in range(100):
        z_no_lesion = transform(z_with_lesion) #transforming latent z (lesion) => latent z (no lesion)
        loss, decoded_nl, decoded_wl = gv_loss(z_no_lesion, z_with_lesion)
        optimizer.zero_grad()
        loss.sum().backward(retain_graph=True)
        optimizer.step()
        
        print("LOSS:", loss.sum().data)

    negative = vae.decode(z_no_lesion)
    g = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    save_image(reconstruction, f"results/groundtruth_{g}.png", )
    save_image(negative, f"results/reconstructed_negative_{g}.png")

    exit() #remove this if you want to generate more than one





