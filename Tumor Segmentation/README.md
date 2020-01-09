# morteza

**current version is v2**

## setup steps:

Data is from https://www.kaggle.com/rhammell/ships-in-satellite-imagery

There are five folders that you need to setup in v2/data. EACH ONE is formatted like "data/ship_raw/train/<PUT IMGS HERE>" 

1) in ship_raw make sure to put the 1000 ship images from the dataset

2) in water_raw make sure to put all the water images you have and duplicate thgem until you have exactly 1000

3) In both_raw_balanced, combine ship_raw and water_raw (2000 total)

4) install requirements.txt by doing `pip install -r requirements.txt`

5) run train_vae.py until finished (feel free to play around with hyperparameters), model will be saved in /saves

6) after vae is finished training, look in /results to make sure the samples are good. then run `generate_vae_data` to plug data into the ship_vae and water_vae folders

7) run `python train_cnn.py` which will train the cnn and save the model in /saves

8) run `python transform.py` to generate 1 pair using our method


## hyperparameter log:

medium-success in v1 using:

**train_cnn.py**

- BATCH_SIZE = 1
- LEARNING_RATE = 1e-4
- EPOCHS = 1
- LOG_INTERVAL = 10

`self.conv1 = nn.Conv2d(3, 5, 3, 1)
        self.bn_1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 10, 3, 1)
        self.bn_2 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(3240, 10)
        self.fc2 = nn.Linear(10, 1)`

**train_ae.py**

BATCH_SIZE = 5
LEARNING_RATE = 1e-4  
EPOCHS = 31
LOG_INTERVAL = 60

 `self.encoder = nn.Sequential(
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
`

**main.py**

BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-3

`def gv_loss(z_no_lesion, z_with_lesion):
    mse_loss = nn.MSELoss(reduce=True)
    nl = vae.decode(z_no_lesion)
    wl = vae.decode(z_with_lesion)
    gv = gv_margin(mse_loss(z_no_lesion, z_with_lesion), torch.tensor([1]).float())
    cnn_loss = 0.1 * cnn(nl)
    print(f"GV: {gv.data}  CNN: {cnn_loss.data}")
    loss =  gv + cnn_loss
    return loss, nl, wl`

    







