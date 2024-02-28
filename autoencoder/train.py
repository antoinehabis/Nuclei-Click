import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from unet import UNet
import torch
from dataloader import *
from augmentation import augmentation
import neptune
from loss_contractive import loss_function
from config import parameters

run = neptune.init_run(
    mode='offline',
    project="aureliensihab/deep-icy",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjdkOTI0Yy1iOGJkLTQyMzEtYmEyOC05MmFmYmFhMWExNTMifQ==",
)  # your credentials

params = {
    "learning_rate": parameters["lr"],
    "optimizer": "Adam",
    "project": "autoencoder",
}

run["parameters"] = params

model = UNet(3, 3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=parameters["lr"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")
loss_contractive = loss_function


def train(model, optimizer, train_dl, val_dl, loss_contractive, loss_mse, epochs=100):
    tmp = (torch.ones(1) * 1e15).cuda()

    for epoch in range(1, epochs + 1):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        model.cuda()
        loss_tot = 0.0

        for batch in train_dl:
            inputs = batch[0].cuda()
            outputs = batch[1].cuda()

            inputs.requires_grad_(True)
            pred_outputs, _, _, _, _, encoding = model(inputs)
            loss_tot = loss_contractive(
                encoding, pred_outputs, inputs, lamda=parameters['contractive'], device=torch.device("cuda")
            )
            inputs.requires_grad_(False)

            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()
            run["train/epoch/loss"].log(loss_tot)  # backward
            torch.save(model.state_dict(), os.path.join(path_pannuke,"weights_autoencoder_CAE"+str(parameters['contractive'])+"_"+str(parameters['n_embedding'])))
    return 0


train(
    model,
    optimizer,
    loader_train,
    loader_val,
    loss_contractive=loss_function,
    loss_mse=loss,
    epochs=4000,
)
