from unet import UNet
import torch
from .dataloader import *
from .augmentation import augmentation
import neptune
from .loss_contractive import loss_function

run = neptune.init_run(
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")
model.load_state_dict(torch.load("weights_autoencoder_CAE"))
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
                encoding, pred_outputs, inputs, lamda=1e-5, device=torch.device("cuda")
            )
            inputs.requires_grad_(False)

            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()
            run["train/epoch/loss"].log(loss_tot)  # backward

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss_tot = 0.0
        num_val_correct = 0
        num_val_examples = 0

        mean = torch.zeros(1).cuda()
        with torch.no_grad():
            for batch in val_dl:
                optimizer.zero_grad()
                inputs = batch[0].cuda()
                outputs = batch[1].cuda()
                pred_outputs = model(inputs)[0]
                val_loss_tot = loss_mse(pred_outputs, outputs)

                run["validation/epoch/loss"].log(val_loss_tot)
                mean += val_loss_tot

            mean = torch.mean(mean)

            if torch.gt(tmp, mean):
                print("the val loss decreased: saving the model...")
                tmp = mean
                torch.save(model.state_dict(), "weights_autoencoder_CAE")
    return 0


train(
    model,
    optimizer,
    loader_train,
    loader_val,
    loss_contractive=loss_function,
    loss_mse=loss,
    epochs=300,
)
