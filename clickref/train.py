import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from config import *
from model import Clickref
import neptune
from dataloader import dataloaders
import torch
from TopoInteraction.TI_Loss import final_loss

token = os.getenv("NEPTUNE_API_TOKEN")
project = os.getenv("NEPTUNE_WORKSPACE")

clickref = Clickref(3, 3)
optimizer = torch.optim.Adam(clickref.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
run = neptune.init_run(
    name="deep-icy",
    mode="offline",
    project=project,
    api_token=token,
)
run["config/Hyperparameters"] = parameters
run["config/optimizer"] = "Adam"


def train_cuda(model, optimizer, train_dl, val_dl, epochs=500):
    tmp = (torch.ones(1) * 1e10).cuda()
    for epoch in range(1, epochs + 1):
        if epoch < 250:
            ti_weights = 0.0
        else:
            ti_weights = 1e-5
        #     final_loss = Finalloss(dice_weight=1.0, ti_weight=1e-4)
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        model.cuda()
        loss = 0.0

        for batch in train_dl:
            optimizer.zero_grad()
            images = batch[0].cuda()
            stardists = batch[1].cuda()
            outputs = batch[2].cuda()
            clicks = batch[3].cuda()
            clicks_max, _ = torch.max(clicks, dim=1)
            pred_outputs = model(images, clicks, stardists)
            loss_click, loss_not_click, ti_loss = final_loss(
                outputs, pred_outputs, stardists, clicks_max
            )
            # backward
            # writer.add_scalar("Loss/train_click", loss_click, i)
            # writer.add_scalar("Loss/train_not_click", 1e-2 *loss_not_click, i)
            run["train/epoch/loss_click"].log(loss_click)
            run["train/epoch/loss_not_click"].log(1e-1 * loss_not_click)
            run["train/epoch/loss_ti"].log(ti_weights * ti_loss)
            loss = loss_click + 1e-2 * loss_not_click + ti_weights * ti_loss
            loss.backward()
            optimizer.step()
            run["train/epoch/loss"].log(loss)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss_tot = 0.0
        num_val_correct = 0
        num_val_examples = 0

        mean = torch.zeros(1).cuda()
        with torch.no_grad():
            for batch in val_dl:
                optimizer.zero_grad()
                images = batch[0].cuda()
                stardists = batch[1].cuda()
                outputs = batch[2].cuda()
                clicks = batch[3].cuda()
                clicks_max, _ = torch.max(clicks, dim=1)
                pred_outputs = model(images, clicks, stardists)
                loss_click, loss_not_click, ti_loss = final_loss(
                    outputs, pred_outputs, stardists, clicks_max
                )
                run["validation/epoch/loss_click"].log(loss_click)
                run["validation/epoch/loss_not_click"].log(1e-2 * loss_not_click)
                run["validation/epoch/loss_not_click"].log(1e-2 * loss_not_click)
                run["validation/epoch/loss_ti"].log(ti_weights * ti_loss)
        
                loss = loss_click + 1e-2 * loss_not_click + ti_weights * ti_loss
                run["validation/epoch/loss"].log(loss)
                mean += loss
            mean = torch.mean(mean)
            if torch.gt(tmp, mean):
                print("the val loss decreased: saving the model...")
                tmp = mean
                torch.save(model.state_dict(), path_weights_clickref)

    return "Training done: the model was trained for " + str(epochs) + " epochs"


train_cuda(clickref, optimizer, dataloaders["train"], dataloaders["val"], epochs=300)
