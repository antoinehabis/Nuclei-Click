import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from config import *
from model import Click_ref
import neptune
from tiloss import Finalloss
from dataloader import *
import torch

final_loss = Finalloss()
click_ref = Click_ref(7, 3)
click_ref.load_state_dict(torch.load(path_weights_click_ref))
optimizer = torch.optim.Adam(click_ref.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


run = neptune.init_run(
    project="aureliensihab/deep-icy",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjdkOTI0Yy1iOGJkLTQyMzEtYmEyOC05MmFmYmFhMWExNTMifQ==",
)
run["config/Hyperparameters"] = parameters
run["config/optimizer"] = "Adam"


def train(model, optimizer, train_dl, val_dl, epochs=300):
    tmp = (torch.ones(1) * 1e15).cuda()
    for epoch in range(1, epochs + 1):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        model.cuda()
        loss_tot = 0.0
        num_train_correct = 0
        num_train_examples = 0

        for batch in train_dl:
            optimizer.zero_grad()

            images = batch[0].cuda()
            stardists = batch[1].cuda()
            outputs = batch[2].cuda()

            pred_outputs = model(images, stardists)
            loss = final_loss(pred_outputs, outputs)
            loss_tot = loss
            # backward
            loss_tot.backward()
            optimizer.step()
            run["train/epoch/loss_tot"].log(loss_tot)

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

                pred_outputs = model(images, stardists)
                loss = final_loss(pred_outputs, outputs)
                val_loss_tot = loss
                mean += val_loss_tot

                run["test/epoch/loss_tot"].log(val_loss_tot)
            mean = torch.mean(mean)
            if torch.gt(tmp, mean):
                print("the val loss decreased: saving the model...")
                tmp = mean
                torch.save(model.state_dict(), path_weights_click_ref)
    return "Training done: the model was trained for " + str(epochs) + " epochs"


train(click_ref, optimizer, dataloaders["train"], dataloaders["val"], epochs=500)
