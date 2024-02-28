import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import tifffile
from config import *
from dataloader import CustomImageDataset
from torch.utils.data import Dataset, DataLoader
import torch
from unet import UNet
from tqdm import tqdm
from glob import glob
import argparse

model = UNet(3, 3)
model.load_state_dict(torch.load(path_weights_autoencoder))
model.cuda()

parser = argparse.ArgumentParser(
    description="Code to get the results of the clickref model "
)
parser.add_argument(
    "-f",
    "--folder",
    help="type of images you want to encode: can be twins or test",
    type=str,
)

args = parser.parse_args()

if args.folder == "twins":
    encodings = np.zeros((len(glob(path_twins + "/*tif")), parameters['n_embedding'], 4))
    df = pd.DataFrame(os.listdir(path_twins), columns = ['filename'])
    path_images = path_twins


elif args.folder == "test":
    encodings = np.zeros(df_test.shape[0], parameters['n_embedding'], 4)
    df = df_test
    path_images = path_images

else: 
    print("Invalid type of data to encode: can be either twins or test")
    sys.exit()

def get_encodings(model, loader):
    list_encoders = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].cuda()
            encoders = model(inputs)[-1]
            list_encoders.append(encoders.cpu().detach().numpy())

    array_encoders = np.concatenate(list_encoders)
    return array_encoders


encodings = np.zeros((len(glob(path_twins + "/*tif")), parameters['n_embedding'], 4))

df = pd.DataFrame(os.listdir(path_twins), columns = ['filename'])
for rot in tqdm(range(encodings.shape[-1])):
    dataset = CustomImageDataset(path_images=path_twins,dataframe = df, rot=rot)
    loader = DataLoader(batch_size=32, dataset=dataset, num_workers=32, shuffle=False)
    encodings[:, :, rot] = get_encodings(model, loader)
    np.save(os.path.join(path_encodings, "encodings"+str(parameters['n_embedding'])+"_"+str(parameters['contractive'])+".npy"), encodings)
