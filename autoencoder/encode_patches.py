import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import tifffile
from config import *
from torch.utils.data import Dataset, DataLoader
import torch
from unet import UNet
from tqdm import tqdm
from glob import glob

model = UNet(3, 3)
model.load_state_dict(torch.load(path_weights_autoencoder))
model.cuda()


class CustomImageDataset(Dataset):

    def __init__(self, path_images, rot=0):

        self.path_images = path_images
        self.rot = rot

    def __getitem__(self, idx):

        path = glob(self.path_images + "/*tif")[idx]

        """read image"""
        image = tifffile.imread(path) / 255
        np.rot90(image, k=self.rot)
        input_ = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return torch.tensor(input_)

    def __len__(self):

        return len(glob(self.path_images + "/*tif"))


def get_encodings(model, loader):
    list_encoders = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            inputs = batch.cuda()
            encoders = model(inputs)[-1]
            list_encoders.append(encoders.cpu().detach().numpy())

    array_encoders = np.concatenate(list_encoders)
    return array_encoders


encodings = np.zeros((len(glob(path_twins + "/*tif")), 256, 4))

for rot in tqdm(range(encodings.shape[-1])):
    dataset = CustomImageDataset(path_images=path_twins, rot=rot)
    loader = DataLoader(batch_size=32, dataset=dataset, num_workers=8, shuffle=False)
    encodings[:, :, rot] = get_encodings(model, loader)
    np.save(os.path.join(path_encodings, "encodings256.npy"), encodings)
