import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from clickref.model import Click_ref
import torch
from torch.utils.data import DataLoader
from clickref.dataloader import CustomImageDataset
from utils import *
import sys
import argparse

parser = argparse.ArgumentParser(
    description="Code to get the results of the clickref model "
)
parser.add_argument(
    "-b",
    "--baseline",
    help="Select the baseline model you want to compare the results with can be 3 types: stardist,hovernet or maskrcnn",
    type=str,
)

parser.add_argument(
    "-s",
    "--size_erase",
    help="select the threshold you want to use in order to not consider nuclei with area <threshold>",
    type=int,
)


args = parser.parse_args()

if args.baseline == "stardist":
    path_baseline = path_stardist
elif args.baseline == "hovernet":
    path_baseline = path_hovernet
elif args.baseline == "maskrcnn":
    path_baseline = path_maskrcnn
else: 
    print("Invalid baseline model")
    sys.exit()

size_erase = args.size_erase
print('processing baseline from {}...'.format(path_baseline))
click_ref = Click_ref(3, 3).cuda()
click_ref.load_state_dict(
    torch.load(path_weights_clickref)
)

dataset_test = CustomImageDataset(
    path_baseline=path_baseline,
    path_images=path_images,
    path_gt=path_gt,
    dataframe=df_test,
    augmenter_bool=False,
)

loader_test = DataLoader(
    batch_size=parameters["batch_size"],
    dataset=dataset_test,
    shuffle=False,
)

print('Prediction of the model...')
arr_baselines, arr_gts, arr_preds, arr_clicks = test_clickref(click_ref, loader_test)

print('Computing table of results...')
get_results(2, size_erase, arr_baselines, arr_gts, arr_preds, arr_clicks)