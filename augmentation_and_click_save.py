import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.augment_merge_split import *
from config import *
from multiprocessing import Pool
from utils.create_gt_grids import Gtgrid
from utils.grids_to_clicks import Grid_to_click
import os
import tifffile
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


args = parser.parse_args()

if args.baseline == "stardist":
    path_baseline = path_stardist
    path_baseline_modified =path_stardist_modified
elif args.baseline == "hovernet":
    path_baseline = path_hovernet
elif args.baseline == "maskrcnn":
    path_baseline = path_maskrcnn
else: 
    print("Invalid baseline model")
    sys.exit()

M = ModifyImg()


def change_mask_and_save(filename):
    """Augment baseline segmentation by generating merge and split and save files"""

    image = tifffile.imread(os.path.join(path_baseline, "baseline", filename))

    patch_new, contour_new = M.modify(image)
    binary = (patch_new > 0).astype(int)
    tifffile.imsave(os.path.join(path_baseline_modified, "binary", filename), binary)
    tifffile.imsave(
        os.path.join(path_baseline_modified, "baseline", filename), patch_new
    )
    tifffile.imsave(
        os.path.join(path_baseline_modified, "contour", filename),
        contour_new,
    )


def generate_clicks_and_save(filename):
    """We created new baseline segmentation, we then have to create new Ground truth clicks"""

    img_gt = tifffile.imread(os.path.join(path_gt, "baseline", filename))
    img_baseline = tifffile.imread(os.path.join(path_baseline, "baseline", filename))
    grid_creator = Gtgrid(img_gt, img_baseline, area=0)
    grid = grid_creator.create_grid()
    click_generator = Grid_to_click(grid, filename, path_baseline)
    click = click_generator.final_click()
    tifffile.imsave(os.path.join(path_baseline, "click", filename), click)

# if path_baseline==path_stardist:
#     print('modifying the segmentation and saving the new segmentation...')
#     if __name__ == "__main__":
#         pool = Pool(processes=16)
#         pool.map(change_mask_and_save, os.listdir(os.path.join(path_baseline,'baseline')))

# print('generating clicks...')
# if __name__ == "__main__":
#     pool = Pool(processes=16)
#     pool.map(
#         generate_clicks_and_save, os.listdir(os.path.join(path_baseline, "baseline"))
#     )
print('generating clicks from {}...'.format(path_baseline))

for filename in os.listdir(os.path.join(path_baseline, "baseline")):
    generate_clicks_and_save(filename)