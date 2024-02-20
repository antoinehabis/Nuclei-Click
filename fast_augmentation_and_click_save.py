from augment_merge_split import *
from config import *
from multiprocessing import Pool
from create_gt_grids import Gtgrid
from grids_to_clicks import Grid_to_click
import os
import tifffile

M = ModifyImg()


def change_mask_and_save(filename):
    """Augment baseline segmentation by generating merge and split and save files"""

    image = tifffile.imread(os.path.join(path_stardist, "baseline", filename))

    patch_new, contour_new = M.modify(image)
    binary = (patch_new > 0).astype(int)
    tifffile.imsave(os.path.join(path_stardist_modified, "binary", filename), binary)
    tifffile.imsave(
        os.path.join(path_stardist_modified, "baseline", filename), patch_new
    )
    tifffile.imsave(
        os.path.join(path_stardist_modified, "contour", filename),
        contour_new,
    )


def generate_clicks_and_save(filename):
    """We created new baseline segmentation, we then have to create new Ground truth clicks"""

    img_gt = tifffile.imread(os.path.join(path_gt, "baseline", filename))
    img_baseline = tifffile.imread(os.path.join(path_stardist_modified, "baseline", filename))
    grid_creator = Gtgrid(img_gt, img_baseline, area=0)
    grid = grid_creator.create_grid()
    click_generator = Grid_to_click(grid, filename, path_stardist_modified)
    click = click_generator.final_click()
    tifffile.imsave(os.path.join(path_stardist_modified, "click", filename), click)


# if __name__ == "__main__":
#     pool = Pool(processes=16)
#     pool.map(change_mask_and_save, os.listdir(os.path.join(path_stardist,'baseline')))


if __name__ == "__main__":
    pool = Pool(processes=16)
    pool.map(
        generate_clicks_and_save, os.listdir(os.path.join(path_stardist, "baseline"))
    )
