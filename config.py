import os
import pandas as pd

path = os.getenv('DATA_PATH')
parameters = {}
parameters["dim"] = 256
parameters["batch_size"] = 32
parameters["lr"] = 1e-4
parameters["n_embedding"] = 512
parameters['contractive'] = 1e-5

#####   PANNUKE DATASET   #####
path_DATA = os.getenv('PATH_DATA')
path_pannuke = os.path.join(path_DATA, 'pannuke')
path_fold1 = os.path.join(path_pannuke, "fold_ train1")
path_fold2 = os.path.join(path_pannuke, "fold_2")
path_fold3 = os.path.join(path_pannuke, "fold_3")

path_mask_fold1 = os.path.join(path_fold1, "masks/masks.npy")
path_mask_fold2 = os.path.join(path_fold2, "masks/masks.npy")
path_mask_fold3 = os.path.join(path_fold3, "masks/masks.npy")

path_image_fold1 = os.path.join(path_fold1, "images/images.npy")
path_image_fold2 = os.path.join(path_fold2, "images/images.npy")
path_image_fold3 = os.path.join(path_fold3, "images/images.npy")

path_images = os.path.join(path_pannuke, "images")
path_gt1 = os.path.join(path_pannuke, "data_gt", "baseline")
path_twins = os.path.join(path_pannuke, "data_twins")
###############################
path_click_project = "."

path_encodings = os.path.join(path_pannuke,'encodings')
path_gt = os.path.join(path_pannuke, "data_gt")
path_stardist = os.path.join(path_pannuke, "data_stardist")
path_stardist_modified = os.path.join(path_pannuke, "data_stardist_modified")
path_maskrcnn = os.path.join(path_pannuke, "data_maskrcnn")
path_hovernet = os.path.join(path_pannuke, "data_hovernet")
path_weights_clickref = os.path.join(path_pannuke, "weights_clickref", "weights")
path_split = os.path.join(path_click_project, "split_train_val_test")


df_train = pd.read_csv(os.path.join(path_split, "train_df.csv"), index_col=0)
df_test = pd.read_csv(os.path.join(path_split, "test_df.csv"), index_col=0)
df_val = pd.read_csv(os.path.join(path_split, "val_df.csv"), index_col=0)

path_weights_autoencoder = os.path.join(path_pannuke,"weights_autoencoder_CAE"+str(parameters['contractive'])+"_"+str(parameters['n_embedding']))


path_baseline = path_maskrcnn