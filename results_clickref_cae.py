from clickref.model import Clickref
import torch
from autoencoder.dataloader import * 
from tqdm import tqdm
from utils import *
from find_same_nucleis import FindSame2
from grids_to_clicks import Grid_to_click
from matching import matching
from clickref.dataloader import CustomImageDataset
from beautifultable import BeautifulTable
from autoencoder.unet import UNet
import argparse
import neptune

token = os.getenv("NEPTUNE_API_TOKEN")
project = os.getenv("NEPTUNE_WORKSPACE")

run = neptune.init_run(
    name="deep-icy",
    mode="offline",
    project=project,
    api_token=token,
)
parser = argparse.ArgumentParser(
    description="Code to evaluate the performance of the overall method "
)
parser.add_argument(
    "-b",
    "--baseline",
    help="Select the baseline model you want to compare the results with can be 3 types: stardist,hovernet or maskrcnn",
    type=str,
)

parser.add_argument(
    "-n",
    "--threshold_nuclei",
    help="select the threshold you want for similar nuclei retrievals.\n The threshold must be between 0 and 1 and nuclei with distance under the threshold will be corrected",
    type=float,
)
parser.add_argument(
    "-p",
    "--threshold_patches",
    help="select the threshold you want for similar patches retrievals.\n The threshold must be between 0 and 1 and patches with similarity above the threshold will be potentially corrected",
    type=float,
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

# th_images, th_nuclei = 0.5, 0.4
th_images, th_nuclei = args.threshold_patches, args.threshold_nuclei
run["config/parameters/th_nuclei"] = th_nuclei
run["config/parameters/th_images"] = th_images

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

autoencoder = UNet(3, 3)
device = torch.device('cpu')
autoencoder.to('cpu')
autoencoder.load_state_dict(
    torch.load(path_weights_autoencoder, map_location=device))
clickref = Clickref(3, 3)
clickref.load_state_dict(
    torch.load(path_weights_clickref, map_location=device)
)

pairwise_distances = np.load('autoencoder/pairwise_distances.npy')

f = FindSame2(df_test,
              autoencoder,
              threshold_images=1e-4,
              threshold_nucleis=1e-2,
              baseline = path_baseline,
              pairwise_similarities=pairwise_distances
              )
def extract_similar_errors(f,index_patch_clicked,th_images, th_nuclei):
    f.extract_errors(index_patch_clicked)
    f.threshold_images = th_images
    f.select_images_near()

    f.threshold_nucleis = th_nuclei
    (
        images_near,
        images_with_same_errors_detected,
        sub_dataset,
        grids,
        filenames_near,
    ) = f.find_same_nuclei()

    return images_near,images_with_same_errors_detected,sub_dataset,grids,filenames_near



def compute_correction_table(images_with_same_errors_detected,grids,loader_test,filenames_near):
    nb_images_to_correct = images_with_same_errors_detected.shape[0]
    results = np.zeros((nb_images_to_correct,2,6))
    for k, index in enumerate(images_with_same_errors_detected):
        filename = filenames_near[index]
        grid = grids[index]

        g = Grid_to_click(grid, filename, path_baseline)
        click = torch.tensor(np.moveaxis(g.final_click(),-1,0))[None].float()
        filename_to_index = loader_test.dataset.dataframe.reset_index(drop=True)
        index = filename_to_index[filename_to_index['filename']==filename].index[0]
        image, baseline, gt, click_manual = loader_test.dataset.__getitem__(index)
        pred_outputs = clickref(image[None], click, baseline[None])[0]

        prediction = post_processing(disk_size=2, size_erase=50, pred = pred_outputs.detach().numpy(), not_pred=baseline.detach().numpy(), is_pred = True).astype(int)
        baseline_post_processed = post_processing(disk_size=1, size_erase=0, pred = None, not_pred=baseline.detach().numpy(), is_pred = False).astype(int)
        gt_post_processed = post_processing(disk_size=1, size_erase=0, pred = None, not_pred=gt.detach().numpy(), is_pred = False).astype(int)
        
        match_pred = matching(gt_post_processed, prediction)
        match_baseline = matching(gt_post_processed, baseline_post_processed)

        results[k,:,0] = match_pred.precision, match_baseline.precision
        results[k,:,1] = match_pred.recall, match_baseline.recall
        results[k,:,2] = match_pred.f1, match_baseline.f1
        results[k,:,3] = match_pred.panoptic_quality, match_baseline.panoptic_quality
        results[k,:,4] = AJI(prediction, gt_post_processed), AJI(baseline_post_processed, gt_post_processed)
        results[k,:,5] = DICE(prediction, gt_post_processed), DICE(baseline_post_processed, gt_post_processed)

    mean_results = np.mean(results,0)
    table = BeautifulTable()
    table.rows.append(['click_ref']+list(mean_results[0]))
    table.rows.append(['baseline']+list(mean_results[1]))

    table.columns.header = [
        "model",
        "Precision",
        "Recall",
        "F1",
        "PANOPTIC",
        "AJI",
        "ADICE",
    ]
    return results, table  
    
nb = df_test.shape[0]
all_res = np.zeros((nb,2,6))
nb_images = 0
nb_nuclei = 0
for i in tqdm(range(nb)):

    images_near,images_with_same_errors_detected,sub_dataset,grids,filenames_near = extract_similar_errors(f,i,th_images, th_nuclei)
    if len(images_with_same_errors_detected) > 0:
        nb_images += len(images_with_same_errors_detected)
        nb_nuclei += len(filenames_near)
        results, table = compute_correction_table(images_with_same_errors_detected,grids,loader_test,filenames_near)
        all_res[i] = np.mean(results, axis = 0)
        print(table)

all_res = np.mean(all_res, axis = 0)

table = BeautifulTable()
table.rows.append(['click_ref']+list(all_res[0]))
table.rows.append(['baseline']+list(all_res[1]))

table.columns.header = [
    "model",
    "Precision",
    "Recall",
    "F1",
    "PANOPTIC",
    "AJI",
    "ADICE",
]
file = 'n'+str(th_images)+'p'+str(th_nuclei)+'b'+str(args.baseline)+'.csv'
table.to_csv(file)

run[str(args.baseline)].upload(file)
# print(table)
# print(nb_images/nb, nb_nuclei/nb)

run["config/results/nb_patches"] = nb_images/nb
run["config/results/nb_nuclei"] = nb_nuclei/nb