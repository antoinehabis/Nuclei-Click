import cv2
from skimage.morphology import binary_dilation, disk
from skimage.measure import label
from matching import matching
from beautifultable import BeautifulTable
import sys
from pathlib import Path
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from config import *
import numpy as np
from tqdm import tqdm



def draw_red_square(image, center_position, square_width, key, count,put_text=True):
    if key == "merge":
        color = [0, 0, 255]
    if key == "split":
        color = [0, 255, 255]
    if key == "fn":
        color = [255, 0, 0]
    if key == "fp":
        color = [0, 255, 0]
    # Calculate the coordinates for the square
    x, y = center_position
    half_width = square_width // 2

    # Draw the red square border
    clip_y_min = np.clip(y - half_width + 1, 0, None)
    clip_x_min = np.clip(x - half_width, 0, None)
    clip_y_max = np.clip(y + half_width, None, 255)
    clip_x_max = np.clip(x + half_width + 1, None, 255)

    image[clip_y_min:clip_y_max, clip_x_min] = color  # Left border
    image[clip_y_min:clip_y_max, clip_x_max] = color  # Right border
    image[clip_y_min, clip_x_min:clip_x_max] = color  # Top border
    image[clip_y_max, clip_x_min:clip_x_max] = color  # Bottom border
    if put_text:
        image = cv2.putText(
            image,
            str(count),
            (clip_x_max - 20, clip_y_min + 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=2,
        )

    return image

def unlarge_connected_component(disk_size,size_erase, connected_components):
    fill = np.zeros(connected_components.shape)
    if len(np.unique(connected_components)[1:])>0:
        for i, cc in enumerate(np.unique(connected_components)[1:]):
            tmp_cc = np.squeeze((connected_components==cc).astype(int))
            if np.sum(tmp_cc)>=size_erase:
                dilated = binary_dilation(tmp_cc,disk(disk_size))
                fill  = np.maximum(fill,dilated*(i+1))
    return fill


def post_processing(disk_size, size_erase,pred, not_pred, is_pred = True):

    if is_pred:
        interior = np.argmax(pred + not_pred, axis=0)==2
        tmp = disk_size
    else: 
        image = not_pred

        tmp = 1
        image = np.moveaxis(image,(0,1,2),(-1,0,1))
        interior = np.squeeze((np.dot(image,np.arange(3)[:,None])==2).astype(int))
    ccs = label(interior)

    ccs_unlarged = unlarge_connected_component(disk_size = tmp,size_erase=size_erase, connected_components = ccs)
    return ccs_unlarged

def test_clickref(model, test_dl):
    list_baselines, list_gts, list_preds, lists_clicks = [], [], [], []
    model.eval()
    for batch in tqdm(test_dl):
        images = batch[0].cuda()
        baselines = batch[1].cuda()
        outputs = batch[2].cuda()
        clicks = batch[3].cuda()
        pred_outputs = model(images, clicks, baselines)
        list_baselines.append(baselines.cpu().detach().numpy())
        list_gts.append(outputs.cpu().detach().numpy())
        list_preds.append(pred_outputs.cpu().detach().numpy())
        lists_clicks.append(clicks.cpu().detach().numpy())
    arr_baselines = np.concatenate(list_baselines)
    arr_gts = np.concatenate(list_gts)
    arr_preds = np.concatenate(list_preds)
    arr_clicks = np.concatenate(lists_clicks)
    return arr_baselines, arr_gts, arr_preds, arr_clicks


def AJI(crop_img_new_seg, crop_img_gt):
    Is = 0
    Us = 0
    intersection = 0
    union = 0

    uniques = np.unique(crop_img_new_seg)[1:]
    uniques_true = np.unique(crop_img_gt)[1:]

    if len(uniques) == 0:
        if len(uniques_true) == 0:
            return 1.0
        else:
            return 0.0

    used = np.zeros(uniques.shape)
    for nuc_truth in uniques_true:
        binary_truth = crop_img_gt == nuc_truth
        IOUs = []
        cards = []
        Is = []
        Us = []
        for nuc in uniques:
            binary_seg = crop_img_new_seg == nuc
            I = np.sum(np.logical_and(binary_seg, binary_truth))
            U = np.sum(np.logical_or(binary_seg, binary_truth))
            Us.append(U)
            Is.append(I)
            IOUs.append(I / U)
        ind = np.argmax(np.array(IOUs))
        used[ind] = 1

        intersection += np.array(Is)[ind]
        union += np.array(Us)[ind]

    for not_used in np.argwhere(np.logical_not(used)):
        union += np.sum(crop_img_new_seg == uniques[not_used])
    if union == 0.0:
        return 1
    else:
        return intersection / union


def DICE(crop_img_new_seg, crop_img_gt):
    Is = 0
    Us = 0
    intersection = 0
    union = 0

    uniques = np.unique(crop_img_new_seg)[1:]
    uniques_true = np.unique(crop_img_gt)[1:]
    if len(uniques) == 0:
        if len(uniques_true) == 0:
            return 1.0
        else:
            return 0.0
    used = np.zeros(uniques.shape)
    for nuc_truth in uniques_true:
        binary_truth = crop_img_gt == nuc_truth
        IOUs = []
        cards = []
        Is = []
        Us = []
        for nuc in uniques:
            binary_seg = crop_img_new_seg == nuc
            I = 2 * np.sum(np.logical_and(binary_seg, binary_truth))
            U = np.sum(binary_seg) + np.sum(binary_truth)
            Us.append(U)
            Is.append(I)
            IOUs.append(I / U)
        ind = np.argmax(np.array(IOUs))
        used[ind] = 1

        intersection += np.array(Is)[ind]
        union += np.array(Us)[ind]

    for not_used in np.argwhere(np.logical_not(used)):
        union += np.sum(crop_img_new_seg == uniques[not_used])
    if union == 0.0:
        return 1
    else:
        return intersection / union



def get_results(disk_size, size_erase, arr_baselines, arr_gts, arr_preds, arr_clicks):
    n = arr_preds.shape[0]
    results = np.ones((n,2,6))

    for i in tqdm(range(n)):
        pred_post_processed = post_processing(disk_size, size_erase, arr_preds[i], arr_baselines[i], is_pred = True).astype(int)
        gt_post_processed = post_processing(disk_size, size_erase, arr_preds[i], arr_gts[i], is_pred = False).astype(int)
        baseline_post_processed = post_processing(disk_size, size_erase, arr_preds[i], arr_baselines[i], is_pred = False).astype(int)
        match_pred = matching(gt_post_processed, pred_post_processed)
        match_baseline = matching(gt_post_processed, baseline_post_processed)

        results[i,:,0] = match_pred.precision, match_baseline.precision
        results[i,:,1] = match_pred.recall, match_baseline.recall
        results[i,:,2] = match_pred.f1, match_baseline.f1
        results[i,:,3] = match_pred.panoptic_quality, match_baseline.panoptic_quality
        results[i,:,4] = AJI(pred_post_processed, gt_post_processed), AJI(baseline_post_processed, gt_post_processed)
        results[i,:,5] = DICE(pred_post_processed, gt_post_processed), DICE(baseline_post_processed, gt_post_processed)
 
    mean_res = np.mean(results, axis = 0)
    res_pred, res_base = mean_res[0], mean_res[1]
    table = BeautifulTable()
    table.rows.append(['click_ref']+list(res_pred))
    table.rows.append(['baseline']+list(res_base))

    table.columns.header = [
        "model",
        "Precision",
        "Recall",
        "F1",
        "PANOPTIC",
        "AJI",
        "ADICE",
    ]
    print(table)
    return table