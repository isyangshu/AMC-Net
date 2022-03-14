import os
import pickle
import warnings

import cv2
import numpy as np
import yaml
from joblib import Parallel, delayed
from prettytable import PrettyTable as ptable
from PIL import Image
import glob
from metrics.vos import f_boundary, jaccard

data_yaml_path = "/workdir/AMCNet/datasets/DAVIS/Annotations/db_info.yml"
mask_data_root = "/workdiir/AMCNet/datasets/FBMS"


def print_all_keys(data_dict, level: int = 0):
    level += 1
    if isinstance(data_dict, dict):
        for k, v in data_dict.items():
            print(f" {'|=' * level}>> {k}")
            print_all_keys(v, level=level)
    elif isinstance(data_dict, (list, tuple)):
        for item in data_dict:
            print_all_keys(item, level=level)
    else:
        return


def get_info_dict_from_yaml(path: str):
    with open(path, encoding="utf-8", mode="r") as f_stream:
        data_info_dict = yaml.load(f_stream, Loader=yaml.FullLoader)
    return data_info_dict


def get_eval_video_name_list(data_dict: dict, eval_set: str = "test"):
    eval_video_name_list = []
    for video_dict in data_dict["sequences"]:
        if video_dict["set"] == eval_set:
            eval_video_name_list.append(video_dict["name"])
    return eval_video_name_list


def get_mean_recall_decay_for_video(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.

    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values[1:-1] > 0.5)

    # Compute decay as implemented in Matlab
    per_frame_values = per_frame_values[1:-1]  # Remove first frame

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i] : ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


def _read_and_eval_file(mask_video_path: str, pred_video_path: str, frame_name: str):
    frame_mask_path = os.path.join(mask_video_path, frame_name)
    # frame_pred_path = os.path.join(pred_video_path, 'frame' + frame_name[1:-3]+'png')
    frame_pred_path = os.path.join(pred_video_path, frame_name)

    frame_mask = cv2.imread(frame_mask_path, 0)  # h, w
    frame_pred = cv2.imread(frame_pred_path, 0)

    if frame_mask.shape != frame_pred.shape:
        h, w = frame_pred.shape
        frame_mask = np.array(Image.fromarray(frame_mask).resize((w,h), 0))
    binary_frame_mask = (frame_mask > 128).astype(np.float32)
    binary_frame_pred = (frame_pred > 0).astype(np.float32)

    J_score = jaccard.db_eval_iou(
        annotation=binary_frame_mask, segmentation=binary_frame_pred
    )
    F_score = f_boundary.db_eval_boundary(
        foreground_mask=binary_frame_pred, gt_mask=binary_frame_mask
    )
    return J_score, F_score


def _eval_video_sequence(
    method_pre_path: str, video_name: str, ignore_head: bool, ignore_tail: bool
):
    print(f"processing {video_name}...")

    mask_video_path = os.path.join(mask_data_root, video_name, 'GroundTruth')
    pred_video_path = os.path.join(method_pre_path, video_name)
    mask_frame_path_list = sorted(os.listdir(mask_video_path))
    mask = []
    for i in mask_frame_path_list:
        if i.endswith('.png'):
            mask.append(i)
    mask_frame_path_list = mask
    if ignore_head:
        mask_frame_path_list = mask_frame_path_list[1:]
    if ignore_tail:
        mask_frame_path_list = mask_frame_path_list[:-1]

    frame_score_list = [
        _read_and_eval_file(
            mask_video_path=mask_video_path,
            pred_video_path=pred_video_path,
            frame_name=frame_name,
        )
        for frame_name in mask_frame_path_list
    ]
    if ignore_head:
        frame_score_list = [[np.nan, np.nan]] + frame_score_list
    if ignore_tail:
        frame_score_list += [[np.nan, np.nan]]
    frame_score_array = np.asarray(frame_score_list)
    M, O, D = zip(
        *[
            get_mean_recall_decay_for_video(frame_score_array[:, i])
            for i in range(frame_score_array.shape[1])
        ]
    )
    return {
        video_name: {
            "pre_frame": frame_score_array,
            "mean": np.asarray(M),
            "recall": np.asarray(O),
            "decay": np.asarray(D),
        }
    }


def get_method_score_dict(
    method_pre_path: str,
    video_name_list: list,
    ignore_head: bool = True,
    ignore_tail: bool = True,
):
    video_score_list = Parallel(n_jobs=10)(
        delayed(_eval_video_sequence)(
            method_pre_path=method_pre_path,
            video_name=video_name,
            ignore_head=ignore_head,
            ignore_tail=ignore_tail,
        )
        for video_name in video_name_list
    )
    video_score_dict = {
        list(kv.keys())[0]: list(kv.values())[0] for kv in video_score_list
    }
    return video_score_dict


def get_method_average_score_dict(method_score_dict: dict):
    # average_score_dict = {"total": 0, "mean": 0, "recall": 0, "decay": 0}
    average_score_dict = {"Average": {"mean": 0, "recall": 0, "decay": 0}}
    for k, v in method_score_dict.items():
        # average_score_item = np.nanmean(v["pre_frame"], axis=0)
        # average_score_dict[k] = average_score_item
        average_score_dict[k] = {
            "mean": v["mean"],
            "recall": v["recall"],
            "decay": v["decay"],
        }
        # average_score_dict["total"] += average_score_item
        average_score_dict["Average"]["mean"] += v["mean"]
        average_score_dict["Average"]["recall"] += v["recall"]
        average_score_dict["Average"]["decay"] += v["decay"]
    # average_score_dict['Average']["total"] /= len(method_score_dict)
    average_score_dict["Average"]["mean"] /= len(method_score_dict)
    average_score_dict["Average"]["recall"] /= len(method_score_dict)
    average_score_dict["Average"]["decay"] /= len(method_score_dict)
    return average_score_dict


def save_to_file(data, save_path: str):
    with open(save_path, mode="wb") as f:
        pickle.dump(data, f)


def read_from_file(file_path: str):
    with open(file_path, mode="rb") as f:
        data = pickle.load(f)
    return data


def convert_data_dict_to_table(data_dict: dict, video_name_list: list):
    table = ptable(["Video", "J(M)", "J(O)", "J(D)", "F(M)", "F(O)", "F(D)"])
    for video_name in video_name_list:
        table.add_row(
            [video_name]
            + [
                f"{data_dict[video_name][x][y]: .3f}"
                for y in range(2)
                for x in ["mean", "recall", "decay"]
            ]
        )
    return "\n" + str(table) + "\n"


def eval_method_from_data(
    method_pre_path: str,
    ignore_head: bool,
    ignore_tail: bool,
    save_path: str = "./output/average.pkl",
):
    # get the list  that will be used to eval the model
    eval_video_name_list = os.listdir(method_pre_path)
    # tervese the each video
    method_score_dict = get_method_score_dict(
        method_pre_path=method_pre_path,
        video_name_list=eval_video_name_list,
        ignore_head=ignore_head,
        ignore_tail=ignore_tail,
    )
    # get the average score
    average_score_dict = get_method_average_score_dict(
        method_score_dict=method_score_dict
    )

    if save_path != None:
        save_to_file(data=average_score_dict, save_path=save_path)

    # show the results
    eval_video_name_list += ["Average"]
    table_str = convert_data_dict_to_table(
        data_dict=average_score_dict, video_name_list=eval_video_name_list
    )
    print(table_str)


def show_results_from_data_file(file_path: str = "./output/average.pkl"):
    average_score_dict = read_from_file(file_path=file_path)

    eval_video_name_list = list(average_score_dict.keys())
    eval_video_name_list[0], eval_video_name_list[-1] = (
        eval_video_name_list[-1],
        eval_video_name_list[0],
    )
    # show the results
    table_str = convert_data_dict_to_table(
        data_dict=average_score_dict, video_name_list=eval_video_name_list
    )
    print(table_str)


if __name__ == "__main__":
    eval_method_from_data(
        method_pre_path="result_path",
        ignore_tail=False,
        ignore_head=False,
        save_path="output/xxxxx.pkl",
    )
