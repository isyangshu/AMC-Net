
# from .utils import Dataset

# -*- coding: utf-8 -*-
import json
import os
import os.path as osp
import pickle

import cv2
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

class YoutubeVOSDataset(Dataset):
    r"""
    ----------------
    dataset_root: str
        path to root of the dataset
    subset: str
        dataset split name (train|val)
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames 
    """
    data_items = []

    default_hyper_params = dict(dataset_root="/home/yang/Documents/Depth_Model/datasets/youtubevos", subsets=["train"], ratio=1.0, max_diff=50)

    def __init__(self):
        r"""
        Create youtube vos dataset
        """
        super(YoutubeVOSDataset, self).__init__()
        dataset_root = self.default_hyper_params["dataset_root"]
        self.default_hyper_params["dataset_root"] = osp.realpath(dataset_root)
        if len(YoutubeVOSDataset.data_items) == 0:
            self._ensure_cache()

    def __getitem__(self, item):
        """
        :param item: int, video id
        :return:
            image_files
            annos
            meta (optional)
        """
        record = YoutubeVOSDataset.data_items[item]
        anno = [[anno_file, record['obj_id']] for anno_file in record["annos"]]
        sequence_data = dict(image=record["image_files"], anno=anno, flow=record['flow'], depth=record['depth'])

        return sequence_data

    def __len__(self):
        return len(YoutubeVOSDataset.data_items)

    def _ensure_cache(self):
        dataset_root = self.default_hyper_params["dataset_root"]
        for subset in self.default_hyper_params["subsets"]:
            image_root = osp.join(dataset_root, subset, "JPEGImages")
            anno_root = osp.join(dataset_root, subset, "Annotations")
            depth_root = osp.join(dataset_root, subset, "Depth")
            flow_root = osp.join(dataset_root, subset, "flow")
            data_anno_list = []
            cache_file = osp.join(dataset_root, "cache/{}.pkl".format(subset))
            if osp.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    YoutubeVOSDataset.data_items += pickle.load(f)
                logger.info("{}: loaded cache file {}".format(
                    YoutubeVOSDataset.__name__, cache_file))
            else:
                meta_file = osp.join(dataset_root, subset, "meta.json")
                with open(meta_file) as f:
                    records = json.load(f)
                records = records["videos"]
                for video_id in records:
                    video = records[video_id]
                    for obj_id in video["objects"]:
                        count = 0
                        for frame_id in video['objects'][obj_id]['frames']:
                            record = {}
                            record['category'] = video['objects'][obj_id]['category']
                            record['video_name'] = video_id
                            record['frame_num'] = frame_id
                            record['obj_id'] = int(obj_id)
                            record['image'] = [
                                osp.join(image_root, video_id, frame_id + '.jpg')
                            ]
                            record['annos'] = [
                                osp.join(anno_root, video_id, frame_id + '.png')
                            ]
                            record['depth'] = [
                                osp.join(depth_root, video_id, frame_id + '.png')
                            ]
                            record['flow'] = [
                                osp.join(flow_root, video_id, frame_id + '.png')
                            ]
                            data_anno_list.append(record)
                cache_dir = osp.dirname(cache_file)
                if not osp.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_anno_list, f)
                logger.info("Youtube VOS dataset: cache dumped at: {}".format(
                    cache_file))
                YoutubeVOSDataset.data_items += data_anno_list
