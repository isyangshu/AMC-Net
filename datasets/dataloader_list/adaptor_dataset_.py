# -*- coding: utf-8 -*
import os
import cv2
import numpy as np
from PIL import Image
from datasets.dataloader_list.transform import Compose, JointResize, RandomHorizontallyFlip, RandomRotate
from torch.utils.data import DataLoader, Dataset
from datasets.dataloader_list.davis import DavisDataset, DavisDatasetVal
from datasets.dataloader_list.ytb_vos import YoutubeVOSDataset

class YTB_DAVIS_Dataset(Dataset):
    def __init__(
            self, split='train', datasets=["Davis"], subsets=["train2016"], inputRes=384):
        self.datasets = datasets
        self.inputRes = inputRes
        self.split = split
        self.modules = []
        self.data_item = []
        if self.split == 'train':
            if "Davis" in self.datasets:
                module = DavisDataset(subsets)
                self.modules.append(module)
            if "YoutubeVOS" in self.datasets:
                module = YoutubeVOSDataset()
                module.data_items = module.data_items[::9]
                self.modules.append(module)
            for i in range(len(self.modules)):
                self.data_item = self.data_item + self.modules[i].data_items
        else:
            module = DavisDatasetVal()
            self.data_item = self.data_item + module.data_items

        self.transform = Compose([JointResize(inputRes)])
        self.img_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
                ]
            )
        self.flow_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
                ]
            )
        self.mask_transform = transforms.ToTensor()
        self.depth_transform = transforms.ToTensor()
        if split == 'train':
            self.transform = Compose([JointResize(inputRes), RandomHorizontallyFlip(), RandomRotate(10)])
            self.img_transform = transforms.Compose(
                [
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
                ]
            )
            self.flow_transform = transforms.Compose(
                [
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
                ]
            )

    def __getitem__(self, item):
        data_item = self.data_item[item]
        image_path = data_item['image'][0]
        mask_path = data_item['annos'][0]
        # depth_path = data_item['depth'][0]
        flow_path = data_item['flow'][0]
        image = Image.open(image_path).convert('RGB')
        flow = Image.open(flow_path).convert('RGB')
        # depth = Image.open(depth_path)
        mask = cv2.imread(mask_path, 0)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

        image, mask, depth, flow = self.transform(image, mask, depth, flow)
        image = self.img_transform(image)
        flow = self.flow_transform(flow)
        mask = self.mask_transform(mask)
        # depth = self.depth_transform(depth)

        # print('mask:', mask.shape)  # mask: torch.Size([1, 384, 384])
        # print('image:', image.shape)  # image: torch.Size([3, 384, 384])
        # print('flow:', flow.shape)  # flow: torch.Size([3, 384, 384])
        # print('depth:', depth.shape)  # depth: torch.Size([1, 384, 384])
        return image, flow, mask

    def __len__(self):
        return len(self.data_item)
        
import torch
from torchvision import transforms
if __name__ == '__main__':
    data = YTB_DAVIS_Dataset(split='val')
    print(len(data))
    # print('dataset processing...')

    dataloader = DataLoader(data,  batch_size=1)
    for ii, (image, flow, mask, depth) in enumerate(dataloader):
        print(torch.unique(depth))
        # print(image.shape)
        # print(depth.shape)
        # print(flow.shape)
        # print(mask.shape)
        print('---')