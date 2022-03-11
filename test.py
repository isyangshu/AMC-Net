import torch
from torchvision import transforms

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.misc import imresize
from utils.utils import make_dir

from model.AMCNet import AMCNet

inputRes = (384, 384)
use_flip = False
Save_video = False
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_transforms = transforms.Compose([to_tensor, normalize])
tensor_transforms = transforms.Compose([to_tensor])
model_name = 'AMCNet'  # specify the model name
snapshot = 'checkpoint_path'

davis_result_dir = '/AMCNet/output/davis16'
make_dir(davis_result_dir)
model = AMCNet()
model.load_state_dict(torch.load(snapshot))
torch.cuda.set_device(device=0)
model.cuda()

model.train(False)

val_set = '/AMCNet/datasets/DAVIS/ImageSets/2016/val.txt'
with open(val_set) as f:
    seqs = f.readlines()
    seqs = [seq.strip() for seq in seqs]

for video in tqdm(seqs):
    davis_root_dir = '/AMCNet/datasets/DAVIS/JPEGImages/480p'
    davis_flow_dir = '/AMCNet/datasets/DAVIS/flow/480p'

    image_dir = os.path.join(davis_root_dir, video)
    flow_dir = os.path.join(davis_flow_dir, video)

    imagefiles = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    flowfiles = sorted(glob.glob(os.path.join(flow_dir, '*.png')))
    image = Image.open(imagefiles[0]).convert('RGB')
    width, height = image.size
    count = 0
    mask_preds = np.zeros((len(imagefiles), height,width)) - 1
    with torch.no_grad():
        for imagefile, flowfile in zip(imagefiles, flowfiles):
            image = Image.open(imagefile).convert('RGB')
            flow = Image.open(flowfile).convert('RGB')
            width, height = image.size

            image = np.array(image.resize(inputRes))
            flow = np.array(flow.resize(inputRes))

            image = image_transforms(image)
            flow = image_transforms(flow)

            image = image.unsqueeze(0)
            flow = flow.unsqueeze(0)

            image, flow = image.cuda(), flow.cuda()
            mask_pred, mask_pred_4, mask_pred_3, mask_pred_2, mask_pred_1, mask_pred_0 = model(image, flow)

            mask_pred = mask_pred[0, 0, :, :]
            mask_pred[mask_pred>=0.5] = 1
            mask_pred[mask_pred<0.5] = 0

            mask_pred = Image.fromarray(mask_pred.cpu().detach().numpy() * 255).convert('L')

            save_folder = '{}/{}/result/{}'.format(davis_result_dir, model_name, video)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_file = os.path.join(save_folder,
                                     os.path.basename(imagefile)[:-4] + '.png')


            mask_pred = mask_pred.resize((width, height))

            mask_preds[count,:,:] = np.array(mask_pred)
            count = count + 1

            mask_pred.save(save_file)


            # mask_pred = mask_pred[0, 0, :, :]
            # mask_pred = Image.fromarray(mask_pred.cpu().detach().numpy() * 255).convert('L')
            # save_folder = '{}/{}/result_1/{}'.format(davis_result_dir, model_name, video)
            # if not os.path.exists(save_folder):
            #     os.makedirs(save_folder)
            #
            # save_file = os.path.join(save_folder,
            #                          os.path.basename(imagefile)[:-4] + '.png')
            # mask_pred = mask_pred.resize((width, height))
            # mask_preds[count,:,:] = np.array(mask_pred)
            # mask_pred = np.array(mask_pred)
            # mask_pred[mask_pred>=127] = 255
            # mask_pred[mask_pred<127] = 0
            # mask_pred = Image.fromarray(mask_pred)
            # count = count + 1
            # mask_pred.save(save_file)