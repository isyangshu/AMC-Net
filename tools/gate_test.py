import torch
from torchvision import transforms
from tools.AMCNet_gate import AMCNet
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.misc import imresize
from utils.utils import make_dir
from os.path import isdir, join

inputRes = (384, 384)
use_flip = False
Save_video = False
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_transforms = transforms.Compose([to_tensor, normalize])
tensor_transforms = transforms.Compose([to_tensor])
model_name = 'AMCNet_gate'  # specify the model name
snapshot = 'checkpoint_path'
davis_result_dir = './output/davis16'
make_dir(davis_result_dir)
model = AMCNet()
model.load_state_dict(torch.load(snapshot))

model.cuda()

model.train(False)

val_set = '/AMCNet/datasets/DAVIS/ImageSets/2016/val.txt'
with open(val_set) as f:
    seqs = f.readlines()
    seqs = [seq.strip() for seq in seqs]

gate = {'RGB_GO':[],'RGB_G1':[],'RGB_G2':[],'RGB_G3':[],'RGB_G4':[],'Flow_GO':[],'Flow_G1':[],'Flow_G2':[],'Flow_G3':[],'Flow_G4':[]}
seqs = ['soapbox']
for video in tqdm(seqs):
    video_gate = {'RGB_GO': [], 'RGB_G1': [], 'RGB_G2': [], 'RGB_G3': [], 'RGB_G4': [], 'Flow_GO': [], 'Flow_G1': [],
            'Flow_G2': [], 'Flow_G3': [], 'Flow_G4': []}
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

            mask_pred, RGB_temp, Flow_temp = model(image, flow)
            video_gate['RGB_GO'].append(RGB_temp['RGB_G0'].item())
            video_gate['RGB_G1'].append(RGB_temp['RGB_G1'].item())
            video_gate['RGB_G2'].append(RGB_temp['RGB_G2'].item())
            video_gate['RGB_G3'].append(RGB_temp['RGB_G3'].item())
            video_gate['RGB_G4'].append(RGB_temp['RGB_G4'].item())
            video_gate['Flow_GO'].append(Flow_temp['Flow_G0'].item())
            video_gate['Flow_G1'].append(Flow_temp['Flow_G1'].item())
            video_gate['Flow_G2'].append(Flow_temp['Flow_G2'].item())
            video_gate['Flow_G3'].append(Flow_temp['Flow_G3'].item())
            video_gate['Flow_G4'].append(Flow_temp['Flow_G4'].item())
            # print(os.path.basename(imagefile)[:-4], RGB_temp['RGB_G0'].item(), RGB_temp['RGB_G1'].item(), RGB_temp['RGB_G2'].item(), RGB_temp['RGB_G3'].item(), RGB_temp['RGB_G4'].item())
            # print(os.path.basename(imagefile)[:-4], Flow_temp['Flow_G0'].item(), Flow_temp['Flow_G1'].item(), Flow_temp['Flow_G2'].item(), Flow_temp['Flow_G3'].item(), Flow_temp['Flow_G4'].item())
            # print("=========================================================")
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
    mRGB_G0 = np.mean(video_gate['RGB_GO'])
    mRGB_G1 = np.mean(video_gate['RGB_G1'])
    mRGB_G2 = np.mean(video_gate['RGB_G2'])
    mRGB_G3 = np.mean(video_gate['RGB_G3'])
    mRGB_G4 = np.mean(video_gate['RGB_G4'])
    mFlow_G0 = np.mean(video_gate['Flow_GO'])
    mFlow_G1 = np.mean(video_gate['Flow_G1'])
    mFlow_G2 = np.mean(video_gate['Flow_G2'])
    mFlow_G3 = np.mean(video_gate['Flow_G3'])
    mFlow_G4 = np.mean(video_gate['Flow_G4'])
    gate['RGB_GO'].append(mRGB_G0)
    gate['RGB_G1'].append(mRGB_G1)
    gate['RGB_G2'].append(mRGB_G2)
    gate['RGB_G3'].append(mRGB_G3)
    gate['RGB_G4'].append(mRGB_G4)
    gate['Flow_GO'].append(mFlow_G0)
    gate['Flow_G1'].append(mFlow_G1)
    gate['Flow_G2'].append(mFlow_G2)
    gate['Flow_G3'].append(mFlow_G3)
    gate['Flow_G4'].append(mFlow_G4)
mRGB_G0 = np.mean(gate['RGB_GO'])
mRGB_G1 = np.mean(gate['RGB_G1'])
mRGB_G2 = np.mean(gate['RGB_G2'])
mRGB_G3 = np.mean(gate['RGB_G3'])
mRGB_G4 = np.mean(gate['RGB_G4'])
mFlow_G0 = np.mean(gate['Flow_GO'])
mFlow_G1 = np.mean(gate['Flow_G1'])
mFlow_G2 = np.mean(gate['Flow_G2'])
mFlow_G3 = np.mean(gate['Flow_G3'])
mFlow_G4 = np.mean(gate['Flow_G4'])
print('RGB-G0：',mRGB_G0)
print('RGB-G1：',mRGB_G1)
print('RGB-G2：',mRGB_G2)
print('RGB-G3：',mRGB_G3)
print('RGB-G4：',mRGB_G4)
print('Flow-G0：',mFlow_G0)
print('Flow-G1：',mFlow_G1)
print('Flow-G2：',mFlow_G2)
print('Flow-G3：',mFlow_G3)
print('Flow-G4：',mFlow_G4)
print('G0：',mRGB_G0/mFlow_G0)
print('G1：',mRGB_G1/mFlow_G1)
print('G2：',mRGB_G2/mFlow_G2)
print('G3：',mRGB_G3/mFlow_G3)
print('G4：',mRGB_G4/mFlow_G4)