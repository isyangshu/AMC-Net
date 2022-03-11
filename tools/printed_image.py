from PIL import Image
import numpy as np
import cv2
import os

Imagefile = '/Users/yang/Downloads/Depth_Model/tools/bear_image/'
Maskfile = '/Users/yang/Downloads/Depth_Model/tools/bear_ann/'
Savefile = '/Users/yang/Downloads/Depth_Model/tools/bear/'
img_list = sorted(os.listdir(Imagefile))
mask_list = sorted(os.listdir(Maskfile))
for img, mk in zip(img_list, mask_list):
    print(img, mk)
    imagename = Imagefile+img
    maskname = Maskfile+mk
    save = Savefile+mk
    image = Image.open(imagename).convert('RGB')
    image = np.array(image).astype(np.float32)
    mask = Image.open(maskname).convert('P')
    mask = np.array(mask)
    weight = mask.copy().astype(np.float32)
    weight[weight==0] = 1
    weight[weight==1] = 0
    w = np.expand_dims(weight,axis=2)
    w = np.concatenate((w, w, w),axis=2)
    mask = np.expand_dims(mask, axis=2)
    m_0 = np.zeros(mask.shape)
    m = np.concatenate((mask, m_0, m_0),axis=2)

    new_image = image * w + (1 - w) /2 * (image + m)

    Image.fromarray(new_image.astype(np.uint8)).save(save)