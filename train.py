import torch
from torch.utils import data
from torchvision import transforms
import cv2
import os
import time
import random
import datetime
import numpy as np
import torch.nn as nn
from model.AMCNet import AMCNet
from utils.args import get_parser
from utils.utils import make_dir
from datasets.dataloader_list.adaptor_dataset import YTB_DAVIS_Dataset
from measures.jaccard import db_eval_iou_multi
import torch.nn.functional as F

def init_dataloaders(args):
    loaders = {}

    # init dataloaders for training and validation
    for split in ['train', 'val']:
        batch_size = args.batch_size
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_transforms = transforms.Compose([to_tensor, normalize])
        target_transforms = transforms.Compose([to_tensor])

        dataset = YTB_DAVIS_Dataset(split=split, datasets=["Davis"], augment=True, transform=image_transforms, target_transform=target_transforms)

        shuffle = True if split == 'train' else False
        loaders[split] = data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=args.num_workers,
                                         drop_last=True)
    return loaders


def trainIters(args):
    print(args)

    model_dir = os.path.join('ckpt/', args.model_name)
    make_dir(model_dir)
    log_path = os.path.join(model_dir, str(datetime.datetime.now()) + '.txt')
    Model = AMCNet()
    open(log_path, 'w').write(str(args) + '\n\n')
    optimizer = torch.optim.SGD([
        {'params': [param for name, param in Model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args.lr},
        {'params': [param for name, param in Model.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr, 'weight_decay': args.weight_decay}], momentum=args.momentum)

    criterion = nn.BCELoss()

    if args.use_gpu:
        Model.cuda()
        criterion.cuda()

    loaders = init_dataloaders(args)
    epoch_resume = 0
    best_iou = 0
    curr_iter = 0
    total_iter = len(loaders['train']) * args.max_epoch
    start = time.time()
    for e in range(epoch_resume, args.max_epoch):
        print("Epoch", e)
        epoch_losses = {'train': {'iou': [], 'output':[], 'mask_loss': []},
                        'val': {'iou': [], 'output':[], 'mask_loss': []}}

        for split in ['train', 'val']:
            if split == 'train':
                Model.train(True)
            else:
                Model.train(False)
            print('Model:---------------', split, '---------------')
            for batch_idx, (image, flow, mask) in enumerate(loaders[split]):
                image, mask, flow = image.cuda(), mask.cuda(), flow.cuda()
                if split == 'train':
                    optimizer.param_groups[0]['lr'] = 2 * args.lr * (1 - float(curr_iter) / total_iter) ** args.lr_decay
                    optimizer.param_groups[1]['lr'] = args.lr * (1 - float(curr_iter) / total_iter) ** args.lr_decay
                    output, c4_attention, c3_attention, c2_attention, c1_attention, c0_attention = Model(image, flow)
                    loss_output = criterion(output, mask)
                    loss_c4 = criterion(c4_attention, 
                                        F.interpolate(mask, size=c4_attention.size()[2:], mode='nearest'))
                    loss_c3 = criterion(c3_attention, 
                                        F.interpolate(mask, size=c3_attention.size()[2:], mode='nearest'))
                    loss_c2 = criterion(c2_attention,
                                        F.interpolate(mask, size=c2_attention.size()[2:], mode='nearest'))
                    loss_c1 = criterion(c1_attention,
                                        F.interpolate(mask, size=c1_attention.size()[2:], mode='nearest'))
                    loss_c0 = criterion(c0_attention,
                                        F.interpolate(mask, size=c0_attention.size()[2:], mode='nearest'))
                    loss = loss_output + loss_c4 + loss_c3 + loss_c2 + loss_c1 + loss_c0

                    iou = db_eval_iou_multi(mask.cpu().detach().numpy(),
                                            output.cpu().detach().numpy())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    curr_iter = curr_iter+1

                else:
                    with torch.no_grad():
                        output, c4_attention, c3_attention, c2_attention, c1_attention, c0_attention = Model(image, flow)
                        loss_output = criterion(output, mask)
                        loss_c4 = criterion(c4_attention,
                                            F.interpolate(mask, size=c4_attention.size()[2:], mode='nearest'))
                        loss_c3 = criterion(c3_attention,
                                            F.interpolate(mask, size=c3_attention.size()[2:], mode='nearest'))
                        loss_c2 = criterion(c2_attention,
                                            F.interpolate(mask, size=c2_attention.size()[2:], mode='nearest'))
                        loss_c1 = criterion(c1_attention,
                                            F.interpolate(mask, size=c1_attention.size()[2:], mode='nearest'))
                        loss_c0 = criterion(c0_attention,
                                            F.interpolate(mask, size=c0_attention.size()[2:], mode='nearest'))
                        loss = loss_output + loss_c4 + loss_c3 + loss_c2 + loss_c1 + loss_c0
                    iou = db_eval_iou_multi(mask.cpu().detach().numpy(), output.cpu().detach().numpy())

                epoch_losses[split]['mask_loss'].append(loss.data.item())
                epoch_losses[split]['output'].append(loss_output.data.item())
                epoch_losses[split]['iou'].append(iou)

                if (batch_idx + 1) % args.print_every == 0:
                    mmask = np.mean(epoch_losses[split]['mask_loss'])
                    mout = np.mean(epoch_losses[split]['output'])
                    miou = np.mean(epoch_losses[split]['iou'])

                    te = time.time() - start
                    log = 'Epoch: [{}/{}][{}/{}]\tLr {:.5f}\tTime {:.3f}s''\tMask Loss: {:.4f}\toutput Loss: {:.4f}\tIOU: {:.4f}'.format(e, args.max_epoch, batch_idx,
                                                 len(loaders[split]), optimizer.param_groups[0]['lr'], te, mmask, mout, miou)
                    print(log)
                    open(log_path, 'a').write(log + '\n')
                    start = time.time()
        miou = np.mean(epoch_losses['val']['iou'])
        if e % 5 == 0:
            torch.save(Model.state_dict(), os.path.join(model_dir, 'RGBF_%d.pth' % e))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'RGBF_%d_optim.pth' % e))
        if e == args.max_epoch-1:
            torch.save(Model.state_dict(), os.path.join(model_dir, 'RGBF_%d.pth' % (e-1)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'RGBF_%d_optim.pth' % (e-1)))
        if miou > best_iou:
            print('=============================')
            print('updata:  ',miou)
            print('=============================')
            best_iou = miou
            torch.save(Model.state_dict(), os.path.join(model_dir, 'best.pth'))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'best_optim.pth'))
    print('best miou:', best_iou)
    log= 'best miou:' + str(best_iou)
    open(log_path, 'a').write(log + '\n')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    args.model_name = 'xxxxxxxxxx'
    args.batch_size = 4
    args.max_epoch = 100
    args.lr = 1e-3
    args.snapshot = ''
    args.pretrain = False

    gpu_id = 0
    print('gpu_id: ', gpu_id)
    print('use_gpu: ', args.use_gpu)
    print('seed: ', args.seed)
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    trainIters(args)