"""
 * @Author: YBIO
 * @Date: 2022-04-11 14:20:07 
 * @Last Modified by:   YBIO
 * @Last Modified time: 2022-04-11 14:20:07 
"""

from tqdm import tqdm
import network
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation, ISPRSSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torchvision
import torch.nn as nn
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
from utils.color_palette import pascal_palette, ade_palette, ISPRS_palette

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from thop import profile


torch.backends.cudnn.benchmark = True

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/data/',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='ISPRS', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16)
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--freeze", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--train_epoch", type=int, default=0,
                        help="epoch number (default: 0")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly',
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='bce_loss',
                        help="loss type (default: False)")
    parser.add_argument("--KD_loss_type", type=str, default='KLDiv_loss',
                         help="KD loss type for ret features")
    parser.add_argument("--use_KD_layer_weight", action='store_true', default=False,
                        help='Whether to apply layer weight for ret feature distillation (default: False)')
    parser.add_argument("--use_KD_class_weight", action='store_true', default=False,
                        help='Whether to apply class weight for ret feature distillation (default: False)')   
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--pseudo", action='store_true', default=False)
    parser.add_argument("--pseudo_thresh", type=float, default=0.7)
    parser.add_argument("--task", type=str, default='15-1')
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", action='store_true', default=False)
    parser.add_argument("--mem_size", type=int, default=0)
    parser.add_argument("--bn_freeze", action='store_true', default=False)
    parser.add_argument("--w_transfer", action='store_true', default=False)
    parser.add_argument("--unknown", action='store_true', default=False)
    
    return parser

def get_dataset(opts):
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if opts.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    if opts.dataset == 'ISPRS':
        dataset = ISPRSSegmentation
        
    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    dataset_dict['val'] = dataset(opts=opts,image_set='val', transform=val_transform, cil_step=opts.curr_step)
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform, 
                                                 cil_step=opts.curr_step, mem_size=opts.mem_size)
    return dataset_dict


def validate(opts, model, loader, device, metrics):

    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(tqdm(loader)):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            ret_features, outputs = model(images)     
            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            if opts.unknown:
                outputs[:, 1] += outputs[:, 0]
                outputs = outputs[:, 1:]
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
                
        score = metrics.get_results()
    return score


def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    if opts.unknown: 
        opts.num_classes = [1, 1, opts.num_classes[0]-1] + opts.num_classes[1:]
    fg_idx = 1 if opts.unknown else 0
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    

    model_map = {
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=opts.bn_freeze)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
        
    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), dataset=opts.dataset)

    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"

    
    model = nn.DataParallel(model)
    mode = model.to(device)

    
    dataset_dict = get_dataset(opts)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    

    report_dict = dict()
    best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step)
    checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
    model.module.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    test_score = validate(opts=opts, model=model, loader=test_loader, 
                          device=device, metrics=metrics)
    print(metrics.to_str(test_score))
    report_dict[f'a_miou'] = test_score['Mean IoU']

    class_iou = list(test_score['Class IoU'].values())

    first_cls = len(get_tasks(opts.dataset, opts.task, 0)) 

    report_dict[f'b_mIoU'] = np.mean(class_iou[:first_cls]) 
    report_dict[f'n_mIoU'] = np.mean(class_iou[first_cls:])  


    print(f" 0-{first_cls-1} : mIoU : %.6f" % np.mean(class_iou[:first_cls]))
    print(f" {first_cls} - {len(class_iou)-1}: mIoU : %.6f" % np.mean(class_iou[first_cls:]))



if __name__ == '__main__':       
    opts = get_argparser().parse_args()
    total_step = len(get_tasks(opts.dataset, opts.task))
    opts.curr_step = total_step-1
    main(opts)