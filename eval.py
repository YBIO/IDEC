"""
 * @Author: HibiscusYB 
 * @Date: 2022-04-11 14:20:07 
 * @Last Modified by:   HibiscusYB
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
from utils.visualize_feature import show_feature_map, draw_features

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from thop import profile


torch.backends.cudnn.benchmark = True

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/data/DB/VOC2012',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'ade', 'ISPRS'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
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
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'],
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
                        choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type (default: False)")
    parser.add_argument("--KD_loss_type", type=str, default='KLDiv_loss',
                        choices=['KLDiv_loss', 'KD_loss', 'L1_loss', 'L2_loss'], help="KD loss type for ret features")
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
    
    # CIL options
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




def convert_from_color_segmentation(seg):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    # palette = pascal_palette() # pascal
    palette = ade_palette() # ade
    # palette = ISPRS_palette() # ISPRS

    for c, i in palette.items():
        color_seg[ seg == i] = c
        
    color_seg = color_seg[..., ::-1]

    return color_seg

def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    
    train_transform = et.ExtCompose([
        #et.ExtResize(size=opts.crop_size),
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
        
    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    elif opts.dataset == 'ISPRS':
        dataset = ISPRSSegmentation
    else:
        raise NotImplementedError
        
    dataset_dict = {}
    dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=train_transform, cil_step=opts.curr_step)
    
    dataset_dict['val'] = dataset(opts=opts,image_set='val', transform=val_transform, cil_step=opts.curr_step)
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=opts.curr_step)
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform, 
                                                 cil_step=opts.curr_step, mem_size=opts.mem_size)

    return dataset_dict


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(tqdm(loader)):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            ret_features, outputs = model(images)
           
            ##===tsne-begin===
            ##==make sure val_size=1
            # feature_tsne = ret_features['feature_l3'].cpu().numpy() #[1,256,33,33] # outputs.cpu().numpy()
            # # feature_tsne = np.resize(feature_tsne,(1,256,17,17))
            # feature_tsne_save = feature_tsne.reshape(1, 256, -1).transpose(0,2,1)
            # np.savetxt('tsne/ISPRS/feature_tsne/feature_tsne_%04d.csv'%(i+2401), feature_tsne_save[0], delimiter=' ')
            # label_tsne = labels.cpu().numpy()
            # # label_tsne =np.resize(label_tsne,(1,256,256)) # do not use np.resize, it changes image content(x_x)
            # label_tsne = cv2.resize(label_tsne[0].astype(np.uint8), (feature_tsne.shape[-2],feature_tsne.shape[-1]))
            # label_tsne = label_tsne.reshape(1,-1)
            # np.savetxt('tsne/ISPRS/label_tsne/label_tsne_%04d.csv'%(i+2401), label_tsne[0], delimiter=' ')
            ##===tsne-end===
         
            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
                    
            # remove unknown label
            if opts.unknown:
                outputs[:, 1] += outputs[:, 0]
                outputs = outputs[:, 1:]
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            ## ===save_pred_visualizations
            save_preds = convert_from_color_segmentation(preds[0]) # BGR to RGB
            save_preds = BGR_to_RGB(save_preds)
            imsave('visualize_results/ade_100-5_step10/%08d.jpg'%(i+1), save_preds)

            ###===save color labels
            # temp0 = labels.cpu().numpy()
            # # temp=cv2.resize(temp0[0].astype(np.uint8), (17,17))
            # temp=temp0[0].astype(np.uint8)
            # save_labels = convert_from_color_segmentation(temp) # BGR to RGB
            # save_labels = BGR_to_RGB(save_labels)
            # imsave('visualize_results/ade_color_labels/%08d.png'%(i+1), save_labels)
            # assert False

            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
                
        score = metrics.get_results()
    return score


def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    if opts.unknown: # [unknown, background, ...]
        opts.num_classes = [1, 1, opts.num_classes[0]-1] + opts.num_classes[1:]
    fg_idx = 1 if opts.unknown else 0
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("  Device: %s" % device)
    print( "  opts : ")
    print(opts)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=opts.bn_freeze)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
        
    # Set up metrics
    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), dataset=opts.dataset)

    if opts.overlap:
        ckpt_str = "checkpoints/100-5_KDoutlogits/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"
    
    model = nn.DataParallel(model)
    mode = model.to(device)

    ##===calculate model params===
    # images = torch.rand(1,3,513,513)
    # from torchstat import stat
    # model = model.module.to(torch.device('cpu'))
    # stat(model, (3,513,513))
    # assert False
    
    dataset_dict = get_dataset(opts)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("... Testing Best Model")
    report_dict = dict()
    # best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, 0)
    best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step)
    print('best_ckpt:', best_ckpt)
    checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
    model.module.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    test_score = validate(opts=opts, model=model, loader=test_loader, 
                          device=device, metrics=metrics)
    print(metrics.to_str(test_score))
    report_dict[f'best/test_all_miou'] = test_score['Mean IoU']

    class_iou = list(test_score['Class IoU'].values())
    class_acc = list(test_score['Class Acc'].values())

    first_cls = len(get_tasks(opts.dataset, opts.task, 0)) 

    report_dict[f'best/test_before_mIoU'] = np.mean(class_iou[:first_cls]) 
    report_dict[f'best/test_after_mIoU'] = np.mean(class_iou[first_cls:])  
    report_dict[f'best/test_before_acc'] = np.mean(class_acc[:first_cls])  
    report_dict[f'best/test_after_acc'] = np.mean(class_acc[first_cls:])  

    print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
    print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
    print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
    print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))


if __name__ == '__main__':
            
    opts = get_argparser().parse_args()
        
    total_step = len(get_tasks(opts.dataset, opts.task))
    # opts.curr_step = 0
    opts.curr_step = total_step-1
    main(opts)

