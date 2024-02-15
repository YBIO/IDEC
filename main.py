"""
 * @Author: YBIO
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
import math

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation, ISPRSSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced
from utils.contrastive_learning import class_contrastive_learning

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/data/DB/VOC2012',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'ade', 'ISPRS'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")
    parser.add_argument("--model", type=str, default='deeplabv3_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet_v2_bubbliiiing', 'deeplabv3plus_mobilenet_v2_bubbliiiing',               
                                 'deeplabv3_berniwal_swintransformer_swin_t', 'deeplabv3plus_berniwal_swintransformer_swin_t',
                                 'deeplabv3_berniwal_swintransformer_swin_s', 'deeplabv3plus_berniwal_swintransformer_swin_s',
                                 'deeplabv3_berniwal_swintransformer_swin_b', 'deeplabv3plus_berniwal_swintransformer_swin_b',
                                 'deeplabv3_berniwal_swintransformer_swin_l', 'deeplabv3plus_berniwal_swintransformer_swin_l',
                                 'deeplabv3_microsoft_swintransformer_swin_t', 'deeplabv3plus_microsoft_swintransformer_swin_t',
                                 'deeplabv3_microsoft_swintransformer_swin_s', 'deeplabv3plus_microsoft_swintransformer_swin_s',
                                 'deeplabv3_microsoft_swintransformer_swin_b', 'deeplabv3plus_microsoft_swintransformer_swin_b',
                                 'deeplabv3_microsoft_swintransformer_swin_l', 'deeplabv3plus_microsoft_swintransformer_swin_l',
                                 'deeplabv3_hrnetv2_32', 'deeplabv3plus_hrnetv2_32',
                                 'deeplabv3_hrnetv2_48', 'deeplabv3plus_hrnetv2_48'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--cutout", action="store_true", default=False)
    parser.add_argument("--cutmix", action="store_true", default=False)
    parser.add_argument("--mixup", action="store_true", default=False)
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--train_epoch", type=int, default=50,
                        help="epoch number")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'],
                        help="learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='bce_loss',
                        choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type")
    parser.add_argument("--KD_loss_type", type=str, default='KLDiv_loss',
                        choices=['KLDiv_loss', 'KD_loss', 'L1_loss', 'L2_loss'], help="KD loss type for ret features")
    parser.add_argument("--use_KD_layer_weight", action='store_true', default=False,
                        help='Whether to apply layer weight for ret feature distillation (default: False)')
    parser.add_argument("--KD_outlogits", action='store_true', default=False,
                        help='Whether to distillation output between old model prediction and new model prediction')
    parser.add_argument("--use_KD_class_weight", action='store_true', default=False,
                        help='Whether to apply class weight for ret feature distillation (default: False)')     
    parser.add_argument('--lamb', type=float, default=1.0, help="initial weight for weight decay")               
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=20,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=200,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--pseudo", action='store_true', help="enable pseudo-labeling")
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help="confidence threshold for pseudo-labeling")
    parser.add_argument("--task", type=str, default='15-1', help="cil task")
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", action='store_true', help="overlap setup (True), disjoint setup (False)")
    parser.add_argument("--mem_size", type=int, default=0, help="size of examplar memory")
    parser.add_argument("--freeze", action='store_true', help="enable network freezing")
    parser.add_argument("--bn_freeze", action='store_true', help="enable batchnorm freezing")
    parser.add_argument("--w_transfer", action='store_true', help="enable weight transfer")
    parser.add_argument("--unknown", action='store_true', help="enable unknown modeling")
    parser.add_argument("--contrastive_learning", action='store_true', help="enable contrastive learning during CIL")
    
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
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels, _, _) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            ret_features, outputs = model(images)
            
            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            elif opts.loss_type == 'focal_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
                    
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
    bn_freeze = opts.bn_freeze if opts.curr_step > 0 else False
        
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


    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("==============================================")


    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    

    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet_v2_bubbliiiing': network.deeplabv3_mobilenet_v2_bubbliiiing,
        'deeplabv3plus_mobilenet_v2_bubbliiiing': network.deeplabv3plus_mobilenet_v2_bubbliiiing,
        'deeplabv3_berniwal_swintransformer_swin_t': network.deeplabv3_berniwal_swintransformer_swin_t,
        'deeplabv3_berniwal_swintransformer_swin_s': network.deeplabv3_berniwal_swintransformer_swin_s,
        'deeplabv3_berniwal_swintransformer_swin_b': network.deeplabv3_berniwal_swintransformer_swin_b,
        'deeplabv3_berniwal_swintransformer_swin_l': network.deeplabv3_berniwal_swintransformer_swin_l,      
        'deeplabv3plus_berniwal_swintransformer_swin_t': network.deeplabv3plus_berniwal_swintransformer_swin_t,
        'deeplabv3plus_berniwal_swintransformer_swin_s': network.deeplabv3plus_berniwal_swintransformer_swin_s,
        'deeplabv3plus_berniwal_swintransformer_swin_b': network.deeplabv3plus_berniwal_swintransformer_swin_b,
        'deeplabv3plus_berniwal_swintransformer_swin_l': network.deeplabv3plus_berniwal_swintransformer_swin_l,            
        'deeplabv3_microsoft_swintransformer_swin_t': network.deeplabv3_microsoft_swintransformer_swin_t,
        'deeplabv3_microsoft_swintransformer_swin_s': network.deeplabv3_microsoft_swintransformer_swin_s,
        'deeplabv3_microsoft_swintransformer_swin_b': network.deeplabv3_microsoft_swintransformer_swin_b,
        'deeplabv3_microsoft_swintransformer_swin_l': network.deeplabv3_microsoft_swintransformer_swin_l,
        'deeplabv3plus_microsoft_swintransformer_swin_t': network.deeplabv3plus_microsoft_swintransformer_swin_t,
        'deeplabv3plus_microsoft_swintransformer_swin_s': network.deeplabv3plus_microsoft_swintransformer_swin_s,
        'deeplabv3plus_microsoft_swintransformer_swin_b': network.deeplabv3plus_microsoft_swintransformer_swin_b,
        'deeplabv3plus_microsoft_swintransformer_swin_l': network.deeplabv3plus_microsoft_swintransformer_swin_l,
        'deeplabv3plus_microsoft_swintransformer_swin_t': network.deeplabv3plus_microsoft_swintransformer_swin_t,
        'deeplabv3plus_microsoft_swintransformer_swin_s': network.deeplabv3plus_microsoft_swintransformer_swin_s,
        'deeplabv3plus_microsoft_swintransformer_swin_b': network.deeplabv3plus_microsoft_swintransformer_swin_b,
        'deeplabv3plus_microsoft_swintransformer_swin_l': network.deeplabv3plus_microsoft_swintransformer_swin_l,
        'deeplabv3_hrnetv2_32': network.deeplabv3_hrnetv2_32,
        'deeplabv3plus_hrnetv2_32': network.deeplabv3plus_hrnetv2_32,
        'deeplabv3_hrnetv2_48': network.deeplabv3_hrnetv2_48,
        'deeplabv3plus_hrnetv2_48': network.deeplabv3plus_hrnetv2_48
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=bn_freeze)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
        
    if opts.curr_step > 0:
        model_prev = model_map[opts.model](num_classes=opts.num_classes[:-1], output_stride=opts.output_stride, bn_freeze=bn_freeze)

        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model_prev.classifier)
        utils.set_bn_momentum(model_prev.backbone, momentum=0.01)
    else:
        model_prev = None
    

    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), dataset=opts.dataset)


    if opts.freeze and opts.curr_step > 0:
        for param in model_prev.parameters():
            param.requires_grad = False

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.head[-1].parameters(): 
            param.requires_grad = True
            
        training_params = [{'params': model.classifier.head[-1].parameters(), 'lr': opts.lr}]
            
        if opts.unknown:
            for param in model.classifier.head[0].parameters(): 
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[0].parameters(), 'lr': opts.lr})
            
            for param in model.classifier.head[1].parameters(): 
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[1].parameters(), 'lr': opts.lr*1e-4})
        
    else:
        training_params = [{'params': model.backbone.parameters(), 'lr': 0.001},
                           {'params': model.classifier.parameters(), 'lr': 0.01}]
        
    optimizer = torch.optim.SGD(params=training_params, 
                                lr=opts.lr, 
                                momentum=0.9, 
                                weight_decay=opts.weight_decay,
                                nesterov=True)
    
    
    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
    
    utils.mkdir('checkpoints')

    best_score = -1
    cur_itrs = 0
    cur_epochs = 0
    
    if opts.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
    
    if opts.curr_step > 0: 
        opts.ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step-1)
        
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))["model_state"]
        model_prev.load_state_dict(checkpoint, strict=True)
        
        if opts.unknown and opts.w_transfer:
            curr_head_num = len(model.classifier.head) - 1

            checkpoint[f"classifier.head.{curr_head_num}.0.weight"] = checkpoint["classifier.head.0.0.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.weight"] = checkpoint["classifier.head.0.1.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.bias"] = checkpoint["classifier.head.0.1.bias"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_mean"] = checkpoint["classifier.head.0.1.running_mean"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_var"] = checkpoint["classifier.head.0.1.running_var"]

            last_conv_weight = model.state_dict()[f"classifier.head.{curr_head_num}.3.weight"]
            last_conv_bias = model.state_dict()[f"classifier.head.{curr_head_num}.3.bias"]

            for i in range(opts.num_classes[-1]):
                last_conv_weight[i] = checkpoint["classifier.head.0.3.weight"]
                last_conv_bias[i] = checkpoint["classifier.head.0.3.bias"]

            checkpoint[f"classifier.head.{curr_head_num}.3.weight"] = last_conv_weight
            checkpoint[f"classifier.head.{curr_head_num}.3.bias"] = last_conv_bias

        model.load_state_dict(checkpoint, strict=False)
        del checkpoint 
    else:
        pass

    model = nn.DataParallel(model)
    mode = model.to(device)
    mode.train()
    
    if opts.curr_step > 0:
        model_prev = nn.DataParallel(model_prev)
        model_prev = model_prev.to(device)
        model_prev.eval()

        if opts.mem_size > 0:
            memory_sampling_balanced(opts, model_prev)
            
    if not opts.crop_val:
        opts.val_batch_size = 1
    
    dataset_dict = get_dataset(opts)
    train_loader = data.DataLoader(
        dataset_dict['train'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        dataset_dict['val'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        memory_loader = data.DataLoader(
            dataset_dict['memory'], batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    total_itrs = opts.train_epoch * len(train_loader)
    val_interval = max(100, total_itrs // 100)
        

    if opts.test_only:
        model.eval()
        test_score = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        
        class_iou = list(test_score['Class IoU'].values())

        first_cls = len(get_tasks(opts.dataset, opts.task, 0)) 
        print(f"0-{first_cls-1}: mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"0-{first_cls} {len(class_iou)-1}: mIoU: %.6f" % np.mean(class_iou[first_cls:]))
        return

    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    elif opts.lr_policy=='warm_poly':
        warmup_iters = int(total_itrs*0.1)
        scheduler = utils.WarmupPolyLR(optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)

    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif opts.loss_type == 'bce_loss':
        criterion = utils.BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, 
                                                           reduction='mean')
    else:
        raise NotImplementedError
    
    if opts.KD_loss_type == 'KLDiv_loss':
        criterion_KD = torch.nn.KLDivLoss(size_average=True, reduce=True)
    elif opts.KD_loss_type == 'KD_loss':
        criterion_KD = utils.loss.KnowledgeDistillationLoss(alpha=2.0)
    elif opts.KD_loss_type == 'L1_loss':
        criterion_KD = torch.nn.L1Loss(size_average=True, reduce=True)
    elif opts.KD_loss_type == 'L2_loss':
        criterion_KD = torch.nn.BCELoss(size_average=True, reduce=True)
    else:
        raise NotImplementedError

    if opts.use_KD_layer_weight:
        KD_layer_weight = {'l1':0.5,'l2':1.0, 'l3':2.0, 'out':1.0}

    scaler = torch.cuda.amp.GradScaler(enabled=opts.amp)
    
    avg_loss = AverageMeter()
    avg_time = AverageMeter()
    avg_KD_loss_ret = AverageMeter()
    avg_CL_loss = AverageMeter()

    KD_loss_ret = torch.tensor(0.)
    KD_loss_ret_l1 = torch.tensor(0.)
    KD_loss_ret_l2 = torch.tensor(0.)
    KD_loss_ret_l3 = torch.tensor(0.)
    KD_loss_ret_out = torch.tensor(0.)
    KD_loss_outlogits = torch.tensor(0.)
    CL_loss = torch.tensor(0.)
    
    model.train()
    save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))
    
    # =====  Train  =====
    while cur_itrs < total_itrs:
        cur_itrs += 1
        optimizer.zero_grad()
        end_time = time.time()
        
        
        try:
            images, labels, _, _ = train_iter.next()
        except:
            train_iter = iter(train_loader)
            images, labels, _, _ = train_iter.next()
            cur_epochs += 1
            avg_loss.reset()
            avg_time.reset()
            avg_KD_loss_ret.reset()
            avg_CL_loss.reset()
            
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        
        if opts.cutout and random.random()>0.5:
            cutout_func = et.Cutout(1, 16) 
            for i in range(opts.batch_size):
                images[i] = cutout_func(images[i])
                labels[i] = cutout_func(labels[i])


        if opts.curr_step > 0 and opts.mem_size > 0:
            try:
                m_images, m_labels, _, _ = mem_iter.next()
            except:
                mem_iter = iter(memory_loader)
                m_images, m_labels, _, _ = mem_iter.next()

            m_images = m_images.to(device, dtype=torch.float32, non_blocking=True)
            m_labels = m_labels.to(device, dtype=torch.long, non_blocking=True)
            
            rand_index = torch.randperm(opts.batch_size)[:opts.batch_size // 2].cuda()
            images[rand_index, ...] = m_images[rand_index, ...]
            labels[rand_index, ...] = m_labels[rand_index, ...]

        

        with torch.cuda.amp.autocast(enabled=opts.amp):

            ret_features, outputs = model(images) 

            if model_prev is not None and opts.curr_step>0:
                with torch.no_grad():
                    ret_features_prev, outputs_prev = model_prev(images)
                
                if opts.contrastive_learning:
                    CL_loss = class_contrastive_learning(outputs, ret_features['feature_out'], outputs_prev, ret_features_prev['feature_out'], \
                                                            num_classes=20)
                lamb = math.pow(opts.lamb, (cur_epochs/opts.train_epoch))
                if opts.use_KD_layer_weight:
                    KD_loss_ret_l1 = KD_layer_weight['l1'] * lamb * criterion_KD(ret_features['feature_l1'], ret_features_prev['feature_l1'])
                    KD_loss_ret_l2 = KD_layer_weight['l2'] * lamb * criterion_KD(ret_features['feature_l2'], ret_features_prev['feature_l2'])
                    KD_loss_ret_l3 = KD_layer_weight['l3'] * lamb * criterion_KD(ret_features['feature_l3'], ret_features_prev['feature_l3'])
                    KD_loss_ret_out = KD_layer_weight['out'] * criterion_KD(ret_features['feature_out'], ret_features_prev['feature_out'])
                else:
                    KD_loss_ret_l1 = criterion_KD(ret_features['feature_l1'], ret_features_prev['feature_l1'])
                    KD_loss_ret_l2 = criterion_KD(ret_features['feature_l2'], ret_features_prev['feature_l2'])
                    KD_loss_ret_l2 = criterion_KD(ret_features['feature_l3'], ret_features_prev['feature_l3'])
                    KD_loss_ret_out = criterion_KD(ret_features['feature_out'], ret_features_prev['feature_out'])

                KD_loss_ret = KD_loss_ret_l1 + KD_loss_ret_l2 + KD_loss_ret_l3 + KD_loss_ret_out 

            if opts.pseudo and opts.curr_step > 0:
                with torch.no_grad():
                    ret_features_prev, outputs_prev = model_prev(images)

                if opts.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs_prev).detach() 
                else:
                    pred_prob = torch.softmax(outputs_prev, 1).detach()
                    
                pred_scores, pred_labels = torch.max(pred_prob, dim=1) 
                pseudo_labels = torch.where( (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= opts.pseudo_thresh), 
                                            pred_labels, 
                                            labels)
                    
                loss = criterion(outputs, pseudo_labels)
            else:
                loss = criterion(outputs, labels)
        
        if opts.contrastive_learning:
            CL_loss.requires_grad_(True)
            scaler.scale(CL_loss.to(device)).backward(retain_graph=True)
            avg_CL_loss.update(CL_loss.item())

        
        if model_prev is not None and opts.curr_step > 0:
            KD_loss_ret.requires_grad_(True)
            scaler.scale(KD_loss_ret).backward(retain_graph=True)  
            avg_KD_loss_ret.update(KD_loss_ret.item())


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        avg_loss.update(loss.item())
        avg_time.update(time.time() - end_time)
        end_time = time.time()

        if (cur_itrs) % 50 == 0:
            print("[Step %d] Epoch %d, Iters %d/%d, Loss=%6f, Time=%.2f , LR=%.8f" %
                  (opts.curr_step, cur_epochs, cur_itrs, total_itrs, 
                   avg_loss.avg,  avg_time.avg*1000, optimizer.param_groups[0]['lr']))

        if val_interval > 0 and (cur_itrs) % val_interval == 0:
            print(">>> Validation")
            model.eval()
            val_score = validate(opts=opts, model=model, loader=val_loader, 
                                 device=device, metrics=metrics)
            print(metrics.to_str(val_score))
            
            model.train()
            
            class_iou = list(val_score['Class IoU'].values())
            val_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] + [class_iou[0]])
            curr_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] )
            print("val_res : %.4f" % (curr_score))
            
            if curr_score > best_score: 
                best_score = curr_score
                save_ckpt(ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step))


    print(">>>End Training")
    
    if opts.curr_step > 0:
        best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step)
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        
        test_score = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        print(metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        first_cls = len(get_tasks(opts.dataset, opts.task, 0))

        print(f"0 - {first_cls-1} : mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"{first_cls} - {len(class_iou)-1} mIoU : %.6f" % np.mean(class_iou[first_cls:]))



if __name__ == '__main__':

    opts = get_argparser().parse_args()
        
    start_step = 0
    total_step = len(get_tasks(opts.dataset, opts.task))
    
    for step in range(start_step, total_step):
        opts.curr_step = step
        main(opts)
        
