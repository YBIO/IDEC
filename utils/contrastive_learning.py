'''
 * @Author: YBIO
 * @Date: 2022-05-07 19:07:31 
 * @Last Modified by:   YBIO
 * @Last Modified time: 2022-05-07 19:07:31 
 '''


import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import cv2
import random

from PIL import Image
from tqdm import tqdm
from utils.loss import CircleLoss, convert_label_to_similarity, ContrastiveLoss



def pixel_contrastive_learning(outputs: torch.tensor, outputs_prev: torch.tensor, pixel_num=10, use_sigmoid=False, loss_type='L1'):
    '''
    Args:
        outputs: model outputs in current step       : [b, curr_numclass, h, w]
        outputs_prev: model outputs in previous step : [b, prev_numclass, h, w]
        pixel_num: select pixel_num pixels to calculate contrastive loss
        use_sigmoid: if True, use torch.sigmoid otherwise torch.softmax
        loss_type: 
    return:
        contrastive loss
    '''
    if use_sigmoid:
        pred_prob = torch.sigmoid(outputs).detach()
        pred_prob_prev = torch.sigmoid(outputs_prev).detach()
    else: 
        pred_prob = torch.softmax(outputs, 1).detach()
        pred_prob_prev = torch.softmax(outputs_prev, 1).detach()


    pred_scores, pred_labels = torch.max(pred_prob, dim=1)  
    pred_scores_prev, pred_labels_prev = torch.max(pred_prob_prev, dim=1)  


    imgsize = outputs.size(2) 

    for i in range(pixel_num):
        # constrative learning in incremental class
        pixel_loc = random.uniform((0, imgsize)) 
        # random a pixel in the image
        anchor_embedding = outputs[0, :, pixel_loc, pixel_loc]
        anchor_label = pred_scores[0, :, pixel_loc, pixel_loc]
        #
        postive_embedding = pred_labels.cpu().numpy()[np.where(pred_labels=anchor_label)]
        negative_embedding = pred_labels_prev.cpu().numpy()
    



def class_contrastive_learning(outputs: torch.tensor, feature: torch.tensor, outputs_prev: torch.tensor, feature_prev: torch.tensor, num_classes=20, min_classes=10, task=15-5, use_sigmoid=True, unknown=True):
    '''
    Args:
        outputs: model outputs in t-step       : [b, curr_numclass, h, w]
        feature: embedding in t-step           : [b, 2048, 33, 33]
        outputs_prev: model outputs in previous step : [b, prev_numclass, h, w]
        feature_prev: embedding in t-1 step    : [b, 2048, 33, 33]
        num_classes: t-step classes number
        task: task format, e.g., 15-5,10-1,100-50,...
        use_sigmoid: if True, use torch.sigmoid otherwise torch.softmax
        
    return:
        contrastive loss
    '''
    if unknown:
        outputs[:, 1] += outputs[:, 0]
        outputs = outputs[:, 1:]
        outputs_prev[:, 1] += outputs_prev[:, 0]
        outputs_prev = outputs_prev[:, 1:]

    if use_sigmoid:
        pred_prob = torch.sigmoid(outputs).detach()
        pred_prob_prev = torch.sigmoid(outputs_prev).detach()
    else: 
        pred_prob = torch.softmax(outputs, 1).detach()
        pred_prob_prev = torch.softmax(outputs_prev, 1).detach()


    criterion = nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')

    pred_scores, pred_labels = torch.max(pred_prob, dim=1)  
    pred_scores_prev, pred_labels_prev = torch.max(pred_prob_prev, dim=1)  

    imgsize = outputs.size(-1)

    feature = F.interpolate(feature, size=imgsize, mode='bilinear', align_corners=False)
    feature_prev = F.interpolate(feature_prev, size=imgsize, mode='bilinear', align_corners=False)
    feature = feature[:,:256,:,:]
    feature_prev = feature_prev[:,:256,:,:]


    mask = torch.zeros(pred_labels_prev[0].size(), dtype=torch.float32)

    device = torch.device('cuda:0')

    contrastive_loss = torch.tensor(0.).to(device)

    for i in range(min(min_classes,num_classes)):

        class_pixel_coord = torch.nonzero(pred_labels_prev==i+random.randint(1,num_classes-i-1)) 

        for coord in class_pixel_coord:
            mask[coord[-2].cpu().numpy(),coord[-1].cpu().numpy()] = 1.

        bool_mask = mask > 0


        class_embedding_anchor = torch.masked_select(feature_prev[0], bool_mask)
        class_embedding_anchor = class_embedding_anchor[class_embedding_anchor>0][:256]
        class_embedding_anchor=class_embedding_anchor.to(device)

        class_embedding_positive = torch.masked_select(feature[0], bool_mask)
        class_embedding_positive = class_embedding_positive[class_embedding_positive>0][:256]
        class_embedding_positive=class_embedding_positive.to(device)

        bool_mask_reverse = bool_mask==False
        class_embedding_negative = torch.masked_select(feature[0], bool_mask_reverse)
        class_embedding_negative = class_embedding_negative[class_embedding_negative>0][:256]
        class_embedding_negative=class_embedding_negative.to(device)

        if class_embedding_negative.size()!= class_embedding_anchor.size():
            continue
        if class_embedding_negative.size()!= class_embedding_positive.size():
            continue
        if class_embedding_positive.size()!= class_embedding_anchor.size():
            continue
        


        contrastive_loss += criterion(class_embedding_anchor.unsqueeze(0), class_embedding_positive.unsqueeze(0), class_embedding_negative.unsqueeze(0))

    contrastive_loss = contrastive_loss / min(min_classes, num_classes)

    return contrastive_loss

        

        
def class_contrastive_learning_new(outputs: torch.tensor, feature: torch.tensor, outputs_prev: torch.tensor, feature_prev: torch.tensor, num_classes=20, min_classes=10, task=15-5, use_sigmoid=True, unknown=True):
    pass

