'''
 * @Author: YuanBo 
 * @Date: 2022-05-07 19:07:31 
 * @Last Modified by:   YuanBo 
 * @Last Modified time: 2022-05-07 19:07:31 
 '''

## contrastive learning during incremental steps
## metric learning + intra-class compactness and inter-class dispersion + Asymmetric Region-wise Contrastive Learning


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

    #[b,c,h,w] -> torch.max() -> [b,1,h,w]
    pred_scores, pred_labels = torch.max(pred_prob, dim=1)  # [b,1,513,513]
    pred_scores_prev, pred_labels_prev = torch.max(pred_prob_prev, dim=1)  # [b,1,513,513]

    # contrastive learning process
    imgsize = outputs.size(2) # h or w
    # random select some pixel-embeddings to proceed contrastive learning
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


    # use metric learning loss 
    criterion = nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
    # loss = criterion(anchor, positive, negative) #input three 2D tensors，each tensor：(B,N)，B is batchsize，N is tensor dimension
    # criterion = CircleLoss(m=0.25, gamma=256)

    #[b,c,h,w] -> torch.max() -> [b,1,h,w]
    pred_scores, pred_labels = torch.max(pred_prob, dim=1)  # [b,1,513,513]  pred_scores is logits, pred_labels is class
    pred_scores_prev, pred_labels_prev = torch.max(pred_prob_prev, dim=1)  # [b,1,513,513]

    imgsize = outputs.size(-1) # h or w
    # feature interpolate [b, 2048, 33, 33] -> [b, 2048, 513, 513]
    feature = F.interpolate(feature, size=imgsize, mode='bilinear', align_corners=False)
    feature_prev = F.interpolate(feature_prev, size=imgsize, mode='bilinear', align_corners=False)
    feature = feature[:,:256,:,:]  # channel dimension reduction
    feature_prev = feature_prev[:,:256,:,:]

    #对每一个类别的相应进行正负样本对的构建，以t-1步模型的类别响应为anchor，以t步模型的相同类别响应
    #构建正样本对，不同类别响应构建负样本对
    
    # create a mask for filtering the specific class pixels
    mask = torch.zeros(pred_labels_prev[0].size(), dtype=torch.float32)
    
    

    # 遍历每个类，以t-1-step model输出的结果作为anchor，t-step model输出的结果作为positive和negative
    device = torch.device('cuda:0')
    # initilize loss value
    contrastive_loss = torch.tensor(0.).to(device)

    for i in range(min(min_classes,num_classes)):
        #寻找torch.max之后对应i类别的坐标索引
        # class_pixel_coord = torch.nonzero(pred_labels_prev==i+1) # skip background class
        class_pixel_coord = torch.nonzero(pred_labels_prev==i+random.randint(1,num_classes-i-1)) # skip background class
        #生成mask，属于i类的像素点值置1，其余为0
        for coord in class_pixel_coord:
            mask[coord[-2].cpu().numpy(),coord[-1].cpu().numpy()] = 1.

        bool_mask = mask > 0
        # mask = mask.unsqueeze(1)
        
        # class_embedding_anchor = feature_prev[0] * mask          # anchor embeding from t-1 step model
        class_embedding_anchor = torch.masked_select(feature_prev[0], bool_mask)
        class_embedding_anchor = class_embedding_anchor[class_embedding_anchor>0][:256]
        class_embedding_anchor=class_embedding_anchor.to(device)
        # print('anchor:',class_embedding_anchor.size())
        # class_embedding_positive = feature[0] * mask             # positive embedding from t step model
        class_embedding_positive = torch.masked_select(feature[0], bool_mask)
        class_embedding_positive = class_embedding_positive[class_embedding_positive>0][:256]
        class_embedding_positive=class_embedding_positive.to(device)
        # print('positive:', class_embedding_positive.size())
        # random_negative_index = random.randint(0,i) if i>1 else random.randint(i+1, num_classes)
        bool_mask_reverse = bool_mask==False
        class_embedding_negative = torch.masked_select(feature[0], bool_mask_reverse)
        class_embedding_negative = class_embedding_negative[class_embedding_negative>0][:256]
        class_embedding_negative=class_embedding_negative.to(device)
        # print('negative:', class_embedding_negative.size())
        if class_embedding_negative.size()!= class_embedding_anchor.size():
            continue
        if class_embedding_negative.size()!= class_embedding_positive.size():
            continue
        if class_embedding_positive.size()!= class_embedding_anchor.size():
            continue
        

        # nn.TripletMarginLoss()输入是anchor, positive, negative三个B*N的张量（表示Batchsize个N为的特征向量），输出triplet loss的值。
        contrastive_loss += criterion(class_embedding_anchor.unsqueeze(0), class_embedding_positive.unsqueeze(0), class_embedding_negative.unsqueeze(0))
        # circle loss
        # contrastive_loss += criterion(convert_label_to_similarity(class_embedding_positive.unsqueeze(0), class_embedding_anchor.unsqueeze(0)))
    contrastive_loss = contrastive_loss / min(min_classes, num_classes)

    return contrastive_loss

        

        
def class_contrastive_learning_new(outputs: torch.tensor, feature: torch.tensor, outputs_prev: torch.tensor, feature_prev: torch.tensor, num_classes=20, min_classes=10, task=15-5, use_sigmoid=True, unknown=True):
    pass

