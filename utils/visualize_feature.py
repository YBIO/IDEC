'''
 * @Author: YuanBo 
 * @Date: 2022-05-07 19:07:31 
 * @Last Modified by:   YuanBo 
 * @Last Modified time: 2022-05-07 19:07:31 
 '''

import network
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import scipy.misc

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


def show_feature_map(feature_map):
    feature_map = feature_map.cpu()                                                                    
    feature_map = feature_map.squeeze(0)
    
    feature_map =feature_map.view(1,feature_map.shape[0],feature_map.shape[1],feature_map.shape[2])#(1,256,33,33)
    upsample = torch.nn.UpsamplingBilinear2d(size=(33,33))
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1],feature_map.shape[2],feature_map.shape[3])
    
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num + 1):

        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1], cmap='gray')
        plt.imsave('visualize_results/ret_features/' + str(index) + ".png",transforms.ToPILImage()(feature_map[index - 1])) 
        plt.axis('off')
    plt.show()


# import this func at network/_deeplab.py, can visualize feature['low_layer_1'], etc.
def draw_features(width,height,x,savepath):
    x=x.cpu().numpy()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  
        img=img.astype(np.uint8) 
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) 
        img = img[:, :, ::-1]
        plt.imshow(img)
        fig.savefig("{}/{}.png".format(savepath,i), dpi=100)


    fig.clf()
    plt.close()
