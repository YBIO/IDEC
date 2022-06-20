
#  @Author: YuanBo 
#  @Date: 2022-05-26 15:31:56 
#  @Last Modified by:   YuanBo 
#  @Last Modified time: 2022-05-26 15:31:56 

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


def show_feature_map(feature_map):#feature_map=torch.Size([1, 256, 33, 33])
    feature_map = feature_map.cpu()                                                                    
    feature_map = feature_map.squeeze(0)#压缩成torch.Size([256, 33, 33])
    
    #以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map =feature_map.view(1,feature_map.shape[0],feature_map.shape[1],feature_map.shape[2])#(1,256,33,33)
    upsample = torch.nn.UpsamplingBilinear2d(size=(33,33))#这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1],feature_map.shape[2],feature_map.shape[3])
    
    feature_map_num = feature_map.shape[0]#返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))#8
    plt.figure()
    for index in range(1, feature_map_num + 1):#通过遍历的方式，将64个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1], cmap='gray')#feature_map[0].shape=torch.Size([33, 33])
        #将上行代码替换成，可显示彩色 
        #plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([33, 33])
        plt.imsave('visualize_results/ret_features/' + str(index) + ".png",transforms.ToPILImage()(feature_map[index - 1])) 
        plt.axis('off')
    plt.show()


# import this func at network/_deeplab.py, can visualize feature['low_layer_1'], etc.
def draw_features(width,height,x,savepath):
    x=x.cpu().numpy()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        # plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        fig.savefig("{}/{}.png".format(savepath,i), dpi=100)
        # if want to save all feature maps to one image, unannotation the following 2 lines
        # print("{}/{}".format(i,width*height))
    # fig.savefig(savepath, dpi=100)
    fig.clf()
    plt.close()
