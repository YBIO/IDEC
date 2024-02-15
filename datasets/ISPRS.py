import os
import sys
import torch.utils.data as data
import numpy as np
import json

import torch
from PIL import Image

from utils.tasks import get_dataset_list, get_tasks

classes = {
        'Imprevious surfaces',
        'Building', 
        'Low vegetation',
        'Tree',
        'Car',
        'Background'
}


def ISPRS_cmap():
    cmap = np.zeros((256, 3), dtype=np.uint8)
    colors = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0]
    ]

    for i in range(len(colors)):
        cmap[i] = colors[i]

    return cmap.astype(np.uint8)

class ISPRSSegmentation(data.Dataset):
    cmap = ISPRS_cmap()
    def __init__(self,
                 opts,
                 image_set='train',
                 transform=None, 
                 cil_step=0, 
                 mem_size=0):

        self.root=opts.data_root        
        self.task=opts.task
        self.overlap=opts.overlap
        self.unknown=opts.unknown
        
        self.image_set = image_set
        self.transform = transform
        
        ISPRS_root =  './datasets/data/ISPRS'
        
        if self.image_set == 'train' or self.image_set == 'memory':
            split = 'training'
        else:
            split = 'validation'
            
        image_dir = os.path.join(self.root, 'images', split)
        mask_dir = os.path.join(self.root, 'annotations', split)
        
        assert os.path.exists(mask_dir), "annotations not found"
            
        self.target_cls = get_tasks('ISPRS', self.task, cil_step)
        self.target_cls += [255] 
            
        if image_set=='test':
            file_names = open(os.path.join(ISPRS_root, 'val_vaihingen.txt'), 'r')
            file_names = file_names.read().splitlines()
            
        elif image_set == 'memory':
            for s in range(cil_step):
                self.target_cls += get_tasks('ISPRS', self.task, s)
            
            memory_json = os.path.join(ISPRS_root, 'memory.json')

            with open(memory_json, "r") as json_file:
                memory_list = json.load(json_file)

            file_names = memory_list[f"step_{cil_step}"]["memory_list"]
            
            while len(file_names) < opts.batch_size:
                file_names = file_names * 2
                
        else:
            file_names = get_dataset_list('ISPRS', self.task, cil_step, image_set, self.overlap)

        self.images = [os.path.join(image_dir, x + ".png") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        self.file_names = file_names
        
        all_steps = get_tasks('ISPRS', self.task)
        all_classes = []
        for i in range(len(all_steps)):
            all_classes += all_steps[i]
            
        self.ordering_map = np.zeros(256, dtype=np.uint8) + 255
        self.ordering_map[:len(all_classes)] = [all_classes.index(x) for x in range(len(all_classes))]
        
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        file_name = self.file_names[index]
        
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        
        target = self.gt_label_mapping(target)
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        if self.image_set == 'train' and self.unknown:
            
            target = torch.where(target == 255, 
                                 torch.zeros_like(target) + 255, 
                                 target+1) 
            
            unknown_area = (target == 1)
            target = torch.where(unknown_area, torch.zeros_like(target), target)

        return img, target.long(), file_name


    def __len__(self):
        return len(self.images)
    
    def gt_label_mapping(self, gt):
        gt = np.array(gt, dtype=np.uint8)
        if self.image_set != 'test':
            gt = np.where(np.isin(gt, self.target_cls), gt, 0)
        gt = self.ordering_map[gt]
        gt = Image.fromarray(gt)
        
        return gt
    
    @classmethod
    def decode_target(cls, mask):
        return cls.cmap[mask]

