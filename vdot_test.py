import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
import torch.utils.data as data
from torch.utils import data
from matplotlib.image import imread
from pycocotools.coco import COCO
from effdet.data.parsers import create_parser

from effdet.config import model_config
from torch.utils.data import Dataset, DataLoader
class VdotTestDataset(data.Dataset):
    def __init__(self, image_dir,  ann_file,  transform=None):
        super(VdotTestDataset, self).__init__
        
        self.transform = transform
        self.image_dir=image_dir
        self.ann_file = ann_file
        self.coco = None
        #ann = open('/home/ekta/AI_current/vdot/vdot/train_annotations/train_annotations.json', 'r')
        ann = open(self.ann_file)
        data_json = json.load(ann)
        self.yxyx = True  
        self.data_json = data_json
        #image_dir = os.listdir('/home/ekta/AI_current/vdot/vdot/train_set')
        image_dir=os.listdir(self.image_dir)
        total_num_images = len(image_dir)
        self.total_num_images = total_num_images
        self.imgs_list, self.annot_list = self.parse_labels(self.data_json)
        
        self._transform = transform
        
    def parse_labels(self, ann_file):

        #annot_lists=[]
        filename_labels = []
        frame_boxes =[]
        det_dict = {}
        for k,v in self.data_json.items():
            filename_labels.append(v['filename'])
            det_dict =  {'img_size': (800, 600)}
            frame_boxes.append(det_dict)
        return filename_labels, frame_boxes
    
    def __len__(self):
       return self.total_num_images
    
    def __getitem__(self, index):
        self.image_name = self.imgs_list[index]
        labels = self.annot_list[index]
        labels['img_id'] = int(index)
        #labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        #img = imread(os.path.join('/home/ekta/AI_current/vdot/vdot/train_set', self.image_name))
        img= Image.open(os.path.join(self.image_dir, self.image_name)).convert('RGB')
        if self.transform is not None:
            img, labels = self.transform(img, labels)
            
        return img, labels