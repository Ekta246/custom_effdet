""" COCO transforms (quick and dirty)

Hacked together by Ross Wightman
"""
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import math
import cv2
import sys
sys.path.append("/home/ekta/AI_current/DataAugmentationForObjectDetection/")
from data_aug.bbox_util import get_corners, rotate_im, rotate_box, get_enclosing_box
#from DataAugmentationForObjectDetection import data_aug

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class ImageToNumpy:

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return np_img, annotations


class ImageToTensor:

    def __init__(self, dtype=torch.float32):
        self.dtype = dtype

    def __call__(self, pil_img, annotations: dict):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.moveaxis(np_img, 2, 0)  # HWC to CHW
        return torch.from_numpy(np_img).to(dtype=self.dtype), annotations


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def clip_boxes_(boxes, img_size):
    height, width = img_size
    clip_upper = np.array([height, width] * 2, dtype=boxes.dtype)
    np.clip(boxes, 0, clip_upper, out=boxes)


def clip_boxes(boxes, img_size):
    clipped_boxes = boxes.copy()
    clip_boxes_(clipped_boxes, img_size)
    return clipped_boxes


def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size
def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
        
def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    
    Parameters
    ----------
    
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`
        
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2` 
    
    """
    
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    mask = (delta_area < (1 - alpha)).astype(int)
    
    bbox = bbox[mask == 1,:]


    return bbox


class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear', fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img, anno: dict):
        width, height = img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        interp_method = _pil_interp(self.interpolation)
        img = img.resize((scaled_w, scaled_h), interp_method)
        new_img.paste(img)

        if 'bbox' in anno:
            # FIXME haven't tested this path since not currently using dataset annotations for train/eval
            bbox = anno['bbox']
            bbox[:, :4] *= img_scale
            clip_boxes_(bbox, (scaled_h, scaled_w))
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        anno['img_scale'] = 1. / img_scale  # back to original
    
        return new_img, anno


class RandomResizePad:

    def __init__(self, target_size: int, scale: tuple = (0.1, 2.0), interpolation: str = 'random',
                 fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.scale = scale
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.fill_color = fill_color

    def _get_params(self, img):
        # Select a random scale factor.
        scale_factor = random.uniform(*self.scale)
        scaled_target_height = scale_factor * self.target_size[0]
        scaled_target_width = scale_factor * self.target_size[1]

        # Recompute the accurate scale_factor using rounded scaled image size.
        width, height = img.size
        img_scale_y = scaled_target_height / height
        img_scale_x = scaled_target_width / width
        img_scale = min(img_scale_y, img_scale_x)

        # Select non-zero random offset (x, y) if scaled image is larger than target size
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)
        offset_y = scaled_h - self.target_size[0]
        offset_x = scaled_w - self.target_size[1]
        offset_y = int(max(0.0, float(offset_y)) * random.uniform(0, 1))
        offset_x = int(max(0.0, float(offset_x)) * random.uniform(0, 1))
        return scaled_h, scaled_w, offset_y, offset_x, img_scale

    def __call__(self, img, anno: dict):
        scaled_h, scaled_w, offset_y, offset_x, img_scale = self._get_params(img)

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        img = img.resize((scaled_w, scaled_h), interpolation)
        right, lower = min(scaled_w, offset_x + self.target_size[1]), min(scaled_h, offset_y + self.target_size[0])
        img = img.crop((offset_x, offset_y, right, lower))
        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        new_img.paste(img)

        if 'bbox' in anno:
            # FIXME not fully tested
            bbox = anno['bbox'].copy()  # FIXME copy for debugger inspection, back to inplace
            bbox[:, :4] *= img_scale
            box_offset = np.stack([offset_y, offset_x] * 2)
            bbox -= box_offset
            clip_boxes_(bbox, (scaled_h, scaled_w))
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]
        
        anno['img_scale'] = 1. / img_scale  # back to original

        return new_img, anno

class RandomRotate:
    """Randomly rotates an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn 
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, angle = 60):
        self.angle = angle
        
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"  
            
        else:
            self.angle = (-self.angle, self.angle)
            
    def __call__(self, img, anno:dict):
    
        angle = random.uniform(*self.angle)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        img = rotate_im(img, angle)
        #corners = get_corners(bboxes)
        #corners = np.hstack((corners, bboxes[:,4:]))
        #corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        #new_bbox = get_enclosing_box(corners)
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        img = cv2.resize(img, (w,h))
        #new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        #bboxes  = new_bbox
        #bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

        if 'bbox' in anno:
            bbox = anno['bbox'].copy()
            corners = get_corners(bbox)
            corners = np.hstack((corners, bbox[:,4:]))
            corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
            new_bbox = get_enclosing_box(corners)
            bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
            bbox=new_bbox 
            #bbox=clip_box(bbox, [0,0,w, h], 0.50)
            #clip_boxes_(bbox, (scaled_h, scaled_w))
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            #anno['cls'] = anno['cls'][valid_indices]
        '''cpt=0
        for i in range(611):
            #new_img.paste(img)
            open_cv_img= np.array(img)
            save_image= open_cv_img[:,:,::-1].copy()
            #annots= np.round(anno['bbox'][:, :4])
            cv2.rectangle(save_image, (int(anno['bbox'][0][1]), int(anno['bbox'][0][0])), (int(anno['bbox'][0][3]), int(anno['bbox'][0][2])), (0,255,0), 2)
            filename_d = "/home/ekta/AI_current/vdot/vdot/saved_images/file_gt%d.jpg"%cpt
            cv2.imwrite(filename_d ,save_image)
            cpt+=1'''
        
        img = Image.fromarray(img)
        return img, anno

class RandomFlip:

    def __init__(self, horizontal=True, vertical=False, prob=0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.prob = prob

    def _get_params(self):
        do_horizontal = random.random() < self.prob if self.horizontal else False
        do_vertical = random.random() < self.prob if self.vertical else False
        return do_horizontal, do_vertical

    def __call__(self, img, annotations: dict):
        do_horizontal, do_vertical = self._get_params()
        width, height = img.size

        def _fliph(bbox):
            x_max = width - bbox[:, 1]
            x_min = width - bbox[:, 3]
            bbox[:, 1] = x_min
            bbox[:, 3] = x_max

        def _flipv(bbox):
            y_max = height - bbox[:, 0]
            y_min = height - bbox[:, 2]
            bbox[:, 0] = y_min
            bbox[:, 2] = y_max

        if do_horizontal and do_vertical:
            img = img.transpose(Image.ROTATE_180)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
                _flipv(annotations['bbox'])
        elif do_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if 'bbox' in annotations:
                _fliph(annotations['bbox'])
        elif do_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if 'bbox' in annotations:
                _flipv(annotations['bbox'])

        return img, annotations


'''
class ZoomImg:


    def zoom_out(image, boxes):
    
        #Zoom out image (max scale = 4)
        #image: A PIL image
        #boxes: bounding boxes, a tensor of dimensions (#objects, 4)
        
        #Out: new_image, new_boxes
    
        original_h = image.size(1)
        original_w = image.size(2)
        max_scale = 4
        scale = random.uniform(1, max_scale)
        new_h = int(scale*original_h)
        new_w = int(scale*original_w)
    
        #Create an image with the filler
        filler = [0.485, 0.456, 0.406]
        filler = torch.FloatTensor(filler) #(3)
        new_image = torch.ones((3, new_h, new_w), dtype= torch.float) * filler.unsqueeze(1).unsqueeze(1)
    

        left = random.randint(0, new_w - original_w)
        right = left + original_w
        top = random.randint(0, new_h - original_h)
        bottom = top + original_h
    
        new_image[:, top:bottom, left:right] = image
    
        #Adjust bounding box
        new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)
    
        return new_image, new_boxes'''

'''class RandomScale:
    """Randomly scales an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn 
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, scale: tuple = (0.1, 2.0), diff = False):
        self.scale = scale

        
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
        
        self.diff = diff

        

    def __call__(self, img, anno:dict):
    
        
        #Chose a random digit toscale by 
        #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        img_size=img.shape
        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x
            
    
        
        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y
        
        img=  cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y )
        
        #bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
        
        
        canvas = np.zeros(img_size, dtype = np.uint8)
        
        y_lim = int(min(resize_scale_y,1)*img_size[0])
        x_lim = int(min(resize_scale_x,1)*img_size[1])
        
        
        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
        
        img = canvas
        #bboxes = clip_box(bboxes, [0,0,1 + img_size[1], img_size[0]], 0.25)

        if 'bbox' in anno:
            # FIXME haven't tested this path since not currently using dataset annotations for train/eval
            bbox = anno['bbox'].copy()
            bbox[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
            clip_box(bbox, [0,0,1 + img_size[1], img_size[0]], 0.75)
            #clip_boxes_(bbox, (y_lim, x_lim))
            valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
            anno['bbox'] = bbox[valid_indices, :]
            anno['cls'] = anno['cls'][valid_indices]

        img = Image.fromarray(img)
        return img, anno'''



def resolve_fill_color(fill_color, img_mean=IMAGENET_DEFAULT_MEAN):
    if isinstance(fill_color, tuple):
        assert len(fill_color) == 3
        fill_color = fill_color
    else:
        try:
            int_color = int(fill_color)
            fill_color = (int_color,) * 3
        except ValueError:
            assert fill_color == 'mean'
            fill_color = tuple([int(round(255 * x)) for x in img_mean])
    return fill_color


class Compose:

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img, annotations: dict):
        for t in self.transforms:
            img, annotations = t(img, annotations)
        return img, annotations


def transforms_coco_eval(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    fill_color = resolve_fill_color(fill_color, mean)

    image_tfl = [
        ResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color),
        ImageToNumpy(),
    ]

    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf


def transforms_coco_train(
        img_size=224,
        interpolation='random',
        use_prefetcher=False,
        fill_color='mean',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):

    fill_color = resolve_fill_color(fill_color, mean)

    image_tfl = [
        RandomFlip(horizontal=True, prob=0.5),
        #RandomRotate(),
        #RandomScale(diff = False),
        RandomResizePad(
            target_size=img_size, interpolation=interpolation, fill_color=fill_color),
        ImageToNumpy(),
    ]


    assert use_prefetcher, "Only supporting prefetcher usage right now"

    image_tf = Compose(image_tfl)
    return image_tf
