import numpy as np
import random
import multiprocessing as mp
import argparse
import os
import glob
from pathlib import Path

import torch as T

import sys
sys.path.insert(1, os.path.join(sys.path[0], 'OneFormer'))

import time
import cv2
import numpy as np
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from demo.defaults import DefaultPredictor

__author__ = "__Dinesh_Dhotrad__"


class OneFormer:
    """ Wraper detectron2 OneFormer model with utility functions
        author: Dinesh Dhotrad
        contact: dxd539@case.edu
    """
    def __init__(self):
        print('preparing OneFormer model ...')
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        cfg.merge_from_file("./OneFormer/configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml")
        cfg.MODEL.WEIGHTS = './OneFormer/PreTrained/COCO/ckpt/150_16_swin_l_oneformer_coco_100ep.pth'
        # cfg.freeze()
        
        self.predictor = DefaultPredictor(cfg)
        
    @T.no_grad()
    def predict(self, image):
        """
        Args:
            image (np.ndarray): [H, W, 3] cv2 default bgr image.
        Returns:
            torch.tensor: [batch_size, num_labels, height, width] logits
            torch.tensor: [batch_size, height, width] segmentaion mask

        Note:
            # sem, pan, inst = outputs.values()
            # type(sem) = torch.Tensor, sem.shape = (133, image.height, image.width), sem.dtype = torch.float32
            # pan is a tuple of id image and id info
            # idimage, info = pan
            # type(idimage) = torch.Tensor, idimage.shape = (image.height, image.width), sem.dtype = torch.int32
            # info = [{'id': id, 'isthing': bool, 'category_id': object_id out of 133 classes, 'area': num of pixels} for id in idimage.max()]
        """
        outputs = self.predictor(image, task = 'semantic')
        return outputs


    
def SegmentImage(input_dir, output_dir, extension="jpg", conf_threshold = 0.017, filter_classes=None):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    
    start_time = time.perf_counter()
    filter_classes = set(filter_classes) if filter_classes is not None else None
    os.makedirs(output_dir, exist_ok=True)
    # root_dir = output_dir.split('masks')[0][:-1]
    viz_dir = os.path.join(output_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    images = glob.glob(f'{input_dir}/*{extension}')
    
    segmentor = OneFormer()
    print('predicting ...')
    total = len(images)
    for i, image in tqdm(enumerate(sorted(images))):
        name = os.path.basename(image)
        image = cv2.imread(image)
        outputs = segmentor.predict(image, )
        sem, pan, inst = outputs.values()
        sem_image = sem.argmax(dim=0).to("cpu")
        
        if conf_threshold:    
            softmax_fn = T.nn.Softmax(dim=0)
            sem_soft = softmax_fn(sem)
            sem_soft_image = T.amax(sem_soft, dim = 0)
            mask = sem_soft_image < conf_threshold
            sem_image[mask] = 133
            
        sem_image = np.array(sem_image.cpu())
        v = Visualizer(image[:, :, ::-1], coco_metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
        semantic_result = v.draw_sem_seg(sem_image).get_image()   
        if filter_classes is not None:
            if len(set(np.unique(sem_image))&set(filter_classes)) == 0: continue
        cv2.imwrite(viz_dir + str('/') + Path(name).stem + '.png', semantic_result)
        cv2.imwrite(output_dir + str('/') + Path(name).stem + '.png', sem_image)

if __name__ == "__main__":
    print("Generating Masks Using OneFormer")
    input_dir = "test_data/rtab/rgb"
    output_dir = "test_data/rtab/masks"
    SegmentImage(input_dir, output_dir)