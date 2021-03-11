# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
import json, glob
from pycocotools.coco import COCO
from pycocotools.mask import merge
from itertools import groupby


img_path = 'examples/'
save = './ours_r50/'
img_save = save + '%s.png'
enable_vis = True


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))
    return rle

def mask_save(masks, save_place):
    num_masks = masks.shape[2]
    new_mask = np.zeros(masks.shape[:2], dtype=np.uint8)
    for idx in range(num_masks):
        mask = masks[:,:,idx] == 1
        new_mask[mask==True] = idx+1
    cv2.imwrite(save_place, new_mask)
        
        

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        # default="../configs/caffe2/e2e_mask_rcnn_R_101_FPN_0.1x.yaml",
        default="../configs/rdpnet/r50-isod-te.yaml",
        metavar="FILE",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )
    import os
    if not os.path.exists(save):
        os.mkdir(save)
    all_masks = []; all_labels = []
    start_time = time.time()

    imgs = glob.glob(img_path + '/*.jpg')
    res = []
    all_masks = []; all_labels = []
    for idx, long_path in enumerate(imgs):
        this_masks = []; this_labels = []
        #print(img_path%img_name)
        img_name = os.path.basename(long_path)
        img = cv2.imread(img_path + img_name)
        composite, masks, labels, scores, t = coco_demo.run_on_opencv_image(img, img_name)
        for j in range(len(labels)):
            this_masks.append(masks[j])
            this_labels.append(0)
        if len(this_masks):
            this_masks = np.array(this_masks).astype(np.uint8).transpose(1,2,0)
        else:
            this_masks = np.zeros((1,1,0))
        all_masks.append(this_masks); all_labels.append(this_labels)
        if enable_vis:
            cv2.imwrite(img_save%img_name[:-4], composite)
        print("\r Time: {:.2f} s / img, {:d}/{:d} images".format((time.time() - start_time)/(idx+1), idx+1, len(imgs)), end="")

if __name__ == "__main__":
    main() 
