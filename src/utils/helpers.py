import torch
import torchvision

import pandas as pd
import cv2
import ast

from torch.utils.data.sampler import SequentialSampler

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def show_image(img_n=0):
    image = cv2.imread(df.loc[img_n, 'images_paths'])
    bbox_str = df.loc[img_n, 'bboxes']
    bbox = ast.literal_eval(bbox_str)
    if bbox != 0:
        for box in bbox:
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)

    figure(figsize=(12, 8), dpi=80)
    plt.imshow(image)


def collate_fn_base(batch):
    return tuple(zip(*batch))
