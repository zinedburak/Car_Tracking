import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from utils.datasets import LoadImages, LoadStreams
from utils.utils import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from PyQt5 import QtGui, QtMultimedia
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap

from functionalities import *

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def rel_bbox(*xyxy):
    """Turning x0,y0 , x1,y1 To x,y,w,h"""
    bbox_l = min([xyxy[0].item(), xyxy[2].item()])
    bbox_t = min([[xyxy[1].item()], xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_l + bbox_w / 2)
    y_c = (bbox_t + bbox_h / 2)
    return x_c, y_c, bbox_w, bbox_h


def label_color(label):
    color = [int((p * label ** 2 - label + 1) % 255) for p in palette]
    return tuple(color)


def draw_bb(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1, x2 = x1 + offset[0], x2 + offset[0]
        y1, y2 = y1 + offset[1], y2 + offset[1]

        id = int(identities[i] if identities is not None else 0)
        color = label_color(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label,cv2.FONT_HERSHEY_PLAIN)
