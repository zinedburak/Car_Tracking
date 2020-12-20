import time
import numpy as np
import cv2
import matplotlib.pyplot as ply

import tensorflow as tf
from yolov5.models import yolo
from utils.utils import convert_boxes

from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

class_names = [c.strip() for c in open('./yolov5/data/coco.names')]
print(class_names)
yolo = yolo(classes=len(class_names))
