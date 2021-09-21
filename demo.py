#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
#
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:

import os
import sys
import random
import math
import cv2

import PIL
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# ## Configurations
#
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
#
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# ## Create Model and Load Trained Weights

# In[3]:

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# ## Class Names
#
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
#
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
#
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
#
# # Print class names
# print(dataset.class_names)
# ```
#
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# ## Run Object Detection

# In[5]:
# Load a random image from the images folder

file_names = next(os.walk(IMAGE_DIR))[2]
l1 = next(os.walk(IMAGE_DIR))[2]
print("li is equal to ",l1)
l1.sort()

for i in l1:
    print(i)
    image = skimage.io.imread(IMAGE_DIR+'/'+i)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    a = r['class_ids']
    print("Print a : ",a)
    p = 0
    a1 = list()
    for y in a:
        if y in {16,17,18,19,20,21,22,23,24} :
            a1.append(p)
        p = p+1
    print("String of a1",a1)

    # Final mask for the image
    finalMask = np.zeros((image.shape[0], image.shape[1]))
    print(image.shape)
    for x in a1 :
        dog_mask = r['masks'][:, :, x]
        print("Value of x: ",x)
        dog_mask_int = 255 * dog_mask.astype(np.uint8)
        finalMask = finalMask + dog_mask_int
        #cv2.imshow('dog_mask_int', dog_mask_int)
        #cv2.imshow('finalMask', finalMask)
        #cv2.waitKey(0)
        p = 'detected_mask'
        q = i.rstrip(".jpng")
        s = IMAGE_DIR + '/' + 'detected_mask' + '/' + q + '_' + p + '.png'
        cv2.imwrite(s, finalMask)

OG_mask = IMAGE_DIR+'/'+'original_mask'
DT_mask = IMAGE_DIR+'/'+'detected_mask'
out= IMAGE_DIR+'/'+'out'+'/'
l2= next(os.walk(OG_mask)) [2]
l2.sort()
print('l2 original mask : ',l2)
l3 = next(os.walk(DT_mask)) [2]
print('l3 detected mask:',l3)
l3.sort()
lst = []
for i,j in zip(l2,l3) :
    print(i)
    print(j)
    fn_mask_gt =OG_mask+'/'+i
    fn_mask_det=DT_mask+'/'+j
    fn_iou_vis_out=out
    mask_gt = cv2.imread(fn_mask_gt,cv2.IMREAD_GRAYSCALE)
    mask_det = cv2.imread(fn_mask_det, cv2.IMREAD_GRAYSCALE)

    #cv2.imshow('Ground truth mask', mask_gt)
    #cv2.imshow('Detected mask', mask_det)
    #cv2.waitKey(40)
    # ------------------

    # Counts of intersection pixels and union pixels
    # ------------------
    num_intersecting = 0
    num_union = 0
    num_rows = np.shape(mask_gt)[0]
    num_cols = np.shape(mask_gt)[1]
    for row in range(num_rows):
        for col in range(num_cols):
            val_gt = mask_gt[row][col]
            val_det = mask_det[row][col]
            if val_gt > 0 and val_det > 0:  # both ground truth and detection
                num_intersecting = num_intersecting + 1
            if val_gt > 0 or val_det > 0:  # either ground truth, or detection (or both)
                num_union = num_union + 1
    # ------------------

    # Calculate intersection over union
    # ------------------
    iou_pcnt = 100 * (num_intersecting / num_union)
    print('num_intersecting: %d, num_union: %d, IoU: %.2f%%:' % (num_intersecting, num_union, iou_pcnt))
    num = float(iou_pcnt)
    lst.append(num)

    #for n in j:
        #lst.append(num)
        #print("Sum:", lst)

    # ------------------

    # Visualize the IoU result (optional)
    # ------------------

    # Initialize three-channel image as zeros (black) - note: OpenCV uses BGR ordering of colour channels
    img_iou_vis = np.zeros((num_rows, num_cols, 3), np.uint8)

    # Cycle through pixels and set colour according to status
    for row in range(num_rows):
        for col in range(num_cols):
            val_gt = mask_gt[row][col]
            val_det = mask_det[row][col]
            if val_gt > 0 and val_det > 0:
                img_iou_vis[row][col] = [255, 255, 255]  # GT and detection: white
            if val_gt > 0 and val_det == 0:
                img_iou_vis[row][col] = [0, 255, 0]  # GT only: green
            if val_gt == 0 and val_det > 0:
                img_iou_vis[row][col] = [0, 0, 255]  # detection only: red

    # Save IoU visualization
    p = 'mask_iou_vis'
    q = i.rstrip(".jpg")
    s = IMAGE_DIR + '/' + 'out' + '/' + q + '_' + p + '.png'
    cv2.imwrite(s, img_iou_vis)

for xy in range(len(lst)):
    Average = sum(lst) / len(lst)
    print("list of iou % : ",xy,lst[xy])
print('average of IOU = ',Average)

    # Display IoU visualization
    #cv2.imshow('IoU visualization', img_iou_vis)
    #cv2.waitKey(0)