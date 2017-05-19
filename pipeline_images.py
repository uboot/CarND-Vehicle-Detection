#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
import cv2
from sklearn.externals import joblib

from features import extract_window_features

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    imcopy = np.copy(img)
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(imcopy, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return imcopy
    
# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, clf, scaler):
    ystart = 400
    ystop = 656
    
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    generator = extract_window_features(img, x_start_stop=[None, None], 
                                        y_start_stop=[ystart, ystop], 
                                        scale=64/96, cells_per_step=2)
    for window, features in generator:
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows
    
svc = joblib.load('model.pkl') 
X_scaler = joblib.load('scaler.pkl') 

for fname in glob.glob('test_images/*.jpg'):
    draw_image = mpimg.imread(fname)
        
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = draw_image.astype(np.float32)/255
    
    hot_windows = search_windows(image, svc, X_scaler)        
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)  
    plt.imshow(window_img)
    plt.show()

    heat = np.zeros_like(image[:,:,0]).astype(np.float)        
    heat = add_heat(heat, hot_windows)    
    plt.imshow(heat, cmap='hot')
    plt.show()
     
    heat = apply_threshold(heat,1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    label_img = draw_labeled_bboxes(draw_image, labels) 
    plt.imshow(label_img)
    plt.show()
