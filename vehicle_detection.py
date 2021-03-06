#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy.ndimage.measurements import label

from features import extract_window_features

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def bboxes_for_labels(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    # Return the bounding boxes
    return bboxes
    
def search_windows(img, clf, scaler):
    """
    Moves sliding windows on two different scales over the image and returns
    those windows which contain features classified as 'car'.
    """
    ystart = 400
    ystop = 656
    
    # Create an empty list to receive positive detection windows
    on_windows = []
    
    # Iterate over all windows in the image on the original scale and classify
    # the features. Append the positive classification results.
    generator = extract_window_features(img, x_start_stop=[None, None], 
                                        y_start_stop=[ystart, ystop], 
                                        scale=64/64, cells_per_step=2)
    for window, features in generator:
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
            
    # Do the same for on a lower scale.
    generator = extract_window_features(img, x_start_stop=[None, None], 
                                        y_start_stop=[ystart, ystop], 
                                        scale=64/96, cells_per_step=2)
    for window, features in generator:
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
            
    return on_windows

def compute_heatmap(img, svc, scaler):
    """
    Computes the heatmap of the classification of the regions of 'img' with the
    given classifier
    """
    # Do the classification (this is an expensive operation)
    hot_windows = search_windows(img, svc, scaler)
    
    # Some up the positive areas
    heat = np.zeros_like(img[:,:,0]).astype(np.float)        
    heat = add_heat(heat, hot_windows)   
    
    return heat

def compute_bboxes(heatmap, threshold=2):
    """
    Computes bounding boxes from the input heatmap.
    """
    # Remove everything below the threshold
    heatmap[heatmap <= threshold] = 0
    
    # Now look for connected regions and their bounding boxes
    labels = label(heatmap)
    bboxes = bboxes_for_labels(labels) 
    
    return bboxes

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws boxes onto an image.
    """
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy