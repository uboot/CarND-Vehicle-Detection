#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from sklearn.externals import joblib

from features import extract_window_features


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
    image = mpimg.imread(fname)
    draw_image = np.copy(image)
        
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255
    
    hot_windows = search_windows(image, svc, X_scaler)                       
    
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    
    plt.imshow(window_img)
    plt.show()
