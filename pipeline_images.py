#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib

from vehicle_detection import compute_heatmap, compute_bboxes, draw_boxes

svc = joblib.load('model.pkl') 
X_scaler = joblib.load('scaler.pkl') 

for fname in glob.glob('test_images/*.jpg'):
    image = mpimg.imread(fname)
        
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = draw_image.astype(np.float32)/255
    
    heat = compute_heatmap(image, svc, X_scaler)  
    plt.imshow(heat, cmap='hot')
    plt.show()
    
    bboxes = compute_bboxes(heat)
    box_image = draw_boxes(image, bboxes)
    plt.imshow(box_image)
    plt.show()
