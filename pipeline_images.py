#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import matplotlib.image as mpimg
from sklearn.externals import joblib

from vehicle_detection import compute_heatmap, compute_bboxes, draw_boxes

# Read the classifier
svc = joblib.load('model.pkl') 
X_scaler = joblib.load('scaler.pkl') 

# Process each image and save the result
for fname in glob.glob('test_images/*.jpg'):
    image = mpimg.imread(fname)
    
    heat = compute_heatmap(image, svc, X_scaler)      
    bboxes = compute_bboxes(heat)
    box_image = draw_boxes(image, bboxes)
    
    out_fname = 'output_images/{0}'.format(os.path.basename(fname))
    mpimg.imsave(out_fname, box_image)
    
