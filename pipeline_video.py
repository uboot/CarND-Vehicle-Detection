#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from moviepy.editor import VideoFileClip
import numpy as np

from vehicle_detection import compute_heatmap, compute_bboxes, draw_boxes

def process_image(image):
    global svc, X_scaler, heatmaps
    
    # Compute heatmap for this frame
    heatmaps.append(compute_heatmap(image, svc, X_scaler))
    if len(heatmaps) > 10:
        heatmaps.pop(0)
    
    # Extract the cars from the averaged heatmaps (last ten frames)
    bboxes = compute_bboxes(np.mean(np.array(heatmaps), axis=0))
    box_image = draw_boxes(image, bboxes)
    return box_image

# Load the classifier
svc = joblib.load('model.pkl') 
X_scaler = joblib.load('scaler.pkl')

# The heatmaps of the previous ten frames 
heatmaps=[]

output_file = 'output.mp4'
input_clip = VideoFileClip('project_video.mp4')
result_clip = input_clip.fl_image(process_image)
result_clip.write_videofile(output_file, audio=False)