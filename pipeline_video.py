#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from moviepy.editor import VideoFileClip
import numpy as np

from vehicle_detection import compute_heatmap, compute_bboxes, draw_boxes

heatmaps=[]

def process_image(image):
    global svc, X_scaler
    
    heatmaps.append(compute_heatmap(image, svc, X_scaler))
    if len(heatmaps) > 5:
        heatmaps.pop(0)
    
    bboxes = compute_bboxes(np.mean(np.array(heatmaps), axis=0))
    box_image = draw_boxes(image, bboxes)
    return box_image


svc = joblib.load('model.pkl') 
X_scaler = joblib.load('scaler.pkl') 

output_file = 'output.mp4'
input_clip = VideoFileClip('test_video.mp4')
result_clip = input_clip.fl_image(process_image)
result_clip.write_videofile(output_file, audio=False)