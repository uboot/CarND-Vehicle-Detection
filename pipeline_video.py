#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

from vehicle_detection import compute_heatmap, compute_bboxes, draw_boxes


def process_image(image):
    global svc, X_scaler
    
    heat = compute_heatmap(image, svc, X_scaler)  
    bboxes = compute_bboxes(heat)
    box_image = draw_boxes(image, bboxes)
    return box_image


svc = joblib.load('model.pkl') 
X_scaler = joblib.load('scaler.pkl') 

output_file = 'output.mp4'
input_clip = VideoFileClip('test_video.mp4')
result_clip = input_clip.fl_image(process_image)
result_clip.write_videofile(output_file, audio=False)