#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.externals import joblib
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
from scipy.ndimage.measurements import label

from vehicle_detection import compute_heatmap, compute_bboxes, draw_boxes, search_windows

def process_image(image):
    global svc, X_scaler, heatmaps
    
    if len(heatmaps) == 10:
        return image
    
    images.append(image)
    boxes.append(search_windows(image, svc, X_scaler))
    heatmaps.append(compute_heatmap(image, svc, X_scaler))
    if len(heatmaps) > 10:
        heatmaps.pop(0)
        images.pop(0)
        boxes.pop(0)
    
    bboxes = compute_bboxes(np.mean(np.array(heatmaps), axis=0))
    box_image = draw_boxes(image, bboxes)
    return box_image

svc = joblib.load('model.pkl') 
X_scaler = joblib.load('scaler.pkl') 
heatmaps=[]
boxes=[]
images=[]

output_file = 'output.mp4'
input_clip = VideoFileClip('test_video.mp4')
result_clip = input_clip.fl_image(process_image)
result_clip.write_videofile(output_file, audio=False)


f, axarr = plt.subplots(5, 4, figsize=(24, 9))
f.tight_layout()
for i in range(5):
    for j in range(2):
        axarr[i, 2*j + 0].imshow(images[5*j + i])
        axarr[i, 2*j + 0].axis('off')
        axarr[i, 2*j + 1].imshow(heatmaps[5*j + i], cmap='hot')
        axarr[i, 2*j + 1].axis('off')
plt.show()

f, axarr = plt.subplots(2, 3, figsize=(24, 9))
f.tight_layout()
for i in range(2):
    for j in range(3):
        axarr[i, j].imshow(draw_boxes(images[i*3 + j], boxes[i*3 + j]))
plt.show()

heatmap = np.mean(np.array(heatmaps), axis=0)
labels, _ = label(heatmap)
plt.imshow(labels, cmap='gray')
plt.show()
bboxes = compute_bboxes(heatmap)
box_image = draw_boxes(images[-1], bboxes)
plt.imshow(box_image)
plt.show()