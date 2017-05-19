#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage.feature import hog


# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     feature_vec=True):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                   visualise=False, feature_vector=feature_vec)
    return features
    
def extract_color_features(image, cspace='RGB', spatial_size=(32, 32),
                           hist_bins=32, hist_range=(0, 256)):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      
    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
    
    return np.concatenate((spatial_features, hist_features))
    
def extract_hog_features(image, cspace='RGB', orient=9, 
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      

    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, feature_vec=True)
    
    return hog_features

def extract_features(image):
    ### TODO: Tweak these parameters and see how the results change.
    spatial = 32
    histbin = 32
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    
    hog_features = extract_hog_features(image, 
                                        cspace=colorspace, 
                                        orient=orient, 
                                        pix_per_cell=pix_per_cell, 
                                        cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel)
    
    color_features = extract_color_features(image,
                                            cspace=colorspace,
                                            spatial_size=(spatial, spatial),
                                            hist_bins=histbin,
                                            hist_range=(0, 256))
    
    return np.concatenate((color_features, hog_features))

def extract_window_features(img, x_start_stop=[None, None], 
                            y_start_stop=[None, None], scale=64/96,
                            xy_overlap=(0.5, 0.5)):

    spatial = 32
    histbin = 32
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    
    
    if colorspace != 'RGB':
        if colorspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif colorspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif colorspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif colorspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    
    new_size = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    feature_image = cv2.resize(feature_image, new_size)   
    
    # If x and/or y start/stop positions not defined, set to image size
    xy_window=(64, 64)
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    # Compute the span of the region to be searched    
    xspan = np.int((x_start_stop[1] - x_start_stop[0]) * scale)
    yspan = np.int((y_start_stop[1] - y_start_stop[0]) * scale)
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + np.int(x_start_stop[0]*scale)
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + np.int(y_start_stop[0]*scale)
            endy = starty + xy_window[1]
            window = ((startx, starty), (endx, endy))
            
            patch = feature_image[window[0][1]:window[1][1], window[0][0]:window[1][0]]
    
            spatial_features = bin_spatial(patch, size=(spatial, spatial))
            hist_features = color_hist(patch, nbins=histbin, bins_range=(0, 256))
            
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(patch.shape[2]):
                    f = get_hog_features(patch[:,:,channel], orient, 
                                         pix_per_cell, cell_per_block, 
                                         feature_vec=True)
                    hog_features.append(f)
                hog_features = np.ravel(hog_features)        
            else:
                f = get_hog_features(patch[:,:,hog_channel], orient, 
                                     pix_per_cell, cell_per_block, 
                                     feature_vec=True)
                hog_features.append(f)
            
            rescaled_window = ((np.int(window[0][0]/scale), np.int(window[0][1]/scale)),
                               (np.int(window[1][0]/scale), np.int(window[1][1]/scale)))
            features = np.concatenate((spatial_features, hist_features, hog_features))
            yield rescaled_window, features

