#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage.feature import hog

spatial = 32
histbin = 32
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

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
    global spatial
    global histbin
    global colorspace
    global orient
    global pix_per_cell
    global cell_per_block
    global hog_channel
    
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
                            cells_per_step=2):
    global spatial
    global histbin
    global colorspace
    global orient
    global pix_per_cell
    global cell_per_block
    global hog_channel
    
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
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    rescaled_x_start_stop = (np.int(x_start_stop[0]*scale), np.int(x_start_stop[1]*scale))
    rescaled_y_start_stop = (np.int(y_start_stop[0]*scale), np.int(y_start_stop[1]*scale))
    
    # Define blocks and steps as above
    nxblocks = (feature_image.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (feature_image.shape[0] // pix_per_cell) - cell_per_block + 1 
    #nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    hog_features_channels = []
    if hog_channel == 'ALL':
        for channel in range(feature_image.shape[2]):
            f = get_hog_features(feature_image[:,:,channel], orient, 
                                 pix_per_cell, cell_per_block, 
                                 feature_vec=False)
            hog_features_channels.append(f)      
    else:
        f = get_hog_features(feature_image[:,:,hog_channel], orient, 
                             pix_per_cell, cell_per_block, 
                             feature_vec=False)
        hog_features_channels.append(f)
        
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            xright = xleft+window
            ybottom = ytop+window
            
            if xleft < rescaled_x_start_stop[0]:
                continue
            if xright > rescaled_x_start_stop[1]:
                continue
            if ytop < rescaled_y_start_stop[0]:
                continue
            if ybottom > rescaled_y_start_stop[1]:
                continue
            
            hog_features = []
            for hog_features_channel in hog_features_channels:
                hog_features.append(hog_features_channel[ypos:ypos+nblocks_per_window, 
                                                         xpos:xpos+nblocks_per_window].ravel())
            hog_features = np.hstack(hog_features)


            # Extract the image patch
            patch = feature_image[ytop:ybottom, xleft:xright]
            
            # Get color features
            spatial_features = bin_spatial(patch, size=(spatial, spatial))
            hist_features = color_hist(patch, nbins=histbin, bins_range=(0, 256))
            features = np.concatenate((spatial_features, hist_features, hog_features))

            rescaled_window = ((np.int(xleft/scale), np.int(ytop/scale)),
                               (np.int(xright/scale), np.int(ybottom/scale)))
            
            yield rescaled_window, features
        
