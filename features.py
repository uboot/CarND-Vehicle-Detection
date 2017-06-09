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

def bin_spatial(img, size=(32, 32)):
    """
    Computes binned color features
    """
    
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Computes color histogram features
    """
    
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     feature_vec=True):
    """
    Computes HOG features
    """
    
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                   visualise=False, feature_vector=feature_vec)
    return features
    
def extract_color_features(image, cspace='RGB', spatial_size=(32, 32),
                           hist_bins=32, hist_range=(0, 256)):
    """
    Converts 'image' to the given color space and extracts binned color
    features and histogram color features
    """
    
    # Apply color conversion if other than 'RGB'
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
    # Combine these features
    return np.concatenate((spatial_features, hist_features))
    
def extract_hog_features(image, cspace='RGB', orient=9, 
                         pix_per_cell=8, cell_per_block=2, hog_channel=0):
    """
    Converts 'image' to the given color space and extracts HOG features for the
    given color channel (or all color channels)
    """
    
    # Apply color conversion if other than 'RGB'
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
        # Concatenate the HOG features for each color channel
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
    """
    Extracts the HOG and color features from 'image'. The parameters for the
    feature extraction are set to the global values defined in this module.
    """
    
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
    """
    Generator function which generates a sequence of windows sliding over the
    given area of the input image. It extracts the HOG and color features for 
    each window and yields the feature vector. The parameters for the
    feature extraction are set to the global values defined in this module.
    Thus, the feature vectors are comparable the vectors returned by the 
    function extract_features().
    """
    
    global spatial
    global histbin
    global colorspace
    global orient
    global pix_per_cell
    global cell_per_block
    global hog_channel
    
    # Convert to the given color space
    if colorspace != 'RGB':
        if colorspace == 'HSV':
            transformed = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif colorspace == 'LUV':
            transformed = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif colorspace == 'HLS':
            transformed = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif colorspace == 'YUV':
            transformed = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif colorspace == 'YCrCb':
            transformed = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: transformed = np.copy(img)
    
    
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    # Clip to the given area
    clipped = transformed[y_start_stop[0]:y_start_stop[1],
                          x_start_stop[0]:x_start_stop[1], :]
    
    # Rescale the image
    new_size = (int(clipped.shape[1]*scale), int(clipped.shape[0]*scale))
    feature_image = cv2.resize(clipped, new_size)   
    
    # Compute the number of blocks which fit inside the image
    nxblocks = (feature_image.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (feature_image.shape[0] // pix_per_cell) - cell_per_block + 1 
    
    # Compute the size of a window in pixels
    window = 8 * pix_per_cell
    
    # Compute the number of sliding steps in each direction
    nblocks_per_window = 8 - cell_per_block + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Now compute (all) the HOG features for the selected and scaled image
    # region
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
        
    # Extract the features for reach window from the precomputed HOG data
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            xright = xleft+window
            ybottom = ytop+window
            
            # Select the HOG features for this window
            hog_features = []
            for hog_features_channel in hog_features_channels:
                hog_features.append(hog_features_channel[ypos:ypos+nblocks_per_window, 
                                                         xpos:xpos+nblocks_per_window].ravel())
            hog_features = np.hstack(hog_features)


            # Extract the image patch
            patch = feature_image[ytop:ybottom, xleft:xright]
            
            # Compute color features
            spatial_features = bin_spatial(patch, size=(spatial, spatial))
            hist_features = color_hist(patch, nbins=histbin, bins_range=(0, 256))
            features = np.concatenate((spatial_features, hist_features, hog_features))

            # Compute the window coordinates with respect to the geometry of the
            # input image
            remapped_window = ((np.int(xleft/scale) + x_start_stop[0],
                                np.int(ytop/scale) + y_start_stop[0]),
                               (np.int(xright/scale)+ x_start_stop[0], 
                                np.int(ybottom/scale) + y_start_stop[0]))
            
            yield remapped_window, features
        
