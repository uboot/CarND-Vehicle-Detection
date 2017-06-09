**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./car.png
[image2]: ./not_car.png
[image3]: ./windows.png
[image5]: ./frames.png
[image6]: ./labels.png
[image7]: ./bboxes.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG and color features from the training images.

The training step is implemented in `train.py`. I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

The script then calls the function `extract_features()` in lines 33 and 40 to compute all image image features. This function is implemented in the module of the module `features.py`. There the HOG features are computed in the lines 78 through 112. In addition to the HOG features I computed spatial color features and color histograms in the lines 16-76.

The parameters for the feature extraction are given in the globals in lines 8 through 14. I explored different settings by changing these values. In case of `skimage.hog()` these are `orient`, `pixels_per_cell`, `hog_channel`, and `cells_per_block`. The color features are determined by `spatial` and `histbin`. I used the same `colorspace` for the HOG and color features.

#### 2. Explain how you settled on your final choice of HOG and color feature parameters.

I tried evaluated the classification accuracy on the test set for various combinations of parameters. It turned out that the initial settings of the HOG parameters (9 orientations, 8 pixels per cell and 2 block per cell, YCrCb color space) were quite good. Using all channels of the image improved the classification result significantly so I chose this approach. 

I used 32 spatial and 32 histogram color features.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 80% of the XXXX training data in line 87 of `train.py`. Evaluating the resulting model on the remaining 20% test data in line 97 showed an accuracy of 99.4%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The function `extract_window_features()` in the lines 143 through 255 in `features.py` computes the features of all windows sliding over a subregion (`x_start_stop` and `y_start_stop`) at a given scale (`scale`) of the input image. For the computation of the HOG features this function computes the HOG on the complete image (lines 211 through 221) and selects the appropriate subregion of the precomputed values for each window (lines 237 and 238).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales (100% and 66%) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The vehicle detection in videos is implemented in `pipeline_video.py`. Here's a [link to my video result](./output.mp4).

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap. Then I averaged the ten previous heatmaps in line 19 and computed the bounding boxes of the vehicles from this heatmap using `compute_bboxes()` in `vehicle_detection.py`.  
There I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are ten frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first step of the project was to train a classifier to distinguish between cars and other texture. It turned out fairly easy to get a classifier which performs better than 99% using HOG and color features and the relatively simple linear SVM. However, the my pipeline for vehicle detection is computationally expensive, i.e. it is not possible to process every frame in real time. Training a classifier which uses less expensive features and still performs well would be an improvement.

Although my classifier has a high accuracy there are cases when false positives are detected on some frames or the detected boxes look wobbly. I think it would beneficial to develop a more advanced averaging over subsequent frames than simply adding the heat maps. On could look at the statistics of the bounding box corners an make sure that there are no outliers on single frames.


