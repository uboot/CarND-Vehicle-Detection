#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import numpy as np
import glob
import time

from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from features import extract_features

# NOTE: the next import is only valid 
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

def collect_data(cars, notcars):
    car_features = []
    for file in cars:
        # Read in each one by one
        image = mpimg.imread(file)
        image = (255*image).astype(np.ubyte)
        car_features.append(extract_features(image))
        
    notcar_features = []
    for file in notcars:
        # Read in each one by one
        image = mpimg.imread(file)
        image = (255*image).astype(np.ubyte)
        notcar_features.append(extract_features(image))
        
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)           
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    return X, y

# Read in car and non-car images
vehicles = glob.glob('training_data/vehicles/**/*.png')
cars = []
for image_file in vehicles:
    cars.append(image_file)

non_vehicles = glob.glob('training_data/non-vehicles/**/*.png')
notcars = []
for image_file in non_vehicles:
    notcars.append(image_file)
    
cars = shuffle(cars)
notcars = shuffle(notcars)
print('# cars:', len(cars))
print('# not cars:', len(notcars))

# Compute the features
X, y = collect_data(cars, notcars)
             
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
    
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()

# save the model
joblib.dump(svc, 'model.pkl') 
joblib.dump(X_scaler, 'scaler.pkl') 

print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

