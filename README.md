# Vehicle Detection

In this project, I implemented a software pipeline to detect vehicles in a video 

The Project
---

The project consists of 3 jupyter notebooks and 1 python file. 
I implemented computer vision pipeline using HOG and SVM based methods. My pipeline is able to detect the cars in test images but yolo detection is performing better when multiple cars are present in the video and can be implemented in Realtime compared to classic Computer vision pipeline. so i also implemented yolo. 

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* I applied a color transform and append binned color features, as well as histograms of color, to my HOG feature vector. 
* Note: for those first two steps I normalized the features and randomized a selection for training and testing.
* Implemented a sliding-window technique and used my trained classifier to search for vehicles in images.
* Implemented a yolo Detection using Keras with Tensorflow Backend.
* Estimated a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.  

