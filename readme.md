# Udacity – Vehicle Detection and Tracking Project

Submitted by: Neeraj Gulia

Date: March-16-2018

## Objective

The objective of this project is to detect and track vehicles on a given video. This project gives us insight on how camera can be used to detect and track a moving vehicle.

## **Project Goals**

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
- Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
- Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

Submitted files:

| File Name | Description |
| --- | --- |
| Writeup.pdf | Writeup document |
| P5.ipynb | Project python notebook |
| p5library.py | Project Python file, contains methods and classed used in the project |
| output images | Folder which contains output images |
| output\_project\_video.mp4 | Output video |

## Rubric Points

## Histogram of Oriented Gradients (HOG)

I started by reading in all the vehicle and non-vehicle images. Here is an example of one of each of the vehicle and non-vehicle classes:

![Car not car](output images\car-notcar.png)

Figure 1 Car and not car example

Then extracted HOG features using skimage.feature&#39;s hog method from the car and not-car images. This was then combined with the Spatial features and Color histogram features

**Methods: extract\_features, color\_hist, bin\_spatial, get\_hog\_features**

HOG Parameters

I did a lot of trial and error and finally settled on following parameters:

| Parameter Name | Values |
| --- | --- |
| color\_space | YCrCb |
| Orientation | 8 |
| pix\_per\_cell | 16 |
| cell\_per\_block | 2 |
| hog\_channel | &#39;ALL&#39; |

### Preprocessing

After feature extraction, created the labeled data –&gt; 1 for car features and 0 for not-car data. Extracted 20% of the data for validation. Used standard Scaler for removing the mean and scaling to unit variance.

**Feature vector length: 1680**

Did data augmentation by adding vertical flipped image of the cars and notcars images. This resulted into a rich dataset of car samples: 17584, notcar samples: 17740

### Classifier training

Used SVM for training, first I tried with LinerSVC but observed a lot of false positives. Finally moved to &#39;rbf&#39; kernel. Used C as 10

## Training Accuracy of SVC = 1.0

## Test Accuracy of SVC = 0.9982

## Sliding Window Search

I used **Hog Sub-Sampling Window Search** method for the sliding window search where overlapping tiles in each test image are classified as vehicle or non-vehicle.

I choose this method over the normal &quot;Sliding Window Implementation&quot; because in this we extract the HOG features only once and then uses sub-sampling of HOG.

**Method: find\_cars**

### **Scaling and height limits used for pipeline**

Following thresholds are used for scaling and height limits

| Height Top | Height Bottom | Scale | Window size (pixels) | Description |
| --- | --- | --- | --- | --- |
| 420 | 486 | .8 | 51.2 | For far out area (small cars) |
| 410 | 480 | 1. | 64 |   |
| 410 | 534 | 1.6 | 102.4 |   |
| 400 | 528 | 2. | 128 |   |
| 420 | 640 | 3.2 | 204.8 | Nearest area – big cars |

## Pipeline working

### Frame processing

1. Detects the car in the given frame and returns rectangles around the identified windows
2. Add the given rectangles to the Store (which stores rectangles of last 12 frames

### Drawing of rectangles on detected car

1. Create a heat map of stored rectangles on the image frame by incrementing the pixel value by 1 for the identified rectangle. This increases the heat for the region where car is detected and more the rectangle identified more the value of heat
2. Threshold applied on heat map image as (0.65 times the length of the stored frames)
3. Clip the heat map between 0 and 255
4. Get labels from the heat map, by using scipy.ndimage.measurements&#39;s label method
5. Draw the final boxes (rectangles) over the frame

![Actual Image](output images\actualimage.png)

Figure 2 Actual Image

![Processed Image](output images\raw-detection.png)

Figure 3 Raw Detection

![Processed Image](output images\heatmap.png)

Figure 4 Heat-map

![Processed Image](output images\final-output.png)

Figure 5 Final Output

### How I improved processing time

The frame processing is done to 1/3 of the frames only, and for remaining 2/3 frames the last calculated rectangles are drawn. This has resulted into smooth video output with very less processing overhead.

**Methods: get\_labels, draw\_boxes, draw\_labeled\_bboxes**

![Processed Image](output images\actual-vs-processed.png)
Figure 6 Actual vs Processed Images

## Pipeline (Video)

The pipeline worked pretty good on the given video.

Output video is: [output project video](https://github.com/neerajgulia/CarND-Vehicle-Detection-and-Training/blob/master/output_project_video.mp4)

## Discussions

The pipeline ran too slow on my normal laptop. I skipped 50% of the frames and used the running average of the rectangles to track the car in the video. Further I reduced the processed frames percentage to 33%. This greatly reduced the processing time of the video without affecting the video output.

I believe instead of Machine Learning emphasis could have been done on doing this project on Deep Learning.

Also, I did not like the fact that most of the code has been provided by Udacity which is killing the creative quotient.

## Problem faced:

I struggled for two days to tweak the hyper parameters. At one moment I was about to abandon the SVM and implement Deep Learning approach.

## References:

[http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

[https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html)