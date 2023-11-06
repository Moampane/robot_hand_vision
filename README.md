# Introduction to Computational Robotics: Computer Vision Project

Authors: [Dexter Friis-Hecht](https://github.com/dfriishecht) & Mo Ampane

## Project Overview

The goal of this project was to use computer vision (CV) to recognize various hand gestures to control a finite state machine using ROS2 in Python. Through the use of opencv and implementation of a hand gesture classifier, we aimed to gain practical experience with CV and machine learning (ML) models. The goal of creating the finite state machine and corresponding robot behaviors was to reinforce our Python and ROS2 skills.

The project can be broadly broken down into hand recognition, gesture recognition, and finite state machine implementation.

## External Libraries

To recognize a single hand, we used Google's ML library [MediaPipe](https://developers.google.com/mediapipe). For real time CV we used OpenCV's Python library. We made use of [computervisioneng](https://github.com/computervisioneng)'s [collect_imgs.py](https://github.com/computervisioneng/sign-language-detector-python/blob/master/collect_imgs.py) to make our classifier's training dataset. Finally, we used sci-kit learn to make our classifier.

## Data Collection and Processing

MediaPipe's hand recognition algorithm represents a hand as 21 hand-knuckle $(x,y,z)$ coordinates in the image frame. Training the hand gesture classifier required creating a dataset of coordinates for each gesture.

![MediaPipe hand landmark diagram](img/hand_landmark.jpg)
Fig 1. Diagram of a [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) hand landmark

Using collect.py we took 200 pictures of each gesture and saved the $(x,y)$ hand-knuckle coordinates of the recognized hand. To make our classifier position agnostic, we normalized the coordinates.

![Raw hand mesh and normalized hand mesh](img/update_particles_with_odom.gif)
Fig 2. Gif of raw hand mesh (red) and normalized hand mesh (blue), normalized hand mesh values are from 0-1 but to be visible for the figure they are from 0-100

![Normalization equation](img/normalization_equation)
Fig 3. Normalization equation

## Training The Classifier

Talk about going from random forest classifier to neural network?

## ROS2 Integration

Image message to opencv image

## Behavior Implementation

Add gifs of behaviors

## Challenges

Location of camera, range of hand detection from camera...

## Stretch Goals

## Learnings
