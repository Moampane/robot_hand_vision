# Introduction to Computational Robotics: Computer Vision Project

Authors: [Dexter Friis-Hecht](https://github.com/dfriishecht) & Mo Ampane

## Project Overview

The goal of this project was to use computer vision (CV) to recognize various hand gestures to control a finite state machine using ROS2 in Python. Through the use of opencv and implementation of a hand gesture classifier, we aimed to gain practical experience with CV and machine learning (ML) models. The goal of creating the finite state machine and corresponding robot behaviors was to reinforce our Python and ROS2 skills.

The project can be broadly broken down into hand recognition, gesture recognition, and finite state machine implementation.

## External Libraries

To recognize a single hand, we used Google's ML library [MediaPipe](https://developers.google.com/mediapipe). For real time CV we used OpenCV's Python library. We made use of [computervisioneng](https://github.com/computervisioneng)'s [collect_imgs.py](https://github.com/computervisioneng/sign-language-detector-python/blob/master/collect_imgs.py) to make our classifier's training dataset. Finally, we used sci-kit learn to make our classifier.

## Data Collection and Processing

MediaPipe's hand recognition algorithm represents a hand as 21 hand-knuckle $(x,y,z)$ coordinates in the image frame. Training the hand gesture classifier required creating a dataset of coordinates for each gesture.

|                                 ![MediaPipe hand landmark diagram](img/hand_landmark.jpg)                                 |
| :-----------------------------------------------------------------------------------------------------------------------: |
| _Fig 1. Diagram of a [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) hand landmark_ |

Using collect.py we took 200 pictures of each gesture and saved the $(x,y)$ hand-knuckle coordinates of the recognized hand. To make our classifier position agnostic, we normalized the coordinates.

|                                                  ![Raw hand mesh and normalized hand mesh](img/normalization.gif)                                                  |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| _Fig 2. Gif of raw hand mesh (red) and normalized hand mesh (blue), normalized hand mesh values are from 0-1 but to be visible for the figure they are from 0-100_ |

| ![Normalization equation](img/normalization_equation.jpg) |
| :-------------------------------------------------------: |
|              _Fig 3. Normalization equation_              |

## The Classifier and Real Time Gesture Recognition

After collecting training data we had to decide what ML model would work best for this classification. We tried sci-kit learn's built in 'RandomForestClassifier()', 'MLPClassifier()', and 'KNeighborsClassifier()'. Using the 'MLPClassifier()' worked best for our application. The neural network we made had 3 hidden layers of size 100, 50, and 10, and went through 15 epochs. The k-nearest neighbors classifier used 10 nearest neighbors and appeared to always be 100% confident in its classification so it did not recognize the no gesture state well. We believe the neural network worked best because of our large training dataset, 200 data points per gesture or 1800 data points.

For classifying a gesture in real time, one hand is recognized, the $(x,y)$ hand-knuckle coordinates are normalized, and a classification is made from the normalized data. There are 9 unique hand gesture classifications and the classifier recognizes when a gesture is not being made when it's less than 60% confident in it's prediction. To communicate what the robot is understanding, a label is shown above the user's hand indicating what the classifier predicts their gesture to be and how confident it is in it's prediction.

| ![Geesture and no gesture recognition](img/gestures.gif) |
| :------------------------------------------------------: |
|    _Fig 4. Gif of Gesture and no gesture recognition_    |

## ROS2 Integration

Our ROS2 node generates a single subscriber and a single publisher. With this, the Node subscribes to the ROS2 Image topic, allowing it to access video footage from our Neato's camera, and publish velocity commands to the cmd_vel topic, allowing the Neato to be controlled based on processed image data.
```mermaid
flowchart TD
A['cv_state_machine' Node]
B[Image Subscriber]
C[Velocity Publisher]
D[image]
E[cmd_vel]

A -->|initializes|B
A -->|initializes|C
B -->|subscribes|D
C -->|publishes|E
```
*Fig 5. The cv_state_machine node's Subscriber and Publisher Structure*

However, OpenCV cannot proccess data from the ROS2 Image topic directly. To amend this, cv_bridge is used to convert the images into Numpy arrays readable by OpenCV.
```mermaid
flowchart TD
A[Image Topic]
B[OpenCV]

A -->|cv_bridge|B
```
*Fig 6. Image Topic to OpenCV Conversion*

## Behavior Implementation

The primary implemented behavior is a teleop implementation toggled with gestural control. Initially, the Neato is in a "locked" state, where the teleop gestures don't actually trigger any behavior. By using the "toggle teleop" gesture, teleop is enabled and the Neato will respond to gestural control. This added layer of toggling ensures that the Neato won't be accidentally toggled by user inputs. Other inputs seperate from Teleop can still be triggered, regardless of Teleop's current state.
The behavior tree is as follows:
```mermaid
flowchart TD
A[Teleop Gesture]
B[Secondary Gesture]
C[Tertiary Gesture]
D[Drive Forward]
E[Drive Backward]
F[Stop]
G[Turn Right]
H[Turn Left]
I[Loop]
J[Process Gesture]

J -->A
A -->|Toggle Teleop|D
A -->|Toggle Teleop|E
A -->|Toggle Teleop|F
A -->|Toggle Teleop|G
A -->|Toggle Teleop|H
B -->|Secondary Behavior|I
C -->|Tertiary Behavior|I
A -->|Toggle Teleop|I
J -->B
J -->C
D -->I
E -->I
F -->I
G -->I
H -->I
```

## Behavior Implementation

Add gifs of behaviors

## Challenges

One challenge we faced was in the implementation of our hand gesture classifier. Figuring out how to collect our training data was difficult, learning how to use MediaPipe and sci-kit learn took a large amount of time (especially converting the hand-knuckle coordinates into a form that could be used by the classifier), and improving the performance of the classifier was a challenge.

Some pain points in the actual use of the system are the location of the camera and its limited range. Using the camera on the robot requires the user to be at the robot's level and to move around with it. The limited range of the camera can sometimes take the user by surprise when a gesture is not registered.

## Stretch Goals

One of our stretch goals was implementing text to speech. We were successful, and as a result, controlling the Neato without the classified label from the laptop screen is significantly easier. Also, one easter egg behavior is a relevant sound clip.

## Learnings

Ultimately we learned how to successfully use CV to implement a hand gesture classifier to control a finite state machine. This included learning when to use different ML models, how to train and optimize them, and data preprocessing. Also, creating a more complex finite state machine reinforced our understanding of ROS2.
