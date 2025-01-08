# Project-Code

The code files in this repository are some of files of my final year university project on robot teleoperation.

The aim of the project was to create a proof of concept for using data from a vision sensor to detect hand gestures
performed by a human operator to control different types of robots in a simulated environment in real-time. As part
of the project different AI methods were analysed and compared to find the most suitable one for this application.

The design of the project had a few key parts:
  - data acquisition
  - feature extraction
  - gesture recognition
  - robot control logic
  - integration with CoppeliaSim environment

The code files in this repository are part of the files used for the feature extraction and gesture recognition steps.

The sensor provides data about the position and orientation of the hand that it detects. The hand gestures are dynamic therefore
for each sample, the data from the sensor is provided as a sequence of frames, where each frame contains the positional tracking
data of the hand at that instant in time.

The functionality of the code firstly reads the data from a database which contains CSV files with data samples for training and 
samples for the offline testing of an AI model. It then preprocesses the data by extracting the elements of the tracking data,
normalising them and calculating features using a sliding window. This input data is then used to train an SVM and test its performance.



