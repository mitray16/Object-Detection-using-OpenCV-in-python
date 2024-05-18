# Object Detection using OpenCV
## Setting Threshold and Video Capture:
thres = 0.45: This threshold determines the confidence level required to consider an object detection result valid.
cap = cv2.VideoCapture(1): Initializes video capture from the camera (usually the default webcam). The argument 1 specifies the camera index (you can change it to 0 if you have only one camera).
## Loading Class Names:
classFile = 'coco.names': This file contains the names of the classes (objects) that the model can detect (e.g., “person,” “car,” “dog,” etc.).
The code reads the class names from this file and stores them in the classNames list.
## Loading Pretrained Model:
configPath and weightsPath point to the configuration file and the frozen inference graph (pretrained model) respectively.
The MobileNetV3 architecture is used for object detection in this case.
Setting Model Input Parameters:
net.setInputSize(320, 320): Sets the input size for the model (width and height).
net.setInputScale(1.0 / 127.5): Scales the input pixel values.
net.setInputMean((127.5, 127.5, 127.5)): Sets the mean subtraction values.
net.setInputSwapRB(True): Swaps the red and blue channels in the input image.
## Object Detection Loop:
The code enters an infinite loop (while True) to continuously process frames from the camera.
For each frame:
cap.read() captures the frame.
net.detect(img, confThreshold=thres) performs object detection on the frame.
Detected objects are represented by classIds, confs (confidence scores), and bounding box coordinates (bbox).
 If any objects are detected:
The bounding boxes are drawn on the frame using cv2.rectangle.
Class names and confidence scores are displayed using cv2.putText.
The processed frame is displayed using cv2.imshow.
cv2.waitKey(1) waits for a key press (1 millisecond) to exit the loop.
## Exiting the Program:
Press any key while the window is active to close the program.