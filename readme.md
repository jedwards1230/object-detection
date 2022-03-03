Playing with realtime computer vision libraries (PyTorch, OpenCV, MediaPipe)

## count_tracked_fingers.py
 - displays count of tracked fingers per hand
 - TODO: ensure proper distinction between left and right from HandDetector
 - TODO: refactor duplicate code
 - TODO: Display results in center of screen

## Detectors.py 
 - uses mediapipe to track hands in a video 

## bb-selector.py
 - uses openCV trackers to track a custom selected bounding box.
 - press 'k' during the stream to select a box, then 'enter' to apply and track

## torch-model.py 
 - uses the fasterrcnn_mobilenet_v3_large_320_fpn detector to classify and provide bounding boxes. This outputs any object predictions with >50% confidence scores.
