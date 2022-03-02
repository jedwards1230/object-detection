import cv2
from imutils.video import FPS
from Detectors import (
    HandDetector,
    PoseDetector)

video_stream = "http://192.168.1.143:56000/mjpeg"

modules = [
    'hand',
    'pose'
]

to_test = modules[1]

if __name__=='__main__':
    vcap = cv2.VideoCapture(video_stream)
    if not vcap.isOpened():
        print("Cannot open camera")
        exit()
        
    if to_test == modules[0]:
        hd = HandDetector()
    elif to_test == modules[1]:
        pd = PoseDetector()
    
    while True:
        success, img = vcap.read()
        if img is not None:
            img = cv2.flip(img, 1)
            fps = FPS().start()
            H, W, C = img.shape
            
            if to_test == modules[0]:
                hd.find_hands(img, draw=True)
                landmark_list = hd.find_position(img, print_coords=False)
            elif to_test == modules[1]:
                pd.find_pose(img, draw=True)
                landmark_list = pd.find_position(img, draw=False, print_coords=False)
            
            fps.update()
            fps.stop()
            fps_text = f'FPS: {round(fps.fps(), 2)}'
            cv2.putText(img, fps_text, (10, H - (20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Image', img)
            
            key = cv2.waitKey(1)
            if key == ord("q"):
                break