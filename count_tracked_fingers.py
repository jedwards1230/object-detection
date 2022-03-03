import cv2
from imutils.video import FPS
from Detectors import (
    HandDetector,
    PoseDetector)

video_stream = "http://192.168.1.143:56000/mjpeg"

w_cam, h_cam = 1200, 768

if __name__=='__main__':
    vcap = cv2.VideoCapture(video_stream)
    if not vcap.isOpened():
        print("Cannot open camera")
        exit()
        
    hd = HandDetector(detection_conf=0.7, track_conf=0.7)
    
    finger_ids = [4, 8, 12, 16, 20]
    
    while True:
        success, img = vcap.read()
        if img is not None:
            # Format image 
            img = cv2.resize(img,(w_cam,h_cam),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            img = cv2.flip(img, 1)
            fps = FPS().start()
            H, W, C = img.shape
            
            hd.find_hands(img, draw=True)
            lm_list = hd.find_position(img, print_coords=False, draw=False)
            hands = hd.get_handedness()
            # TODO: refactor duplicate code
            if len(hands) > 0:
                for hand in hands:
                    if hand == 'Left':
                        left_result = []
                        if len(lm_list) > 0:
                            for i in range(5):
                                if lm_list[finger_ids[i]][2] < lm_list[finger_ids[i]-2][2] and lm_list[finger_ids[i-1]][2] < lm_list[finger_ids[i]-3][2]:
                                    left_result.append(i)
                        # Display result
                        cv2.putText(img, str(len(left_result)), (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
                    if hand == 'Right':
                        right_result = []
                        if len(lm_list) > 0:
                            for i in range(5):
                                if lm_list[finger_ids[i]][2] < lm_list[finger_ids[i]-2][2] and lm_list[finger_ids[i-1]][2] < lm_list[finger_ids[i]-3][2]:
                                    right_result.append(i)
                        # Display result
                        cv2.putText(img, str(len(right_result)), (W - 100, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
                # TODO: Display result in center
                if len(hands) == 2:
                    result = len(left_result) + len(right_result)
                    text_x = w_cam
                    cv2.putText(img, str(result), (text_x, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
            
            # Display FPS
            fps.update()
            fps.stop()
            fps_text = f'FPS: {round(fps.fps(), 2)}'
            cv2.putText(img, fps_text, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display Frame
            cv2.imshow('Image', img)
            
            # Press q to quit
            key = cv2.waitKey(1)
            if key == ord("q"):
                break