import cv2
import mediapipe as mp
from imutils.video import FPS

video_stream = "http://192.168.1.143:56000/mjpeg"

class HandDetector():
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detection_conf, self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        H, W, C = img.shape
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def find_position(self, img, hand_n=0, draw=False, print_coords=False):
        landmark_list = []
        H, W, C = img.shape
        
        if self.results.multi_hand_landmarks:
            found_hand = self.results.multi_hand_landmarks[hand_n]
            for id, landmark in enumerate(found_hand.landmark):
                cx, cy = int(landmark.x * W), int(landmark.y * H)
                landmark_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        if print_coords and len(landmark_list) > 0:
            print(landmark_list)
                
        return landmark_list

if __name__=='__main__':
    vcap = cv2.VideoCapture(video_stream)
    hd = HandDetector()
    
    while True:
        success, img = vcap.read()
        if img is not None:
            img = cv2.flip(img, 1)
            fps = FPS().start()
            H, W, C = img.shape
            
            img = hd.find_hands(img, draw=True)
            landmark_list = hd.find_position(img, print_coords=False)
            
            fps.update()
            fps.stop()
            fps_text = f'FPS: {round(fps.fps(), 2)}'
            cv2.putText(img, fps_text, (10, H - (20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Image', img)
            
            key = cv2.waitKey(1)
            if key == ord("q"):
                break