import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

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
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        H, W, C = img.shape
        if self.results.multi_hand_landmarks and draw:
            for landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
    
    # TODO: ensure left/right are each sent out
    def find_position(self, img, max_hands=2, draw=False, print_coords=False):
        landmark_list = []
        H, W, C = img.shape
        
        if self.results.multi_hand_landmarks:
            for j in range(len(self.results.multi_hand_landmarks)):
                found_hand = self.results.multi_hand_landmarks[j]
                for id, landmark in enumerate(found_hand.landmark):
                    cx, cy = int(landmark.x * W), int(landmark.y * H)
                    landmark_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        
        if print_coords and len(landmark_list) > 0:
            print(landmark_list)
                
        return landmark_list
    
    def get_handedness(self):
        n = self.results.multi_handedness
        #print(n)
        h = []
        if n is not None:
            for i, hand in enumerate(n):
                h_dict = MessageToDict(hand)
                h.append(h_dict['classification'][0]['label'])
        return h

class PoseDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, segmentation=False, smooth_segmentation=True, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.complexity, self.smooth, self.segmentation, self.smooth_segmentation, self.detection_conf, self.track_conf)
        self.mp_draw = mp.solutions.drawing_utils
    
    def find_pose(self, img, draw=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        H, W, C = img.shape
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
    
    def find_position(self, img, draw=False, print_coords=False):
        landmark_list = []
        H, W, C = img.shape
        
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * W), int(landmark.y * H)
                landmark_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        if print_coords and len(landmark_list) > 0:
            print(landmark_list)
                
        return landmark_list