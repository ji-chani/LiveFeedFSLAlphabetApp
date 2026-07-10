"""
LiveFeedAlphabetClassifier

--------
detects the current frame and predicts the corresponding letter
of the alphabet. Requires scaler and classifier to be specified.

"""

import cv2
import pickle
import string
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
])


class LiveFeedFSLAlphabet:
    def __init__(self,
                 model_path:str,
                 scaler_path:str,
                 classifier_path:str,
                 labels:list,
                 max_hands:int=2,
                 detection_conf:float=0.5,
                 tracking_conf:float=0.5,
                 ):
        
        self.labels = labels
        self.max_hands = max_hands
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=tracking_conf,
            min_tracking_confidence=tracking_conf
        )

        # setting up models
        self.landmarker = HandLandmarker.create_from_options(options)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)

    def classify_hand_to_letter(self,
                                bgr_frame,
                                frame_ts_ms:int,
                                show_landmarks:bool=True):
        
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=rgb_frame)
        
        result = self.landmarker.detect_for_video(mp_image, frame_ts_ms)
        
        annotated_frame = bgr_frame.copy()

        for hand_idx, world_landmarks in enumerate(result.hand_world_landmarks):
            
            # classify hand using landmarks as features
            features = np.array(
                [[lm.x, lm.y, lm.z] for lm in world_landmarks]
            ).flatten().reshape(1, -1)
            
            
            features_scaled = self.scaler.transform(features)
            prediction = self.classifier.predict(features_scaled)[0]
            label = self.labels[prediction] if isinstance(prediction, (int, np.integer)) else prediction

            # draw bbox and label
            # bbox = {center': (cx,cy), 'bbox': [],'label': label}
            annotated_frame, _ = draw_bbox_label(annotated_frame, result, hand_idx, label)
            
            # draw landmarks
            if show_landmarks:
                annotated_frame = draw_hand_landmarks(annotated_frame, result, hand_idx)

        # annotated_frame is in bgr format
        return annotated_frame

def draw_bbox_label(annotated_frame:np.ndarray, result, hand_idx:int, label, 
                    buffer:int=20,
                    bbox_color:tuple=(204,0,204),
                    label_color:tuple=(204,0,204)):
    """ Draw bounding box and label on detected hand. """
    h, w, _ = annotated_frame.shape
    landmarks = result.hand_landmarks[hand_idx]
    

    xlist = [int(lm.x*w) for lm in landmarks]
    ylist = [int(lm.y*h) for lm in landmarks]

    # bbox
    xmin, xmax = min(xlist), max(xlist)
    ymin, ymax = min(ylist), max(ylist)
    boxW, boxH = xmax - xmin, ymax - ymin
    bbox = xmin, ymin, boxW, boxH
    cx, cy = bbox[0] + (bbox[2] // 2), \
            bbox[1] + (bbox[3] // 2)

    # save bbox features
    bbox_dict = {'center': (cx,cy), 'bbox': bbox}
    
    cv2.rectangle(annotated_frame,
                  (bbox[0]-buffer, bbox[1]-buffer),
                  (bbox[0]+bbox[2]+buffer, bbox[1]+bbox[3]+buffer),
                  bbox_color,2)
    cv2.putText(annotated_frame,
                label,
                (bbox[0]-buffer-10, bbox[1]-buffer-10),
                cv2.FONT_HERSHEY_PLAIN,
                2,label_color,2)

    return annotated_frame, bbox_dict

def draw_hand_landmarks(annotated_frame:np.ndarray, result, hand_idx:int,
                        line_color:tuple=(255,255,255),
                        dot_color:tuple=(0,0,0)) -> np.ndarray:
    h, w, _ = annotated_frame.shape
    
    landmarks = result.hand_landmarks[hand_idx]
    # draw connections
    for a, b in HAND_CONNECTIONS:
        x1, y1 = int(landmarks[a].x*w), int(landmarks[a].y*h)
        x2, y2 = int(landmarks[b].x*w), int(landmarks[b].y*h)
        cv2.line(img=annotated_frame,
                 pt1=(x1,y1), pt2=(x2,y2),
                 color=line_color,thickness=2)
        
    # draw dots
    for lm in landmarks:
        cx, cy = int(lm.x*w), int(lm.y*h)
        cv2.circle(img=annotated_frame, center=(cx,cy),
                   radius=4, color=dot_color, thickness=-1)  # -1 thickness = FILLED
    
    return annotated_frame


if __name__ == "__main__":
    
    # ------ GLOBAL VARIABLES ----------------
    LANDMARKER_MODEL_PATH = './resources/hand_landmarker.task'
    CLASSIFIER_PATH = './trained_models/svm.sav'
    SCALER_PATH = './trained_models/scaler.sav'
    NUM_HANDS = 2
    CAMERA_INDEX = 0
    LABELS = [c for c in string.ascii_uppercase if c not in ("J", "Z")]
    # ----------------------------------------
    
    # initiate live-feed classifier
    livefeedclassifier = LiveFeedFSLAlphabet(model_path=LANDMARKER_MODEL_PATH,
                                             scaler_path=SCALER_PATH,
                                             classifier_path=CLASSIFIER_PATH,
                                             labels=LABELS,
                                             max_hands=NUM_HANDS)
    # webcam loop
    cap = cv2.VideoCapture(CAMERA_INDEX)
    frame_timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        bgr_frame = cv2.flip(frame,1)  # flip horizontally
        annotated_frame = livefeedclassifier.classify_hand_to_letter(
            bgr_frame, frame_timestamp_ms, show_landmarks=False
        )
        frame_timestamp_ms += 1

        if annotated_frame is not None:
            cv2.imshow("Live-Feed FSL Alphabet Classifier", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    livefeedclassifier.landmarker.close()
