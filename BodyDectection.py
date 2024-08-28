import cv2
import mediapipe as mp
import time

class FaceDetectionModule:
    def __init__(self, detection_confidence=0.5):
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=detection_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def detect_faces(self, imgRGB):
        return self.faceDetection.process(imgRGB)

class HandTrackingModule:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def track_hands(self, imgRGB):
        return self.hands.process(imgRGB)

class PoseModule:
    def __init__(self, detection_confidence=0.5, track_confidence=0.5):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=False,
            smooth_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=track_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils

    def estimate_pose(self, imgRGB):
        return self.pose.process(imgRGB)

class CombinedDetection:
    def __init__(self, video_source=0, detection_confidence=0.5, track_confidence=0.5):
        self.face_detector = FaceDetectionModule(detection_confidence=detection_confidence)
        self.hand_tracker = HandTrackingModule()
        self.pose_detector = PoseModule(detection_confidence=detection_confidence, track_confidence=track_confidence)
        self.video_source = video_source
        self.pTime = 0

    def process_video(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Error: Could not read frame.")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Face Detection
            face_results = self.face_detector.detect_faces(imgRGB)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            # Hand Tracking
            hand_results = self.hand_tracker.track_hands(imgRGB)
            if hand_results.multi_hand_landmarks:
                for handLms in hand_results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, _ = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    self.hand_tracker.mpDraw.draw_landmarks(img, handLms, self.hand_tracker.mpHands.HAND_CONNECTIONS)

            # Pose Detection
            pose_results = self.pose_detector.estimate_pose(imgRGB)
            if pose_results.pose_landmarks:
                self.pose_detector.mpDraw.draw_landmarks(img, pose_results.pose_landmarks, self.pose_detector.mpPose.POSE_CONNECTIONS)

            # Calculate and display FPS
            cTime = time.time()
            fps = int(1 / (cTime - self.pTime)) if (cTime - self.pTime) > 0 else 0
            self.pTime = cTime
            cv2.putText(img, f'FPS: {fps}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            
            cv2.imshow("Combined Detection", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = CombinedDetection(video_source=0)  # 0 for webcam
    detector.process_video()
