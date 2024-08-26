import cv2
import mediapipe as mp
import time

class PoseDetector:
    def __init__(self, video_source=0, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        # Initialize MediaPipe Pose module
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=mode,
            smooth_landmarks=smooth,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.video_source = video_source

    def process_video(self):
        # Initialize video capture (0 for webcam, or provide a video file path)
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        pTime = 0

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                print("Error: Could not read frame.")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(imgRGB)
            
            if results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            
            # Calculate and display FPS
            cTime = time.time()
            fps = int(1 / (cTime - pTime)) if (cTime - pTime) > 0 else 0
            pTime = cTime
            cv2.putText(img, f'FPS: {fps}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            
            cv2.imshow("Pose Detection", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PoseDetector(video_source=0)  # 0 for webcam
    detector.process_video()
