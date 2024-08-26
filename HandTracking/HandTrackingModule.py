# hand_tracker_module.py

import cv2
import mediapipe as mp
import time

class HandTracker:
    def __init__(self, video_source=0):
        """
        Initializes the HandTracker with the given video source.

        :param video_source: Index of the webcam (0 for default webcam) or a file path for video files.
        """
        # Initialize MediaPipe Hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.video_source = video_source
        self.pTime = 0

    def process_video(self):
        """
        Processes video from the specified source (webcam or file) and performs hand tracking.
        """
        # Initialize video capture (0 for webcam, or provide a video file path)
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
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

            # Display FPS
            cTime = time.time()
            fps = int(1 / (cTime - self.pTime)) if (cTime - self.pTime) > 0 else 0
            self.pTime = cTime

            cv2.putText(img, f'FPS: {fps}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage for testing the module directly
if __name__ == "__main__":
    # You can test it with webcam (0) or a video file path ("path/to/video.mp4")
    detector = HandTracker(video_source=0)  # Change to video file path if needed
    detector.process_video()
