# face_detection_module.py

import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, video_source=0, detection_confidence=0.5):
        """
        Initializes the FaceDetector with the given video source and detection confidence.

        :param video_source: Index of the webcam (0 for default webcam) or a file path for video files.
        :param detection_confidence: Confidence level for face detection.
        """
        # Initialize MediaPipe Face Detection
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=detection_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.video_source = video_source
        self.pTime = 0

    def process_video(self):
        """
        Processes video from the specified source (webcam or file) and performs face detection.
        """
        # Initialize video capture (0 for webcam, or provide a video file path)
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        while True:
            success, img = cap.read()
            if not success:
                print("Error: Could not read frame.")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.faceDetection.process(imgRGB)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)

                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            # Display the FPS
            cTime = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (cTime - self.pTime) if (cTime - self.pTime) > 0 else 0
            self.pTime = cTime

            cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

            # Show the image
            cv2.imshow("Image", img)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage for testing the module directly
if __name__ == "__main__":
    # You can test it with webcam (0) or a video file path ("path/to/video.mp4")
    detector = FaceDetector(video_source=0)  # Change to video file path if needed
    detector.process_video()
