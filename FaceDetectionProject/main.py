# main.py

from FaceDetectionModule import FaceDetector  # Import the PoseDetector class

def main():
    # Path to the video file or 0 for webcam
    video_source = "videos/1.mp4"  # Update with your actual path or use 0 for webcam

    # Create an instance of PoseDetector
    detector = FaceDetector(video_source=video_source)
    
    # Process the video source (either webcam or video file)
    detector.process_video()

if __name__ == "__main__":
    main()
