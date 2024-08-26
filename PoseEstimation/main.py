# main.py

from PoseModule import PoseDetector

def main():
    video_source = 0
     # Use 0 for webcam, or replace with video file path
    detector = PoseDetector(video_source=video_source)
    detector.process_video()

if __name__ == "__main__":
    main()
