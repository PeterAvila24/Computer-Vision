# main_script.py

from HandTrackingModule import HandTracker

def main():
    # Initialize HandTracker with default webcam
    tracker = HandTracker(video_source="1.mp4")
    tracker.process_video()

if __name__ == "__main__":
    main()
