import cv2
from ultralytics import YOLO
from typing import Dict, List
import numpy as np
from config import PLAYER_CLASS_ID, BALL_CLASS_ID

def process_video_detections(video_path: str, model_path: str) -> Dict[int, Dict[str, List[np.ndarray]]]:
    """Processes a video to detect players and the ball in each frame using a YOLO model. Includes a progress indicator."""
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # get total number of frames for progress calculation
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_detections = {}
        frame_idx = 0

        # proper logging
        print(f"-> Starting detection on: {video_path}")
        print(f"   Total frames to process: {total_frames}")
        print("   Processing frames...")
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, verbose=False)[0]
            
            detections_in_frame = {'players': [], 'ball': []}
            for box in results.boxes:
                class_id = int(box.cls[0])
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                
                if class_id == PLAYER_CLASS_ID:
                    detections_in_frame['players'].append(bbox)
                elif class_id == BALL_CLASS_ID:
                    detections_in_frame['ball'].append(bbox)
            
            frame_detections[frame_idx] = detections_in_frame
            frame_idx += 1

            # PROGRESS BAR , for better user experience 
            percentage = (frame_idx / total_frames) * 100
            print(f"   Progress: {percentage:.1f}% ({frame_idx}/{total_frames})", end='\r')

            
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
    

    print("\n   ...Detection complete for this video.") # print A line to indicate completion
    return frame_detections