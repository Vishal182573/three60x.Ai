import cv2

MODEL_PATH = "models/player_ball_detector.pt"
TACTICAM_VIDEO_PATH = "data/tacticam.mp4"
BROADCAST_VIDEO_PATH = "data/broadcast.mp4"
# --- CHANGE THIS LINE ---
OUTPUT_VIDEO_PATH = "output/final_output.avi" 

# synchronization settings
SYNC_POINT_TACTICAM_MS = 6500
SYNC_POINT_BROADCAST_MS = 4500

# mapping parameters 
#weight given to appearance vs position. higher means appearance is more important
APPEARANCE_WEIGHT = 0.85
# Player and Ball class IDs from the yolo model
PLAYER_CLASS_ID = 0
BALL_CLASS_ID = 1

# visualization settings
BOX_COLOR = (0, 255, 0) # Green colot
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2