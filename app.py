import sys
import os
import cv2
import numpy as np


import config
from src.detection.video_processor import process_video_detections
from src.features.feature_extractor import FeatureExtractor
from src.mapping.player_mapper import map_players_in_frame
from scipy.spatial.distance import cdist


def main():
    """ main pipeline to run player re-identification with id persistence"""
    print("starting pipeline...") 

    # object detection using yolo model
    print("processing video detections...")
    t_detections = process_video_detections(config.TACTICAM_VIDEO_PATH, config.MODEL_PATH)
    b_detections = process_video_detections(config.BROADCAST_VIDEO_PATH, config.MODEL_PATH)
    print("   ...detection complete.")

    # initialize tools and id memory
    print("initializing feature extractor and video streams...")
    feature_extractor = FeatureExtractor()
    cap_t = cv2.VideoCapture(config.TACTICAM_VIDEO_PATH)
    cap_b = cv2.VideoCapture(config.BROADCAST_VIDEO_PATH)

    w = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_b.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

    known_player_features = {}
    next_player_id = 0
    RE_ID_THRESHOLD = 0.7

    time_offset = config.SYNC_POINT_TACTICAM_MS - config.SYNC_POINT_BROADCAST_MS
    print("initialization complete")

    # frame-by-frame processing
    print("starting frame-by-frame mapping and rendering...")
    frame_idx = 0
    frames_written = 0 #counter
    try:
        while cap_t.isOpened():
            success_t, frame_t = cap_t.read()
            if not success_t: break

            t_msec = cap_t.get(cv2.CAP_PROP_POS_MSEC)
            cap_b.set(cv2.CAP_PROP_POS_MSEC, t_msec - time_offset)
            success_b, frame_b = cap_b.read()
            if not success_b: continue

            # print(f"Processing Frame {frame_idx}... Detections: Tacticam={len(t_dets.get('players',[]))}, Broadcast={len(b_dets.get('players',[]))}")
            
            b_frame_idx = int(cap_b.get(cv2.CAP_PROP_POS_FRAMES))

            t_dets = t_detections.get(frame_idx, {})
            b_dets = b_detections.get(b_frame_idx, {})

            if not t_dets.get('players') or not b_dets.get('players'):
                out.write(frame_b)
                frames_written += 1
                frame_idx += 1
                continue

            t_features = feature_extractor.extract(frame_t, t_dets['players'])
            b_features = feature_extractor.extract(frame_b, b_dets['players'])

            t_ball_bbox = t_dets['ball'][0] if t_dets.get('ball') else None
            b_ball_bbox = b_dets['ball'][0] if b_dets.get('ball') else None
            
            t_data = {'features': t_features, 'bboxes': t_dets['players'], 'ball_bbox': t_ball_bbox, 'frame_shape': frame_t.shape[:2]}
            b_data = {'features': b_features, 'bboxes': b_dets['players'], 'ball_bbox': b_ball_bbox, 'frame_shape': frame_b.shape[:2]}

            matches = map_players_in_frame(t_data, b_data, config.APPEARANCE_WEIGHT)

            for t_idx, b_idx in matches:
                current_feature = t_data['features'][t_idx]
                
                found_match = False
                if known_player_features:
                    known_features = np.array(list(known_player_features.values()))
                    distances = cdist(np.expand_dims(current_feature, 0), known_features, 'cosine')[0]
                    
                    best_match_idx = np.argmin(distances)
                    if 1 - distances[best_match_idx] > RE_ID_THRESHOLD:
                        pid = list(known_player_features.keys())[best_match_idx]
                        known_player_features[pid] = current_feature
                        found_match = True

                if not found_match:
                    pid = next_player_id
                    known_player_features[pid] = current_feature
                    next_player_id += 1
                
                x1, y1, x2, y2 = b_data['bboxes'][b_idx]
                cv2.rectangle(frame_b, (x1, y1), (x2, y2), config.BOX_COLOR, config.BOX_THICKNESS)
                cv2.putText(frame_b, f"P-{pid}", (x1, y1 - 10), config.FONT, 
                            config.FONT_SCALE, config.BOX_COLOR, config.FONT_THICKNESS)

            out.write(frame_b)
            frames_written += 1
            frame_idx += 1
    finally:
        print("finalizing and cleaning up...")
        print(f"   total frames written to output: {frames_written}") # Final check
        cap_t.release()
        cap_b.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"   ...pipeline finished. output saved to {config.OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()