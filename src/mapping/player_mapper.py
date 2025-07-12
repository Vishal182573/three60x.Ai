import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple

def _get_center(bbox: np.ndarray) -> np.ndarray:
    """Calculates the center of a bounding box."""
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

def map_players_in_frame(
    t_data: Dict, 
    b_data: Dict, 
    appearance_weight: float
) -> List[Tuple[int, int]]:
    """Maps players between two views for a single synchronized frame pair"""
    t_features, t_bboxes = np.array(t_data['features']), t_data['bboxes']
    b_features, b_bboxes = np.array(b_data['features']), b_data['bboxes']
    
    if t_features.shape[0] == 0 or b_features.shape[0] == 0:
        return []

    # these processing is done on each frame pair

    # appearance most matrix used to compare feature vectors of a playr
    appearance_cost = cdist(t_features, b_features, 'cosine')

    # positional cost matrix used to compare distances of players from the ball 
    position_cost = np.zeros_like(appearance_cost)
    position_weight = 1.0 - appearance_weight

    # only calculate positional cost if the ball is visible in both views, in case of missing ball, we rely 100% on appearance
    has_ball_t = t_data['ball_bbox'] is not None
    has_ball_b = b_data['ball_bbox'] is not None

    if has_ball_t and has_ball_b and position_weight > 0:
        t_norm = np.linalg.norm(t_data['frame_shape'])
        b_norm = np.linalg.norm(b_data['frame_shape'])
        
        t_ball_center = _get_center(t_data['ball_bbox'])
        b_ball_center = _get_center(b_data['ball_bbox'])

        for i, t_bbox in enumerate(t_bboxes):
            t_dist = np.linalg.norm(_get_center(t_bbox) - t_ball_center) / t_norm
            for j, b_bbox in enumerate(b_bboxes):
                b_dist = np.linalg.norm(_get_center(b_bbox) - b_ball_center) / b_norm
                position_cost[i, j] = np.abs(t_dist - b_dist)
    else:
        # If ball is missing, rely 100% on appearance code
        appearance_weight = 1.0
        position_weight = 0.0

    # total cost
    total_cost = (appearance_weight * appearance_cost) + (position_weight * position_cost)
    row_ind, col_ind = linear_sum_assignment(total_cost)
    
    return list(zip(row_ind, col_ind))