import json
import numpy as np
import cv2

COURT_RoI = np.array([(345, 320), (935, 320), (1140, 720), (145, 720)], dtype=np.int32)
MIDLINE_Y = 475

def classify_player(player_bbox):
    '''
    Args:
        player_bbox(list): list of bbox for player.
    Returns:
        str: Classification for the player ['Top Half', 'Bottom Half', 'Outside'].
    '''
    x1, y1, x2, y2 = player_bbox
    center_bottom = ((x1 + x2) // 2, y2)
    
    is_inside = cv2.pointPolygonTest(COURT_RoI, center_bottom, False) >= 0
    
    if not is_inside:
        return "Outside Court"
    else:
        if center_bottom[1] < MIDLINE_Y:
            return "Top Half"
        else:
            return "Bottom Half"
        

if __name__=='__main__':
    with open('results.json') as file:
        data = json.load(file)
        for img_name, detections in data.items():
            for detection in detections:
                print(img_name, ": ", classify_player(detection["bbox"]))