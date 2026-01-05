import cv2
import numpy as np
import time


# created background for when running the program. My goal was to have a gray background in order to detect the movement from my camera
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y,x].T
    
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    img_br = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_br, lines, 0, (0, 255, 0))
    
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_br, (x1, y1), 1, (0, 255, 0), -1)
        
    return img_br