import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry

##### module scope #####
masks = None
scores = None
state = 0

def register_sam(img = None):
    if img is None:
        print("Warning: You did not provide an image for SAM")
        print("Failed to register SAM")
        return None

    if isinstance(img, np.ndarray):
        sam = sam_model_registry["vit_h"](checkpoint = r"/Users/jantheiss/Labeling/sam_vit_h_4b8939.pth")
        predictor = SamPredictor(sam)
        predictor.set_image(img)
        return predictor
        
    else: 
        print("Warning: Invalid Data Type for initializing SAM")
        print("Failed to register SAM")
        return None

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at coordinates (x={x}, y={y})")
        predict_mask(param, x, y)
    
def predict_mask(predictor, x, y):
    global masks, scores

    masks, scores, _= predictor.predict(
        point_coords = np.array([[x,y]]),
        point_labels  = np.array([1]),
        multimask_output = True
        )
    
    #key = cv2.waitKey(0) & 0xFF

    print("press (n) to save the mask")
    print("press (y) if you are done with the image")
    print("press (esc) if you want to close the program without saving the process of the current cob")
    print("or just click on the image to create a new mask without saving the current")

    key = cv2.waitKey(0) & 0xFF
    process_key_press(key)

def process_key_press(key):
    if key == ordn('n'):
        state = 1
    elif key == ord('y'):
        state = 2
    elif key == 27:
        state = -1
    else:
        print("Invalid Key")