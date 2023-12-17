import cv2
import numpy as np
import os
import glob
from segment_anything import SamPredictor, sam_model_registry

#### GLOBAL ####
mask_index = 0
predictor = None
output_directory = "selected_masks"
img = []
m_img = []
kill = False

def build(current_path,files):
    global img, m_img 

    #getting my paths
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f"Directory '{output_directory}' created.")
        img_path = os.path.join(current_path, files[0] )
        m_img_path = os.path.join(current_path, files[1])
    else:
        print(f"Directory '{output_directory}' already exists.")
        img_path = os.path.join(current_path, files[0] )
        m_img_path = os.path.join(current_path, output_directory, f"SelectedInstances_2.png")

    print("THIS IS THE IMG PATH: ", img_path)
    print("THIS IS THE M PATH: ", m_img_path)
    
    #building the Windows
    img = cv2.imread(img_path)
    m_img = cv2.imread(m_img_path)
    cv2.imshow("IMAGE", m_img)
    cv2.imshow("MASKE", img)
    first_window_width = cv2.getWindowImageRect("IMAGE")[2]
    cv2.moveWindow("MASKE", first_window_width, 0)
    
def register_sam():
    global predictor
    #/Users/jantheiss/Labeling/sam_vit_b_01ec64.pth
    #/Users/jantheiss/Labeling/sam_vit_h_4b8939.pth
    sam = sam_model_registry["vit_h"](checkpoint = <ENTER PATH TO MODEL CHECKPOINTS>)
    predictor = SamPredictor(sam)
    predictor.set_image(img)

def random_color():
    return np.random.randint(0, 255, 3).tolist()

def overlay_mask(original_img, mask, alpha=0.5):
    mask = mask.astype(np.uint8)
    green_mask = cv2.merge([np.zeros_like(mask), 255 * (mask > 0).astype(np.uint8), np.zeros_like(mask)])
    result = cv2.addWeighted(original_img, 1 - alpha, green_mask, alpha, 0)
    return result

def new_mask(original_img, mask, color):
    mask = mask.astype(np.uint8)
    color_mask = original_img.copy()
    color_mask[mask > 0] = color
    return color_mask

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at coordinates (x={x}, y={y})")
        predict_mask(x, y)
        
def predict_mask(x, y):
    global mask_index, m_img, kill

    masks, scores, _= predictor.predict(
            point_coords = np.array([[x,y]]),
            point_labels  = np.array([1]),
            multimask_output = True
        )
    sorted_indices = np.argsort(scores)[::-1] 
    highest_score_index = sorted_indices[0]

    #overlay_result = overlay_mask(img, masks[highest_score_index])
    cv2.imshow("MASKE", overlay_mask(img, masks[highest_score_index]))

    print("press (n) to save the mask")
    print("press (y) if you are done with the image")
    print("or just click on the image to create a new mask without saving the current")
    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        selected_mask = masks[highest_score_index]
        output_path = os.path.join(output_directory, f"selected_mask_{mask_index}.png")
        cv2.imwrite(output_path, selected_mask * 255)
        print(f"Selected mask {mask_index} saved to {output_path}")
        mask_index += 1
        cv2.imshow("MASKE", img)
        m_img = new_mask(m_img, masks[highest_score_index], random_color())
        cv2.imshow("IMAGE", m_img)
    elif key == ord('y'):
        print("ALL DONE! Saving selected instances")
        output_path = os.path.join(output_directory, f"SelectedInstances_2.png")
        cv2.imwrite(output_path, m_img)
        kill = True
    else:
        print("INVALID KEY")

current_path = os.getcwd()
files = glob.glob("*.png")
build(current_path,files)
register_sam()

cv2.setMouseCallback("IMAGE", on_mouse_click)
            
while not kill:

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        kill = True
        
cv2.destroyAllWindows()
    
