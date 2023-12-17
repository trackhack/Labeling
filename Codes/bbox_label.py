import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator


start_point = (-1, -1)
end_point = (-1, -1)

#flag
first_click = False

input_box = np.array([0, 0, 0, 0])

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, first_click, input_box
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if not first_click:
            start_point = (x, y)
            first_click = True
        else:
            end_point = (x, y)
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Rectangle Drawing", img)
            input_box = np.array([start_point[0], start_point[1], end_point[0], end_point[1]])
            first_click = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if first_click:
            img_copy = img.copy()
            cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("Rectangle Drawing", img_copy)

# Create a black image window
img = cv2.imread(<PATH TO YOUR IMAGE>)

# Display the initial black image
cv2.imshow("Rectangle Drawing", img)

cv2.setMouseCallback("Rectangle Drawing", mouse_callback)

while True:
    key = cv2.waitKey(1) & 0xFF

    # Press 'esc' to exit the program
    if key == 27:
        break
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(input_box)

cv2.destroyAllWindows()

###################################### SAM-PREDICTOR USING BBOX PROMT ######################################

sam = sam_model_registry["vit_h"](checkpoint = <ENTER PATH TO MODEL CHECKPOINTS>)
predictor = SamPredictor(sam)
predictor.set_image(img)
input_label = np.array([1])

masks, _, _, = predictor.predict(
    point_coords = None,
    point_labels = None,
    box = input_box[None, :],
    multimask_output = True
)

# Display each mask
for i, mask in enumerate(masks):
    cv2.imshow(f"Mask {i}", mask.astype(np.uint8) * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

############################################################################################################

####################### SAM-AUTOMATIC MASK GENERATOR USING THE CROPPED IMAGE SEGMENT #######################

#if input_box[0] != input_box[2] and input_box[1] != input_box[3]:
#    roi = img[int(input_box[1]):int(input_box[3]), int(input_box[0]):int(input_box[2])]
#    cv2.imshow("selected_rectangle.png", roi)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
#sam = sam_model_registry["vit_h"](checkpoint = <ENTER PATH TO MODEL CHECKPOINTS>)
#mask_generator = SamAutomaticMaskGenerator(sam)
#masks = mask_generator.generate(roi)

#for it_masks in masks:
#    segmentation_mask = it_masks['segmentation']
#    segmentation_mask_uint8 = segmentation_mask.astype(np.uint8) * 255
   
#    overlay_image = roi.copy()
#    overlay_image[segmentation_mask] = [0, 255, 0]  # Assuming green color for the mask
#    alpha = 0.5  # Adjust this value for the transparency of the overlay
#    cv2.addWeighted(overlay_image, alpha, roi, 1 - alpha, 0, overlay_image)

#    cv2.imshow("Window", overlay_image)
#    cv2.waitKey(0)
#    cv2.destroyWindow("Window")

############################################################################################################
