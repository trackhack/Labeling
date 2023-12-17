import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

###read image###
img = cv2.imread(<PATH TO AN IMAGE OF YOUR CHOICE>)
cv2.imshow("SCHLOSS", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

###register sam###
sam = sam_model_registry["vit_h"](checkpoint = <ENTER PATH TO MODEL CHECKPOINTS>)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)

###show masks###
for it_masks in masks:
    segmentation_mask = it_masks['segmentation']
    segmentation_mask_uint8 = segmentation_mask.astype(np.uint8) * 255
    overlay_image = img.copy()

    overlay_image[segmentation_mask] = [0, 255, 0]

    alpha = 0.5 
    cv2.addWeighted(overlay_image, alpha, img, 1 - alpha, 0, overlay_image)

    cv2.imshow("Window", overlay_image)
    cv2.waitKey(0)
    cv2.destroyWindow("Window")
