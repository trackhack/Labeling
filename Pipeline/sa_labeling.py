import os
from pickletools import uint8
from pydoc import doc
import cv2
import numpy as np
import crawler_module as cm
import segment_anything
import tkinter as tk
from typing import Tuple
 
#### module scope ####
annotation_img = None
original_img = None
#selected_masks = []
original_size = None
masks = None
highest_score_index = None

def get_screen_width() -> int:
    """Using tkinter to get the maximum screen width."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    screen_width = root.winfo_screenwidth()
    root.destroy()  # Destroy the hidden window

    return screen_width

def rotate_and_resize(img: np.ndarray, recover: bool = False) -> np.ndarray:  
    """Rotates the image horizontally and rescales it to
      the maximum screen size for easier handling. Passing recover
      allows for undoing rotation and resizing in order to get the
      original image features.
    """
    global original_size
    
    if recover:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(img, (int(original_size[0]), int(original_size[1])))
    else:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        height, width, _ = (img.shape + (1,))[:3]
        original_size = (height, width)
        factor= get_screen_width() / width
        img = cv2.resize(img, (int(width*factor), int(height*factor)))

    return img

def build(path: str, image_list: list[str]):
    """Reads the images and builds window layout."""
    global annotation_img, original_img

    annotation_img = cv2.imread(os.path.join(path, image_list[1]))
    original_img = cv2.imread(os.path.join(path, image_list[0]))
    annotation_img = rotate_and_resize(annotation_img)
    original_img = rotate_and_resize(original_img)

    cv2.namedWindow("Annotations", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Mask Predictions", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Annotations", annotation_img)
    #positioning Windows underneath each other
    cv2.moveWindow("Mask Predictions", 0, annotation_img.shape[0])
    cv2.imshow("Mask Predictions", original_img)


def register_sam(img: np.ndarray = None
                 ) -> segment_anything.predictor.SamPredictor:
    """Initializing segmen anything predictor."""
    if img is None:
        print("Warning: You did not provide an image for SAM")
        print("Failed to register SAM")
        return None

    if isinstance(img, np.ndarray):
        sam = segment_anything.sam_model_registry["vit_h"](
            checkpoint = <ENTER CHECKPOINT PATH HERE>
            )
        predictor = segment_anything.SamPredictor(sam)
        predictor.set_image(img)
        return predictor
    else: 
        print("Warning: Invalid Data Type for initializing SAM")
        print("Failed to register SAM")
        return None
    
def overlay_mask(original_img: np.ndarray, 
                 mask: np.ndarray, alpha: int = 0.5) -> np.ndarray:
    """Merges a mask over an image."""
    mask = mask.astype(np.uint8)
    green_mask = cv2.merge([np.zeros_like(mask), 
                            255 * (mask > 0).astype(np.uint8), 
                            np.zeros_like(mask)]
                            )
    result = cv2.addWeighted(original_img, 1 - alpha, green_mask, alpha, 0)

    return result

def new_mask_marker(original_img: np.ndarray,
                    mask: np.ndarray, color: list[int]) -> np.ndarray:
    """Updates an image with a mask on top, similar to the function 
    overlay_mask.
    """
    mask = mask.astype(np.uint8)
    color_mask = original_img.copy()
    color_mask[mask > 0] = color

    return color_mask

def random_color() -> list[int]:
    """Get a random RGB color."""
    return np.random.randint(0, 255, 3).tolist()

def on_mouse_click(event, x, y, flags, param):
    """Defaul opencv function to handle mouseclick events."""
    if event == cv2.EVENT_LBUTTONDOWN:
        get_selected_masks(x, y, param)

def get_selected_masks(x: int, y: int, 
                       predictor: segment_anything.predictor.SamPredictor):
    """Get the mask predictions from segment anything for a mouseclick 
    position. We continue using the highest scoring mask, so we get the
    list index for this mask. Updating the "Mask Predictions" Window for
    visualization.
    """
    global masks, highest_score_index

    masks, scores, _= predictor.predict(
            point_coords = np.array([[x,y]]),
            point_labels  = np.array([1]),
            multimask_output = True
        )
    
    sorted_indices = np.argsort(scores)[::-1] 
    highest_score_index = sorted_indices[0]
    cv2.imshow("Mask Predictions", overlay_mask(original_img, 
                                                masks[highest_score_index]))

def folder_build() -> Tuple[str, list[str], list[str]]:
    """Processing folder structures to get the necessary paths for 
    further use.
    """
    exclude_list = ['.DS_Store'] 
    path = cm.get_current_path()
    items = cm.get_items(path, exclude_list)
    supervised_folder = cm.filter_items(items, "supervised")
    o_path = os.path.join(path, supervised_folder[0])
    items = cm.get_items(o_path, 
                         exclude_list, 
                         numerical_sort = True
                         )
    
    return o_path, items, exclude_list

def main():
    """Main function with the following rundown:
    1. Sourcing all necessary paths
    2. Iterating over .png of the images that need further labeling
    3. Checking if the image is already finally annotated
    4. Building cv2 Window structure
    5. Registering SAM
    6. Setting mousecallback and processing mouse clicks
    7. Saving masks and the finally annotated image
    8. Next iteration step until all images are processed
    """
    global annotation_img, masks, highest_score_index
    #1
    o_path, items, exclude_list = folder_build()
    selected_masks = []
    #2
    for item in items: 
        path = os.path.join(o_path, item)
        print(path)
        items = cm.get_items(path, exclude_list)
        image_list = cm.filter_items(items, ".png")
        #3
        file_appendix = "_SelectedInstances_2.png"
        check_str = path+file_appendix
        if os.path.basename(check_str) in image_list: 
            print("This folder has already been checked")
            continue

        image_list.sort(key = len)  
        #4
        build(path, image_list)
        #5
        predictor = register_sam(original_img)
        #6
        cv2.setMouseCallback("Annotations", on_mouse_click, predictor)
        print("press (n) to save the mask")
        print("press (y) if you are done with the image")
        print("or just click on the image to create "
              "a new mask without saving the current")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                selected_masks.append(
                rotate_and_resize(masks[highest_score_index].astype(np.uint8), 
                                  recover=True)
                )
                cv2.imshow("Mask Predictions", original_img)
                annotation_img = new_mask_marker(annotation_img, 
                                                 masks[highest_score_index],
                                                 random_color()
                                         )
                cv2.imshow("Annotations", annotation_img)
                #print(len(selected_masks))
            elif key == ord('y'):
                print("ALL DONE! Saving selected instances")
                break
            elif key == 27:
                cv2.destroyAllWindows
                quit()
        #7
        output_path = os.path.join(path, f"masks")
        num_already_existing_masks = cm.count_elements(output_path)
        for i in range(len(selected_masks)):
            mask_appendix = "_mask"+str(i+num_already_existing_masks)+".png"
            os.path.join(output_path, os.path.basename(path)+mask_appendix)
            cv2.imwrite(os.path.join(output_path, 
                                     os.path.basename(path)+mask_appendix), 
                                     selected_masks[i].astype(np.uint8)*255
                                     )
        annotation_img = rotate_and_resize(annotation_img, recover=True)
        cv2.imwrite(os.path.join(path, 
                                 os.path.basename(path)+file_appendix), 
                                 annotation_img
                                 )
        cv2.destroyAllWindows()
        selected_masks.clear()

if __name__ == "__main__":
    main()
