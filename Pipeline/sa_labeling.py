import os
import cv2
import numpy as np
import crawler_module
#import mask_handling_module as mhm
#import predictor_module
from segment_anything import SamPredictor, sam_model_registry

#### module scope ####
annotation_img = None
original_img = None
key = None
selected_masks = []
predictor = None

def build(path, image_list):
    global annotation_img, original_img

    annotation_img = cv2.imread(os.path.join(path, image_list[1]))
    original_img = cv2.imread(os.path.join(path, image_list[0]))

    cv2.imshow("Annotations", annotation_img)
    cv2.imshow("Mask Predictions", original_img)

    first_window_width = cv2.getWindowImageRect("Annotations")[2]
    cv2.moveWindow("Mask Predictions", first_window_width, 0) 

def register_sam(img = None):
    global predictor
    if img is None:
        print("Warning: You did not provide an image for SAM")
        print("Failed to register SAM")

    if isinstance(img, np.ndarray):
        sam = sam_model_registry["vit_h"](checkpoint = <ENTER PATH TO MODEL CHECKPOINTS>)
        predictor = SamPredictor(sam)
        predictor.set_image(img)
        print("SAM BUILD CORRECTLY")
        
    else: 
        print("Warning: Invalid Data Type for initializing SAM")
        print("Failed to register SAM")
    
def overlay_mask(original_img, mask, alpha=0.5):
    mask = mask.astype(np.uint8)
    green_mask = cv2.merge([np.zeros_like(mask), 255 * (mask > 0).astype(np.uint8), np.zeros_like(mask)])
    result = cv2.addWeighted(original_img, 1 - alpha, green_mask, alpha, 0)
    return result

def new_mask_marker(original_img, mask, color):
    mask = mask.astype(np.uint8)
    color_mask = original_img.copy()
    color_mask[mask > 0] = color
    return color_mask

def random_color():
    return np.random.randint(0, 255, 3).tolist()

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at coordinates (x={x}, y={y})")
        get_selected_masks(x, y)

def get_selected_masks(x, y):
    global annotation_img, selected_masks

    masks, scores, _= predictor.predict(
            point_coords = np.array([[x,y]]),
            point_labels  = np.array([1]),
            multimask_output = True
        )
    
    sorted_indices = np.argsort(scores)[::-1] 
    highest_score_index = sorted_indices[0]

    cv2.imshow("Mask Predictions", overlay_mask(original_img, masks[highest_score_index]))

    print("press (n) to save the mask")
    print("press (y) if you are done with the image")
    print("or just click on the image to create a new mask without saving the current")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        selected_masks.append(masks[highest_score_index])
        cv2.imshow("Mask Predictions", original_img)
        annotation_img = new_mask_marker(annotation_img, masks[highest_score_index], random_color())
        cv2.imshow("Annotations", annotation_img)
        print(selected_masks)

def folder_build():
    exclude_list = exclude_list = ['.DS_Store'] ###store elements that should get filtered out. in this case: macOS element
        
    path = crawler_module.get_current_path()

    items = crawler_module.get_items(path, exclude_list)

    supervised_folder = crawler_module.filter_items(items, "supervised")

    o_path = os.path.join(path, supervised_folder[0])

    items = crawler_module.get_items(o_path, exclude_list, numerical_sort = True)
    return o_path, items, exclude_list

def main():
    global selected_masks
    o_path, items, exclude_list = folder_build()

    for item in items: 
        path = os.path.join(o_path, item)
        items = crawler_module.get_items(path, exclude_list)
        image_list = crawler_module.filter_items(items, ".png")
            
        file_appendix = "_SelectedInstances_2.png"
        check_str = path + file_appendix

        if os.path.basename(check_str) in image_list: #checking if there already is an image
            print("This folder has already been checked")
            continue

        image_list.sort(key = len) #sorting the filenames by lenght to make sure that following indices 

        build(path, image_list)

        register_sam(original_img)

        cv2.setMouseCallback("Annotations", on_mouse_click)

        while True:
            
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                selected_masks = None
                break
            elif key == ord('y'):
                print("ALL DONE! Saving selected instances")
                print(selected_masks)
                break

        output_path = os.path.join(path, f"masks")
        num_already_existing_masks = crawler_module.count_items_in_folder(output_path)
        print(num_already_existing_masks)
        for i in range(len(selected_masks)):
            mask_appendix = "_mask" + str(i+num_already_existing_masks) + ".png"
            os.path.join(output_path, os.path.basename(path) + mask_appendix)
            cv2.imwrite(os.path.join(output_path, os.path.basename(path) + mask_appendix), selected_masks[i].astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(path, os.path.basename(path) + file_appendix), annotation_img)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
