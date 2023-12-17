import os
import cv2
import numpy as np
import predictor_module

def get_current_path():
    current_path = os.getcwd()
    return current_path

def get_items(path, exclude_items=None, numerical_sort=False):
    items = os.listdir(path)
    
    if exclude_items:
        items = [item for item in items if item not in exclude_items]
    
    if numerical_sort:
        items.sort(key=lambda x: [int(part) if part.isdigit() else part for part in x.split('_')])
    
    return items

def filter_items(items, filter_criteria):
    if isinstance(filter_criteria, str):
        filter_criteria = [filter_criteria]

    filtered_items = [item for item in items if any(item.endswith(criteria) for criteria in filter_criteria)]
    return filtered_items

def count_items_in_folder(path):
    with os.scandir(path) as entries:
        return sum(1 for entry in entries)

def build(path, image_list):
    annotation_img = cv2.imread(os.path.join(path, image_list[1]))
    original_img = cv2.imread(os.path.join(path, image_list[0]))

    cv2.imshow("Annotations", annotation_img)
    cv2.imshow("Mask Predictions", original_img)

    first_window_width = cv2.getWindowImageRect("Annotations")[2]
    cv2.moveWindow("Mask Predictions", first_window_width, 0) 
    return annotation_img, original_img

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

def main():

    exclude_list = exclude_list = ['.DS_Store'] ###store elements that should get filtered out. in this case: macOS element
    
    path = get_current_path()

    items = get_items(path, exclude_list)

    supervised_folder = filter_items(items, "supervised")

    o_path = os.path.join(path, supervised_folder[0])

    items = get_items(o_path, exclude_list, numerical_sort = True)

    for item in items: 
        path = os.path.join(o_path, item)
        items = get_items(path, exclude_list)
        image_list = filter_items(items, ".png")
        
        file_appendix = "_SelectedInstances_2.png"
        check_str = path + file_appendix

        if check_str in image_list: #checking if there already is an image
            #print("This folder has already been checked")
            continue

        image_list.sort(key = len) #sorting the filenames by lenght to make sure that following indices 
        
        annotation_img, original_img = build(path, image_list)
        
        predictor = predictor_module.register_sam(original_img)

        cv2.setMouseCallback("Annotations", predictor_module.on_mouse_click, predictor)
        cv2.waitKey(0)
        
        sorted_indices = np.argsort(predictor_module.scores)[::-1] 
        highest_score_index = sorted_indices[0]

        cv2.imshow("Mask Predictions", overlay_mask(original_img, predictor_module.masks[highest_score_index]))

        #print("press (n) to save the mask")
        #print("press (y) if you are done with the image")
        #print("press (esc) if you want to close the program without saving the process of the current cob")
        #print("or just click on the image to create a new mask without saving the current")

        selected_masks = []

        if predictor_module.state == 1:
            selected_masks.append(predictor_module.masks[highest_score_index])
            cv2.imshow("Mask Predictions", original_img)
            annotation_img = new_mask_marker(annotation_img, predictor_module.masks[highest_score_index], random_color())
            cv2.imshow("Annotations", annotation_img)
        elif predictor_module.state == 2:
            print("ALL DONE! Saving selected instances")
            output_path = os.path.join(path, f"masks")
            num_already_existing_masks = count_items_in_folder(output_path)
            for i in range(len(selected_masks)):
                mask_appendix = "_mask" + str(i+num_already_existing_masks) + ".png"
                cv2.imwrite(os.path.join(output_path, mask_appendix), selected_masks[i])
            cv2.imwrite(os.path.join(path, file_appendix), annotation_img)
            cv2.destroyAllWindows()
        elif predictor_module.state == -1:
            cv2.destroyAllWindows()
            return

if __name__ == "__main__":
    main()