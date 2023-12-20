import os

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
        #return sum(1 for entry in entries)
        return sum(1 for entry in entries if entry.is_file())