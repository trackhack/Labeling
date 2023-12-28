import os
import glob

def get_current_path() -> str:
    """Retrieves the current working directory."""
    current_path = os.getcwd()
    return current_path

def get_items(path: str, exclude_items: list[str] = None, 
              numerical_sort: bool = False) -> list[str]:
    """This function extracts all items inside the given folder (path) and 
    returns them in an alphabetically sorted list.
    You can manually exclude unwanted items by passing a list of strings
    that correspond to the item names of set items.
    In addition to that, you can pass numerical_sort as an input argument
    to make sure the elements are sorted numerically instead.
    """
    items = sorted(os.listdir(path))
    
    if exclude_items:
        # here is a more detailed explanation of what the statement does:
        # if exclude_items:
        #    for item in items:
        #        if item not in exclude_items:
        #            items.append(item)

        items = [item for item in items if item not in exclude_items]
    
    if numerical_sort:
        # here is a more detailed explanation of what the statemend does:
        # sorting the list by the numerical part of the strings
        # 
        # example strings: this_is_string_1 (1), this_is_string_2 (2), 
        #                  this_is_string_11 (3)
        # 
        # when using the sorted() method for a alphabetical sort, 
        # we would get the liste sorted in order 1,3,2
        #
        # however we want to sort the original order
        # therefore we split the strings with every underscore and check 
        # if the part is a string of numbers:
        # for part in x.split("_"):
        #     if part.isdigit()
        # 
        # we then use the numercial parts of the strings as the key argument
        # for sorting the items

        items = sorted(
            items,
            key=lambda x: [
                int(part) if part.isdigit() else part for part in x.split("_")
                ]
        )

        
    return items

def filter_items(items: list[str], 
                 filter_criteria: str | list[str]) -> list[str]: 
    """Filters elements of a provided list in respect to provided 
    filter criteria. You can pass a single filter criterion as a string
    as well as multiple filter criterions in a list of strings.
    """
    if isinstance(filter_criteria, str):
        filter_criteria = [filter_criteria]

    filtered_items = [
        item 
        for item in items 
        if any(item.endswith(criteria) for criteria in filter_criteria)
        ]
    
    return filtered_items

def count_elements(path: str) -> int:
    """Counts all elements inside a folder."""
    elements = glob.glob(os.path.join(path, "*"))
    return len(elements)
